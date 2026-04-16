#!/usr/bin/env python3
"""
DrivoR HUGSIM FIFO adapter (client.py)

- Reads pickled (obs, info) messages from obs_pipe
- Builds navsim.AgentInput (reusing navsim dataclasses)
- Uses DrivoRAgent.get_feature_builders() (DrivoRFeatureBuilder) to compute features
- Runs agent.forward(features) to get proposals and scores
- Writes a plan payload to plan_pipe (pickled, with 8-byte length prefix) compatible with HUGSIM

Assumptions (review before running):
- HUGSIM message format: same as RAP adapter: pickled tuple (obs, info)
  - obs['rgb'][cam_name] -> H x W x 3 uint8 images
  - info contains 'cam_params'[cam_name] with {intrinsic: {W,H,cx,cy,fovx,fovy}, l2c_rot: [3 deg], l2c_trans: [3]}
  - info contains ego_pos/ego_rot or timestamp fields used to compute velocities
- Camera mapping to DrivoR sensors (edit MAP_HUGSIM_TO_DRIVOR if needed)
"""
import argparse
import logging
import math
import os
import pickle
import struct
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import inspect

# Bridge the pytree registration API for older torch versions.
# Newer transformers expects torch>=2.2's public pytree registration name.
# DrivoR environments may still run with torch 2.1, which exposes the private helper.
try:
    from torch.utils import _pytree as _torch_pytree

    if hasattr(_torch_pytree, "_register_pytree_node"):
        _raw_register_pytree_node = _torch_pytree._register_pytree_node
        _raw_signature = inspect.signature(_raw_register_pytree_node)

        def _compat_register_pytree_node(cls, flatten_fn, unflatten_fn, **kwargs):
            supported_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in _raw_signature.parameters
            }
            return _raw_register_pytree_node(cls, flatten_fn, unflatten_fn, **supported_kwargs)

        _torch_pytree.register_pytree_node = _compat_register_pytree_node
except Exception:
    pass

# Import navsim dataclasses (AgentInput, Camera, Cameras, Lidar, EgoStatus)
# We add repo root to path based on env var DRIVOR_REPO_ROOT (set by HUGSIM launch)
DRIVOR_REPO_ROOT = os.environ.get("DRIVOR_REPO_ROOT", "")
if not DRIVOR_REPO_ROOT:
    raise RuntimeError("DRIVOR_REPO_ROOT must be set in environment")
sys.path.insert(0, str(Path(DRIVOR_REPO_ROOT).resolve()))

from navsim.common.dataclasses import AgentInput, Cameras, Camera, Lidar, EgoStatus  # type: ignore
from navsim.agents.drivoR.drivor_agent import DrivoRAgent  # type: ignore

LOG = logging.getLogger("drivor_adapter")

DEFAULT_CAM_ORDER = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Map HUGSIM camera names to navsim Cameras field names
MAP_HUGSIM_TO_DRIVOR = {
    "CAM_FRONT": "cam_f0",
    "CAM_BACK": "cam_b0",
    "CAM_FRONT_LEFT": "cam_l0",
    "CAM_FRONT_RIGHT": "cam_r0",
    "CAM_BACK_LEFT": "cam_l1",
    "CAM_BACK_RIGHT": "cam_r1",
}

# history frames used to build AgentInput (DrivoR config often expects 4)
EGO_HISTORY_FRAMES = 4

TOPK = 8


def parse_args():
    parser = argparse.ArgumentParser(description="DrivoR FIFO client for HUGSIM")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "drivor_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def read_obs(obs_pipe: Path):
    """Read pickled object from pipe: 8-byte length prefix + payload"""
    with open(obs_pipe, "rb") as pipe:
        header = pipe.read(8)
        if len(header) != 8:
            raise EOFError(f"Incomplete pipe header from {obs_pipe}")
        payload_size = struct.unpack("<Q", header)[0]
        payload = bytearray()
        while len(payload) < payload_size:
            chunk = pipe.read(payload_size - len(payload))
            if not chunk:
                raise EOFError(f"Incomplete pipe payload from {obs_pipe}")
            payload.extend(chunk)
    return pickle.loads(payload)


def write_plan(plan_pipe: Path, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plan_pipe, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def euler_deg_to_rot_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """
    Convert Euler angles in degrees to rotation matrix.
    Assumes angles are [roll, pitch, yaw] in degrees and uses R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    This ordering is a common convention but may need adjustment to match HUGSIM.
    """
    roll, pitch, yaw = np.deg2rad(angles_deg[:3])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]], dtype=np.float32)
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return R


def build_camera_from_hugsim(cam_name: str, rgb_image: np.ndarray, cam_params: Dict) -> Camera:
    """
    Build navsim.common.dataclasses.Camera from HUGSIM image and cam_params.
    cam_params expected shape:
      cam_params[cam_name]["intrinsic"] with W,H,cx,cy,fovx,fovy
      cam_params[cam_name]["l2c_rot"] (3 angles deg) and l2c_trans (3 floats) OR cam_params[cam_name]['l2c'] a 4x4 matrix
    """
    # intrinsics -> 3x3 matrix
    intr = cam_params.get("intrinsic", {})
    W = float(intr.get("W", cam_params.get("W", 800)))
    H = float(intr.get("H", cam_params.get("H", 450)))
    cx = float(intr.get("cx", intr.get("cx", W / 2.0)))
    cy = float(intr.get("cy", intr.get("cy", H / 2.0)))
    fovx = float(intr.get("fovx", 60.0))
    fovy = float(intr.get("fovy", 40.0))
    # compute fx, fy as RAP did
    fx = W / (2.0 * math.tan(math.radians(fovx) / 2.0))
    fy = H / (2.0 * math.tan(math.radians(fovy) / 2.0))
    cam_intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Compute sensor2lidar rotation and translation by inverting lidar->cam if available
    # HUGSIM sometimes provides l2c_rot and l2c_trans (degrees and meters)
    sensor2lidar_rot = None
    sensor2lidar_trans = None
    if "l2c" in cam_params:
        # assume provided full 4x4 matrix l2c
        l2c = np.array(cam_params["l2c"], dtype=np.float32)
        if l2c.shape == (4, 4):
            cam2lidar = np.linalg.inv(l2c)
            sensor2lidar_rot = cam2lidar[:3, :3].astype(np.float32)
            sensor2lidar_trans = cam2lidar[:3, 3].astype(np.float32)
    else:
        # try l2c_rot / l2c_trans keys (degrees + meters)
        rot_angles = cam_params.get("l2c_rot", None)
        trans = cam_params.get("l2c_trans", None)
        if rot_angles is not None and trans is not None:
            R_l2c = euler_deg_to_rot_matrix(rot_angles)
            t_l2c = np.array(trans, dtype=np.float32)
            # lidar -> cam : [R_l2c, t_l2c], invert to get cam -> lidar
            R_c2l = R_l2c.T
            t_c2l = -R_c2l @ t_l2c
            sensor2lidar_rot = R_c2l.astype(np.float32)
            sensor2lidar_trans = t_c2l.astype(np.float32)

    # fallback to identity / zero if missing
    if sensor2lidar_rot is None:
        sensor2lidar_rot = np.eye(3, dtype=np.float32)
    if sensor2lidar_trans is None:
        sensor2lidar_trans = np.zeros(3, dtype=np.float32)

    cam = Camera(
        image=(rgb_image if rgb_image is not None else None),
        sensor2lidar_rotation=sensor2lidar_rot,
        sensor2lidar_translation=sensor2lidar_trans,
        intrinsics=cam_intrinsic,
        distortion=None,
    )
    return cam


def compute_local_velocity(info_history: Sequence[Dict], index: int) -> np.ndarray:
    """Compute local velocity (forward, left) using RAP's approach (finite differences)"""
    if len(info_history) <= 1:
        return np.zeros(2, dtype=np.float32)

    curr_info = info_history[index]
    curr_pos = np.asarray(curr_info["ego_pos"], dtype=np.float32)
    curr_yaw = float(np.asarray(curr_info["ego_rot"], dtype=np.float32)[1])

    def forward_left_basis(yaw: float):
        forward_dir = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        left_dir = np.array([-math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        return forward_dir, left_dir

    def world_delta_to_local_components(delta_world: np.ndarray, yaw: float) -> np.ndarray:
        fwd, left = forward_left_basis(yaw)
        return np.array([float(np.dot(delta_world, fwd)), float(np.dot(delta_world, left))], dtype=np.float32)

    if index > 0:
        prev_info = info_history[index - 1]
        prev_pos = np.asarray(prev_info["ego_pos"], dtype=np.float32)
        dt = float(curr_info["timestamp"]) - float(prev_info["timestamp"])
        if dt <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        delta_world = np.array([curr_pos[0] - prev_pos[0], curr_pos[2] - prev_pos[2]], dtype=np.float32)
        return world_delta_to_local_components(delta_world, curr_yaw) / dt

    # forward diff
    next_info = info_history[index + 1]
    next_pos = np.asarray(next_info["ego_pos"], dtype=np.float32)
    dt = float(next_info["timestamp"]) - float(curr_info["timestamp"])
    if dt <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    delta_world = np.array([next_pos[0] - curr_pos[0], next_pos[2] - curr_pos[2]], dtype=np.float32)
    return world_delta_to_local_components(delta_world, curr_yaw) / dt


def compute_local_acceleration(info_history: Sequence[Dict], index: int) -> np.ndarray:
    if len(info_history) <= 2:
        return np.zeros(2, dtype=np.float32)
    curr_vel = compute_local_velocity(info_history, index)
    if index > 0:
        prev_vel = compute_local_velocity(info_history, index - 1)
        dt = float(info_history[index]["timestamp"]) - float(info_history[index - 1]["timestamp"])
        if dt <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (curr_vel - prev_vel) / dt
    next_vel = compute_local_velocity(info_history, index + 1)
    dt = float(info_history[index + 1]["timestamp"]) - float(info_history[index]["timestamp"])
    if dt <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (next_vel - curr_vel) / dt


def make_command_one_hot(command: int) -> np.ndarray:
    # HUGSIM commands: 0=right, 1=left, 2=forward
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def build_agent_input_from_hugsim(obs: Dict, info_history: List[Dict], num_history: int = EGO_HISTORY_FRAMES) -> AgentInput:
    """
    Convert HUGSIM obs + info_history into navsim.AgentInput.
    - obs is expected to be {'rgb': {cam_name: HxWx3 array}, ...}
    - info_history is a list of info dicts (most recent last) with keys: 'cam_params', 'ego_pos', 'ego_rot', 'timestamp', 'command' ...
    """
    # ensure length
    while len(info_history) < num_history:
        info_history.insert(0, info_history[0].copy())

    ego_statuses = []
    cameras_list = []
    lidars_list = []

    for idx in range(-num_history, 0):
        info = info_history[idx]
        # compute ego_pose local: DrivoR feature builder expects local ego pose; the AgentInput factory normally converts global to local
        # Here we set ego_pose to [0,0,0] (local origin) for the most recent frame and relative for older frames is not strictly required by builder.
        # Simpler: set ego_pose to [0,0,0] for every history step (builder primarily uses relative ego status).
        ego_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        ego_velocity = compute_local_velocity(info_history[-num_history:], idx + num_history)
        ego_accel = compute_local_acceleration(info_history[-num_history:], idx + num_history)
        cmd = info.get("command", -1)
        driving_command = make_command_one_hot(cmd)
        ego_status = EgoStatus(ego_pose.astype(np.float64), ego_velocity.astype(np.float32), ego_accel.astype(np.float32), driving_command)
        ego_statuses.append(ego_status)

        # Build Cameras dataclass (all camera fields)
        cam_dict = {}
        cam_params = info.get("cam_params", {})
        rgb = obs.get("rgb", {})
        # create Camera objects for navsim Cameras
        cams_kwargs = {}
        for hug_name, drv_field in MAP_HUGSIM_TO_DRIVOR.items():
            img = rgb.get(hug_name, None)
            params = cam_params.get(hug_name, {})
            cam_obj = build_camera_from_hugsim(hug_name, img, params)
            cams_kwargs[drv_field] = cam_obj
        # fill missing fields with empty Camera
        # Construct Cameras dataclass by positional order
        cameras_dataclass = Cameras(
            cam_f0=cams_kwargs.get("cam_f0"),
            cam_l0=cams_kwargs.get("cam_l0"),
            cam_l1=cams_kwargs.get("cam_l1"),
            cam_l2=cams_kwargs.get("cam_l2"),
            cam_r0=cams_kwargs.get("cam_r0"),
            cam_r1=cams_kwargs.get("cam_r1"),
            cam_r2=cams_kwargs.get("cam_r2"),
            cam_b0=cams_kwargs.get("cam_b0"),
        )
        cameras_list.append(cameras_dataclass)
        # no lidar provided -> push empty Lidar
        lidars_list.append(Lidar())

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras_list, lidars=lidars_list)


def navsim_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def build_plan_payload_from_model_output(predictions: Dict, output_num_poses: int = 8, topk: int = TOPK) -> Dict:
    """
    Identify proposals & scores in model output (attempt several common keys).
    Expected predictions dict may include:
      - 'trajectory' : tensor [B, P, T, 3]
      - 'score' or 'scores' : tensor [B, P] or [B, P]
    We handle batch size 1
    """
    # find trajectory tensor
    traj = None
    scores = None
    for key in ["trajectory", "proposals", "proposals_traj", "trajectories"]:
        if key in predictions:
            traj = predictions[key]
            break
    for key in ["score", "scores", "prob", "logits"]:
        if key in predictions:
            scores = predictions[key]
            break

    if traj is None:
        # try to find any 4D tensor
        for v in predictions.values():
            if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[-1] == 3:
                traj = v
                break

    if traj is None:
        raise RuntimeError("Model output does not contain a recognizable trajectory tensor")

    traj_np = traj.detach().cpu().numpy()
    # assume batch dim first
    if traj_np.ndim == 4:
        traj_np = traj_np[0]  # [P, T, 3]
    else:
        raise RuntimeError("Unexpected trajectory tensor shape")

    if scores is not None:
        scores_np = scores.detach().cpu().numpy()
        if scores_np.ndim == 2:
            scores_np = scores_np[0]
    else:
        # fallback: use zeros
        scores_np = np.zeros(len(traj_np), dtype=np.float32)

    # choose topk
    topk_idx = np.argsort(scores_np)[-topk:][::-1]
    selected_idx = int(topk_idx[0])
    selected_traj = traj_np[selected_idx, :output_num_poses, :]

    payload = {
        "selected_idx": selected_idx,
        "selected_score": float(scores_np[selected_idx]),
        "selected_plan": navsim_to_hugsim_plan(selected_traj),
        "topk_indices": [int(i) for i in topk_idx.tolist()],
        "topk_scores": [float(scores_np[i]) for i in topk_idx.tolist()],
        "topk_plans": [navsim_to_hugsim_plan(traj_np[i, :output_num_poses, :]) for i in topk_idx.tolist()],
    }
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    setup_logging(output_dir)
    LOG.info("Starting DrivoR adapter")

    # Env vars
    repo_root = Path(os.environ["DRIVOR_REPO_ROOT"]).expanduser().resolve()
    checkpoint = os.environ["DRIVOR_CHECKPOINT"]
    dino = os.environ.get("DRIVOR_DINO", "")
    device_name = os.environ.get("DRIVOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    LOG.info("Repo root: %s, checkpoint: %s, dino: %s, device: %s", repo_root, checkpoint, dino, device)

    # Add repo root to sys.path (already done above)
    sys.path.insert(0, str(repo_root))

    # Instantiate a DrivoR agent directly via constructor pattern used in repo.
    # Minimal constructor params to match DrivoRAgent signature: (config, lr_args, checkpoint_path, ...)
    # We will build a minimal config dict familiar to the agent. If you have a Hydra config you'd prefer,
    # replace this block with hydra compose & instantiate.
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception:
        LOG.warning("omegaconf not found; proceeding with minimal dict config where needed")

    # Minimal config: ideally you'd load your drivoR.yaml via OmegaConf to preserve all settings.
    # Try to load a drivoR config file if DRIVOR_CONFIG is specified; otherwise build a tiny config
    drivo_config = {}
    # TODO: optionally load a full hydra config here if you want; for now use defaults in agent's code path
    lr_args = {"name": "AdamW", "base_lr": 5e-4, "base_batch_size": 64}

    # Create agent instance
    # We pass in the checkpoint path; DrivoRAgent.initialize() will load it.
    agent = DrivoRAgent(config=drivo_config, lr_args=lr_args, checkpoint_path=checkpoint, progress_bar=False)
    LOG.info("DrivoRAgent instance created")

    # Initialize agent (this loads checkpoint if checkpoint_path != "")
    try:
        agent.initialize()
    except Exception:
        LOG.exception("Failed to initialize DrivoRAgent (checkpoint loading may fail)")
        return 1
    LOG.info("Agent initialized OK")

    # Set device
    # The agent's internal ModelLoader would normally pick device; here ensure model on device
    try:
        agent._drivor_model.to(device)
        agent._drivor_model.eval()
    except Exception:
        LOG.warning("Could not move model to device; continuing")

    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"

    info_history: deque = deque(maxlen=EGO_HISTORY_FRAMES)

    while True:
        try:
            message = read_obs(obs_pipe)
            if message == "Done":
                LOG.info("Received shutdown signal")
                break

            obs, info = message
            # info_history append and pad
            info_history.append(dict(info))
            while len(info_history) < EGO_HISTORY_FRAMES:
                info_history.appendleft(dict(info_history[0]))

            # Build AgentInput from HUGSIM data
            agent_input = build_agent_input_from_hugsim(obs, list(info_history), num_history=EGO_HISTORY_FRAMES)

            # Use DrivoR's feature builders (native) - get_feature_builders returns builder instances
            builders = agent.get_feature_builders()
            # DrivoRFeatureBuilder expects AgentInput; compute features
            features = {}
            for b in builders:
                # compute_features returns a dict of torch tensors (no batch dim)
                f = b.compute_features(agent_input)
                features.update(f)

            # Add batch dimension and move to device
            features_batched = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features_batched[k] = v.unsqueeze(0).to(device)
                else:
                    # if numpy arrays, convert and add batch dim
                    try:
                        t = torch.from_numpy(np.array(v))
                        features_batched[k] = t.unsqueeze(0).to(device)
                    except Exception:
                        # leave as-is if not tensor-like
                        features_batched[k] = v

            # Run model forward (DrivoRAgent.forward delegates to DrivoRModel)
            with torch.no_grad():
                try:
                    pred = agent.forward(features_batched)
                except Exception:
                    # fallback: call internal model directly
                    pred = agent._drivor_model(features_batched)

            # Build plan payload (select best proposal)
            try:
                plan_payload = build_plan_payload_from_model_output(pred, output_num_poses=agent._config.get("num_poses", 8) if hasattr(agent, "_config") and isinstance(agent._config, dict) else 8)
            except Exception as e:
                LOG.exception("Failed to interpret model output: %s", e)
                # try to salvage by examining common keys
                try:
                    if "trajectory" in pred:
                        traj = pred["trajectory"].detach().cpu().numpy()
                        if traj.ndim == 4:
                            traj_np = traj[0, 0, :, :2]  # first proposal
                            plan = navsim_to_hugsim_plan(traj_np)
                            plan_payload = {"selected_idx": 0, "selected_score": 0.0, "selected_plan": plan}
                        else:
                            raise RuntimeError("unexpected shape")
                    else:
                        raise RuntimeError("no trajectory key and fallback failed")
                except Exception:
                    LOG.exception("Fallback failed - writing None plan")
                    write_plan(plan_pipe, None)
                    continue

            write_plan(plan_pipe, plan_payload)
        except Exception:
            LOG.error("Adapter loop failed")
            LOG.error(traceback.format_exc())
            try:
                write_plan(plan_pipe, None)
            except Exception:
                LOG.error("Failed to notify HUGSIM about adapter failure")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())