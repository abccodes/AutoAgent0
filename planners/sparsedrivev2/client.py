#!/usr/bin/env python3
"""
SparseDrive HUGSIM FIFO adapter (client.py)

- Reads pickled (obs, info) messages from obs_pipe
- Builds navsim.AgentInput (reusing navsim dataclasses)
- Uses SparseDriveAgent.get_feature_builders() (SparseDriveFeatureBuilder) to compute features
- Runs the SparseDrive model to get a single trajectory proposal
- Writes a plan payload to plan_pipe (pickled, with 8-byte length prefix) compatible with HUGSIM

Assumptions (review before running):
- HUGSIM message format: same as RAP adapter: pickled tuple (obs, info)
  - obs['rgb'][cam_name] -> H x W x 3 uint8 images
  - info contains 'cam_params'[cam_name] with {intrinsic: {W,H,cx,cy,fovx,fovy}, l2c_rot: [3 deg], l2c_trans: [3]}
  - info contains ego_pos/ego_rot or timestamp fields used to compute velocities
- Camera mapping to SparseDrive sensors (edit MAP_HUGSIM_TO_SPARSEDRIVE if needed)
"""
import argparse
import logging
import math
import os
import pickle
import struct
import sys
import time
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
# SparseDrive environments may still run with torch 2.1, which exposes the private helper.
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

from scipy.spatial.transform import Rotation as SCR
from planners.common.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from planners.common.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    get_prefixed_env_value,
)

# Import navsim dataclasses (AgentInput, Camera, Cameras, Lidar, EgoStatus)
# We add repo root to path based on env var SPARSEDRIVE_REPO_ROOT (set by HUGSIM launch)
SPARSEDRIVE_REPO_ROOT = os.environ.get("SPARSEDRIVE_REPO_ROOT")
if not SPARSEDRIVE_REPO_ROOT:
    raise RuntimeError("SPARSEDRIVE_REPO_ROOT must be set in environment")
sys.path.insert(0, str(Path(SPARSEDRIVE_REPO_ROOT).resolve()))

from navsim.common.dataclasses import AgentInput, Cameras, Camera, Lidar, EgoStatus  # type: ignore
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent  # type: ignore
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig  # type: ignore

LOG = logging.getLogger("sparsedrive_adapter")

DEFAULT_CAM_ORDER = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Map HUGSIM camera names to navsim Cameras field names.
# SparseDriveConfig uses three camera views by default.
MAP_HUGSIM_TO_SPARSEDRIVE = {
    "CAM_FRONT_LEFT": "cam_l0",
    "CAM_FRONT": "cam_f0",
    "CAM_FRONT_RIGHT": "cam_r0",
}

# history frames used to build AgentInput
EGO_HISTORY_FRAMES = 4

TOPK = 10

PLAN_DT_SEC = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="SparseDrive FIFO client for HUGSIM")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    return parser.parse_args()


#helper functions to retreive environment variables, and replace them with defaults if not found
def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

def _coerce_env_value(raw_value, default_value):
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)

#automatically resolves VLM config values based on vlm_env.py
def resolve_vlm_config() -> VLMSelectorConfig:
    values = {}
    sparsedrive_python_bin = os.environ.get("SPARSEDRIVE_PYTHON_BIN")
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = sparsedrive_python_bin
        raw_value = get_prefixed_env_value(suffix, default=default_value)
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sparsedrive_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )

# no need for load_model since it's abstracted to SparseDriveAgent.initialize()

def make_command_one_hot(command: int) -> np.ndarray:
    # HUGSIM commands: 0=right, 1=left, 2=forward
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

# no need for preprocess_image because DrivoR uses its own feature builders to perform preprocessing

#helper functions for camera and ego state related calculations/transformations
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

    if rgb_image is None:
        LOG.warning(
            "Missing HUGSIM camera image for %s; using blank %dx%d black frame",
            cam_name,
            int(H),
            int(W),
        )
        rgb_image = np.zeros((int(H), int(W), 3), dtype=np.uint8)

    cam = Camera(
        image=rgb_image,
        sensor2lidar_rotation=sensor2lidar_rot,
        sensor2lidar_translation=sensor2lidar_trans,
        intrinsics=cam_intrinsic,
        distortion=None,
    )
    return cam


def save_sparsedrive_frame_to_tmp(img: Optional[np.ndarray], timestamp: object, hug_name: str) -> Optional[Path]:
    """
    Save a HUGSIM RGB frame to a temporary directory under the SparseDrive repo and
    return the saved Path. This is a skeleton helper used to implement the
    SparseDrive compatibility change: SparseDriveFeatureBuilder expects an
    on-disk `image_path` on each Camera instance.

    Implementation notes (what to implement here):
    - Create `Path(os.environ['SPARSEDRIVE_REPO_ROOT']) / 'tmp_hugsim_frames'` and
      ensure it exists.
    - Convert `img` (HxWx3 uint8 numpy) to a PIL Image and save as JPG/PNG.
    - Use `timestamp` and `hug_name` to form a unique filename
      (e.g. `{int(timestamp)}_{hug_name}.jpg`).
    - Return the Path on success or None on failure.

    This function currently contains a working minimal implementation using
    PIL if available; it's intentionally permissive so it can be replaced or
    extended in future edits.
    """
    if img is None:
        return None
    try:
        from PIL import Image  # type: ignore

        tmp_dir = Path(SPARSEDRIVE_REPO_ROOT) / "tmp_hugsim_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            ts = int(float(timestamp))
        except Exception:
            ts = int(time.time())
        fname = f"{ts}_{hug_name}.jpg"
        path = tmp_dir / fname
        # Ensure RGB uint8 ordering
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(path)
        return path
    except Exception:
        LOG.exception("Failed to save HUGSIM frame for %s", hug_name)
        return None


def sparsedrive_predict_proposals(agent, features_batched: Dict[str, object]) -> Optional[Tuple[np.ndarray, np.ndarray, object]]:
    """
    Try to run a SparseDrive-style model forward and extract a single selected
    trajectory if present. This is a compatibility shim described in the
    adapter migration notes.

    Behavior (skeleton / best-effort):
        - If agent has a SparseDrive model, call it through `agent.forward`.
    - If the model returns a tuple/list, extract `outputs = ret[0]`.
    - If `outputs` is a dict and contains key `trajectory` (Tensor [B,T,3]
      or [T,3]), convert to numpy and return (proposals, scores, outputs)
      where `proposals` is shape [1,T,3] and `scores` is [1] placeholder.
    - On any failure or if the model is not present, return None so
      the adapter can fall back to its existing `agent.forward` path.

    The function contains a minimal implementation so the adapter remains
    functional by default; expand or harden it if you switch entirely to
    SparseDrive internals.
    """
    try:
        if not hasattr(agent, "forward"):
            return None
        ret = agent.forward(features_batched)
    except:
        return None

    outputs = ret[0] if isinstance(ret, (tuple, list)) else ret
    if isinstance(outputs, dict) and "trajectory" in outputs:
        try:
            traj_t = outputs["trajectory"]
            traj_np = traj_t.detach().cpu().numpy()
            # If batch dim present, take first
            if traj_np.ndim == 3 and traj_np.shape[0] > 1:
                traj_np = traj_np[0]
            if traj_np.ndim == 3 and traj_np.shape[0] == 1:
                traj_np = traj_np[0]
            # Ensure shape [T,3] -> proposals [1,T,3]
            if traj_np.ndim == 2 and traj_np.shape[1] == 3:
                proposals = np.expand_dims(traj_np, axis=0).astype(np.float32)
                scores = np.array([0.0], dtype=np.float32)
                return proposals, scores, outputs
        except Exception:
            LOG.exception("Failed to extract 'trajectory' from SparseDrive outputs")
            return None
    return None




def forward_left_basis(yaw: float):
        forward_dir = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        left_dir = np.array([-math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        return forward_dir, left_dir

def world_delta_to_local_components(delta_world: np.ndarray, yaw: float) -> np.ndarray:
    fwd, left = forward_left_basis(yaw)
    return np.array([float(np.dot(delta_world, fwd)), float(np.dot(delta_world, left))], dtype=np.float32)


#need to figure out what this does
def timestamp_delta_seconds(prev_info: Dict[str, object], next_info: Dict[str, object], default_dt: float = 0.25) -> float:
    dt = float(next_info["timestamp"]) - float(prev_info["timestamp"])
    if dt <= 1e-6:
        return default_dt
    return dt

def compute_local_velocity(info_history: Sequence[Dict[str, object]], index: int) -> np.ndarray:
    """Compute local velocity (forward, left) using RAP's approach (finite differences)"""
    if len(info_history) <= 1:
        return np.zeros(2, dtype=np.float32)

    curr_info = info_history[index]
    curr_pos = np.asarray(curr_info["ego_pos"], dtype=np.float32)
    curr_yaw = float(np.asarray(curr_info["ego_rot"], dtype=np.float32)[1])

    if index > 0:
        prev_info = info_history[index - 1]
        prev_pos = np.asarray(prev_info["ego_pos"], dtype=np.float32)
        dt = timestamp_delta_seconds(prev_info, curr_info)
        delta_world = np.array(
            [curr_pos[0] - prev_pos[0], curr_pos[2] - prev_pos[2]],
            dtype=np.float32)
        return world_delta_to_local_components(delta_world, curr_yaw) / dt

    # forward diff
    next_info = info_history[index + 1]
    next_pos = np.asarray(next_info["ego_pos"], dtype=np.float32)
    dt = timestamp_delta_seconds(curr_info, next_info)
    delta_world = np.array(
        [next_pos[0] - curr_pos[0], next_pos[2] - curr_pos[2]], 
        dtype=np.float32,
    )
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



#comparable to build_features for RAP
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

    # Debug: log high-level obs/cam_params info to help diagnose missing images
    try:
        rgb_keys = list(obs.get("rgb", {}).keys()) if isinstance(obs, dict) else []
        LOG.info("build_agent_input_from_hugsim: num_history=%d, obs rgb keys=%s", num_history, rgb_keys)
    except Exception:
        LOG.exception("Failed to summarize obs in build_agent_input_from_hugsim")

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
        # per-frame debug: timestamp and available camera keys
        try:
            LOG.debug("Frame idx=%d timestamp=%s obs.rgb keys=%s cam_params keys=%s", idx, info.get("timestamp"), list(rgb.keys()) if isinstance(rgb, dict) else None, list(cam_params.keys()) if isinstance(cam_params, dict) else None)
        except Exception:
            LOG.exception("Failed to log frame-level debug info")

        # create Camera objects for navsim Cameras
        cams_kwargs = {}
        for hug_name, drv_field in MAP_HUGSIM_TO_SPARSEDRIVE.items():
            img = rgb.get(hug_name, None)
            params = cam_params.get(hug_name, {})
            # debug: missing/invalid image diagnostics
            if img is None:
                LOG.warning(
                    "HUGSIM missing image for %s at frame idx=%d timestamp=%s; obs.rgb keys=%s; cam_params for this cam: %s",
                    hug_name,
                    idx,
                    info.get("timestamp"),
                    list(rgb.keys()) if isinstance(rgb, dict) else None,
                    cam_params.get(hug_name),
                )
            else:
                try:
                    if isinstance(img, np.ndarray):
                        LOG.debug("HUGSIM image %s shape=%s dtype=%s min=%s max=%s", hug_name, img.shape, img.dtype, int(img.min()) if img.size else None, int(img.max()) if img.size else None)
                    else:
                        LOG.warning("HUGSIM image for %s has unexpected type %s", hug_name, type(img))
                except Exception:
                    LOG.exception("Failed to inspect image for %s", hug_name)
            # SparseDrive's pipeline expects an on-disk `image_path` on Camera
            # instances. We keep the in-memory image for debugging, but also save
            # a temp file so the feature builder can load the frame from disk.
            cam_obj = build_camera_from_hugsim(hug_name, img, params)

            try:
                tmp_path = save_sparsedrive_frame_to_tmp(img, info.get("timestamp", 0), hug_name)
                if tmp_path is not None:
                    try:
                        cam_obj.image_path = str(tmp_path)
                    except Exception:
                        cam_obj.image_path = tmp_path
            except Exception:
                LOG.exception("Failed to save temp frame for camera %s", hug_name)

            cams_kwargs[drv_field] = cam_obj

        # Ensure all Cameras dataclass fields are populated, but only with enabled cameras.
        all_fields = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        enabled_fields = ["cam_f0", "cam_l0", "cam_r0"]
        for f in all_fields:
            if f not in cams_kwargs:
                if f in enabled_fields:
                    # Enabled camera but no HUGSIM source: create black frame
                    cams_kwargs[f] = build_camera_from_hugsim(f, None, {})
                else:
                    # Disabled camera: create a black frame Camera so downstream
                    # lists do not contain `None` which would produce object-dtype
                    # arrays when stacked. Use default intrinsics fallback.
                    cams_kwargs[f] = build_camera_from_hugsim(f, None, {})
        
        # Construct Cameras dataclass by positional order
        cameras_dataclass = Cameras(
            cam_f0=cams_kwargs.get("cam_f0"),
            cam_l0=cams_kwargs.get("cam_l0"),
            cam_l1=cams_kwargs.get("cam_l1"),  # None - disabled in config
            cam_l2=cams_kwargs.get("cam_l2"),  # None - disabled in config
            cam_r0=cams_kwargs.get("cam_r0"),
            cam_r1=cams_kwargs.get("cam_r1"),  # None - disabled in config
            cam_r2=cams_kwargs.get("cam_r2"),  # None - disabled in config
            cam_b0=cams_kwargs.get("cam_b0"),
        )
        # Ensure every Camera object has an `image_path` attribute (string).
        for k, cam_obj in cams_kwargs.items():
            try:
                if not hasattr(cam_obj, "image_path") or cam_obj.image_path is None:
                    cam_obj.image_path = ""
                else:
                    cam_obj.image_path = str(cam_obj.image_path)
            except Exception:
                LOG.exception("Failed to normalize image_path for camera %s", k)
        cameras_list.append(cameras_dataclass)
        # no lidar provided -> push empty Lidar
        lidars_list.append(Lidar())

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras_list, lidars=lidars_list)

# comparable to rap_to_hugsim_plan
def sparsedrive_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def batch_and_move_feature_tree(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0).to(device)
    if isinstance(value, dict):
        return {key: batch_and_move_feature_tree(subvalue, device) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [batch_and_move_feature_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(batch_and_move_feature_tree(item, device) for item in value)
    try:
        tensor = torch.from_numpy(np.asarray(value))
        return tensor.unsqueeze(0).to(device)
    except Exception:
        return value


def prepare_sparsedrive_features(feature_builder, agent_input: AgentInput):
    features = feature_builder.compute_features(agent_input)
    targets: Dict[str, torch.Tensor] = {}
    token = None
    try:
        # Normalize any pathlib.Path objects (or other non-string scalars) to plain strings
        def _normalize_paths(obj):
            if isinstance(obj, dict):
                return {k: _normalize_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_normalize_paths(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_normalize_paths(v) for v in obj)
            try:
                from pathlib import Path as _Path
            except Exception:
                _Path = None
            if _Path is not None and isinstance(obj, _Path):
                return str(obj)
            return obj

        features = _normalize_paths(features)
        # Keep `image_path` present; SparseDrive feature builder expects it.
        features, targets, token = feature_builder.pipeline(features, targets, token, test_mode=True, vis=False)
        return features, targets, token
    except Exception:
        LOG.exception("feature_builder.pipeline failed; dumping features diagnostics")
        try:
            def _diag(prefix: str, v, depth: int = 0):
                if depth > 6:
                    return
                LOG.error("Diag %s: type=%s", prefix, type(v))
                if isinstance(v, dict):
                    for k, sv in v.items():
                        _diag(f"{prefix}.{k}", sv, depth + 1)
                elif isinstance(v, list):
                    elem_types = {type(x) for x in v}
                    LOG.error("Diag %s: list len=%d elem_types=%s", prefix, len(v), elem_types)
                    for i, item in enumerate(v[:10]):
                        _diag(f"{prefix}[{i}]", item, depth + 1)
                elif isinstance(v, np.ndarray):
                    try:
                        LOG.error("Diag %s: ndarray dtype=%s shape=%s", prefix, v.dtype, v.shape)
                    except Exception:
                        LOG.error("Diag %s: ndarray (unable to inspect)", prefix)
                else:
                    try:
                        arr = np.asarray(v)
                        LOG.error("Diag %s: asarray dtype=%s shape=%s", prefix, arr.dtype, getattr(arr, 'shape', None))
                    except Exception:
                        LOG.error("Diag %s: cannot convert to ndarray (type=%s)", prefix, type(v))

            _diag("features", features)
            # Targeted camera_feature inspection to identify non-numeric elements
            try:
                cf = features.get("camera_feature") if isinstance(features, dict) else None
                if isinstance(cf, list):
                    for fi, frame in enumerate(cf):
                        try:
                            LOG.error("Camera feature frame %d type=%s", fi, type(frame))
                            if isinstance(frame, dict):
                                for cam_k, cam_v in frame.items():
                                    try:
                                        if isinstance(cam_v, np.ndarray):
                                            LOG.error("  %s: ndarray dtype=%s shape=%s", cam_k, cam_v.dtype, cam_v.shape)
                                        elif cam_v is None:
                                            LOG.error("  %s: None", cam_k)
                                        else:
                                            LOG.error("  %s: type=%s repr=%s", cam_k, type(cam_v), repr(cam_v)[:200])
                                    except Exception:
                                        LOG.exception("Failed inspecting camera_feature.%s in frame %d", cam_k, fi)
                            else:
                                LOG.error("  frame is not dict, repr=%s", repr(frame)[:300])
                        except Exception:
                            LOG.exception("Failed inspecting camera_feature frame %d", fi)
            except Exception:
                LOG.exception("Failed targeted camera_feature diagnostics")
        except Exception:
            LOG.exception("Failed dumping feature diagnostics")
        raise

#gotta figure out what the functions up to world_points_to_current_local do

def info_to_pose(info: Dict[str, object]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = SCR.from_euler(
        "XYZ",
        np.asarray(info["ego_rot"], dtype=np.float32),
        degrees=False,
    ).as_matrix().astype(np.float32)
    pose[:3, 3] = np.asarray(info["ego_pos"], dtype=np.float32)
    return pose


def local_plan_to_world(plan_traj: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    origin = ego_pose[:3, 3]
    right_dir = ego_pose[:3, 0]
    forward_dir = ego_pose[:3, 2]
    points_world = [
        origin + float(right) * right_dir + float(forward) * forward_dir
        for right, forward in plan_traj
    ]
    return np.asarray(points_world, dtype=np.float32)


def world_points_to_current_local(points_world: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    ego_points = (np.linalg.inv(ego_pose) @ homogeneous.T).T[:, :3]
    return np.stack([ego_points[:, 0], ego_points[:, 2]], axis=1).astype(np.float32)


def path_length(plan_traj: np.ndarray) -> float:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(plan_traj, axis=0), axis=1).sum())



def curvature_cost(plan_traj: np.ndarray) -> float:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) < 3:
        return 0.0
    diffs = np.diff(plan_traj, axis=0)
    headings = np.arctan2(diffs[:, 0], np.clip(diffs[:, 1], 1e-6, None))
    heading_deltas = np.diff(headings)
    heading_deltas = np.arctan2(np.sin(heading_deltas), np.cos(heading_deltas))
    return float(np.mean(np.abs(heading_deltas)))


# def compute_q_score(
#     plan_traj: np.ndarray,
#     proposal_score_norm: float,
#     vlm_cfg: VLMSelectorConfig,
#     is_carry: bool,
# ) -> float:
#     plan_traj = np.asarray(plan_traj, dtype=np.float32)
#     if len(plan_traj) == 0:
#         return -1e6

#     path = path_length(plan_traj)
#     endpoint = plan_endpoint(plan_traj)
#     progress = max(0.0, float(endpoint[1]))
#     offcenter = abs(float(endpoint[0]))
#     curvature = curvature_cost(plan_traj)
#     shortfall = max(0.0, 1.0 - path)

#     score = 0.0
#     score += vlm_cfg.q_weight_rap_score * float(proposal_score_norm)
#     score += vlm_cfg.q_weight_progress * progress
#     score -= vlm_cfg.q_weight_offcenter * offcenter
#     score -= vlm_cfg.q_weight_curvature * curvature
#     score -= vlm_cfg.q_weight_shortplan * shortfall
#     if is_carry and vlm_cfg.q_carry_score_decay > 0.0:
#         score -= vlm_cfg.q_carry_score_decay
#     return float(score)


def build_carry_plan_candidate(
    previous_plan: Optional[np.ndarray],
    previous_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_timestamp: Optional[float],
    current_info: Dict[str, object],
    vlm_cfg: VLMSelectorConfig,
) -> Optional[Dict[str, object]]:
    if not vlm_cfg.carry_previous_enabled or previous_plan is None or previous_pose is None or previous_timestamp is None:
        return None

    current_timestamp = float(current_info.get("timestamp", previous_timestamp))
    elapsed_sec = max(0.0, current_timestamp - float(previous_timestamp))
    elapsed_pose_steps = int(round(elapsed_sec / PLAN_DT_SEC))
    if elapsed_pose_steps >= len(previous_plan):
        return None

    trimmed_plan = np.asarray(previous_plan[elapsed_pose_steps:], dtype=np.float32)
    if len(trimmed_plan) < vlm_cfg.carry_previous_min_points:
        return None

    points_world = local_plan_to_world(trimmed_plan, np.asarray(previous_pose, dtype=np.float32))
    current_local = world_points_to_current_local(points_world, info_to_pose(current_info))

    valid_mask = current_local[:, 1] > 0.0
    if not np.any(valid_mask):
        return None
    first_valid_idx = int(np.argmax(valid_mask))
    current_local = current_local[first_valid_idx:]

    if len(current_local) < vlm_cfg.carry_previous_min_points:
        return None
    if path_length(current_local) < vlm_cfg.carry_previous_min_path_m:
        return None

    return {
        "source": "carry_prev",
        "proposal_index": None,
        "proposal_score": 0.0,
        "proposal_score_norm": 0.0,
        "origin_selected_score_raw": None if previous_selected_score is None else float(previous_selected_score),
        "local_plan": current_local.astype(np.float32),
        "execution_plan": current_local.astype(np.float32),
        "carry_elapsed_sec": elapsed_sec,
        "carry_elapsed_pose_steps": elapsed_pose_steps,
    }


def truncate_plan(plan_traj: np.ndarray, num_poses: int) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if num_poses <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(plan_traj[: min(len(plan_traj), int(num_poses))], dtype=np.float32)

def plan_endpoint(plan_traj: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(plan_traj[-1], dtype=np.float32)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.float32)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max - score_min < 1e-6:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - score_min) / (score_max - score_min)


def get_default_trajectories(num_poses: int) -> np.ndarray:
    num_poses = max(2, int(num_poses))
    t = np.linspace(0.0, 1.0, num_poses, dtype=np.float32)
    forward = np.stack([np.zeros_like(t), 40.0 * t], axis=1)
    slight_left = np.stack([-5.0 * (t ** 2), 38.0 * t], axis=1)
    slight_right = np.stack([5.0 * (t ** 2), 38.0 * t], axis=1)
    sharp_left = np.stack([-25.0 * (t ** 3), 30.0 * t], axis=1)
    sharp_right = np.stack([25.0 * (t ** 3), 30.0 * t], axis=1)
    return np.stack([forward, slight_left, slight_right, sharp_left, sharp_right], axis=0).astype(np.float32)

#reads from HUGSIM FIFO pipe
def read_obs(obs_pipe: Path):
    """ pickled object from pipe: 8-byte length prefix + payload"""
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


def read_obs_file(pipe):
    header = pipe.read(8)
    if len(header) != 8:
        raise EOFError("Incomplete pipe header from open obs pipe handle")
    payload_size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < payload_size:
        chunk = pipe.read(payload_size - len(payload))
        if not chunk:
            raise EOFError("Incomplete pipe payload from open obs pipe handle")
        payload.extend(chunk)
    return pickle.loads(payload)

#writes DrivoR plan back to HUGSIM
def write_plan(plan_pipe: Path, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plan_pipe, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def write_plan_file(pipe, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()

# ============================================================================
# NEW FUNCTIONS: RAP-compatible candidate generation and payload building
# ============================================================================

def extract_proposals_and_scores_from_predictions(
    predictions: Dict,
    output_num_poses: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract trajectory and score tensors from model predictions and normalize to [P,T,3] and [P].
    
    Mirrors model output extraction from build_plan_payload_from_model_output,
    but returns only proposals and scores without payload construction.
    
    Returns:
        proposals: np.ndarray of shape [P, T, 3] (proposals, timesteps, xyz with padded heading)
        scores: np.ndarray of shape [P] (proposal scores)
    """
    # Prefer the full proposal set over the already-selected single trajectory.
    traj = None
    scores = None
    for key in ["proposals", "proposals_traj", "trajectories", "trajectory"]:
        if key in predictions:
            traj = predictions[key]
            break
    for key in ["pdm_score", "score", "scores", "prob", "logits"]:
        if key in predictions:
            scores = predictions[key]
            break

    # Fallback: search for any tensor with right dimensionality
    if traj is None:
        for v in predictions.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[-1] in [2, 3]:
                traj = v
                break

    if traj is None:
        raise RuntimeError(f"Model output has no trajectory tensor. Available keys: {list(predictions.keys())}")

    # Convert to numpy and normalize shape to [P, T, D]
    traj_np = traj.detach().cpu().numpy()
    if traj_np.ndim == 4:
        traj_np = traj_np[0]  # [B, P, T, D] -> [P, T, D]
    elif traj_np.ndim == 3:
        pass  # Already [P, T, D]
    else:
        raise RuntimeError(f"Unexpected trajectory shape: {traj_np.shape}, expected [B,P,T,D] or [P,T,D]")

    # Pad heading dimension if needed
    if traj_np.shape[-1] == 2:
        traj_np = np.pad(traj_np, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

    if scores is not None:
        scores_np = scores.detach().cpu().numpy()
        if scores_np.ndim == 2:
            scores_np = scores_np[0]  # [B, P] -> [P]
    else:
        scores_np = np.zeros(len(traj_np), dtype=np.float32)

    if scores_np.ndim != 1:
        scores_np = np.asarray(scores_np).reshape(-1)

    return traj_np, scores_np


def build_sparsedrive_candidate_rows(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    vlm_cfg: VLMSelectorConfig,
    current_info: Dict[str, object],
    previous_selected_plan: Optional[np.ndarray],
    previous_selected_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_selected_timestamp: Optional[float],
    previous_selected_source: Optional[str],
) -> Tuple[List[Dict[str, object]], bool]:
    """
    Build candidate rows from top-k proposals, carry_prev, and optional default fallbacks.
    Mirrors RAP's build_vlm_candidate_rows but adapted for SparseDrive model outputs.
    
    All returned plans are in HUGSIM format [x_right, y_forward] and truncated to shared_horizon.
    
    Returns:
        candidate_rows: List of dicts with keys: source, proposal_index, proposal_score, 
                       proposal_score_norm, local_plan, execution_plan, q_score
        allow_carry_previous: bool indicating if carry_prev was allowed and generated
    """
    # Check if we can carry previous plan
    allow_carry_previous = not (
        previous_selected_source is not None
        and str(previous_selected_source).startswith("default_fallback_")
    )

    # Build carry_prev candidate if eligible
    carry_candidate = build_carry_plan_candidate(
        previous_plan=previous_selected_plan if allow_carry_previous else None,
        previous_pose=previous_selected_pose if allow_carry_previous else None,
        previous_selected_score=previous_selected_score if allow_carry_previous else None,
        previous_timestamp=previous_selected_timestamp if allow_carry_previous else None,
        current_info=current_info,
        vlm_cfg=vlm_cfg,
    )

    # Build top-k candidates from current model output
    sorted_indices = np.argsort(scores)[::-1]
    # [PLACEHOLDER] vlm_cfg.candidate_limit may need to be added to VLMSelectorConfig if not present
    # For now, fallback to a sensible default if missing
    candidate_limit = getattr(vlm_cfg, "candidate_limit", 8)
    carry_slot_count = 1 if carry_candidate is not None else 0
    current_candidate_limit = max(1, int(candidate_limit) - carry_slot_count)
    current_candidate_limit = min(current_candidate_limit, int(len(sorted_indices)))
    candidate_indices = sorted_indices[:current_candidate_limit]

    candidate_rows: List[Dict[str, object]] = []

    # Insert carry_prev if valid
    if carry_candidate is not None:
        candidate_rows.append(carry_candidate)

    # Add top-k current proposals (converted to HUGSIM coords)
    for idx in candidate_indices:
        full_plan = sparsedrive_to_hugsim_plan(proposals[idx, :output_num_poses])
        candidate_rows.append(
            {
                "source": "current_sparsedrive",
                "proposal_index": int(idx),
                "proposal_score": float(scores[idx]),
                "local_plan": full_plan,
                "execution_plan": full_plan.copy(),
            }
        )

    # Optionally add default fallback candidates
    if getattr(vlm_cfg, "include_default_candidates", False):
        for default_idx, default_plan in enumerate(get_default_trajectories(output_num_poses)):
            candidate_rows.append(
                {
                    "source": f"default_fallback_{default_idx}",
                    "proposal_index": None,
                    "proposal_score": 0.0,
                    "local_plan": default_plan,
                    "execution_plan": default_plan.copy(),
                }
            )

    # Compute shared horizon (min of carry_prev length or output_num_poses)
    carry_row = next((row for row in candidate_rows if row.get("source") == "carry_prev"), None)
    shared_horizon = len(carry_row["local_plan"]) if carry_row is not None else output_num_poses
    shared_horizon = max(1, int(shared_horizon))

    # Truncate all local_plan to shared_horizon; preserve full execution_plan
    for row in candidate_rows:
        execution_plan = np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32)
        row["execution_plan"] = execution_plan
        row["local_plan"] = truncate_plan(execution_plan, shared_horizon)

    return candidate_rows, allow_carry_previous


def build_plan_payload(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    selected_idx: Optional[int] = None,
    selected_source: str = "sparsedrive_argmax",
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int = TOPK,
) -> Dict[str, object]:
    """
    Build complete plan payload with candidate pool, defaults, and VLM metadata.
    Mirrors RAP's build_plan_payload but adapted for SparseDrive models.
    
    Handles both direct proposal selection and VLM-selected plan override.
    """
    # Determine topk and extract top indices
    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]

    # Determine selected_idx
    if selected_idx is None and selected_plan_override is None:
        selected_idx = int(top_indices[0])
    else:
        selected_idx = None if selected_idx is None else int(selected_idx)

    # Determine selected plan and score
    if selected_plan_override is not None:
        selected_plan = np.asarray(selected_plan_override, dtype=np.float32)
        selected_score = float(selected_score_override) if selected_score_override is not None else None
    else:
        assert selected_idx is not None
        selected_traj = proposals[selected_idx, :output_num_poses]
        selected_plan = sparsedrive_to_hugsim_plan(selected_traj)
        selected_score = float(scores[selected_idx])

    # Build candidate pool (either from provided rows or derived from top-k)
    if candidate_pool_rows is not None:
        candidate_pool_plans = [
            np.asarray(row["local_plan"], dtype=np.float32).tolist()
            for row in candidate_pool_rows
        ]
        candidate_pool_execution_plans = [
            np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32).tolist()
            for row in candidate_pool_rows
        ]
        candidate_pool_scores = [float(row.get("proposal_score", 0.0)) for row in candidate_pool_rows]
        candidate_pool_q_scores = [
            None if row.get("q_score") is None else float(row["q_score"])
            for row in candidate_pool_rows
        ]
        candidate_pool_sources = [str(row.get("source", "current_drivor")) for row in candidate_pool_rows]
        candidate_pool_proposal_indices = [
            None if row.get("proposal_index") is None else int(row["proposal_index"])
            for row in candidate_pool_rows
        ]
    else:
        # Fallback: derive candidate pool from top-k proposals
        candidate_pool_plans = [
            sparsedrive_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = ["current_sparsedrive" for _ in top_indices]
        candidate_pool_proposal_indices = [int(idx) for idx in top_indices]

    # Build default overlay plans if requested
    default_overlay_plans = None
    default_overlay_sources = None
    if bool(selection_debug and selection_debug.get("display_default_trajectories")):
        default_overlay_plans = [traj.tolist() for traj in get_default_trajectories(output_num_poses)]
        default_overlay_sources = [f"default_fallback_{idx}" for idx in range(len(default_overlay_plans))]

    # Assemble final payload
    payload = {
        "selected_idx": selected_idx,
        "selected_score": selected_score,
        "selected_source": selected_source,
        "selected_plan": selected_plan,
        "topk_indices": [int(idx) for idx in top_indices],
        "topk_scores": [float(scores[idx]) for idx in top_indices],
        "topk_plans": [
            sparsedrive_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ],
        "candidate_pool_plans": candidate_pool_plans,
        "candidate_pool_execution_plans": candidate_pool_execution_plans,
        "candidate_pool_scores": candidate_pool_scores,
        "candidate_pool_q_scores": candidate_pool_q_scores,
        "candidate_pool_sources": candidate_pool_sources,
        "candidate_pool_proposal_indices": candidate_pool_proposal_indices,
        "default_overlay_plans": default_overlay_plans,
        "default_overlay_sources": default_overlay_sources,
    }

    # Merge selection_debug metadata into payload
    if selection_debug:
        payload.update(selection_debug)

    return payload


def build_plain_sparsedrive_plan_result(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
) -> Dict[str, object]:
    """
    Plain argmax fallback for SparseDrive when VLM is disabled.
    Returns the selected_plan, selected_score, selected_score_raw, selected_row, and a plan_payload.
    """
    best_idx = int(np.argmax(scores))
    selected_traj = proposals[best_idx, :output_num_poses]
    selected_plan = sparsedrive_to_hugsim_plan(selected_traj)
    selected_score = float(scores[best_idx])
    selected_score_raw = float(selected_score)
    plan_payload = build_plan_payload(
        proposals=proposals,
        scores=scores,
        output_num_poses=output_num_poses,
        selected_idx=best_idx,
        selected_source="sparsedrive_argmax",
        selection_debug={
            "vlm_invoked": False,
            "display_default_trajectories": False,
            "include_default_candidates": False,
        },
        selected_plan_override=selected_plan,
        selected_score_override=selected_score,
        candidate_pool_rows=None,
        topk=TOPK,
    )
    return {
        "selected_plan": selected_plan,
        "selected_score": selected_score,
        "selected_score_raw": selected_score_raw,
        "selected_row": {"source": "current_sparsedrive", "proposal_index": best_idx},
        "plan_payload": plan_payload,
    }




def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    setup_logging(output_dir)
    LOG.info("Starting SparseDrive adapter")

    # Env vars
    repo_root = Path(os.environ.get("SPARSEDRIVE_REPO_ROOT", os.environ.get("DRIVOR_REPO_ROOT", ""))).expanduser().resolve()
    checkpoint = os.environ.get("SPARSEDRIVE_CHECKPOINT", os.environ.get("DRIVOR_CHECKPOINT", ""))
    dino = os.environ.get("SPARSEDRIVE_DINO", "")
    device_name = os.environ.get("SPARSEDRIVE_DEVICE", os.environ.get("DRIVOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)

    LOG.info("Repo root: %s, checkpoint: %s, dino: %s, device: %s", repo_root, checkpoint, dino, device)

    # Add repo root to sys.path (already done above)
    sys.path.insert(0, str(repo_root))

    # Instantiate a SparseDrive agent directly via constructor pattern used in repo.
    try:
        from omegaconf import OmegaConf  # type: ignore
        omega_available = True
    except Exception:
        omega_available = False
        LOG.warning("omegaconf not available; Hydra configs cannot be composed here")

    # Load SparseDrive config if provided; otherwise use the repo defaults.
    sparsedrive_config = SparseDriveConfig()
    sparsedrive_config_path = os.environ.get("SPARSEDRIVE_CONFIG").strip()
    LOG.info("SPARSEDRIVE_CONFIG env: '%s', OmegaConf available: %s", sparsedrive_config_path, omega_available)

    if sparsedrive_config_path:
        if omega_available:
            try:
                loaded = OmegaConf.load(sparsedrive_config_path)
                loaded = OmegaConf.to_container(loaded, resolve=True)
                LOG.info("Loaded SparseDrive config from %s (type=%s)", sparsedrive_config_path, type(loaded))
            except Exception:
                LOG.exception("Failed to load SPARSEDRIVE_CONFIG=%s; falling back to defaults", sparsedrive_config_path)
                loaded = {}
        else:
            # Try to load as plain YAML to surface parse errors for easier debugging
            try:
                import yaml  # type: ignore

                with open(sparsedrive_config_path, "r") as f:
                    loaded = yaml.safe_load(f)
                LOG.info("Loaded plain YAML SPARSEDRIVE_CONFIG from %s (type=%s)", sparsedrive_config_path, type(loaded))
            except Exception:
                LOG.exception("Failed to read SPARSEDRIVE_CONFIG=%s as YAML; falling back to defaults", sparsedrive_config_path)
                loaded = {}
    else:
        LOG.info("No SPARSEDRIVE_CONFIG specified; using default SparseDriveConfig.")

    if isinstance(sparsedrive_config_path, str) and sparsedrive_config_path:
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key == "trajectory_sampling" and isinstance(value, dict):
                    if "time_horizon" in value:
                        sparsedrive_config.trajectory_sampling.time_horizon = float(value["time_horizon"])
                    if "interval_length" in value:
                        sparsedrive_config.trajectory_sampling.interval_length = float(value["interval_length"])
                elif hasattr(sparsedrive_config, key):
                    setattr(sparsedrive_config, key, value)

    for attr_name in ("path_anchor", "velocity_anchor", "trajectory_anchor", "bkb_path"):
        attr_value = getattr(sparsedrive_config, attr_name)
        if isinstance(attr_value, str) and not os.path.isabs(attr_value):
            setattr(sparsedrive_config, attr_name, str((repo_root / attr_value).resolve()))

    LOG.info("Creating SparseDriveAgent with config type=%s", type(sparsedrive_config))
    agent = SparseDriveAgent(config=sparsedrive_config, lr=5e-4, checkpoint_path=checkpoint)
    LOG.info("SparseDriveAgent created")

    feature_builder = agent.get_feature_builders()[0]

    # Initialize agent (this loads checkpoint if checkpoint_path != "")
    try:
        agent.initialize()
    except Exception:
        LOG.exception("Failed to initialize SparseDriveAgent (checkpoint loading may fail)")
        return 1
    LOG.info("Agent initialized OK")

    # Set device
    try:
        current_model_device = "unknown"
        try:
            first_param = next(agent.parameters())
            current_model_device = str(first_param.device)
        except StopIteration:
            current_model_device = "no-parameters"
        except Exception:
            LOG.exception("Failed to inspect SparseDrive model device before inference")

        LOG.info("SparseDrive model device before optional move: current=%s target=%s", current_model_device, device)
        target_device = str(device)
        if current_model_device.startswith(target_device) or (
            target_device.startswith("cuda") and current_model_device.startswith("cuda")
        ):
            LOG.info("Skipping redundant SparseDrive model.to(%s)", device)
        else:
            LOG.info("Moving SparseDrive model to device %s", device)
            agent.to(device)
        agent.eval()
        try:
            first_param = next(agent.parameters())
            LOG.info("SparseDrive model ready for inference on device %s", first_param.device)
        except Exception:
            LOG.info("SparseDrive model ready for inference")
    except Exception:
        LOG.warning("Could not move model to device; continuing")

    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    LOG.info("Waiting for scene FIFOs to appear: obs=%s plan=%s", obs_pipe, plan_pipe)
    while not obs_pipe.exists() or not plan_pipe.exists():
        time.sleep(0.1)
    obs_pipe_reader = os.fdopen(os.open(obs_pipe, os.O_RDWR), "rb", buffering=0)
    plan_pipe_writer = os.fdopen(os.open(plan_pipe, os.O_RDWR), "wb", buffering=0)
    LOG.info("Opened persistent scene FIFOs for obs and plan exchange")

    info_history: deque[Dict[str, object]] = deque(maxlen=EGO_HISTORY_FRAMES)

    # VLM selector setup
    vlm_cfg = resolve_vlm_config()
    vlm_selector = VLMPlanSelector(vlm_cfg, output_dir)
    vlm_selector.preload()
    LOG.info(
        "VLM selector configured enabled=%s intervention_enabled=%s backend=%s device=%s",
        getattr(vlm_cfg, "enabled", False),
        getattr(vlm_cfg, "intervention_enabled", False),
        getattr(vlm_cfg, "backend", "unknown"),
        getattr(vlm_cfg, "device", "unknown"),
    )
    frame_index = 0
    previous_selected_plan: Optional[np.ndarray] = None
    previous_selected_pose: Optional[np.ndarray] = None
    previous_selected_score: Optional[float] = None
    previous_selected_timestamp: Optional[float] = None
    previous_selected_source: Optional[str] = None

    try:
        LOG.info("Entering adapter read loop; waiting for observations on %s", obs_pipe)
        LOG.info("Attempting to read preflight diagnostic message")
        message = read_obs_file(obs_pipe_reader)
        if isinstance(message, dict) and message.get("message_type") == "hugsim_preflight":
                    LOG.info(
                        "Received HUGSIM preflight diagnostic: output_dir=%s obs_pipe=%s plan_pipe=%s include_privileged_pipe=%s camera_count=%s timestamp=%s",
                        message.get("output_dir"),
                        message.get("obs_pipe"),
                        message.get("plan_pipe"),
                        message.get("include_privileged_pipe"),
                        message.get("camera_count"),
                        message.get("timestamp"),
                    )
        
        else:
            LOG.info("preflight unsuccessful...message: %s, message type: %s", message, type(message))
        while True:
            try:
                LOG.info("Waiting for next observation payload on %s", obs_pipe)
                message = read_obs_file(obs_pipe_reader)
                LOG.info("Received observation payload from %s; type=%s", obs_pipe, type(message))
                try:
                    if message == "Done":
                        LOG.info("Received shutdown signal")
                        break

                    # Handle stray preflight diagnostics that may appear here
                    if isinstance(message, dict) and message.get("message_type") == "hugsim_preflight":
                        LOG.info("Received HUGSIM preflight diagnostic (inside read loop); skipping")
                        continue

                    # Accept either (obs, info) or (obs, info, privileged_info)
                    if isinstance(message, (tuple, list)):
                        msg_len = len(message)
                        LOG.info("Observation message is sequence length=%d", msg_len)
                        if msg_len == 2:
                            obs, info = message
                        elif msg_len == 3:
                            obs, info, privileged_msg = message
                            # attach privileged info to info dict if possible for downstream code
                            try:
                                if isinstance(info, dict):
                                    info["privileged"] = privileged_msg
                            except Exception:
                                LOG.exception("Failed to attach privileged info to info dict")
                        else:
                            LOG.error("Unexpected observation sequence length=%d; skipping", msg_len)
                            write_plan_file(plan_pipe_writer, None)
                            continue
                    else:
                        LOG.error("Unexpected observation message type %s; skipping", type(message))
                        write_plan_file(plan_pipe_writer, None)
                        continue
                except Exception:
                    LOG.exception("Failed to parse incoming observation message")
                    write_plan_file(plan_pipe_writer, None)
                    continue
                # info_history append and pad
                info_history.append(dict(info))
                while len(info_history) < EGO_HISTORY_FRAMES:
                    info_history.appendleft(dict(info_history[0]))

                # Build AgentInput from HUGSIM data
                agent_input = build_agent_input_from_hugsim(obs, list(info_history), num_history=EGO_HISTORY_FRAMES)

                # SparseDrive feature pipeline: compute raw features, then preprocess them.
                features, targets, token = prepare_sparsedrive_features(feature_builder, agent_input)
                features_batched = batch_and_move_feature_tree(features, device)
                targets_batched = batch_and_move_feature_tree(targets, device)

                with torch.no_grad():
                    # Debug: log shapes and dtypes of features before forwarding to the model
                    try:
                        def _log_feature_tree(prefix: str, value):
                            if isinstance(value, torch.Tensor):
                                LOG.info("Feature '%s': tensor shape=%s dtype=%s", prefix, tuple(value.shape), value.dtype)
                            elif isinstance(value, dict):
                                for key, subvalue in value.items():
                                    _log_feature_tree(f"{prefix}.{key}", subvalue)
                            else:
                                LOG.info("Feature '%s': type=%s", prefix, type(value))

                        for key, value in features_batched.items():
                            _log_feature_tree(key, value)
                    except Exception:
                        LOG.exception("Failed to iterate features_batched for debug")

                    output_num_poses = int(agent._config.trajectory_sampling.num_poses)

                    proposals = None
                    scores = None
                    pred = None

                    sparse_try = sparsedrive_predict_proposals(agent, features_batched)
                    if sparse_try is not None:
                        proposals, scores, pred = sparse_try
                    else:
                        try:
                            pred = agent.forward(features_batched, targets_batched)
                        except Exception:
                            LOG.exception("agent.forward failed; dumping feature diagnostics and calling internal model")
                            try:
                                for k, v in features_batched.items():
                                    if isinstance(v, torch.Tensor):
                                        LOG.error("DIAG feature '%s' shape=%s dtype=%s", k, tuple(v.shape), v.dtype)
                                    else:
                                        LOG.error("DIAG feature '%s' type=%s", k, type(v))
                            except Exception:
                                LOG.exception("Failed diag dump of features_batched")

                if proposals is None or scores is None:
                    try:
                        proposals, scores = extract_proposals_and_scores_from_predictions(pred, output_num_poses=output_num_poses)
                    except Exception as e:
                        LOG.exception("Failed to extract proposals/scores from model output: %s", e)
                        write_plan_file(plan_pipe_writer, None)
                        continue

                # Build candidate rows (includes carry_prev, top-k, and optional defaults)
                try:
                    candidate_rows, allow_carry_prev = build_sparsedrive_candidate_rows(
                        proposals=proposals,
                        scores=scores,
                        output_num_poses=output_num_poses,
                        vlm_cfg=vlm_cfg,
                        current_info=info,
                        previous_selected_plan=previous_selected_plan,
                        previous_selected_pose=previous_selected_pose,
                        previous_selected_score=previous_selected_score,
                        previous_selected_timestamp=previous_selected_timestamp,
                        previous_selected_source=previous_selected_source,
                    )
                except Exception as e:
                    LOG.exception("Failed to build candidate rows: %s", e)
                    write_plan_file(plan_pipe_writer, None)
                    continue

                # Determine default selection for VLM fallback
                # Find the index of the model's best candidate (argmax by score)
                try:
                    best_idx = int(np.argmax(scores))
                    default_selected_index = None
                    for idx, row in enumerate(candidate_rows):
                        if row.get("proposal_index") is not None and int(row.get("proposal_index")) == best_idx:
                            default_selected_index = idx
                            break
                    if default_selected_index is None:
                        default_selected_index = 0  # Fallback to first candidate
                    default_selected_source = "sparsedrive_argmax"
                except Exception as e:
                    LOG.exception("Failed to determine default selection: %s", e)
                    default_selected_index = 0
                    default_selected_source = "sparsedrive_argmax"

                # Call VLM selector (or use default if disabled)
                try:
                    # If VLM disabled, use plain argmax fallback similar to RAP's plain result
                    if not getattr(vlm_cfg, "enabled", False):
                        plain_result = build_plain_sparsedrive_plan_result(proposals, scores, output_num_poses)
                        selected_plan = np.asarray(plain_result["selected_plan"], dtype=np.float32)
                        selected_score = float(plain_result["selected_score"])
                        selected_score_raw = float(plain_result.get("selected_score_raw", selected_score))
                        selected_idx = int(plain_result["selected_row"]["proposal_index"]) if plain_result["selected_row"].get("proposal_index") is not None else None
                        selected_source = "sparsedrive_argmax"
                        selection_debug = {
                            "vlm_invoked": False,
                            "fallback_selected_idx": int(default_selected_index),
                            "fallback_selected_source": default_selected_source,
                            "display_default_trajectories": bool(getattr(vlm_cfg, "display_default_trajectories", False)),
                            "include_default_candidates": bool(getattr(vlm_cfg, "include_default_candidates", False)),
                        }
                        
                    else:
                        camera_images = obs.get("rgb", {}) if isinstance(obs, dict) else {}
                        selection_result = vlm_selector.maybe_select(
                            frame_index=frame_index,
                            camera_images=camera_images,
                            info=info,
                            candidate_rows=candidate_rows,
                            default_selected_index=default_selected_index,
                            default_selected_source=default_selected_source,
                        )
                        frame_index += 1

                        selected_row = selection_result["selected_candidate_row"]
                        selected_plan = np.asarray(
                            selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32
                        )
                        selected_idx = selected_row.get("proposal_index")
                        selected_score = float(selected_row.get("proposal_score", 0.0))
                        # origin_selected_score_raw if present preserves raw value for carry logic
                        selected_score_raw = (
                            float(selected_row.get("origin_selected_score_raw"))
                            if selected_row.get("origin_selected_score_raw") is not None
                            else float(selected_score)
                        )
                        selected_source = str(selection_result.get("selected_source", "sparsedrive_vlm"))

                        # Build VLM selection debug metadata (non-Q fields only)
                        selection_debug = {
                            "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
                            "vlm_confidence": selection_result.get("vlm_confidence"),
                            "vlm_reasoning": selection_result.get("vlm_reasoning"),
                            "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
                            "vlm_error": selection_result.get("vlm_error"),
                            "scoring_invoked": selection_result.get("scoring_invoked"),
                            "intervention_invoked": selection_result.get("intervention_invoked"),
                            "intervention_should_intervene": selection_result.get("intervention_should_intervene"),
                            "intervention_severity_score": selection_result.get("intervention_severity_score"),
                            "intervention_severity_band": selection_result.get("intervention_severity_band"),
                            "intervention_corrective_action": selection_result.get("intervention_corrective_action"),
                            "intervention_confidence": selection_result.get("intervention_confidence"),
                            "intervention_reasoning": selection_result.get("intervention_reasoning"),
                            "intervention_elapsed_sec": selection_result.get("intervention_elapsed_sec"),
                            "intervention_error": selection_result.get("intervention_error"),
                            "adaptive_replan_decision": selection_result.get("adaptive_replan_decision"),
                            "carry_previous_valid": selection_result.get("carry_previous_valid"),
                            "latency_timeline_record": selection_result.get("latency_timeline_record"),
                            "vlm_failed": selection_result.get("vlm_failed"),
                            "fallback_selected_idx": int(default_selected_index),
                            "fallback_selected_source": default_selected_source,
                            "display_default_trajectories": bool(getattr(vlm_cfg, "display_default_trajectories", False)),
                            "include_default_candidates": bool(getattr(vlm_cfg, "include_default_candidates", False)),
                            "carry_previous_allowed": bool(allow_carry_prev),
                            "previous_selected_source": previous_selected_source,
                            "selected_score_raw": float(selected_score_raw),
                        }
                        # Build final payload below (outside this block)
                except Exception as e:
                    LOG.exception("VLM selection failed: %s", e)
                    # Fallback to model selection
                    best_idx = int(np.argmax(scores))
                    selected_traj = proposals[best_idx, :output_num_poses]
                    selected_plan = sparsedrive_to_hugsim_plan(selected_traj)
                    selected_idx = best_idx
                    selected_score = float(scores[best_idx])
                    selected_score_raw = float(selected_score)
                    selected_source = "sparsedrive_argmax_fallback"
                    selection_debug = {
                        "vlm_error": str(e),
                        "fallback_selected_idx": int(default_selected_index),
                        "fallback_selected_source": default_selected_source,
                    }

                # Build final payload with VLM-selected plan and candidate pool metadata
                try:
                    plan_payload = build_plan_payload(
                        proposals=proposals,
                        scores=scores,
                        output_num_poses=output_num_poses,
                        selected_idx=selected_idx,
                        selected_source=selected_source,
                        selection_debug=selection_debug,
                        selected_plan_override=selected_plan,
                        selected_score_override=selected_score,
                        candidate_pool_rows=candidate_rows,
                        topk=TOPK,
                    )
                except Exception as e:
                    LOG.exception("Failed to build plan payload: %s", e)
                    write_plan_file(plan_pipe_writer, None)
                    continue

                # Write final plan to HUGSIM
                write_plan_file(plan_pipe_writer, plan_payload)

                # Save previous selection for carry-prev support
                try:
                    previous_selected_plan = np.asarray(plan_payload["selected_plan"], dtype=np.float32).copy()
                    previous_selected_pose = info_to_pose(info)
                    # Preserve raw selected score (from VLM or fallback) similar to RAP
                    previous_selected_score = float(selected_score_raw) if 'selected_score_raw' in locals() else float(plan_payload.get("selected_score", 0.0))
                    previous_selected_timestamp = float(info.get("timestamp", 0.0))
                    previous_selected_source = selected_source
                except Exception:
                    # keep previous selections as-is on failure
                    LOG.exception("Failed to save previous_selected state")
            except Exception:
                LOG.error("Adapter loop failed")
                LOG.error(traceback.format_exc())
                try:
                    write_plan_file(plan_pipe_writer, None)
                except Exception:
                    LOG.error("Failed to notify HUGSIM about adapter failure")
                return 1
    finally:
        try:
            obs_pipe_reader.close()
        except Exception:
            pass
        try:
            plan_pipe_writer.close()
        except Exception:
            pass
        try:
            vlm_selector.finalize()
        except Exception:
            LOG.exception("Error finalizing VLM selector")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

