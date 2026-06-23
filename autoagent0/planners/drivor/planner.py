#!/usr/bin/env python3
"""DrivoR planner core: observation -> trajectory proposals + scores.

Pure-inference half of the former ``planners/drivor/client.py``: build a navsim
AgentInput from the HUGSIM observation, run the DrivoR agent, and return
proposals (HUGSIM local coordinates) plus scores. Selection / payload happen
pipeline-side.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from autoagent0.adapters.hugsim.geometry import (
    compute_local_acceleration as shared_compute_local_acceleration,
    compute_local_velocity,
)
from autoagent0.planners.base import PlannerResult, PlannerService

# Bridge the pytree registration API for older torch versions (DrivoR may run
# torch 2.1 which exposes the private helper).
try:
    import inspect
    from torch.utils import _pytree as _torch_pytree

    if hasattr(_torch_pytree, "_register_pytree_node"):
        _raw_register_pytree_node = _torch_pytree._register_pytree_node
        _raw_signature = inspect.signature(_raw_register_pytree_node)

        def _compat_register_pytree_node(cls, flatten_fn, unflatten_fn, **kwargs):
            supported_kwargs = {
                key: value for key, value in kwargs.items() if key in _raw_signature.parameters
            }
            return _raw_register_pytree_node(cls, flatten_fn, unflatten_fn, **supported_kwargs)

        _torch_pytree.register_pytree_node = _compat_register_pytree_node
except Exception:
    pass

# DrivoR lives in its own repo; add it to the path (env set by launch.sh) and
# import the navsim dataclasses + agent. This mirrors the legacy client.
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

# Map HUGSIM camera names to navsim Cameras field names (only enabled slots).
MAP_HUGSIM_TO_DRIVOR = {
    "CAM_FRONT": "cam_f0",
    "CAM_BACK": "cam_b0",
    "CAM_FRONT_LEFT": "cam_l0",
    "CAM_FRONT_RIGHT": "cam_r0",
}

EGO_HISTORY_FRAMES = 4


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "drivor_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def make_command_one_hot(command: int) -> np.ndarray:
    # HUGSIM commands: 0=right, 1=left, 2=forward
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def euler_deg_to_rot_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(angles_deg[:3])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]], dtype=np.float32)
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def build_camera_from_hugsim(cam_name: str, rgb_image: np.ndarray, cam_params: Dict) -> Camera:
    intr = cam_params.get("intrinsic", {})
    W = float(intr.get("W", cam_params.get("W", 800)))
    H = float(intr.get("H", cam_params.get("H", 450)))
    cx = float(intr.get("cx", intr.get("cx", W / 2.0)))
    cy = float(intr.get("cy", intr.get("cy", H / 2.0)))
    fovx = float(intr.get("fovx", 60.0))
    fovy = float(intr.get("fovy", 40.0))
    fx = W / (2.0 * math.tan(math.radians(fovx) / 2.0))
    fy = H / (2.0 * math.tan(math.radians(fovy) / 2.0))
    cam_intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    sensor2lidar_rot = None
    sensor2lidar_trans = None
    if "l2c" in cam_params:
        l2c = np.array(cam_params["l2c"], dtype=np.float32)
        if l2c.shape == (4, 4):
            cam2lidar = np.linalg.inv(l2c)
            sensor2lidar_rot = cam2lidar[:3, :3].astype(np.float32)
            sensor2lidar_trans = cam2lidar[:3, 3].astype(np.float32)
    else:
        rot_angles = cam_params.get("l2c_rot", None)
        trans = cam_params.get("l2c_trans", None)
        if rot_angles is not None and trans is not None:
            R_l2c = euler_deg_to_rot_matrix(rot_angles)
            t_l2c = np.array(trans, dtype=np.float32)
            R_c2l = R_l2c.T
            t_c2l = -R_c2l @ t_l2c
            sensor2lidar_rot = R_c2l.astype(np.float32)
            sensor2lidar_trans = t_c2l.astype(np.float32)

    if sensor2lidar_rot is None:
        sensor2lidar_rot = np.eye(3, dtype=np.float32)
    if sensor2lidar_trans is None:
        sensor2lidar_trans = np.zeros(3, dtype=np.float32)

    if rgb_image is None:
        LOG.warning("Missing HUGSIM camera image for %s; using blank %dx%d frame", cam_name, int(H), int(W))
        rgb_image = np.zeros((int(H), int(W), 3), dtype=np.uint8)

    return Camera(
        image=rgb_image,
        sensor2lidar_rotation=sensor2lidar_rot,
        sensor2lidar_translation=sensor2lidar_trans,
        intrinsics=cam_intrinsic,
        distortion=None,
    )


def build_agent_input_from_hugsim(obs: Dict, info_history: List[Dict], num_history: int = EGO_HISTORY_FRAMES) -> AgentInput:
    while len(info_history) < num_history:
        info_history.insert(0, info_history[0].copy())

    ego_statuses = []
    cameras_list = []
    lidars_list = []

    for idx in range(-num_history, 0):
        info = info_history[idx]
        ego_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ego_velocity = compute_local_velocity(info_history[-num_history:], idx + num_history)
        ego_accel = shared_compute_local_acceleration(
            info_history[-num_history:], idx + num_history, zero_on_nonpositive_dt=True,
        )
        driving_command = make_command_one_hot(info.get("command", -1))
        ego_statuses.append(
            EgoStatus(ego_pose.astype(np.float64), ego_velocity.astype(np.float32), ego_accel.astype(np.float32), driving_command)
        )

        cam_params = info.get("cam_params", {})
        rgb = obs.get("rgb", {})
        cams_kwargs = {}
        for hug_name, drv_field in MAP_HUGSIM_TO_DRIVOR.items():
            img = rgb.get(hug_name, None)
            params = cam_params.get(hug_name, {})
            cams_kwargs[drv_field] = build_camera_from_hugsim(hug_name, img, params)

        all_fields = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        enabled_fields = ["cam_f0", "cam_l0", "cam_r0", "cam_b0"]
        for f in all_fields:
            if f not in cams_kwargs:
                if f in enabled_fields:
                    cams_kwargs[f] = build_camera_from_hugsim(f, None, {})
                else:
                    cams_kwargs[f] = Camera(
                        image=None,
                        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
                        sensor2lidar_translation=np.zeros(3, dtype=np.float32),
                        intrinsics=np.eye(3, dtype=np.float32),
                        distortion=None,
                    )

        cameras_list.append(
            Cameras(
                cam_f0=cams_kwargs.get("cam_f0"),
                cam_l0=cams_kwargs.get("cam_l0"),
                cam_l1=cams_kwargs.get("cam_l1"),
                cam_l2=cams_kwargs.get("cam_l2"),
                cam_r0=cams_kwargs.get("cam_r0"),
                cam_r1=cams_kwargs.get("cam_r1"),
                cam_r2=cams_kwargs.get("cam_r2"),
                cam_b0=cams_kwargs.get("cam_b0"),
            )
        )
        lidars_list.append(Lidar())

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras_list, lidars=lidars_list)


def drivor_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def extract_proposals_and_scores_from_predictions(
    predictions: Dict,
    output_num_poses: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    if traj is None:
        for v in predictions.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[-1] in [2, 3]:
                traj = v
                break
    if traj is None:
        raise RuntimeError(f"Model output has no trajectory tensor. Available keys: {list(predictions.keys())}")

    traj_np = traj.detach().cpu().numpy()
    if traj_np.ndim == 4:
        traj_np = traj_np[0]
    elif traj_np.ndim != 3:
        raise RuntimeError(f"Unexpected trajectory shape: {traj_np.shape}, expected [B,P,T,D] or [P,T,D]")
    if traj_np.shape[-1] == 2:
        traj_np = np.pad(traj_np, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

    if scores is not None:
        scores_np = scores.detach().cpu().numpy()
        if scores_np.ndim == 2:
            scores_np = scores_np[0]
    else:
        scores_np = np.zeros(len(traj_np), dtype=np.float32)
    if scores_np.ndim != 1:
        scores_np = np.asarray(scores_np).reshape(-1)

    return traj_np, scores_np


class DrivorPlanner(PlannerService):
    history_frames = EGO_HISTORY_FRAMES

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.agent = None
        self.device = None
        self.output_num_poses = 8

    def setup(self) -> None:
        setup_logging(self.output_dir)
        repo_root = Path(os.environ["DRIVOR_REPO_ROOT"]).expanduser().resolve()
        checkpoint = os.environ["DRIVOR_CHECKPOINT"]
        dino = os.environ.get("DRIVOR_DINO", "")
        device_name = os.environ.get("DRIVOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        LOG.info("Starting DrivoR planner repo=%s checkpoint=%s dino=%s device=%s", repo_root, checkpoint, dino, self.device)
        sys.path.insert(0, str(repo_root))

        try:
            from omegaconf import OmegaConf
            omega_available = True
        except Exception:
            omega_available = False
            LOG.warning("omegaconf not available; Hydra configs cannot be composed here")

        drivo_config: Any = {}
        drivor_config_path = os.environ.get("DRIVOR_CONFIG", "").strip()
        if drivor_config_path:
            if omega_available:
                try:
                    loaded = OmegaConf.load(drivor_config_path)
                    drivo_config = OmegaConf.create(loaded) if isinstance(loaded, dict) else loaded
                    LOG.info("Loaded DrivoR config from %s", drivor_config_path)
                except Exception:
                    LOG.exception("Failed to load DRIVOR_CONFIG=%s; using minimal dict", drivor_config_path)
                    drivo_config = {}
            else:
                try:
                    import yaml
                    with open(drivor_config_path, "r") as f:
                        loaded = yaml.safe_load(f)
                    drivo_config = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    LOG.exception("Failed to read DRIVOR_CONFIG=%s as YAML; using minimal dict", drivor_config_path)
                    drivo_config = {}

        if omega_available and OmegaConf.is_config(drivo_config):
            if "num_poses" not in drivo_config and "config" in drivo_config:
                drivo_config = drivo_config.config
        elif isinstance(drivo_config, dict):
            if "num_poses" not in drivo_config and "config" in drivo_config:
                drivo_config = drivo_config["config"]

        lr_args = {"name": "AdamW", "base_lr": 5e-4, "base_batch_size": 64}
        agent = DrivoRAgent(config=drivo_config, lr_args=lr_args, checkpoint_path=checkpoint, progress_bar=False)
        agent.initialize()
        LOG.info("DrivoRAgent initialized OK")

        try:
            current_model_device = str(next(agent._drivor_model.parameters()).device)
        except Exception:
            current_model_device = "unknown"
        target_device = str(self.device)
        if not (current_model_device.startswith(target_device)
                or (target_device.startswith("cuda") and current_model_device.startswith("cuda"))):
            agent._drivor_model.to(self.device)
        agent._drivor_model.eval()

        self.agent = agent
        try:
            self.output_num_poses = int(agent._config.get("num_poses", 8)) if hasattr(agent, "_config") and isinstance(agent._config, dict) else 8
        except Exception:
            self.output_num_poses = 8
        LOG.info("DrivoR ready, output_num_poses=%d", self.output_num_poses)

    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> PlannerResult:
        agent_input = build_agent_input_from_hugsim(obs, list(info_history), num_history=EGO_HISTORY_FRAMES)
        features: Dict[str, Any] = {}
        for builder in self.agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features_batched = {}
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features_batched[key] = value.unsqueeze(0).to(self.device)
            else:
                try:
                    features_batched[key] = torch.from_numpy(np.array(value)).unsqueeze(0).to(self.device)
                except Exception:
                    features_batched[key] = value

        with torch.no_grad():
            try:
                pred = self.agent.forward(features_batched)
            except Exception:
                LOG.exception("agent.forward failed; falling back to internal model call")
                pred = self.agent._drivor_model(features_batched)

        proposals_raw, scores = extract_proposals_and_scores_from_predictions(pred, output_num_poses=self.output_num_poses)
        proposals = np.stack(
            [drivor_to_hugsim_plan(proposals_raw[i, :self.output_num_poses]) for i in range(proposals_raw.shape[0])],
            axis=0,
        ).astype(np.float32)
        return PlannerResult(proposals=proposals, scores=np.asarray(scores, dtype=np.float32))

    def finalize(self) -> None:
        pass
