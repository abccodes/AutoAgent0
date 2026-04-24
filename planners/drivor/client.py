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

from scipy.spatial.transform import Rotation as SCR
from planners.common.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from planners.common.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    get_prefixed_env_value,
)

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
# NOTE: Only map cameras that are enabled in the model config.
# DrivoR config has: cam_f0:[3], cam_l0:[3], cam_r0:[3], cam_b0:[3] (enabled)
#                   cam_l1:[], cam_l2:[], cam_r1:[], cam_r2:[] (disabled/empty)
# To avoid feature mismatch (8 cameras vs 4 scene tokens), we only populate enabled slots.
MAP_HUGSIM_TO_DRIVOR = {
    "CAM_FRONT": "cam_f0",
    "CAM_BACK": "cam_b0",
    "CAM_FRONT_LEFT": "cam_l0",
    "CAM_FRONT_RIGHT": "cam_r0",
    # Disabled in model config: cam_l1, cam_l2, cam_r1, cam_r2
}

# history frames used to build AgentInput (DrivoR config often expects 4)
EGO_HISTORY_FRAMES = 4

TOPK = 8

PLAN_DT_SEC = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="DrivoR FIFO client for HUGSIM")
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
    drivor_python_bin = os.environ.get("DRIVOR_PYTHON_BIN", "")
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = drivor_python_bin
        raw_value = get_prefixed_env_value(suffix, default=default_value)
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)

# don't need resolve_config since DrivoR uses Hydra for it's configs rather than env vars

# def env_get(primary: str, fallback: str, default: str) -> str:
#     raw = os.environ.get(primary)
#     if raw is not None:
#         return raw
#     raw = os.environ.get(fallback)
#     if raw is not None:
#         return raw
#     return default


# def env_flag_compat(primary: str, fallback: str, default: bool = False) -> bool:
#     raw = os.environ.get(primary)
#     if raw is None:
#         raw = os.environ.get(fallback)
#     if raw is None:
#         return default
#     return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# def resolve_vlm_config() -> VLMSelectorConfig:
#     return VLMSelectorConfig(
#         enabled=env_flag_compat("PLANNER_VLM_ENABLED", "DRIVOR_VLM_ENABLED", False),
#         backend=env_get("PLANNER_VLM_BACKEND", "DRIVOR_VLM_BACKEND", "qwen3_vl"),
#         model_id=env_get("PLANNER_VLM_MODEL_ID", "DRIVOR_VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"),
#         device=env_get("PLANNER_VLM_DEVICE", "DRIVOR_VLM_DEVICE", "auto"),
#         max_new_tokens=int(env_get("PLANNER_VLM_MAX_NEW_TOKENS", "DRIVOR_VLM_MAX_NEW_TOKENS", "300")),
#         candidate_limit=int(env_get("PLANNER_VLM_CANDIDATE_LIMIT", "DRIVOR_VLM_CANDIDATE_LIMIT", "5")),
#         timeout_sec=float(env_get("PLANNER_VLM_TIMEOUT_SEC", "DRIVOR_VLM_TIMEOUT_SEC", "10.0")),
#         save_debug_artifacts=env_flag_compat("PLANNER_VLM_SAVE_DEBUG_ARTIFACTS", "DRIVOR_VLM_SAVE_DEBUG_ARTIFACTS", True),
#         debug_dir_name=env_get("PLANNER_VLM_DEBUG_DIR_NAME", "DRIVOR_VLM_DEBUG_DIR_NAME", "vlm_debug"),
#         carry_previous_enabled=env_flag_compat("PLANNER_VLM_CARRY_PREVIOUS_ENABLED", "DRIVOR_VLM_CARRY_PREVIOUS_ENABLED", True),
#         carry_previous_min_path_m=float(env_get("PLANNER_VLM_CARRY_PREVIOUS_MIN_PATH_M", "DRIVOR_VLM_CARRY_PREVIOUS_MIN_PATH_M", "0.5")),
#         carry_previous_min_points=int(env_get("PLANNER_VLM_CARRY_PREVIOUS_MIN_POINTS", "DRIVOR_VLM_CARRY_PREVIOUS_MIN_POINTS", "2")),
#         adaptive_replan_mode=env_get("PLANNER_VLM_ADAPTIVE_REPLAN_MODE", "DRIVOR_VLM_ADAPTIVE_REPLAN_MODE", "log_only"),
#         latency_tracking_mode=env_get("PLANNER_VLM_LATENCY_TRACKING_MODE", "DRIVOR_VLM_LATENCY_TRACKING_MODE", "full_timeline"),
#         q_enabled=env_flag_compat("PLANNER_VLM_Q_ENABLED", "DRIVOR_VLM_Q_ENABLED", True),
#         q_switch_margin=float(env_get("PLANNER_VLM_Q_SWITCH_MARGIN", "DRIVOR_VLM_Q_SWITCH_MARGIN", "0.05")),
#         q_weight_rap_score=float(env_get("PLANNER_VLM_Q_WEIGHT_RAP_SCORE", "DRIVOR_VLM_Q_WEIGHT_RAP_SCORE", "0.55")),
#         q_weight_progress=float(env_get("PLANNER_VLM_Q_WEIGHT_PROGRESS", "DRIVOR_VLM_Q_WEIGHT_PROGRESS", "0.30")),
#         q_weight_offcenter=float(env_get("PLANNER_VLM_Q_WEIGHT_OFFCENTER", "DRIVOR_VLM_Q_WEIGHT_OFFCENTER", "0.10")),
#         q_weight_curvature=float(env_get("PLANNER_VLM_Q_WEIGHT_CURVATURE", "DRIVOR_VLM_Q_WEIGHT_CURVATURE", "0.08")),
#         q_weight_shortplan=float(env_get("PLANNER_VLM_Q_WEIGHT_SHORTPLAN", "DRIVOR_VLM_Q_WEIGHT_SHORTPLAN", "0.18")),
#         q_carry_score_decay=float(env_get("PLANNER_VLM_Q_CARRY_SCORE_DECAY", "DRIVOR_VLM_Q_CARRY_SCORE_DECAY", "0.0")),
#         display_default_trajectories=env_flag_compat("PLANNER_VLM_DISPLAY_DEFAULT_TRAJECTORIES", "DRIVOR_VLM_DISPLAY_DEFAULT_TRAJECTORIES", False),
#         include_default_candidates=env_flag_compat("PLANNER_VLM_INCLUDE_DEFAULT_CANDIDATES", "DRIVOR_VLM_INCLUDE_DEFAULT_CANDIDATES", False),
#     )


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "drivor_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )

#no need for load_rap_model since it's abstracted to DrivoRAgent.initialize()

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
        for hug_name, drv_field in MAP_HUGSIM_TO_DRIVOR.items():
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
            cam_obj = build_camera_from_hugsim(hug_name, img, params)
            cams_kwargs[drv_field] = cam_obj

        # Ensure all Cameras dataclass fields are populated, but only with enabled cameras.
        # Disabled camera slots (per drivor.yaml config) are set to None to prevent feature mismatch.
        # Enabled cameras: cam_f0, cam_l0, cam_r0, cam_b0 (4 cameras)
        # Disabled cameras: cam_l1, cam_l2, cam_r1, cam_r2 (set to None so feature builder skips them)
        all_fields = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        enabled_fields = ["cam_f0", "cam_l0", "cam_r0", "cam_b0"]
        for f in all_fields:
            if f not in cams_kwargs:
                if f in enabled_fields:
                    # Enabled camera but no HUGSIM source: create black frame
                    cams_kwargs[f] = build_camera_from_hugsim(f, None, {})
                else:
                    # Disabled camera: create a Camera object with image=None so the
                    # DrivoR feature builder can check `cam.image is None` safely
                    cams_kwargs[f] = Camera(
                        image=None,
                        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
                        sensor2lidar_translation=np.zeros(3, dtype=np.float32),
                        intrinsics=np.eye(3, dtype=np.float32),
                        distortion=None,
                    )
        
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
        cameras_list.append(cameras_dataclass)
        # no lidar provided -> push empty Lidar
        lidars_list.append(Lidar())

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras_list, lidars=lidars_list)

#comparable to rap_to_hugsim_plan
def drivor_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)

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

#writes DrivoR plan back to HUGSIM
def write_plan(plan_pipe: Path, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plan_pipe, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)

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
    # Find trajectory tensor (try multiple keys)
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

    return traj_np, scores_np


def build_drivor_candidate_rows(
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
    Mirrors RAP's build_vlm_candidate_rows but adapted for DrivoR model outputs.
    
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
    current_candidate_limit = max(1, int(candidate_limit) - 1)
    current_candidate_limit = min(current_candidate_limit, int(len(sorted_indices)))
    candidate_indices = sorted_indices[:current_candidate_limit]

    candidate_rows: List[Dict[str, object]] = []

    # Insert carry_prev if valid
    if carry_candidate is not None:
        candidate_rows.append(carry_candidate)

    # Add top-k current proposals (converted to HUGSIM coords)
    for idx in candidate_indices:
        full_plan = drivor_to_hugsim_plan(proposals[idx, :output_num_poses])
        candidate_rows.append(
            {
                "source": "current_drivor",
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
    selected_source: str = "drivor_argmax",
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int = TOPK,
) -> Dict[str, object]:
    """
    Build complete plan payload with candidate pool, defaults, and VLM metadata.
    Mirrors RAP's build_plan_payload but adapted for DrivoR models.
    
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
        selected_plan = drivor_to_hugsim_plan(selected_traj)
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
            drivor_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = ["current_drivor" for _ in top_indices]
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
            drivor_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
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


def build_plain_drivor_plan_result(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
) -> Dict[str, object]:
    """
    Plain argmax fallback for DrivoR when VLM is disabled.
    Returns the selected_plan, selected_score, selected_score_raw, selected_row, and a plan_payload.
    """
    best_idx = int(np.argmax(scores))
    selected_traj = proposals[best_idx, :output_num_poses]
    selected_plan = drivor_to_hugsim_plan(selected_traj)
    selected_score = float(scores[best_idx])
    selected_score_raw = float(selected_score)
    plan_payload = build_plan_payload(
        proposals=proposals,
        scores=scores,
        output_num_poses=output_num_poses,
        selected_idx=best_idx,
        selected_source="drivor_argmax",
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
        "selected_row": {"source": "current_drivor", "proposal_index": best_idx},
        "plan_payload": plan_payload,
    }


# ============================================================================
# OLD FUNCTIONS (commented out for reference and comparison)
# ============================================================================

# def build_plan_payload_from_model_output_OLD(
#         predictions: Dict, 
#         output_num_poses: int = 8, 
#         topk: int = TOPK
# ) -> Dict:
#     """
#     OLD VERSION: Identify proposals & scores in model output (attempt several common keys).
#     DrivoR may output trajectories in multiple formats:
#       - 'trajectory' : tensor [B, P, T, 3] or [P, T, 3] or [B, P, T]
#       - 'score' or 'scores' : tensor [B, P] or [P]
#     We handle batch size 1 and flatten as needed.
#     """
#     # find trajectory tensor
#     traj = None
#     scores = None
#     for key in ["trajectory", "proposals", "proposals_traj", "trajectories"]:
#         if key in predictions:
#             traj = predictions[key]
#             break
#     for key in ["score", "scores", "prob", "logits"]:
#         if key in predictions:
#             scores = predictions[key]
#             break
#
#     if traj is None:
#         # try to find any tensor that could be trajectory
#         for v in predictions.values():
#             if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[-1] in [2, 3]:
#                 traj = v
#                 break
#
#     if traj is None:
#         raise RuntimeError(f"Model output does not contain a recognizable trajectory tensor. Available keys: {list(predictions.keys())}")
#
#     traj_np = traj.detach().cpu().numpy()
#     
#     # Normalize to [P, T, 3] (proposals, timesteps, 3D coords)
#     # Handle batch dimension if present
#     if traj_np.ndim == 4:
#         # [B, P, T, D] -> take batch 0
#         traj_np = traj_np[0]  # [P, T, D]
#     elif traj_np.ndim == 3:
#         # Already [P, T, D] - good
#         pass
#     else:
#         raise RuntimeError(f"Unexpected trajectory tensor shape: {traj_np.shape}, expected [B,P,T,D] or [P,T,D]")
#     
#     # Ensure last dim is 3 (x, y, heading); if it's 2, pad with zeros
#     if traj_np.shape[-1] == 2:
#         traj_np = np.pad(traj_np, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
#
#     if scores is not None:
#         scores_np = scores.detach().cpu().numpy()
#         # Normalize scores to [P] (proposals)
#         if scores_np.ndim == 2:
#             scores_np = scores_np[0]  # [P]
#     else:
#         # fallback: use zeros or argsort by norm
#         scores_np = np.zeros(len(traj_np), dtype=np.float32)
#
#     # choose topk
#     topk_idx = np.argsort(scores_np)[-topk:][::-1]
#     selected_idx = int(topk_idx[0])
#     selected_traj = traj_np[selected_idx, :output_num_poses, :]
#
#     payload = {
#         "selected_idx": selected_idx,
#         "selected_score": float(scores_np[selected_idx]),
#         "selected_plan": drivor_to_hugsim_plan(selected_traj),
#         "topk_indices": [int(i) for i in topk_idx.tolist()],
#         "topk_scores": [float(scores_np[i]) for i in topk_idx.tolist()],
#         "topk_plans": [drivor_to_hugsim_plan(traj_np[i, :output_num_poses, :]) for i in topk_idx.tolist()],
#     }
#     return payload

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
    # Attempt to import OmegaConf for Hydra-style configs; fall back gracefully.
    try:
        from omegaconf import OmegaConf  # type: ignore
        omega_available = True
    except Exception:
        omega_available = False
        LOG.warning("omegaconf not available; Hydra configs cannot be composed here")

    # Minimal config: prefer loading a Hydra/OmegaConf config if provided via env var
    # Set DRIVOR_CONFIG to a YAML file path (Hydra/omega format) to have it loaded here.
    drivo_config = {}
    drivor_config_path = os.environ.get("DRIVOR_CONFIG", "").strip()
    LOG.info("DRIVOR_CONFIG env: '%s', OmegaConf available: %s", drivor_config_path, omega_available)

    if drivor_config_path:
        if omega_available:
            try:
                loaded = OmegaConf.load(drivor_config_path)
                # If OmegaConf returned a plain dict (PyYAML backend), convert it to a DictConfig
                if isinstance(loaded, dict):
                    drivo_config = OmegaConf.create(loaded)
                else:
                    drivo_config = loaded
                LOG.info("Loaded DrivoR config from %s (type=%s)", drivor_config_path, type(drivo_config))
            except Exception:
                LOG.exception("Failed to load DRIVOR_CONFIG=%s; falling back to minimal dict", drivor_config_path)
                drivo_config = {}
        else:
            # Try to load as plain YAML to surface parse errors for easier debugging
            try:
                import yaml  # type: ignore

                with open(drivor_config_path, "r") as f:
                    loaded = yaml.safe_load(f)
                drivo_config = loaded if isinstance(loaded, dict) else {}
                LOG.info("Loaded plain YAML DRIVOR_CONFIG from %s (type=%s)", drivor_config_path, type(drivo_config))
            except Exception:
                LOG.exception("Failed to read DRIVOR_CONFIG=%s as YAML; falling back to minimal dict", drivor_config_path)
                drivo_config = {}
    else:
        LOG.info("No DRIVOR_CONFIG specified; using minimal config dict. Set DRIVOR_CONFIG to a yaml to pass a full config.")

    #unwrap config 
    if omega_available and OmegaConf.is_config(drivo_config):
        if "num_poses" not in drivo_config and "config" in drivo_config:
            drivo_config = drivo_config.config
    elif isinstance(drivo_config, dict):
        if "num_poses" not in drivo_config and "config" in drivo_config:
            drivo_config = drivo_config["config"]

    # learning rate / optimizer args (still a plain dict)
    lr_args = {"name": "AdamW", "base_lr": 5e-4, "base_batch_size": 64}

    # Create agent instance
    # We pass in the checkpoint path; DrivoRAgent.initialize() will load it.
    LOG.info("DrivoRAgent instance going to be created using config var of type: %s", type(drivo_config))
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

    info_history: deque[Dict[str, object]] = deque(maxlen=EGO_HISTORY_FRAMES)

    # VLM selector setup
    vlm_cfg = resolve_vlm_config()
    vlm_selector = VLMPlanSelector(vlm_cfg, output_dir)
    frame_index = 0
    previous_selected_plan: Optional[np.ndarray] = None
    previous_selected_pose: Optional[np.ndarray] = None
    previous_selected_score: Optional[float] = None
    previous_selected_timestamp: Optional[float] = None
    previous_selected_source: Optional[str] = None

    try:
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
                    # Debug: log shapes and dtypes of features before forwarding to the model
                    try:
                        for k, v in features_batched.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    LOG.info("Feature '%s': tensor shape=%s dtype=%s", k, tuple(v.shape), v.dtype)
                                else:
                                    LOG.info("Feature '%s': type=%s", k, type(v))
                            except Exception:
                                LOG.exception("Failed to describe feature %s", k)
                    except Exception:
                        LOG.exception("Failed to iterate features_batched for debug")

                    try:
                        pred = agent.forward(features_batched)
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
                        # fallback: call internal model directly
                        pred = agent._drivor_model(features_batched)

                # Extract proposals and scores from model predictions
                try:
                    output_num_poses = (
                        int(agent._config.get("num_poses", 8))
                        if hasattr(agent, "_config") and isinstance(agent._config, dict)
                        else 8
                    )
                    proposals, scores = extract_proposals_and_scores_from_predictions(pred, output_num_poses=output_num_poses)
                except Exception as e:
                    LOG.exception("Failed to extract proposals/scores from model output: %s", e)
                    write_plan(plan_pipe, None)
                    continue

                # Build candidate rows (includes carry_prev, top-k, and optional defaults)
                try:
                    candidate_rows, allow_carry_prev = build_drivor_candidate_rows(
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
                    write_plan(plan_pipe, None)
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
                    default_selected_source = "drivor_argmax"
                except Exception as e:
                    LOG.exception("Failed to determine default selection: %s", e)
                    default_selected_index = 0
                    default_selected_source = "drivor_argmax"

                # Call VLM selector (or use default if disabled)
                try:
                    # If VLM disabled, use plain argmax fallback similar to RAP's plain result
                    if not getattr(vlm_cfg, "enabled", False):
                        plain_result = build_plain_drivor_plan_result(proposals, scores, output_num_poses)
                        selected_plan = np.asarray(plain_result["selected_plan"], dtype=np.float32)
                        selected_score = float(plain_result["selected_score"])
                        selected_score_raw = float(plain_result.get("selected_score_raw", selected_score))
                        selected_idx = int(plain_result["selected_row"]["proposal_index"]) if plain_result["selected_row"].get("proposal_index") is not None else None
                        selected_source = "drivor_argmax"
                        selection_debug = {
                            "vlm_invoked": False,
                            "fallback_selected_idx": int(default_selected_index),
                            "fallback_selected_source": default_selected_source,
                            "display_default_trajectories": bool(getattr(vlm_cfg, "display_default_trajectories", False)),
                            "include_default_candidates": bool(getattr(vlm_cfg, "include_default_candidates", False)),
                        }
                        
                    else:
                        front_image = obs.get("rgb", {}).get("CAM_FRONT") if isinstance(obs, dict) else None
                        selection_result = vlm_selector.maybe_select(
                            frame_index=frame_index,
                            front_image=front_image,
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
                        selected_source = str(selection_result.get("selected_source", "drivor_vlm"))

                        # Build VLM selection debug metadata (non-Q fields only)
                        selection_debug = {
                            "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
                            "vlm_confidence": selection_result.get("vlm_confidence"),
                            "vlm_reasoning": selection_result.get("vlm_reasoning"),
                            "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
                            "vlm_error": selection_result.get("vlm_error"),
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
                    selected_plan = drivor_to_hugsim_plan(selected_traj)
                    selected_idx = best_idx
                    selected_score = float(scores[best_idx])
                    selected_score_raw = float(selected_score)
                    selected_source = "drivor_argmax_fallback"
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
                    write_plan(plan_pipe, None)
                    continue

                # Write final plan to HUGSIM
                write_plan(plan_pipe, plan_payload)

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
                    write_plan(plan_pipe, None)
                except Exception:
                    LOG.error("Failed to notify HUGSIM about adapter failure")
                return 1
    finally:
        try:
            vlm_selector.finalize()
        except Exception:
            LOG.exception("Error finalizing VLM selector")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())