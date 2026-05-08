#!/usr/bin/env python3
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

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

from planners.common.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from planners.common.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    get_prefixed_env_value,
)

# Newer transformers expects torch>=2.2's public pytree registration name.
# RAP currently runs with torch 2.1 in this env, which still exposes the
# private helper. Bridge the name before importing RAP/transformers modules.
try:
    import inspect
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


DEFAULT_CAM_ORDER = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]
DEFAULT_OUTPUT_POSES = 10
TOPK_PROPOSALS_TO_SEND = 20
EGO_HISTORY_FRAMES = 4
PLAN_DT_SEC = 0.5

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class AdapterConfig:
    output_dir: Path
    rap_repo_root: Path
    checkpoint_path: Path
    camera_order: Sequence[str]
    image_scale: float
    device: torch.device
    debug_diagnostics: bool
    use_scene_rig_lidar2img: bool
    output_num_poses: int
    vlm: VLMSelectorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAP FIFO client for HUGSIM")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    return parser.parse_args()


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


def resolve_vlm_config() -> VLMSelectorConfig:
    values = {}
    rap_python_bin = os.environ.get("RAP_PYTHON_BIN", "")
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = rap_python_bin
        raw_value = get_prefixed_env_value(suffix, default=default_value)
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)


def resolve_config(args: argparse.Namespace) -> AdapterConfig:
    output_dir = Path(args.output).resolve()
    rap_repo_root_raw = os.environ.get("RAP_REPO_ROOT", "").strip()
    checkpoint_path_raw = os.environ.get("RAP_CHECKPOINT", "").strip()
    image_scale = float(os.environ.get("RAP_IMAGE_SCALE", "0.4"))
    device_name = os.environ.get("RAP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if not rap_repo_root_raw:
        raise ValueError("RAP_REPO_ROOT is not set")
    if not checkpoint_path_raw:
        raise ValueError("RAP_CHECKPOINT is not set")
    rap_repo_root = Path(rap_repo_root_raw).expanduser()
    checkpoint_path = Path(checkpoint_path_raw).expanduser()
    return AdapterConfig(
        output_dir=output_dir,
        rap_repo_root=rap_repo_root.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        camera_order=list(DEFAULT_CAM_ORDER),
        image_scale=image_scale,
        device=torch.device(device_name),
        debug_diagnostics=env_flag("RAP_DEBUG_DIAGNOSTICS", False),
        use_scene_rig_lidar2img=env_flag("RAP_USE_SCENE_RIG_LIDAR2IMG", False),
        output_num_poses=DEFAULT_OUTPUT_POSES,
        vlm=resolve_vlm_config(),
    )


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rap_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_rap_model(cfg: AdapterConfig):
    sys.path.insert(0, str(cfg.rap_repo_root))

    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from navsim.agents.rap_dino.rap_model import RAPModel
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    raw_state_dict = checkpoint.get("state_dict", checkpoint)

    inferred_num_poses = None
    pose_shape_keys = [
        "agent._rap_model._trajectory_head.0.Bev_refiner.positional_encoding.col_embed.weight",
        "_trajectory_head.0.Bev_refiner.positional_encoding.col_embed.weight",
    ]
    for key in pose_shape_keys:
        weight = raw_state_dict.get(key)
        if weight is not None:
            inferred_num_poses = int(weight.shape[0])
            break

    config = RAPConfig(
        cache_data=False,
        distill_feature=False,
        pdm_scorer=False,
    )
    if inferred_num_poses is not None:
        config.trajectory_sampling = TrajectorySampling(
            num_poses=inferred_num_poses,
            interval_length=config.trajectory_sampling.interval_length,
        )
        cfg.output_num_poses = inferred_num_poses
    model = RAPModel(config)
    model.progress = 1.0
    model.batch_size = 0

    state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("agent._rap_model."):
            state_dict[key.removeprefix("agent._rap_model.")] = value
        elif key.startswith("_rap_model."):
            state_dict[key.removeprefix("_rap_model.")] = value
        elif key.startswith("model."):
            state_dict[key.removeprefix("model.")] = value
        else:
            state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logging.warning("Missing RAP weights: %s", missing_keys)
    if unexpected_keys:
        logging.warning("Unexpected RAP weights: %s", unexpected_keys)

    model.to(cfg.device)
    model.eval()
    return model


def make_command_one_hot(command: int) -> np.ndarray:
    # HUGSIM commands: 0=right, 1=left, 2=forward.
    # RAP training uses one-hot order: (left, forward, right, unknown).
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def preprocess_image(image: np.ndarray, image_scale: float) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    image = image.astype(np.float32)
    image = (image - MEAN) / STD
    scaled_w = max(1, int(round(image.shape[1] * image_scale)))
    scaled_h = max(1, int(round(image.shape[0] * image_scale)))
    image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    pad_h = int(math.ceil(image.shape[0] / 32.0) * 32)
    pad_w = int(math.ceil(image.shape[1] / 32.0) * 32)
    padded = np.zeros((pad_h, pad_w, image.shape[2]), dtype=np.float32)
    padded[: image.shape[0], : image.shape[1]] = image
    return padded, padded.shape


def compute_lidar2img(cam_params: Dict[str, Dict[str, np.ndarray]], cam_name: str, image_scale: float) -> np.ndarray:
    params = cam_params[cam_name]
    intrinsic = params["intrinsic"]
    lidar2cam = np.array(params["l2c"], dtype=np.float32)

    fx = intrinsic["W"] / (2.0 * math.tan(intrinsic["fovx"] / 2.0))
    fy = intrinsic["H"] / (2.0 * math.tan(intrinsic["fovy"] / 2.0))
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[0, 0] = fx * image_scale
    viewpad[1, 1] = fy * image_scale
    viewpad[0, 2] = cx * image_scale
    viewpad[1, 2] = cy * image_scale

    return viewpad @ lidar2cam


def compute_scene_rig_lidar2img(
    cam_params: Dict[str, Dict[str, np.ndarray]],
    cam_name: str,
    image_scale: float,
) -> np.ndarray:
    params = cam_params[cam_name]
    intrinsic = params["intrinsic"]
    front2cam = np.array(params["front2cam"], dtype=np.float32)
    front_v2c = np.array(cam_params["CAM_FRONT"]["v2c"], dtype=np.float32)

    fx = intrinsic["W"] / (2.0 * math.tan(intrinsic["fovx"] / 2.0))
    fy = intrinsic["H"] / (2.0 * math.tan(intrinsic["fovy"] / 2.0))
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[0, 0] = fx * image_scale
    viewpad[1, 1] = fy * image_scale
    viewpad[0, 2] = cx * image_scale
    viewpad[1, 2] = cy * image_scale

    # HUGSIM local coordinates are [right, forward, up] around the rear axle.
    # The scene rig is expressed relative to the rendered front camera, so we
    # must first shift from rear-axle origin into the front-camera origin, then
    # rotate into the front-camera frame before applying the per-camera rig
    # transform.
    camera_in_vehicle = np.linalg.inv(front_v2c)[:3, 3]
    camera_in_local = np.array(
        [-camera_in_vehicle[1], camera_in_vehicle[0], camera_in_vehicle[2]],
        dtype=np.float32,
    )
    local_to_front_cam = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    local_to_front_cam[:3, 3] = -(local_to_front_cam[:3, :3] @ camera_in_local)
    return viewpad @ np.linalg.inv(front2cam) @ local_to_front_cam


def normalize_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def forward_left_basis(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    forward_dir = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    left_dir = np.array([-math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    return forward_dir, left_dir


def world_delta_to_local_components(delta_world: np.ndarray, yaw: float) -> np.ndarray:
    forward_dir, left_dir = forward_left_basis(yaw)
    return np.array(
        [
            float(np.dot(delta_world, forward_dir)),
            float(np.dot(delta_world, left_dir)),
        ],
        dtype=np.float32,
    )


def timestamp_delta_seconds(prev_info: Dict[str, object], next_info: Dict[str, object], default_dt: float = 0.25) -> float:
    dt = float(next_info["timestamp"]) - float(prev_info["timestamp"])
    if dt <= 1e-6:
        return default_dt
    return dt


def compute_local_velocity(info_history: Sequence[Dict[str, object]], index: int) -> np.ndarray:
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
            dtype=np.float32,
        )
        return world_delta_to_local_components(delta_world, curr_yaw) / dt

    next_info = info_history[index + 1]
    next_pos = np.asarray(next_info["ego_pos"], dtype=np.float32)
    dt = timestamp_delta_seconds(curr_info, next_info)
    delta_world = np.array(
        [next_pos[0] - curr_pos[0], next_pos[2] - curr_pos[2]],
        dtype=np.float32,
    )
    return world_delta_to_local_components(delta_world, curr_yaw) / dt


def compute_local_acceleration(info_history: Sequence[Dict[str, object]], index: int) -> np.ndarray:
    if len(info_history) <= 2:
        return np.zeros(2, dtype=np.float32)

    curr_vel = compute_local_velocity(info_history, index)

    if index > 0:
        prev_vel = compute_local_velocity(info_history, index - 1)
        dt = timestamp_delta_seconds(info_history[index - 1], info_history[index])
        return (curr_vel - prev_vel) / dt

    next_vel = compute_local_velocity(info_history, index + 1)
    dt = timestamp_delta_seconds(info_history[index], info_history[index + 1])
    return (next_vel - curr_vel) / dt


def build_features(
    obs: Dict[str, Dict[str, np.ndarray]],
    info_history: Sequence[Dict[str, object]],
    cfg: AdapterConfig,
) -> Dict[str, torch.Tensor]:
    info = info_history[-1]
    rgb_obs = obs["rgb"]
    cam_params = info["cam_params"]

    camera_images: List[np.ndarray] = []
    img_shapes: List[Tuple[int, int, int]] = []
    lidar2img: List[np.ndarray] = []
    for cam_name in cfg.camera_order:
        if cam_name not in rgb_obs:
            raise KeyError(f"Missing camera {cam_name} in HUGSIM observation")
        image, img_shape = preprocess_image(rgb_obs[cam_name], cfg.image_scale)
        camera_images.append(np.transpose(image, (2, 0, 1)))
        img_shapes.append(img_shape)
        if cfg.use_scene_rig_lidar2img and "front2cam" in cam_params[cam_name]:
            lidar2img.append(compute_scene_rig_lidar2img(cam_params, cam_name, cfg.image_scale))
        else:
            lidar2img.append(compute_lidar2img(cam_params, cam_name, cfg.image_scale))

    current_pos = np.asarray(info["ego_pos"], dtype=np.float32)
    current_rot = np.asarray(info["ego_rot"], dtype=np.float32)
    current_yaw = float(current_rot[1])
    forward_dir, left_dir = forward_left_basis(current_yaw)

    ego_status_history = []
    for index, hist_info in enumerate(info_history):
        hist_pos = np.asarray(hist_info["ego_pos"], dtype=np.float32)
        hist_rot = np.asarray(hist_info["ego_rot"], dtype=np.float32)
        hist_yaw = float(hist_rot[1])
        delta_world = np.array(
            [hist_pos[0] - current_pos[0], hist_pos[2] - current_pos[2]],
            dtype=np.float32,
        )
        rel_forward = float(np.dot(delta_world, forward_dir))
        rel_left = float(np.dot(delta_world, left_dir))
        rel_yaw = normalize_angle(hist_yaw - current_yaw)
        ego_velocity = compute_local_velocity(info_history, index)
        ego_acceleration = compute_local_acceleration(info_history, index)
        ego_status = np.concatenate(
            [
                np.array([rel_forward, rel_left, rel_yaw], dtype=np.float32),
                ego_velocity,
                ego_acceleration,
                make_command_one_hot(hist_info.get("command", -1)),
            ]
        )
        ego_status_history.append(ego_status)

    features = {
        "camera_feature": torch.from_numpy(np.stack(camera_images, axis=0)).unsqueeze(0).to(cfg.device),
        "ego_status": torch.from_numpy(np.stack(ego_status_history, axis=0)[None]).to(cfg.device),
        "img_shape": torch.tensor(np.array(img_shapes, dtype=np.float32)).unsqueeze(0).to(cfg.device),
        "lidar2img": torch.tensor(np.array(lidar2img, dtype=np.float32)).unsqueeze(0).to(cfg.device),
    }
    return features


def rap_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # RAP predicts [x_forward, y_left, heading] in ego coordinates.
    # HUGSIM expects [x_right, y_forward] in lidar-style local coordinates.
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


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


def build_carry_plan_candidate(
    previous_plan: Optional[np.ndarray],
    previous_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_timestamp: Optional[float],
    current_info: Dict[str, object],
    cfg: AdapterConfig,
) -> Optional[Dict[str, object]]:
    if not cfg.vlm.carry_previous_enabled or previous_plan is None or previous_pose is None or previous_timestamp is None:
        return None

    current_timestamp = float(current_info.get("timestamp", previous_timestamp))
    elapsed_sec = max(0.0, current_timestamp - float(previous_timestamp))
    elapsed_pose_steps = int(round(elapsed_sec / PLAN_DT_SEC))
    if elapsed_pose_steps >= len(previous_plan):
        return None

    trimmed_plan = np.asarray(previous_plan[elapsed_pose_steps:], dtype=np.float32)
    if len(trimmed_plan) < cfg.vlm.carry_previous_min_points:
        return None

    points_world = local_plan_to_world(trimmed_plan, np.asarray(previous_pose, dtype=np.float32))
    current_local = world_points_to_current_local(points_world, info_to_pose(current_info))

    valid_mask = current_local[:, 1] > 0.0
    if not np.any(valid_mask):
        return None
    first_valid_idx = int(np.argmax(valid_mask))
    current_local = current_local[first_valid_idx:]

    if len(current_local) < cfg.vlm.carry_previous_min_points:
        return None
    if path_length(current_local) < cfg.vlm.carry_previous_min_path_m:
        return None

    return {
        "source": "carry_prev",
        "proposal_index": None,
        # Do not give the carry path stale numeric advantage in the next decision.
        # Keep the original selected score only as raw debug metadata.
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


def read_obs(obs_pipe: Path):
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


def build_plan_payload(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    selected_idx: Optional[int] = None,
    selected_source: str = "rap_argmax",
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int = TOPK_PROPOSALS_TO_SEND,
) -> Dict[str, object]:
    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]
    if selected_idx is None and selected_plan_override is None:
        selected_idx = int(top_indices[0])
    else:
        selected_idx = None if selected_idx is None else int(selected_idx)

    if selected_plan_override is not None:
        selected_plan = np.asarray(selected_plan_override, dtype=np.float32)
        selected_score = float(selected_score_override) if selected_score_override is not None else None
    else:
        assert selected_idx is not None
        selected_traj = proposals[selected_idx, :output_num_poses]
        selected_plan = rap_to_hugsim_plan(selected_traj)
        selected_score = float(scores[selected_idx])

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
        candidate_pool_sources = [str(row.get("source", "current_rap")) for row in candidate_pool_rows]
        candidate_pool_proposal_indices = [
            None if row.get("proposal_index") is None else int(row["proposal_index"])
            for row in candidate_pool_rows
        ]
    else:
        candidate_pool_plans = [
            rap_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = ["current_rap" for _ in top_indices]
        candidate_pool_proposal_indices = [int(idx) for idx in top_indices]

    default_overlay_plans = None
    default_overlay_sources = None
    if bool(selection_debug and selection_debug.get("display_default_trajectories")):
        default_overlay_plans = [traj.tolist() for traj in get_default_trajectories(output_num_poses)]
        default_overlay_sources = [f"default_fallback_{idx}" for idx in range(len(default_overlay_plans))]

    payload = {
        "selected_idx": selected_idx,
        "selected_score": selected_score,
        "selected_source": selected_source,
        "selected_plan": selected_plan,
        "topk_indices": [int(idx) for idx in top_indices],
        "topk_scores": [float(scores[idx]) for idx in top_indices],
        "topk_plans": [
            rap_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
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
    if selection_debug:
        payload.update(selection_debug)
    return payload


def build_plain_rap_plan_result(
    proposals: np.ndarray,
    scores: np.ndarray,
    cfg: AdapterConfig,
) -> Dict[str, object]:
    best_idx = int(np.argmax(scores))
    selected_plan = rap_to_hugsim_plan(proposals[best_idx, :cfg.output_num_poses])
    selected_score = float(scores[best_idx])
    return {
        "selected_plan": selected_plan,
        "selected_score": selected_score,
        "selected_score_raw": selected_score,
        "selected_row": {
            "source": "current_rap",
            "proposal_index": best_idx,
        },
        "plan_payload": build_plan_payload(
            proposals,
            scores,
            output_num_poses=cfg.output_num_poses,
            selected_idx=best_idx,
            selected_source="rap_argmax",
            topk=10,
        ),
    }


def build_vlm_candidate_rows(
    proposals: np.ndarray,
    scores: np.ndarray,
    cfg: AdapterConfig,
    current_info: Dict[str, object],
    previous_selected_plan: Optional[np.ndarray],
    previous_selected_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_selected_timestamp: Optional[float],
    previous_selected_source: Optional[str],
) -> Tuple[List[Dict[str, object]], bool]:
    allow_carry_previous = not (
        previous_selected_source is not None
        and str(previous_selected_source).startswith("default_fallback_")
    )
    carry_candidate = build_carry_plan_candidate(
        previous_plan=previous_selected_plan if allow_carry_previous else None,
        previous_pose=previous_selected_pose if allow_carry_previous else None,
        previous_selected_score=previous_selected_score if allow_carry_previous else None,
        previous_timestamp=previous_selected_timestamp if allow_carry_previous else None,
        current_info=current_info,
        cfg=cfg,
    )

    sorted_indices = np.argsort(scores)[::-1]
    current_candidate_limit = max(1, int(cfg.vlm.candidate_limit) - 1)
    current_candidate_limit = min(current_candidate_limit, int(len(sorted_indices)))
    candidate_indices = sorted_indices[:current_candidate_limit]
    candidate_rows: List[Dict[str, object]] = []
    if carry_candidate is not None:
        candidate_rows.append(carry_candidate)

    for idx in candidate_indices:
        full_plan = rap_to_hugsim_plan(proposals[idx, :cfg.output_num_poses])
        candidate_rows.append(
            {
                "source": "current_rap",
                "proposal_index": int(idx),
                "proposal_score": float(scores[idx]),
                "local_plan": full_plan,
                "execution_plan": full_plan.copy(),
            }
        )

    if cfg.vlm.include_default_candidates:
        for default_idx, default_plan in enumerate(get_default_trajectories(cfg.output_num_poses)):
            candidate_rows.append(
                {
                    "source": f"default_fallback_{default_idx}",
                    "proposal_index": None,
                    "proposal_score": 0.0,
                    "local_plan": default_plan,
                    "execution_plan": default_plan.copy(),
                }
            )

    carry_row = next((row for row in candidate_rows if row.get("source") == "carry_prev"), None)
    shared_horizon = len(carry_row["local_plan"]) if carry_row is not None else cfg.output_num_poses
    shared_horizon = max(1, int(shared_horizon))
    for row in candidate_rows:
        execution_plan = np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32)
        row["execution_plan"] = execution_plan
        row["local_plan"] = truncate_plan(execution_plan, shared_horizon)

    return candidate_rows, allow_carry_previous


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    setup_logging(cfg.output_dir)
    logging.info("Starting RAP adapter with repo=%s checkpoint=%s", cfg.rap_repo_root, cfg.checkpoint_path)
    logging.info("RAP lidar2img mode: %s", "scene_rig" if cfg.use_scene_rig_lidar2img else "static_l2c")

    try:
        model = load_rap_model(cfg)
    except Exception:
        logging.exception("Failed to initialize RAP model")
        return 1

    obs_pipe = cfg.output_dir / "obs_pipe"
    plan_pipe = cfg.output_dir / "plan_pipe"
    info_history: deque[Dict[str, object]] = deque(maxlen=EGO_HISTORY_FRAMES)
    vlm_selector = VLMPlanSelector(cfg.vlm, cfg.output_dir)
    vlm_selector.preload()
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
                    logging.info("Received shutdown signal")
                    break

                obs, info = message
                info_history.append(dict(info))
                while len(info_history) < EGO_HISTORY_FRAMES:
                    info_history.appendleft(dict(info_history[0]))
                features = build_features(obs, list(info_history), cfg)
                with torch.no_grad():
                    predictions = model(features, targets=None, return_score=True)
                    scores = predictions["score"][0].detach().cpu().numpy()
                    proposals = predictions["trajectory"][0].detach().cpu().numpy()
                    best_idx = int(np.argmax(scores))
                    trajectory = proposals[best_idx, :cfg.output_num_poses]
                    best_xy = trajectory[:, :2]
                    step_norms = np.linalg.norm(np.diff(best_xy, axis=0), axis=1) if len(best_xy) > 1 else np.zeros((0,), dtype=np.float32)

                    if cfg.debug_diagnostics:
                        top_indices = np.argsort(scores)[-3:][::-1]
                        top_summary = []
                        for idx in top_indices:
                            proposal_xy = proposals[idx, :cfg.output_num_poses, :2]
                            proposal_steps = np.linalg.norm(np.diff(proposal_xy, axis=0), axis=1)
                            top_summary.append(
                                (
                                    int(idx),
                                    float(scores[idx]),
                                    float(np.linalg.norm(proposal_xy[-1] - proposal_xy[0])),
                                    float(proposal_steps.sum()),
                                )
                            )
                        logging.info(
                            "ts=%.2f cmd=%s velo=%.3f best=%d score_max=%.4f score_mean=%.4f "
                            "traj_extent=%.3f traj_path=%.3f min_step=%.4f max_step=%.4f top3=%s",
                            float(info["timestamp"]),
                            info.get("command"),
                            float(info["ego_velo"]),
                            best_idx,
                            float(scores[best_idx]),
                            float(scores.mean()),
                            float(np.linalg.norm(best_xy[-1] - best_xy[0])),
                            float(step_norms.sum()),
                            float(step_norms.min(initial=0.0)),
                            float(step_norms.max(initial=0.0)),
                            top_summary,
                        )

                    traj_path = float(step_norms.sum()) if len(step_norms) > 0 else 0.0
                    traj_extent = float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])) if len(trajectory) > 1 else 0.0
                    if traj_path < 0.5:
                        logging.warning(
                            "collapsed_plan ts=%.2f cmd=%s velo=%.3f accel=%.3f path=%.3f extent=%.3f "
                            "head=%s proposal0=%s",
                            float(info["timestamp"]),
                            info.get("command"),
                            float(info["ego_velo"]),
                            float(info.get("accelerate", 0.0)),
                            traj_path,
                            traj_extent,
                            np.round(trajectory[:5, :2], 3).tolist(),
                            np.round(proposals[0, :5, :2], 3).tolist(),
                        )
                if not cfg.vlm.enabled:
                    plain_result = build_plain_rap_plan_result(proposals, scores, cfg)
                    selected_plan = plain_result["selected_plan"]
                    selected_score = plain_result["selected_score"]
                    selected_score_raw = plain_result["selected_score_raw"]
                    selected_row = plain_result["selected_row"]
                    plan_payload = plain_result["plan_payload"]
                else:
                    candidate_rows, allow_carry_previous = build_vlm_candidate_rows(
                        proposals=proposals,
                        scores=scores,
                        cfg=cfg,
                        current_info=info,
                        previous_selected_plan=previous_selected_plan,
                        previous_selected_pose=previous_selected_pose,
                        previous_selected_score=previous_selected_score,
                        previous_selected_timestamp=previous_selected_timestamp,
                        previous_selected_source=previous_selected_source,
                    )
                    rap_best_idx = int(np.argmax(scores))
                    default_selected_index = next(
                        idx
                        for idx, row in enumerate(candidate_rows)
                        if row.get("source") == "current_rap" and row.get("proposal_index") == rap_best_idx
                    )

                    selection_result = vlm_selector.maybe_select(
                        frame_index=frame_index,
                        camera_images=obs["rgb"],
                        info=info,
                        candidate_rows=candidate_rows,
                        default_selected_index=default_selected_index,
                        default_selected_source="fallback_rap_argmax",
                    )
                    frame_index += 1

                    selected_row = selection_result["selected_candidate_row"]
                    selected_plan = np.asarray(selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32)
                    selected_idx = selected_row.get("proposal_index")
                    selected_score_value = selected_row.get("proposal_score")
                    if selected_score_value is None:
                        selected_score_value = selected_row.get("rap_score", 0.0)
                    selected_score = float(selected_score_value)

                    selected_score_raw_value = selected_row.get("origin_selected_score_raw")
                    if selected_score_raw_value is None:
                        selected_score_raw_value = selected_score
                    selected_score_raw = float(selected_score_raw_value)

                    selection_debug = {
                        "q_selected_idx": None,
                        "q_selected_source": None,
                        "q_candidate_scores": None,
                        "q_carry_score": None,
                        "q_best_current_score": None,
                        "q_score_gap": None,
                        "q_switch_margin": None,
                        "q_selected_path_length": None,
                        "q_best_current_path_length": None,
                        "fallback_selected_idx": int(default_selected_index),
                        "fallback_selected_source": "fallback_rap_argmax",
                        "display_default_trajectories": bool(cfg.vlm.display_default_trajectories),
                        "include_default_candidates": bool(cfg.vlm.include_default_candidates),
                        "q_invoked_vlm": bool(cfg.vlm.enabled),
                        "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
                        "vlm_confidence": selection_result.get("vlm_confidence"),
                        "vlm_reasoning": selection_result.get("vlm_reasoning"),
                        "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
                        "vlm_error": selection_result.get("vlm_error"),
                        "vlm_candidate_count": selection_result.get("vlm_candidate_count"),
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
                        "vlm_q_valid": selection_result.get("vlm_q_valid"),
                        "vlm_timed_out": selection_result.get("vlm_timed_out"),
                        "vlm_q_candidate_scores": selection_result.get("vlm_q_candidate_scores"),
                        "vlm_q_best_candidate_index": selection_result.get("vlm_q_best_candidate_index"),
                        "vlm_q_score_gap_to_carry": selection_result.get("vlm_q_score_gap_to_carry"),
                        "vlm_q_score_gap_top2": selection_result.get("vlm_q_score_gap_top2"),
                        "vlm_q_best_current_score": selection_result.get("vlm_q_best_current_score"),
                        "vlm_q_carry_score": selection_result.get("vlm_q_carry_score"),
                        "adaptive_replan_decision": selection_result.get("adaptive_replan_decision"),
                        "carry_previous_valid": selection_result.get("carry_previous_valid"),
                        "carry_previous_allowed": bool(allow_carry_previous),
                        "previous_selected_source": previous_selected_source,
                        "selected_score_raw": selected_score_raw,
                        "latency_timeline_record": selection_result.get("latency_timeline_record"),
                        "vlm_failed": selection_result.get("vlm_failed"),
                    }
                    plan_payload = build_plan_payload(
                        proposals,
                        scores,
                        output_num_poses=cfg.output_num_poses,
                        selected_idx=None if selected_idx is None else int(selected_idx),
                        selected_source=str(selection_result["selected_source"]),
                        selection_debug=selection_debug,
                        selected_plan_override=selected_plan,
                        selected_score_override=selected_score,
                        candidate_pool_rows=candidate_rows,
                    )
                write_plan(plan_pipe, plan_payload)

                previous_selected_plan = selected_plan.copy()
                previous_selected_pose = info_to_pose(info)
                previous_selected_score = selected_score_raw
                previous_selected_timestamp = float(info.get("timestamp", 0.0))
                previous_selected_source = str(
                    selected_row.get(
                        "source",
                        "rap_argmax" if not cfg.vlm.enabled else "vlm_selected",
                    )
                )
            except Exception:
                logging.error("Inference failure in RAP adapter")
                logging.error(traceback.format_exc())
                try:
                    write_plan(plan_pipe, None)
                except Exception:
                    logging.error("Failed to notify HUGSIM about planner failure")
                return 1
    finally:
        vlm_selector.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
