#!/usr/bin/env python3
"""RAP planner core: observation -> trajectory proposals + scores.

This is the pure-inference half of the former ``planners/rap/client.py``. It
builds RAP features, runs the model, and returns proposals (in HUGSIM local
coordinates) plus scores. Candidate selection, VLM/AutoAgent0 reasoning and
payload construction live on the pipeline side.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from autoagent0.adapters.hugsim.geometry import (
    compute_local_acceleration,
    compute_local_velocity,
    forward_left_basis,
    normalize_angle,
)
from autoagent0.planners.base import PlannerResult, PlannerService

# Newer transformers expects torch>=2.2's public pytree registration name.
# RAP runs with torch 2.1, which still exposes the private helper. Bridge the
# name before importing RAP/transformers modules.
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


DEFAULT_CAM_ORDER = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]
DEFAULT_OUTPUT_POSES = 10
EGO_HISTORY_FRAMES = 4

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


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_config(output_dir: Path) -> AdapterConfig:
    output_dir = Path(output_dir).resolve()
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
    return AdapterConfig(
        output_dir=output_dir,
        rap_repo_root=Path(rap_repo_root_raw).expanduser().resolve(),
        checkpoint_path=Path(checkpoint_path_raw).expanduser().resolve(),
        camera_order=list(DEFAULT_CAM_ORDER),
        image_scale=image_scale,
        device=torch.device(device_name),
        debug_diagnostics=env_flag("RAP_DEBUG_DIAGNOSTICS", False),
        use_scene_rig_lidar2img=env_flag("RAP_USE_SCENE_RIG_LIDAR2IMG", False),
        output_num_poses=DEFAULT_OUTPUT_POSES,
    )


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rap_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
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

    config = RAPConfig(cache_data=False, distill_feature=False, pdm_scorer=False)
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
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D camera image, got shape {image.shape}")

    # Some scenes surface RGB frames in CHW layout instead of HWC. Normalize
    # here so the RAP backbone always sees [num_cams, 3, H, W].
    if image.shape[-1] not in (3, 4) and image.shape[0] in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported camera image layout {image.shape}")

    if image.shape[-1] == 4:
        image = image[..., :3]

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


def pad_camera_batch(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("Expected at least one camera image")

    max_h = max(int(image.shape[0]) for image in images)
    max_w = max(int(image.shape[1]) for image in images)
    channels = int(images[0].shape[2])
    batch = np.zeros((len(images), max_h, max_w, channels), dtype=np.float32)
    for idx, image in enumerate(images):
        h, w, c = image.shape
        if c != channels:
            raise ValueError(f"Inconsistent channel count in camera batch: {c} vs {channels}")
        batch[idx, :h, :w] = image
    return batch


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
    # first shift from rear-axle origin into the front-camera origin, then rotate
    # into the front-camera frame before applying the per-camera rig transform.
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
        camera_images.append(image)
        img_shapes.append(img_shape)
        if cfg.use_scene_rig_lidar2img and "front2cam" in cam_params[cam_name]:
            lidar2img.append(compute_scene_rig_lidar2img(cam_params, cam_name, cfg.image_scale))
        else:
            lidar2img.append(compute_lidar2img(cam_params, cam_name, cfg.image_scale))

    camera_batch = pad_camera_batch(camera_images)

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
        "camera_feature": torch.from_numpy(np.transpose(camera_batch, (0, 3, 1, 2))).unsqueeze(0).to(cfg.device),
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


class RAPPlanner(PlannerService):
    history_frames = EGO_HISTORY_FRAMES

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.cfg: Optional[AdapterConfig] = None
        self.model = None

    def setup(self) -> None:
        cfg = resolve_config(self.output_dir)
        setup_logging(self.output_dir)
        logging.info("Starting RAP planner with repo=%s checkpoint=%s", cfg.rap_repo_root, cfg.checkpoint_path)
        logging.info("RAP lidar2img mode: %s", "scene_rig" if cfg.use_scene_rig_lidar2img else "static_l2c")
        self.cfg = cfg
        self.model = load_rap_model(cfg)

    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> PlannerResult:
        cfg = self.cfg
        features = build_features(obs, list(info_history), cfg)
        with torch.no_grad():
            predictions = self.model(features, targets=None, return_score=True)
            scores = predictions["score"][0].detach().cpu().numpy()
            proposals_raw = predictions["trajectory"][0].detach().cpu().numpy()

        if cfg.debug_diagnostics:
            best_idx = int(np.argmax(scores))
            best_xy = proposals_raw[best_idx, :cfg.output_num_poses, :2]
            step_norms = (
                np.linalg.norm(np.diff(best_xy, axis=0), axis=1)
                if len(best_xy) > 1 else np.zeros((0,), dtype=np.float32)
            )
            logging.info(
                "ts=%.2f cmd=%s velo=%.3f best=%d score_max=%.4f score_mean=%.4f traj_path=%.3f",
                float(info.get("timestamp", 0.0)), info.get("command"), float(info.get("ego_velo", 0.0)),
                best_idx, float(scores[best_idx]), float(scores.mean()), float(step_norms.sum()),
            )

        # Convert every proposal to HUGSIM local coordinates and truncate to the
        # model's pose horizon: [N, output_num_poses, 2].
        proposals = np.stack(
            [rap_to_hugsim_plan(proposals_raw[i, :cfg.output_num_poses]) for i in range(proposals_raw.shape[0])],
            axis=0,
        ).astype(np.float32)
        return PlannerResult(proposals=proposals, scores=np.asarray(scores, dtype=np.float32))

    def finalize(self) -> None:
        pass
