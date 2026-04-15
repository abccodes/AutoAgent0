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

from vlm_selector import VLMPlanSelector, VLMSelectorConfig

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
        vlm=VLMSelectorConfig(
            enabled=env_flag("RAP_VLM_ENABLED", False),
            backend=os.environ.get("RAP_VLM_BACKEND", "qwen3_vl"),
            model_id=os.environ.get("RAP_VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"),
            device=os.environ.get("RAP_VLM_DEVICE", "auto"),
            max_new_tokens=int(os.environ.get("RAP_VLM_MAX_NEW_TOKENS", "300")),
            candidate_limit=int(os.environ.get("RAP_VLM_CANDIDATE_LIMIT", "8")),
            timeout_sec=float(os.environ.get("RAP_VLM_TIMEOUT_SEC", "10.0")),
            save_debug_artifacts=env_flag("RAP_VLM_SAVE_DEBUG_ARTIFACTS", True),
            debug_dir_name=os.environ.get("RAP_VLM_DEBUG_DIR_NAME", "vlm_debug"),
        ),
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

    fx = intrinsic["W"] / (2.0 * math.tan(intrinsic["fovx"] / 2.0))
    fy = intrinsic["H"] / (2.0 * math.tan(intrinsic["fovy"] / 2.0))
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[0, 0] = fx * image_scale
    viewpad[1, 1] = fy * image_scale
    viewpad[0, 2] = cx * image_scale
    viewpad[1, 2] = cy * image_scale

    # HUGSIM local coordinates are [right, forward, up]. The scene rig is
    # expressed relative to the rendered front camera, so convert into that
    # camera-aligned frame before applying the per-camera extrinsic.
    local_to_front_cam = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
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
    topk: int = TOPK_PROPOSALS_TO_SEND,
) -> Dict[str, object]:
    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]
    if selected_idx is None:
        selected_idx = int(top_indices[0])
    else:
        selected_idx = int(selected_idx)
    selected_traj = proposals[selected_idx, :output_num_poses]
    payload = {
        "selected_idx": selected_idx,
        "selected_score": float(scores[selected_idx]),
        "selected_source": selected_source,
        "selected_plan": rap_to_hugsim_plan(selected_traj),
        "topk_indices": [int(idx) for idx in top_indices],
        "topk_scores": [float(scores[idx]) for idx in top_indices],
        "topk_plans": [
            rap_to_hugsim_plan(proposals[idx, :output_num_poses])
            for idx in top_indices
        ],
    }
    if selection_debug:
        payload.update(selection_debug)
    return payload


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
    frame_index = 0

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
            sorted_indices = np.argsort(scores)[::-1]
            candidate_limit = max(1, min(int(cfg.vlm.candidate_limit), int(len(sorted_indices))))
            candidate_indices = sorted_indices[:candidate_limit]
            candidate_plans = [
                rap_to_hugsim_plan(proposals[idx, :cfg.output_num_poses])
                for idx in candidate_indices
            ]
            candidate_scores = [float(scores[idx]) for idx in candidate_indices]
            selection_result = vlm_selector.maybe_select(
                frame_index=frame_index,
                front_image=obs["rgb"]["CAM_FRONT"],
                info=info,
                candidate_plans=candidate_plans,
                candidate_indices=[int(idx) for idx in candidate_indices],
                candidate_scores=candidate_scores,
                rap_argmax_index=best_idx,
            )
            frame_index += 1

            selection_debug = {
                "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
                "vlm_confidence": selection_result.get("vlm_confidence"),
                "vlm_reasoning": selection_result.get("vlm_reasoning"),
                "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
                "vlm_error": selection_result.get("vlm_error"),
                "vlm_candidate_count": selection_result.get("vlm_candidate_count"),
            }
            plan_payload = build_plan_payload(
                proposals,
                scores,
                output_num_poses=cfg.output_num_poses,
                selected_idx=int(selection_result["selected_index"]),
                selected_source=str(selection_result["selected_source"]),
                selection_debug=selection_debug,
            )
            write_plan(plan_pipe, plan_payload)
        except Exception:
            logging.error("Inference failure in RAP adapter")
            logging.error(traceback.format_exc())
            try:
                write_plan(plan_pipe, None)
            except Exception:
                logging.error("Failed to notify HUGSIM about planner failure")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
