#!/usr/bin/env python3
import argparse
import logging
import math
import os
import pickle
import struct
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAP FIFO client for HUGSIM")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    return parser.parse_args()


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
        # Keep HUGSIM aligned with RAP's training-time camera order.
        camera_order=list(DEFAULT_CAM_ORDER),
        image_scale=image_scale,
        device=torch.device(device_name),
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
    # HUGSIM commands: 0=right, 1=left, 2=forward. RAP uses a 4-way one-hot.
    mapping = {
        2: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
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


def build_features(
    obs: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, object],
    cfg: AdapterConfig,
) -> Dict[str, torch.Tensor]:
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
        lidar2img.append(compute_lidar2img(cam_params, cam_name, cfg.image_scale))

    ego_speed = float(info["ego_velo"])
    ego_acc = float(info.get("accelerate", 0.0))
    ego_status = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            np.array([ego_speed, 0.0], dtype=np.float32),
            np.array([ego_acc, 0.0], dtype=np.float32),
            make_command_one_hot(info.get("command", -1)),
        ]
    )

    features = {
        "camera_feature": torch.from_numpy(np.stack(camera_images, axis=0)).unsqueeze(0).to(cfg.device),
        "ego_status": torch.from_numpy(ego_status[None, None, :]).to(cfg.device),
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


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    setup_logging(cfg.output_dir)
    logging.info("Starting RAP adapter with repo=%s checkpoint=%s", cfg.rap_repo_root, cfg.checkpoint_path)

    try:
        model = load_rap_model(cfg)
    except Exception:
        logging.exception("Failed to initialize RAP model")
        return 1

    obs_pipe = cfg.output_dir / "obs_pipe"
    plan_pipe = cfg.output_dir / "plan_pipe"

    while True:
        try:
            message = read_obs(obs_pipe)
            if message == "Done":
                logging.info("Received shutdown signal")
                break

            obs, info = message
            features = build_features(obs, info, cfg)
            with torch.no_grad():
                predictions = model(features, targets=None, return_score=False)
            trajectory = predictions["trajectory"][0].detach().cpu().numpy()
            plan = rap_to_hugsim_plan(trajectory)
            write_plan(plan_pipe, plan)
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
