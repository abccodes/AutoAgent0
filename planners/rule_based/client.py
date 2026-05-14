#!/usr/bin/env python3
"""
Rule-based Planner HUGSIM FIFO adapter (client.py)

- Reads pickled (obs, info, privileged_info) messages from obs_pipe
- Uses PrivilegedPlannerService.process() to generate trajectory plans
- Converts Trajectory objects to proposal format compatible with HUGSIM
- Writes a plan payload to plan_pipe (pickled, with 8-byte length prefix) compatible with HUGSIM

Assumptions (review before running):
- HUGSIM message format: pickled tuple (obs, info, privileged_info)
  - obs['rgb'][cam_name] -> H x W x 3 uint8 images
  - info contains 'cam_params'[cam_name] with {intrinsic: {W,H,cx,cy,fovx,fovy}, l2c_rot: [3 deg], l2c_trans: [3]}
  - info contains ego_pos/ego_rot or timestamp fields used to compute velocities
  - privileged_info contains ground-truth agent/vehicle info from closed-loop simulator
- Environment variables: RULE_BASED_REPO_ROOT (path to rule-based planner repo)
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
import select
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import fcntl

from scipy.spatial.transform import Rotation as SCR
from planners.common.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from planners.common.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    get_prefixed_env_value,
)

# Import custom rule-based planner
# Add repo root to path based on env var RULE_BASED_REPO_ROOT (set by HUGSIM launch)
RULE_BASED_REPO_ROOT = os.environ.get("RULE_BASED_REPO_ROOT", "")
if not RULE_BASED_REPO_ROOT:
    raise RuntimeError("RULE_BASED_REPO_ROOT must be set in environment")
sys.path.insert(0, str(Path(RULE_BASED_REPO_ROOT).resolve()))

try:
    # from planners.rule_based.planner_service import PlannerService
    from privileged_planner.service import PrivilegedPlannerService
except ImportError as e:
    raise RuntimeError(
        f"PrivilegedPlannerService not found in {RULE_BASED_REPO_ROOT}. "
        "Ensure planners.rule_based.planner_service module exists. Error: {e}"
    ) from e

LOG = logging.getLogger("rule_based_adapter")

DEFAULT_CAM_ORDER = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Trajectory horizon and time step
EGO_HISTORY_FRAMES = 4
DEFAULT_OUTPUT_POSES = 8

TOPK = 10

PLAN_DT_SEC = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based planner FIFO client for HUGSIM")
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
    rule_based_python_bin = os.environ.get("RULE_BASED_PYTHON_BIN", "")
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = rule_based_python_bin
        raw_value = get_prefixed_env_value(
            suffix,
            default=default_value,
            prefixes=("RULE_BASED_VLM_", "PLANNER_VLM_"),
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rule_based_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def _log_fifo_fd(pipe, label: str) -> None:
    try:
        fd = pipe.fileno()
        stat_result = os.fstat(fd)
        # fetch open flags
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        except Exception:
            flags = None
        LOG.info(
            "%s fd=%s inode=%s mode=%s flags=%s",
            label,
            fd,
            stat_result.st_ino,
            oct(stat_result.st_mode),
            str(flags),
        )
        try:
            LOG.info("%s fd link=%s", label, os.readlink(f"/proc/{os.getpid()}/fd/{fd}"))
        except Exception:
            pass
    except Exception:
        LOG.exception("Failed to inspect FIFO fd for %s", label)

# no need for load_rap_model since this adapter owns its own planner bootstrap

def make_command_one_hot(command: int) -> np.ndarray:
    # HUGSIM commands: 0=right, 1=left, 2=forward
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

# no need for preprocess_image because rule-based planner handles its own preprocessing

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



# comparable to rap_to_hugsim_plan
def rule_based_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


# ============================================================================
# Rule-based planner adapter helpers
# ============================================================================

def trajectory_to_proposals(traj: Any, output_num_poses: int) -> np.ndarray:
    """
    Convert Trajectory object from rule-based planner to proposal format.
    
    Args:
        traj: Trajectory object with .states [T, 2] (x_forward, y_left in ego frame)
        output_num_poses: horizon length for padding/truncation
        
    Returns:
        proposals: np.ndarray of shape [1, output_num_poses, 3] 
                  (single proposal with x_forward, y_left, heading=0)
    """
    states = np.asarray(traj.states, dtype=np.float32)  # [T, 2]
    
    # Pad or truncate to output_num_poses
    if states.shape[0] < output_num_poses:
        pad_len = output_num_poses - states.shape[0]
        last_state = states[-1] if len(states) > 0 else np.zeros(2, dtype=np.float32)
        pad = np.tile(last_state, (pad_len, 1)).astype(np.float32)
        states = np.concatenate([states, pad], axis=0)
    else:
        states = states[:output_num_poses]
    
    # Pad heading dimension (set to 0)
    headings = np.zeros((output_num_poses, 1), dtype=np.float32)
    traj_3d = np.concatenate([states[:, :2], headings], axis=1)  # [T, 3]
    
    # Return in rule-based planner format [proposals, timesteps, xyz]
    return np.expand_dims(traj_3d, axis=0)  # [1, T, 3]


def trajectory_to_scores(debug_info: Dict[str, Any]) -> np.ndarray:
    """
    Extract score from planner debug info.
    
    Args:
        debug_info: Debug dict from planner with 'best_score' key
        
    Returns:
        scores: np.ndarray of shape [1] with single planner score
    """
    best_score = float(debug_info.get("best_score", 1.0))
    return np.array([best_score], dtype=np.float32)

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


def read_obs_file(pipe, timeout_sec: float = 30.0):
    LOG.info("entered read_obs_file function %s at t=%s", pipe, time.time())
    fd = pipe.fileno()
    LOG.info("select waiting on fd=%s timeout=%.1f at t=%s", fd, timeout_sec, time.time())
    ready, _, _ = select.select([fd], [], [], timeout_sec)
    if not ready:
        try:
            stat_result = os.fstat(fd)
            LOG.error(
                "Timeout waiting for FIFO readability after %.1fs: fd=%s inode=%s mode=%s",
                timeout_sec,
                fd,
                stat_result.st_ino,
                oct(stat_result.st_mode),
            )
        except Exception:
            LOG.error("Timeout waiting for FIFO readability after %.1fs", timeout_sec)
        raise TimeoutError(f"No obs_pipe data became readable within {timeout_sec} seconds")

    LOG.info("about to read header at t=%s", time.time())
    header = pipe.read(8)
    LOG.info("read header len=%s at t=%s", len(header) if header is not None else None, time.time())
    if len(header) != 8:
        raise EOFError("Incomplete pipe header from open obs pipe handle")
    payload_size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < payload_size:
        LOG.info("loading chunks, payload is size: %s", len(payload))
        LOG.info("about to read chunk remaining=%s at t=%s", payload_size - len(payload), time.time())
        chunk = pipe.read(payload_size - len(payload))
        if not chunk:
            raise EOFError("Incomplete pipe payload from open obs pipe handle")
        payload.extend(chunk)
    return pickle.loads(payload)

# writes planner plan back to HUGSIM
def write_plan(plan_pipe: Path, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plan_pipe, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def write_plan_file(pipe, plan) -> None:
    try:
        payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
        fd = pipe.fileno()
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        except Exception:
            flags = None
        LOG.info("write_plan_file: fd=%s flags=%s bytes=%s t=%s", fd, str(flags), len(payload), time.time())
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)
        pipe.flush()
    except Exception:
        LOG.exception("Failed writing plan to pipe")

# ============================================================================
# NEW FUNCTIONS: RAP-compatible candidate generation and payload building
# ============================================================================


def build_rule_based_candidate_rows(
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
    Mirrors RAP's build_vlm_candidate_rows but adapted for rule-based planner outputs.
    
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
        full_plan = rule_based_to_hugsim_plan(proposals[idx, :output_num_poses])
        candidate_rows.append(
            {
                "source": "rule_based_planner",
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
    selected_source: str = "rule_based_argmax",
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int = TOPK,
) -> Dict[str, object]:
    """
    Build complete plan payload with candidate pool, defaults, and VLM metadata.
    Mirrors RAP's build_plan_payload but adapted for rule-based planner outputs.
    
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
        selected_plan = rule_based_to_hugsim_plan(selected_traj)
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
        candidate_pool_sources = [str(row.get("source", "current_rule_based")) for row in candidate_pool_rows]
        candidate_pool_proposal_indices = [
            None if row.get("proposal_index") is None else int(row["proposal_index"])
            for row in candidate_pool_rows
        ]
    else:
        # Fallback: derive candidate pool from top-k proposals
        candidate_pool_plans = [
            rule_based_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = ["rule_based_planner" for _ in top_indices]
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
            rule_based_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
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


def build_plain_rule_based_plan_result(
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
) -> Dict[str, object]:
    """
    Plain argmax fallback for rule-based planner.
    Returns the selected_plan, selected_score, selected_score_raw, selected_row, and a plan_payload.
    """
    best_idx = int(np.argmax(scores))
    selected_traj = proposals[best_idx, :output_num_poses]
    selected_plan = rule_based_to_hugsim_plan(selected_traj)
    selected_score = float(scores[best_idx])
    selected_score_raw = float(selected_score)
    plan_payload = build_plan_payload(
        proposals=proposals,
        scores=scores,
        output_num_poses=output_num_poses,
        selected_idx=best_idx,
        selected_source="rule_based_argmax",
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
        "selected_row": {"source": "rule_based_planner", "proposal_index": best_idx},
        "plan_payload": plan_payload,
    }



def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    setup_logging(output_dir)
    LOG.info("Starting rule-based planner adapter")

    # Env vars
    repo_root_value = os.environ.get("RULE_BASED_REPO_ROOT", "").strip()
    if not repo_root_value:
        raise RuntimeError("RULE_BASED_REPO_ROOT is not set")
    repo_root = Path(repo_root_value).expanduser().resolve()
    # don't need torch just yet
    # device_name = os.environ.get(
    #     "RULE_BASED_DEVICE",
    #     os.environ.get("RULE_BASED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    # )
    # device = torch.device(device_name)

    LOG.info(
        "Repo root: %s, device: %s, config: %s",
        repo_root,
        os.environ.get("RULE_BASED_DEVICE", "cuda"),
        os.environ.get("RULE_BASED_CONFIG", ""),
    )

    # Add repo root to sys.path (already done above)
    sys.path.insert(0, str(repo_root))

    # Load optional config from environment or file
    planner_config = None
    planner_config_path = os.environ.get("PLANNER_CONFIG", "").strip()
    if planner_config_path:
        try:
            import yaml  # type: ignore
            with open(planner_config_path, "r") as f:
                planner_config = yaml.safe_load(f)
            LOG.info("Loaded planner config from %s", planner_config_path)
        except Exception:
            LOG.exception("Failed to load PLANNER_CONFIG=%s; using None", planner_config_path)
    else:
        LOG.info("No PLANNER_CONFIG specified; using default PrivilegedPlannerService initialization")

    # Create planner instance
    try:
        planner = PrivilegedPlannerService(config=planner_config)
        LOG.info("PrivilegedPlannerService initialized OK")
    except Exception:
        LOG.exception("Failed to initialize PrivilegedPlannerService")
        return 1

    # Determine output_num_poses from planner config or use default
    try:
        output_num_poses = int(
            planner_config.get("horizon", DEFAULT_OUTPUT_POSES) 
            if planner_config and isinstance(planner_config, dict) 
            else DEFAULT_OUTPUT_POSES
        )
    except Exception:
        output_num_poses = DEFAULT_OUTPUT_POSES
    LOG.info("Using output_num_poses=%d", output_num_poses)

    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    LOG.info("Waiting for scene FIFOs to appear: obs=%s plan=%s", obs_pipe, plan_pipe)
    while not obs_pipe.exists() or not plan_pipe.exists():
        LOG.info("obs_pipe and plan_pipe dont exist, sleeping for 0.1s")
        time.sleep(0.1)
    # Match the older DrivoR FIFO semantics: O_RDWR avoids startup deadlocks
    # when the writer and reader come up in different orders.
    obs_pipe_reader = os.fdopen(os.open(obs_pipe, os.O_RDWR), "rb", buffering=0)
    plan_pipe_writer = os.fdopen(os.open(plan_pipe, os.O_RDWR), "wb", buffering=0)
    LOG.info("Opened persistent scene FIFOs for obs and plan exchange")
    _log_fifo_fd(obs_pipe_reader, "obs_pipe_reader")
    _log_fifo_fd(plan_pipe_writer, "plan_pipe_writer")

    # Ensure the opened obs fd corresponds to the current path inode.
    # If the writer removed and recreated the FIFO after we opened it,
    # our fd will point to an unlinked inode and won't see new writes.
    try:
        def _ensure_fd_matches_path(fd_obj, path, attempts=10, delay=0.2):
            for attempt in range(attempts):
                try:
                    fd = fd_obj.fileno()
                    fd_inode = os.fstat(fd).st_ino
                    path_inode = os.stat(path).st_ino
                    if fd_inode == path_inode:
                        LOG.info("obs fd inode matches path inode: %s", fd_inode)
                        return True
                    LOG.warning("FD inode %s != path inode %s; reopening (attempt %d)", fd_inode, path_inode, attempt + 1)
                    try:
                        fd_obj.close()
                    except Exception:
                        pass
                    # Reopen fresh
                    new_fd = os.fdopen(os.open(path, os.O_RDWR), "rb", buffering=0)
                    fd_obj.__init__(new_fd.fileno(), new_fd.mode)
                except FileNotFoundError:
                    LOG.warning("Path %s disappeared while ensuring inode match; retrying", path)
                except Exception:
                    LOG.exception("Error ensuring fd/path inode match")
                time.sleep(delay)
            return False

        _ensure_fd_matches_path(obs_pipe_reader, str(obs_pipe))
    except Exception:
        LOG.exception("Failed during obs_pipe inode sanity check")

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
                message = read_obs_file(obs_pipe_reader, timeout_sec=30.0)
                LOG.info("Received observation payload from %s", obs_pipe)
                if message == "Done":
                    LOG.info("Received shutdown signal")
                    break

                try:
                    # REFAC: prefer privileged 3-tuple payloads, but fall back to
                    # legacy 2-tuple messages so older scene drivers keep working.
                    if isinstance(message, (tuple, list)) and len(message) == 3:
                        obs, info, privileged_info = message
                    elif isinstance(message, (tuple, list)) and len(message) == 2:
                        obs, info = message
                        privileged_info = None
                    else:
                        raise ValueError(
                            f"unexpected payload length: {len(message) if isinstance(message, (tuple, list)) else 'n/a'}"
                        )
                except (ValueError, TypeError):
                    LOG.error("Message format error: expected (obs, info) or (obs, info, privileged_info), got %s", type(message))
                    write_plan_file(plan_pipe_writer, None)
                    continue
                
                # info_history append and pad
                info_history.append(dict(info))
                while len(info_history) < EGO_HISTORY_FRAMES:
                    info_history.appendleft(dict(info_history[0]))

                # Run planner on current observation
                try:
                    traj, planner_debug = planner.process(
                        obs=obs,
                        info=info,
                        info_history=info_history,
                        privileged_agents=privileged_info
                    )
                    LOG.info(
                        "Planner generated trajectory: behavior=%s, states_shape=%s",
                        traj.behavior if hasattr(traj, 'behavior') else 'unknown',
                        np.asarray(traj.states).shape if hasattr(traj, 'states') else None
                    )
                except Exception as e:
                    LOG.exception("Planner.process() failed: %s", e)
                    write_plan_file(plan_pipe_writer, None)
                    continue

                # Convert planner trajectory to proposal format
                try:
                    proposals = trajectory_to_proposals(traj, output_num_poses)
                    scores = trajectory_to_scores(planner_debug)
                    LOG.info("Converted trajectory to proposals shape=%s, scores=%s", proposals.shape, scores)
                except Exception as e:
                    LOG.exception("Failed to convert planner trajectory: %s", e)
                    write_plan_file(plan_pipe_writer, None)
                    continue

                # Build candidate rows (includes carry_prev, top-k, and optional defaults)
                try:
                    candidate_rows, allow_carry_prev = build_rule_based_candidate_rows(
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
                    default_selected_source = "rule_based_argmax"
                except Exception as e:
                    LOG.exception("Failed to determine default selection: %s", e)
                    default_selected_index = 0
                    default_selected_source = "rule_based_argmax"

                # Call VLM selector (or use default if disabled)
                try:
                    # If VLM disabled, use plain argmax fallback similar to RAP's plain result
                    if not getattr(vlm_cfg, "enabled", False):
                        plain_result = build_plain_rule_based_plan_result(proposals, scores, output_num_poses)
                        selected_plan = np.asarray(plain_result["selected_plan"], dtype=np.float32)
                        selected_score = float(plain_result["selected_score"])
                        selected_score_raw = float(plain_result.get("selected_score_raw", selected_score))
                        selected_idx = int(plain_result["selected_row"]["proposal_index"]) if plain_result["selected_row"].get("proposal_index") is not None else None
                        selected_source = "rule_based_argmax"
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
                        selected_source = str(selection_result.get("selected_source", "rule_based_vlm"))

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
                    selected_plan = rule_based_to_hugsim_plan(selected_traj)
                    selected_idx = best_idx
                    selected_score = float(scores[best_idx])
                    selected_score_raw = float(selected_score)
                    selected_source = "rule_based_argmax_fallback"
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
