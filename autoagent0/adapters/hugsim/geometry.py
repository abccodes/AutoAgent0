from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


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


def compute_local_acceleration(
    info_history: Sequence[Dict[str, object]],
    index: int,
    *,
    zero_on_nonpositive_dt: bool = False,
) -> np.ndarray:
    if len(info_history) <= 2:
        return np.zeros(2, dtype=np.float32)

    curr_vel = compute_local_velocity(info_history, index)

    if index > 0:
        prev_vel = compute_local_velocity(info_history, index - 1)
        if zero_on_nonpositive_dt:
            dt = float(info_history[index]["timestamp"]) - float(info_history[index - 1]["timestamp"])
            if dt <= 1e-6:
                return np.zeros(2, dtype=np.float32)
        else:
            dt = timestamp_delta_seconds(info_history[index - 1], info_history[index])
        return (curr_vel - prev_vel) / dt

    next_vel = compute_local_velocity(info_history, index + 1)
    if zero_on_nonpositive_dt:
        dt = float(info_history[index + 1]["timestamp"]) - float(info_history[index]["timestamp"])
        if dt <= 1e-6:
            return np.zeros(2, dtype=np.float32)
    else:
        dt = timestamp_delta_seconds(info_history[index], info_history[index + 1])
    return (next_vel - curr_vel) / dt


def info_to_pose(info: Dict[str, object]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SCR

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
