from __future__ import annotations

from typing import Any, Dict, Sequence


def summarize_demo_overlay(task_records: Sequence[Dict[str, Any]], task_info: Dict[str, Any] | None, final_goal_status: Any) -> Dict[str, Any]:
    if not task_info:
        return {}

    projection_states = [str(record.get("projection_state", "inactive")) for record in task_records]
    visible_frames = [int(record["frame_idx"]) for record in task_records if record.get("visible")]
    return {
        "task_active": True,
        "task_type": task_info.get("task_type"),
        "task_instruction": task_info.get("task_instruction"),
        "task_target_pose_local": task_info.get("task_target_pose_local"),
        "task_target_world": task_info.get("task_target_world"),
        "projection_states": projection_states,
        "marker_visible_frame_count": int(len(visible_frames)),
        "first_visible_frame": None if not visible_frames else int(visible_frames[0]),
        "last_visible_frame": None if not visible_frames else int(visible_frames[-1]),
        "final_goal_status": final_goal_status,
        "task_completion_reason": task_info.get("task_completion_reason"),
    }


def is_stop_task_complete(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    if str(info.get("task_type", "")).strip() != "stop_at_target":
        return False
    goal_status = info.get("task_goal_status")
    if not isinstance(goal_status, dict) or not bool(goal_status.get("reached")):
        return False
    try:
        ego_speed_mps = abs(float(info.get("ego_velo", 0.0)))
    except (TypeError, ValueError):
        return False
    try:
        stop_speed_threshold_mps = float(info.get("task_stop_speed_threshold_mps", 0.5))
    except (TypeError, ValueError):
        stop_speed_threshold_mps = 0.5
    return ego_speed_mps <= stop_speed_threshold_mps


def is_park_task_complete(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    if str(info.get("task_type", "")).strip() != "park_at_target":
        return False
    goal_status = info.get("task_goal_status")
    if not isinstance(goal_status, dict) or not bool(goal_status.get("reached")):
        return False
    try:
        ego_speed_mps = abs(float(info.get("ego_velo", 0.0)))
    except (TypeError, ValueError):
        return False
    try:
        park_speed_threshold_mps = float(info.get("task_park_speed_threshold_mps", 0.75))
    except (TypeError, ValueError):
        park_speed_threshold_mps = 0.75
    return ego_speed_mps <= park_speed_threshold_mps


def apply_demo_task_action_override(action: Dict[str, Any], info: Any) -> Dict[str, Any]:
    if not isinstance(info, dict):
        return action
    if str(info.get("task_type", "")).strip() != "park_at_target":
        return action
    goal_status = info.get("task_goal_status")
    if not isinstance(goal_status, dict):
        return action
    try:
        position_error_m = float(goal_status.get("position_error_m", 1e9))
    except (TypeError, ValueError):
        return action
    try:
        brake_distance_m = float(info.get("task_park_brake_distance_m", 10.0))
    except (TypeError, ValueError):
        brake_distance_m = 10.0
    if position_error_m > brake_distance_m:
        return action

    overridden = dict(action)
    try:
        ego_speed_mps = abs(float(info.get("ego_velo", 0.0)))
    except (TypeError, ValueError):
        ego_speed_mps = None
    try:
        park_speed_threshold_mps = float(info.get("task_park_speed_threshold_mps", 0.75))
    except (TypeError, ValueError):
        park_speed_threshold_mps = 0.75
    try:
        brake_accel_mps2 = float(info.get("task_park_brake_accel_mps2", -2.0))
    except (TypeError, ValueError):
        brake_accel_mps2 = -2.0
    if ego_speed_mps is not None and ego_speed_mps <= park_speed_threshold_mps:
        overridden["acc"] = 0.0
        overridden["steer_rate"] = 0.0
        return overridden

    overridden["acc"] = min(float(overridden.get("acc", 0.0)), brake_accel_mps2)

    try:
        position_tolerance_m = float(info.get("task_position_tolerance_m", 3.0))
    except (TypeError, ValueError):
        position_tolerance_m = 3.0
    if position_error_m <= max(position_tolerance_m * 1.5, 4.0):
        overridden["steer_rate"] = 0.0
    return overridden
