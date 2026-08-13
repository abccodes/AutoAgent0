"""Per-frame binary safety labels derived from the score_calculator output."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


UNSAFE_NC_THRESHOLD = 1.0
UNSAFE_DAC_THRESHOLD = 1.0


def _per_timestep_score_array(frame: Dict[str, Any], key: str) -> np.ndarray:
    details = frame.get("score_details")
    if isinstance(details, dict):
        seq = details.get(key)
        if isinstance(seq, (list, tuple, np.ndarray)):
            return np.asarray(seq, dtype=np.float64)
    value = frame.get(key)
    if isinstance(value, (int, float)):
        return np.asarray([float(value)], dtype=np.float64)
    return np.zeros((0,), dtype=np.float64)


def _frame_is_unsafe(frame: Dict[str, Any]) -> bool:
    components = frame_failure_components(frame)
    return any(components.values())


def _executed_collision(frame: Dict[str, Any]) -> bool:
    outcome = frame.get("execution_outcome")
    if isinstance(outcome, dict) and "collision" in outcome:
        return bool(outcome["collision"])
    if "collision_after_step" in frame:
        return bool(frame["collision_after_step"])
    return bool(frame.get("collision", False))


def _oriented_rectangle(box: Sequence[float]) -> ShapelyPolygon:
    x, y, _z, width, length, _height, yaw = box
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x_offsets = np.asarray([length / 2, length / 2, -length / 2, -length / 2])
    y_offsets = np.asarray([width / 2, -width / 2, -width / 2, width / 2])
    points = np.stack(
        [
            x + x_offsets * cos_yaw - y_offsets * sin_yaw,
            y + x_offsets * sin_yaw + y_offsets * cos_yaw,
        ],
        axis=1,
    )
    return ShapelyPolygon(points)


def planned_object_overlap_evidence(
    frames: Sequence[Dict[str, Any]], frame_idx: int
) -> Optional[bool]:
    """Replay only the object-box portion of evaluator NC for one saved plan.

    ``False`` on an evaluator NC failure implies a static-background failure.
    ``True`` only proves that an object overlap exists somewhere in the plan;
    static geometry may still be the evaluator's first failure.
    """

    try:
        frame = frames[frame_idx]
        ego_box = np.asarray(frame["ego_box"], dtype=np.float64)
        plan_payload = frame["planned_traj"]
        trajectory = np.asarray(plan_payload["traj"], dtype=np.float64)
        timestep = float(plan_payload["timestep"])
        timestamp = float(frame["time_stamp"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if trajectory.ndim != 2 or trajectory.shape[0] < 1 or trajectory.shape[1] < 3:
        return None

    planned = np.concatenate((ego_box[[0, 1, 6]][None], trajectory[:, :3]), axis=0)
    if np.linalg.norm(planned[-1, :2] - planned[0, :2]) < 1.0:
        planned[:, 2] = planned[0, 2]

    planned_last_timestamp = timestamp + len(trajectory) * timestep
    observation_lists: List[List[Sequence[float]]] = []
    current_timestamp = timestamp
    current_frame_idx = frame_idx
    while current_timestamp <= planned_last_timestamp + 1e-5:
        if current_frame_idx >= len(frames):
            break
        observed_frame = frames[current_frame_idx]
        observed_timestamp = observed_frame.get("time_stamp")
        if observed_timestamp is not None and abs(
            current_timestamp - float(observed_timestamp)
        ) < 1e-5:
            boxes = observed_frame.get("obj_boxes") or []
            names = observed_frame.get("obj_names") or ["car"] * len(boxes)
            observation_lists.append(
                [box for box, name in zip(boxes, names) if name == "car"]
            )
            current_timestamp += timestep
        current_frame_idx += 1

    ego_width, ego_length, ego_height = ego_box[3], ego_box[4], ego_box[5]
    for step, (x, y, yaw) in enumerate(planned):
        ego_polygon = _oriented_rectangle(
            [x, y, ego_box[2], ego_width, ego_length, ego_height, yaw]
        )
        if not observation_lists:
            obstacles = []
        else:
            obstacles = observation_lists[min(step, len(observation_lists) - 1)]
        if any(ego_polygon.intersects(_oriented_rectangle(box)) for box in obstacles):
            return True
    return False


def frame_failure_components(frame: Dict[str, Any]) -> Dict[str, bool]:
    """Return the auditable collision/NC/DAC failure flags for one frame."""

    nc = _per_timestep_score_array(frame, "nc")
    dac = _per_timestep_score_array(frame, "dac")
    return {
        "collision": _executed_collision(frame),
        "nc_failure": bool(nc.size and float(nc.min()) < UNSAFE_NC_THRESHOLD),
        "dac_failure": bool(dac.size and float(dac.min()) < UNSAFE_DAC_THRESHOLD),
    }


def compute_future_failure_labels(
    frames: Sequence[Dict[str, Any]],
    *,
    horizon_steps: int = 20,
) -> Dict[str, List[Any]]:
    """Return decomposed current/future failures and time to the next unsafe frame."""

    components = [frame_failure_components(frame) for frame in frames]
    unsafe = [any(item.values()) for item in components]
    horizon = max(1, int(horizon_steps))
    labels: Dict[str, List[Any]] = {
        "unsafe_now": [int(value) for value in unsafe],
        "collision_now": [int(item["collision"]) for item in components],
        "nc_failure_now": [int(item["nc_failure"]) for item in components],
        "dac_failure_now": [int(item["dac_failure"]) for item in components],
        "future_unsafe": [],
        "future_collision": [],
        "future_nc_failure": [],
        "future_dac_failure": [],
        "steps_to_unsafe": [],
    }
    future_keys = {
        "future_collision": "collision",
        "future_nc_failure": "nc_failure",
        "future_dac_failure": "dac_failure",
    }
    for idx in range(len(frames)):
        window_end = min(len(frames), idx + horizon)
        labels["future_unsafe"].append(int(any(unsafe[idx:window_end])))
        for output_key, component_key in future_keys.items():
            labels[output_key].append(
                int(any(item[component_key] for item in components[idx:window_end]))
            )
        next_offsets = [
            offset
            for offset, value in enumerate(unsafe[idx:window_end])
            if value
        ]
        labels["steps_to_unsafe"].append(next_offsets[0] if next_offsets else None)
    return labels


def compute_future_unsafe_label(
    frames: Sequence[Dict[str, Any]],
    *,
    horizon_steps: int = 20,
) -> List[int]:
    """Return a 0/1 label per frame: 1 iff any frame in [i, i+horizon_steps) is unsafe.

    Unsafe is `collision == True` OR per-timestep `nc < 1` OR per-timestep `dac < 1`.
    The per-timestep arrays land in `frame['score_details']` when score_calculator
    has annotated the run; otherwise we fall back to the scalar `collision` flag.
    """

    return compute_future_failure_labels(
        frames, horizon_steps=horizon_steps
    )["future_unsafe"]


def annotate_frames_with_score_details(
    frames: Sequence[Dict[str, Any]],
    details: Dict[float, Dict[str, Any]],
) -> None:
    """Merge per-timestamp score dict from eval.json into the corresponding frames.

    `details` is keyed by timestamp (float). Each frame has its own `time_stamp`.
    Numeric score terms remain one-element arrays for compatibility with older
    logs. Evaluator provenance such as NC failure type and step stays typed.
    """

    if not details:
        return
    by_ts = {float(ts): scores for ts, scores in details.items()}
    for frame in frames:
        ts = frame.get("time_stamp")
        if ts is None:
            continue
        scores = by_ts.get(float(ts))
        if scores is None:
            continue
        frame["score_details"] = {
            key: [float(value)] if key in {"nc", "dac", "ttc", "c", "pdms"} else value
            for key, value in scores.items()
        }
