"""Per-frame binary safety labels derived from the score_calculator output."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


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


def frame_failure_components(frame: Dict[str, Any]) -> Dict[str, bool]:
    """Return the auditable collision/NC/DAC failure flags for one frame."""

    nc = _per_timestep_score_array(frame, "nc")
    dac = _per_timestep_score_array(frame, "dac")
    return {
        "collision": bool(frame.get("collision", False)),
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
    details: Dict[float, Dict[str, float]],
) -> None:
    """Merge per-timestamp score dict from eval.json into the corresponding frames.

    `details` is keyed by timestamp (float). Each frame has its own `time_stamp`.
    We attach the matching score dict under `frame['score_details']` so that
    `_frame_is_unsafe` finds the per-timestep arrays.
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
            key: [float(value)] for key, value in scores.items()
        }
