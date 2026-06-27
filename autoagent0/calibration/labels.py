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
    if bool(frame.get("collision", False)):
        return True
    nc = _per_timestep_score_array(frame, "nc")
    dac = _per_timestep_score_array(frame, "dac")
    if nc.size and float(nc.min()) < UNSAFE_NC_THRESHOLD:
        return True
    if dac.size and float(dac.min()) < UNSAFE_DAC_THRESHOLD:
        return True
    return False


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

    n = len(frames)
    raw = [_frame_is_unsafe(frames[i]) for i in range(n)]
    horizon = max(1, int(horizon_steps))
    labels: List[int] = []
    for i in range(n):
        window_end = min(n, i + horizon)
        labels.append(int(any(raw[i:window_end])))
    return labels


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
