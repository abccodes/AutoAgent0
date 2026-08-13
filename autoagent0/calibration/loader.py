"""Load per-frame uncertainty + outcome labels from a corpus of HUGSIM runs."""

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from autoagent0.calibration.labels import (
    annotate_frames_with_score_details,
    compute_future_failure_labels,
    frame_failure_components,
    planned_object_overlap_evidence,
)
from autoagent0.calibration.features import (
    post_selection_features,
    raw_observation_features,
    score_features,
    temporal_change,
)


REQUIRED_FILES = ("data.pkl", "eval.json")


def find_run_dirs(root: str) -> List[str]:
    """Return run directories under `root` that contain both data.pkl and eval.json."""

    runs: List[str] = []
    if not os.path.isdir(root):
        return runs
    for dirpath, _dirnames, filenames in os.walk(root):
        if all(name in filenames for name in REQUIRED_FILES):
            runs.append(dirpath)
    return sorted(runs)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _frame_uncertainty(frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    debug = frame.get("planner_debug") or {}
    payload = debug.get("autoagent0_frame_uncertainty")
    if isinstance(payload, dict):
        return payload
    return None


def _max_silhouette(meta: Dict[str, Any]) -> Optional[float]:
    modes = meta.get("modes") if isinstance(meta, dict) else None
    if not isinstance(modes, dict):
        return None
    sil_per_k = modes.get("silhouette_per_k")
    if not isinstance(sil_per_k, dict):
        return None
    candidates: List[float] = []
    for k, score in sil_per_k.items():
        try:
            if int(k) <= 1:
                continue
        except (TypeError, ValueError):
            continue
        v = _safe_float(score)
        if v is not None:
            candidates.append(v)
    if not candidates:
        return None
    return max(candidates)


def _critic_rejected(frame: Dict[str, Any]) -> Optional[bool]:
    debug = frame.get("planner_debug") or {}
    payload = debug.get("autoagent0_default_critique")
    if not isinstance(payload, dict):
        return None
    rejected = payload.get("autoagent0_critique_rejected")
    if isinstance(rejected, bool):
        return rejected
    accepted = payload.get("autoagent0_critique_accepted")
    if isinstance(accepted, bool):
        return not accepted
    return None


def _execution_outcome(frame: Dict[str, Any]) -> Dict[str, Any]:
    outcome = frame.get("execution_outcome")
    return outcome if isinstance(outcome, dict) else {}


def _score_detail(frame: Dict[str, Any], key: str) -> Any:
    details = frame.get("score_details")
    return details.get(key) if isinstance(details, dict) else None


def load_run(run_dir: str, *, horizon_steps: int = 20) -> pd.DataFrame:
    data_pkl = os.path.join(run_dir, "data.pkl")
    eval_json = os.path.join(run_dir, "eval.json")

    with open(data_pkl, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, list) and data:
        save_data = data[0]
    else:
        save_data = data
    frames = save_data.get("frames", []) if isinstance(save_data, dict) else []

    with open(eval_json, "r") as fh:
        eval_payload = json.load(fh)
    details = eval_payload.get("details") if isinstance(eval_payload, dict) else None
    if isinstance(details, dict):
        annotate_frames_with_score_details(frames, details)

    labels = compute_future_failure_labels(frames, horizon_steps=horizon_steps)
    run_id = os.path.relpath(run_dir, os.path.dirname(os.path.dirname(run_dir)))
    run_name = os.path.basename(run_dir)
    scene_group = run_name.split("_", 1)[0]

    rows: List[Dict[str, Any]] = []
    previous_selected_world: Optional[np.ndarray] = None
    for idx, frame in enumerate(frames):
        unc = _frame_uncertainty(frame)
        if unc is None:
            continue
        meta = unc.get("metadata") or {}
        debug = frame.get("planner_debug") or {}
        outcome = _execution_outcome(frame)
        failure_components = frame_failure_components(frame)
        object_overlap_evidence = (
            planned_object_overlap_evidence(frames, idx)
            if labels["nc_failure_now"][idx]
            else None
        )
        change, previous_selected_world = temporal_change(
            debug.get("local_plan"),
            debug.get("overlay_plan_origin_pose"),
            previous_selected_world,
        )
        rows.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "scene_group": scene_group,
                "frame_idx": int(idx),
                "timestamp_sec": _safe_float(frame.get("time_stamp")),
                "intra_m": _safe_float(unc.get("intra_learned_m")),
                "cross_m": _safe_float(unc.get("cross_family_m")),
                "mode_count": int(unc.get("mode_count") or 1),
                "max_silhouette": _max_silhouette(meta),
                "zone": str(unc.get("routing_zone") or ""),
                "future_unsafe": int(labels["future_unsafe"][idx]),
                "future_collision": int(labels["future_collision"][idx]),
                "future_nc_failure": int(labels["future_nc_failure"][idx]),
                "future_dac_failure": int(labels["future_dac_failure"][idx]),
                "unsafe_now": int(labels["unsafe_now"][idx]),
                "nc_failure_now": int(labels["nc_failure_now"][idx]),
                "dac_failure_now": int(labels["dac_failure_now"][idx]),
                "steps_to_unsafe": labels["steps_to_unsafe"][idx],
                "critic_rejected": _critic_rejected(frame),
                "collision_now": failure_components["collision"],
                "execution_outcome_available": bool(outcome),
                "terminated_after_step": bool(outcome.get("terminated", False)),
                "truncated_after_step": bool(outcome.get("truncated", False)),
                "runner_timeout_after_step": bool(outcome.get("runner_timeout", False)),
                "termination_reason": outcome.get("termination_reason"),
                "nc_failure_type": _score_detail(frame, "nc_failure_type"),
                "nc_failure_step": _safe_float(_score_detail(frame, "nc_failure_step")),
                "nc_object_overlap_evidence": object_overlap_evidence,
                "temporal_selected_change_m": change,
                **score_features(debug.get("topk_scores")),
                **post_selection_features(debug),
                **raw_observation_features(debug),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "scene_group",
                "frame_idx",
                "timestamp_sec",
                "intra_m",
                "cross_m",
                "mode_count",
                "max_silhouette",
                "zone",
                "future_unsafe",
                "future_collision",
                "future_nc_failure",
                "future_dac_failure",
                "unsafe_now",
                "nc_failure_now",
                "dac_failure_now",
                "steps_to_unsafe",
                "critic_rejected",
                "collision_now",
                "execution_outcome_available",
                "terminated_after_step",
                "truncated_after_step",
                "runner_timeout_after_step",
                "termination_reason",
                "nc_failure_type",
                "nc_failure_step",
                "nc_object_overlap_evidence",
                "temporal_selected_change_m",
                "score_entropy",
                "effective_candidate_count",
                "score_margin_normalized",
                "post_dispersion_m",
                "post_pairwise_mean_m",
                "post_pairwise_p90_m",
                "post_endpoint_mean_m",
                "post_primitive_distance_m",
                "raw_candidate_count",
                "raw_dispersion_m",
                "raw_dispersion_normalized",
                "raw_pairwise_mean_m",
                "raw_pairwise_p90_m",
                "raw_endpoint_mean_m",
                "raw_score_entropy",
                "raw_effective_candidate_count",
                "raw_score_margin_normalized",
                "raw_primitive_distance_m",
            ]
        )
    return pd.DataFrame(rows)


def load_corpus(roots: Iterable[str], *, horizon_steps: int = 20) -> pd.DataFrame:
    run_dirs: List[str] = []
    for root in roots:
        if any(ch in root for ch in "*?["):
            for match in sorted(glob.glob(root, recursive=True)):
                if os.path.isdir(match):
                    run_dirs.extend(find_run_dirs(match))
        else:
            run_dirs.extend(find_run_dirs(root))

    seen: set = set()
    deduped: List[str] = []
    for path in run_dirs:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)

    frames: List[pd.DataFrame] = []
    for run_dir in deduped:
        try:
            df = load_run(run_dir, horizon_steps=horizon_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"[loader] skip {run_dir}: {exc!r}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    populated = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(populated, ignore_index=True).reindex(columns=columns)
