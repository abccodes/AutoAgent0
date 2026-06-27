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
    compute_future_unsafe_label,
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
    if isinstance(payload, dict):
        action = payload.get("action")
        if isinstance(action, str):
            return action.lower() in ("redesign", "reject", "intervene")
    return None


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

    labels = compute_future_unsafe_label(frames, horizon_steps=horizon_steps)
    run_id = os.path.relpath(run_dir, os.path.dirname(os.path.dirname(run_dir)))

    rows: List[Dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        unc = _frame_uncertainty(frame)
        if unc is None:
            continue
        meta = unc.get("metadata") or {}
        rows.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "frame_idx": int(idx),
                "intra_m": _safe_float(unc.get("intra_learned_m")),
                "cross_m": _safe_float(unc.get("cross_family_m")),
                "mode_count": int(unc.get("mode_count") or 1),
                "max_silhouette": _max_silhouette(meta),
                "zone": str(unc.get("routing_zone") or ""),
                "future_unsafe": int(labels[idx]) if idx < len(labels) else 0,
                "critic_rejected": _critic_rejected(frame),
                "collision_now": bool(frame.get("collision", False)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "frame_idx",
                "intra_m",
                "cross_m",
                "mode_count",
                "max_silhouette",
                "zone",
                "future_unsafe",
                "critic_rejected",
                "collision_now",
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
    return pd.concat(frames, ignore_index=True)
