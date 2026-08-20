#!/usr/bin/env python3
"""Compare repeated HUGSIM uncertainty-off and passive-observe controls."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METRICS = ("nc", "dac", "ttc", "c", "pdms", "rc", "hdscore")
LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", nargs="+", required=True, help="Repeated off run roots")
    parser.add_argument("--observe", nargs="+", required=True, help="Repeated observe run roots")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan-tolerance-m", type=float, default=1e-6)
    return parser.parse_args()


def _load_frames(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    return list(payload.get("frames", [])) if isinstance(payload, dict) else []


def _log_duration(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    timestamps: List[dt.datetime] = []
    loop_start: Optional[dt.datetime] = None
    shutdown: Optional[dt.datetime] = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_TIME.match(line)
        if match:
            timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            timestamps.append(timestamp)
            if "Opened persistent scene FIFOs" in line:
                loop_start = timestamp
            elif "Received shutdown signal" in line:
                shutdown = timestamp
    if loop_start is not None and shutdown is not None and shutdown >= loop_start:
        return (shutdown - loop_start).total_seconds()
    if len(timestamps) < 2:
        return None
    return (timestamps[-1] - timestamps[0]).total_seconds()


def _run_rows(condition: str, roots: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    rows: List[Dict[str, Any]] = []
    frames_by_run: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for repeat, root_text in enumerate(roots, start=1):
        root = Path(root_text)
        for eval_path in sorted(root.glob("*/eval.json")):
            route = eval_path.parent.name
            frames = _load_frames(eval_path.parent / "data.pkl")
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            outcome = frames[-1].get("execution_outcome", {}) if frames else {}
            duration = _log_duration(eval_path.parent / "drivor_client.log")
            row: Dict[str, Any] = {
                "condition": condition,
                "repeat": repeat,
                "route": route,
                "run_root": str(root),
                "frame_count": len(frames),
                "termination_reason": outcome.get("termination_reason"),
                "collision": bool(outcome.get("collision", False)),
                "route_departure": bool(outcome.get("route_departure", False)),
                "route_complete": bool(outcome.get("route_complete", False)),
                "wallclock_sec": duration,
                "wallclock_sec_per_frame": duration / len(frames) if duration is not None and frames else None,
            }
            row.update({metric: evaluation.get(metric) for metric in METRICS})
            rows.append(row)
            frames_by_run[(condition, f"{repeat}:{route}")] = frames
    return rows, frames_by_run


def _local_plan(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    plan = (frame.get("planner_debug") or {}).get("local_plan")
    try:
        array = np.asarray(plan, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if array.ndim == 2 and array.shape[1] >= 2 and len(array) else None


def _trajectory_comparison(
    left: List[Dict[str, Any]], right: List[Dict[str, Any]], tolerance: float
) -> Dict[str, Any]:
    distances: List[float] = []
    first_divergence: Optional[int] = None
    for frame_idx, (left_frame, right_frame) in enumerate(zip(left, right)):
        left_plan, right_plan = _local_plan(left_frame), _local_plan(right_frame)
        if left_plan is None or right_plan is None:
            continue
        horizon = min(len(left_plan), len(right_plan))
        distance = float(
            np.linalg.norm(left_plan[:horizon, :2] - right_plan[:horizon, :2], axis=1).mean()
        )
        distances.append(distance)
        if first_divergence is None and distance > tolerance:
            first_divergence = frame_idx
    return {
        "common_frames": min(len(left), len(right)),
        "compared_plan_frames": len(distances),
        "first_plan_divergence_frame": first_divergence,
        "mean_selected_plan_distance_m": float(np.mean(distances)) if distances else None,
        "max_selected_plan_distance_m": float(np.max(distances)) if distances else None,
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if finite else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    off_rows, off_frames = _run_rows("off", args.off)
    observe_rows, observe_frames = _run_rows("observe", args.observe)
    rows = off_rows + observe_rows
    if not off_rows or not observe_rows:
        raise SystemExit("both conditions must contain completed eval.json runs")

    route_columns = list(rows[0].keys())
    with (out_dir / "route_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=route_columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["route"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (condition, route), group in sorted(grouped.items()):
        reasons = {str(row["termination_reason"]) for row in group}
        summary: Dict[str, Any] = {
            "condition": condition,
            "route": route,
            "repeats": len(group),
            "terminal_outcome_agreement": len(reasons) == 1,
            "terminal_outcomes": ",".join(sorted(reasons)),
            "frame_count_mean": _mean(row["frame_count"] for row in group),
            "frame_count_std": _std(row["frame_count"] for row in group),
            "wallclock_sec_per_frame_mean": _mean(row["wallclock_sec_per_frame"] for row in group),
            "wallclock_sec_per_frame_std": _std(row["wallclock_sec_per_frame"] for row in group),
        }
        for metric in METRICS:
            summary[f"{metric}_mean"] = _mean(row[metric] for row in group)
            summary[f"{metric}_std"] = _std(row[metric] for row in group)
        summary_rows.append(summary)

    with (out_dir / "route_condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    trajectory_rows: List[Dict[str, Any]] = []
    for condition, frames_by_run in (("off", off_frames), ("observe", observe_frames)):
        by_route: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = defaultdict(list)
        for (_condition, key), frames in frames_by_run.items():
            repeat_text, route = key.split(":", 1)
            by_route[route].append((int(repeat_text), frames))
        for route, repeats in sorted(by_route.items()):
            for (left_id, left), (right_id, right) in itertools.combinations(sorted(repeats), 2):
                trajectory_rows.append(
                    {
                        "condition": condition,
                        "route": route,
                        "left_repeat": left_id,
                        "right_repeat": right_id,
                        **_trajectory_comparison(left, right, args.plan_tolerance_m),
                    }
                )

    if trajectory_rows:
        with (out_dir / "trajectory_repeatability.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_rows)

    conditions: Dict[str, Dict[str, Optional[float]]] = {}
    for condition in ("off", "observe"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            **{f"{metric}_mean": _mean(row[metric] for row in condition_rows) for metric in METRICS},
            **{f"{metric}_std": _std(row[metric] for row in condition_rows) for metric in METRICS},
            "wallclock_sec_per_frame_mean": _mean(row["wallclock_sec_per_frame"] for row in condition_rows),
            "terminal_outcome_agreement_rate": _mean(
                float(summary["terminal_outcome_agreement"])
                for summary in summary_rows
                if summary["condition"] == condition
            ),
        }

    route_summaries = {(row["condition"], row["route"]): row for row in summary_rows}
    common_routes = sorted(
        {route for condition, route in route_summaries if condition == "off"}
        & {route for condition, route in route_summaries if condition == "observe"}
    )
    deltas = {
        metric: [
            route_summaries[("observe", route)][f"{metric}_mean"]
            - route_summaries[("off", route)][f"{metric}_mean"]
            for route in common_routes
        ]
        for metric in METRICS
    }
    result = {
        "off_roots": args.off,
        "observe_roots": args.observe,
        "routes": common_routes,
        "conditions": conditions,
        "observe_minus_off_route_mean": {metric: _mean(values) for metric, values in deltas.items()},
        "observe_minus_off_route_std": {metric: _std(values) for metric, values in deltas.items()},
        "trajectory_pair_count": len(trajectory_rows),
        "trajectory_first_divergence_frame_median": _median(
            row["first_plan_divergence_frame"] for row in trajectory_rows
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Uncertainty repeatability controls",
        "",
        f"- Off repeats: **{len(args.off)}**",
        f"- Passive-observe repeats: **{len(args.observe)}**",
        f"- Common routes: **{len(common_routes)}**",
        "",
        "| metric | off mean | observe mean | route-mean delta | delta std |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {_fmt(conditions['off'][f'{metric}_mean'])} | "
            f"{_fmt(conditions['observe'][f'{metric}_mean'])} | "
            f"{_fmt(result['observe_minus_off_route_mean'][metric])} | "
            f"{_fmt(result['observe_minus_off_route_std'][metric])} |"
        )
    lines.extend(
        [
            "",
            "## Runtime and determinism",
            "",
            f"- Off wall time/frame: **{_fmt(conditions['off']['wallclock_sec_per_frame_mean'], 3)} s**",
            f"- Observe wall time/frame: **{_fmt(conditions['observe']['wallclock_sec_per_frame_mean'], 3)} s**",
            f"- Off route-outcome agreement: **{_fmt(conditions['off']['terminal_outcome_agreement_rate'], 3)}**",
            f"- Observe route-outcome agreement: **{_fmt(conditions['observe']['terminal_outcome_agreement_rate'], 3)}**",
            f"- Within-condition trajectory pairs: **{len(trajectory_rows)}**",
            "",
            "A passive effect is not identifiable when its cross-condition delta is comparable to within-condition variance or when terminal outcomes are not repeatable.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[repeatability] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
