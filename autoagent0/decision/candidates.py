from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from autoagent0.adapters.hugsim.candidate_visuals import get_candidate_visual_style


PLAN_DT_SEC = 0.5


def summarize_candidate(
    points: List[Sequence[float]],
    current_ego_speed_mps: Optional[float] = None,
) -> Dict[str, object]:
    if not points:
        return {
            "num_points": 0,
            "start": None,
            "end": None,
            "min_x": None,
            "max_x": None,
            "min_y": None,
            "max_y": None,
            "delta_x": None,
            "delta_y": None,
            "path_length_m": None,
            "forward_progress_m": None,
            "first_step_m": None,
            "first_step_speed_mps": None,
            "avg_speed_mps": None,
            "max_step_speed_mps": None,
            "speed_delta_vs_ego_mps": None,
            "first_step_accel_vs_ego_mps2": None,
        }

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    start = [round(xs[0], 3), round(ys[0], 3)]
    end = [round(xs[-1], 3), round(ys[-1], 3)]
    points_arr = np.asarray(points, dtype=np.float32)
    step_distances = np.linalg.norm(np.diff(points_arr, axis=0), axis=1) if len(points_arr) > 1 else np.zeros((0,), dtype=np.float32)
    path_length_m = float(step_distances.sum())
    first_step_m = float(step_distances[0]) if len(step_distances) > 0 else 0.0
    first_step_speed_mps = first_step_m / PLAN_DT_SEC
    avg_speed_mps = path_length_m / (len(step_distances) * PLAN_DT_SEC) if len(step_distances) > 0 else 0.0
    max_step_speed_mps = float(step_distances.max()) / PLAN_DT_SEC if len(step_distances) > 0 else 0.0
    speed_delta_vs_ego_mps = None
    first_step_accel_vs_ego_mps2 = None
    if current_ego_speed_mps is not None:
        speed_delta_vs_ego_mps = first_step_speed_mps - float(current_ego_speed_mps)
        first_step_accel_vs_ego_mps2 = speed_delta_vs_ego_mps / PLAN_DT_SEC
    return {
        "num_points": len(points),
        "start": start,
        "end": end,
        "min_x": round(min(xs), 3),
        "max_x": round(max(xs), 3),
        "min_y": round(min(ys), 3),
        "max_y": round(max(ys), 3),
        "delta_x": round(end[0] - start[0], 3),
        "delta_y": round(end[1] - start[1], 3),
        "path_length_m": round(path_length_m, 3),
        "forward_progress_m": round(max(0.0, end[1] - start[1]), 3),
        "first_step_m": round(first_step_m, 3),
        "first_step_speed_mps": round(first_step_speed_mps, 3),
        "avg_speed_mps": round(avg_speed_mps, 3),
        "max_step_speed_mps": round(max_step_speed_mps, 3),
        "speed_delta_vs_ego_mps": None if speed_delta_vs_ego_mps is None else round(speed_delta_vs_ego_mps, 3),
        "first_step_accel_vs_ego_mps2": None if first_step_accel_vs_ego_mps2 is None else round(first_step_accel_vs_ego_mps2, 3),
    }


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def format_candidate_text(candidate_rows: Sequence[Dict[str, object]]) -> str:
    lines = []
    for row in candidate_rows:
        s = row["summary"]
        line = (
            f"- candidate_{row['candidate_index']} | color={row['color_name']} | source={row['source']} | "
        )
        line += (
            f"num_points={s['num_points']} | "
            f"start={s['start']} | end={s['end']} | "
            f"x_range=[{s['min_x']},{s['max_x']}] | y_range=[{s['min_y']},{s['max_y']}] | "
            f"delta=({s['delta_x']},{s['delta_y']}) | "
            f"path_length_m={s['path_length_m']} | "
            f"forward_progress_m={s['forward_progress_m']}"
        )
        lines.append(line)
    return "\n".join(lines)


def build_candidate_rows(
    candidates: Sequence[Dict[str, object]],
    current_ego_speed_mps: Optional[float] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    current_rank = 0
    for rank, candidate in enumerate(candidates):
        source = str(candidate.get("source", "current_rap"))
        style = get_candidate_visual_style(source, current_rank)
        if source != "carry_prev":
            current_rank += 1
        plan = np.asarray(candidate["local_plan"], dtype=np.float32)
        row = {
            "candidate_index": rank,
            "candidate_rank": rank,
            "proposal_index": None if candidate.get("proposal_index") is None else int(candidate["proposal_index"]),
            "source": source,
            "color_name": style.color_name,
            "color_bgr": list(style.color_bgr),
            "local_plan": plan.tolist(),
            "execution_plan": np.asarray(candidate.get("execution_plan", candidate["local_plan"]), dtype=np.float32).tolist(),
            "proposal_score": float(candidate.get("proposal_score", 0.0)),
            "rap_score": float(candidate.get("proposal_score", 0.0)),
            "origin_selected_score_raw": (
                None if candidate.get("origin_selected_score_raw") is None else float(candidate["origin_selected_score_raw"])
            ),
            "q_score": None if candidate.get("q_score") is None else float(candidate["q_score"]),
        }
        row["summary"] = summarize_candidate(row["local_plan"], current_ego_speed_mps=current_ego_speed_mps)
        rows.append(row)
    return rows


def select_representative_candidate_row(candidate_rows: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not candidate_rows:
        return None
    current_rows = [
        row
        for row in candidate_rows
        if str(row.get("source", "")).strip().lower() not in {"carry_prev"}
        and not str(row.get("source", "")).strip().lower().startswith("default_fallback_")
    ]
    if current_rows:
        candidate_rows = current_rows
    for row in candidate_rows:
        try:
            if int(row.get("candidate_rank", -1)) == 0:
                return dict(row)
        except Exception:
            continue
    return dict(candidate_rows[0])


def family_rows_for_planner_gate(candidate_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [
        dict(row)
        for row in candidate_rows
        if str(row.get("source", "")).strip().lower() not in {"carry_prev"}
        and not str(row.get("source", "")).strip().lower().startswith("default_fallback_")
    ]
    if rows:
        return rows
    return [dict(row) for row in candidate_rows]


def dedupe_gate_candidates(candidate_rows: Sequence[Dict[str, object]], *, limit: int = 3) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    seen = set()
    for row in candidate_rows:
        plan = np.asarray(row.get("local_plan", []), dtype=np.float32)
        summary = summarize_candidate(plan.tolist())
        key = (
            round(float(summary.get("path_length_m", 0.0) or 0.0), 1),
            round(float(summary.get("forward_progress_m", 0.0) or 0.0), 1),
            tuple(round(float(v), 1) for v in (summary.get("end") or [0.0, 0.0])),
        )
        if key in seen:
            continue
        seen.add(key)
        row_copy = dict(row)
        row_copy["summary"] = summary
        deduped.append(row_copy)
        if len(deduped) >= max(1, int(limit)):
            break
    return deduped


def planner_gate_family_debug(
    candidate_rows: Sequence[Dict[str, object]],
    *,
    limit: int = 10,
) -> List[Dict[str, object]]:
    debug_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(candidate_rows[: max(1, int(limit))]):
        plan = np.asarray(row.get("local_plan", []), dtype=np.float32)
        summary = summarize_candidate(plan.tolist())
        debug_rows.append(
            {
                "option_index": idx,
                "candidate_index": row.get("candidate_index"),
                "candidate_rank": row.get("candidate_rank"),
                "source": row.get("source"),
                "proposal_index": row.get("proposal_index"),
                "proposal_score": float(row.get("proposal_score", 0.0)),
                "summary": summary,
            }
        )
    return debug_rows

