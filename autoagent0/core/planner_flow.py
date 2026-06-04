from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LearnedPlannerSelection:
    selected_row: Dict[str, object]
    selected_plan: np.ndarray
    selected_idx: Optional[int]
    selected_score: float
    selected_score_raw: float
    selected_source: str
    selected_planner: str
    candidate_rows: List[Dict[str, object]]
    planner_gate_result: Dict[str, object]
    default_selected_index: int
    default_selected_source: str
    selection_debug: Dict[str, object]


def disabled_planner_gate_result() -> Dict[str, object]:
    return {
        "selected_planner": "learned",
        "confidence": None,
        "reasoning": "planner_gate_disabled",
        "elapsed_sec": 0.0,
        "error": None,
        "timed_out": False,
        "prompt_char_count": 0,
        "image_count": 0,
    }


def choose_candidate_family(
    *,
    frame_index: int,
    camera_images: Dict[str, Any],
    info: Dict[str, object],
    vlm_selector: Any,
    learned_candidate_rows: Sequence[Dict[str, object]],
    rule_based_candidate_rows: Sequence[Dict[str, object]],
    rule_based_merge_enabled: bool,
    planner_gate_enabled: bool,
) -> tuple[List[Dict[str, object]], str, Dict[str, object]]:
    planner_gate_result = disabled_planner_gate_result()
    selected_planner = "learned"
    candidate_rows = list(learned_candidate_rows)

    if rule_based_merge_enabled and not planner_gate_enabled and rule_based_candidate_rows:
        candidate_rows = list(learned_candidate_rows) + list(rule_based_candidate_rows)

    if planner_gate_enabled:
        planner_gate_result = vlm_selector.maybe_select_planner(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
        )
        selected_planner = str(planner_gate_result.get("selected_planner", "learned"))
        if selected_planner == "rule_based" and rule_based_candidate_rows:
            candidate_rows = list(rule_based_candidate_rows)
        else:
            selected_planner = "learned"
            candidate_rows = list(learned_candidate_rows)

    return candidate_rows, selected_planner, planner_gate_result


def default_selection_for_family(
    *,
    candidate_rows: Sequence[Dict[str, object]],
    selected_planner: str,
    scores: np.ndarray,
    learned_source_name: str,
    learned_default_source: str,
    strict_learned_argmax_lookup: bool = False,
) -> tuple[int, str]:
    if selected_planner == "rule_based":
        return (
            int(max(range(len(candidate_rows)), key=lambda idx: float(candidate_rows[idx].get("proposal_score", 0.0)))),
            "fallback_rule_based_argmax",
        )

    best_idx = int(np.argmax(scores))
    for idx, row in enumerate(candidate_rows):
        if row.get("source") == learned_source_name and row.get("proposal_index") == best_idx:
            return int(idx), learned_default_source
        if not strict_learned_argmax_lookup and row.get("proposal_index") is not None:
            try:
                if int(row.get("proposal_index")) == best_idx:
                    return int(idx), learned_default_source
            except Exception:
                pass

    if strict_learned_argmax_lookup:
        raise StopIteration(f"Could not find learned argmax row for proposal_index={best_idx}")
    return 0, learned_default_source


def _selected_score_from_row(selected_row: Dict[str, object], *, fallback_key: str) -> tuple[float, float]:
    selected_score_value = selected_row.get("proposal_score")
    if selected_score_value is None:
        selected_score_value = selected_row.get(fallback_key, 0.0)
    selected_score = float(selected_score_value)

    selected_score_raw_value = selected_row.get("origin_selected_score_raw")
    if selected_score_raw_value is None:
        selected_score_raw_value = selected_score
    return selected_score, float(selected_score_raw_value)


def _planner_gate_debug(
    *,
    planner_gate_result: Dict[str, object],
    selected_planner: str,
    default_selected_index: int,
    default_selected_source: str,
    display_default_trajectories: bool,
    include_default_candidates: bool,
    allow_carry_previous: bool,
    previous_selected_source: Optional[str],
    selected_score_raw: float,
    q_key_prefix: bool,
) -> Dict[str, object]:
    debug = {
        "scoring_invoked": False,
        "execution_mode": (
            "planner_gate_failed_base_policy_fallback"
            if planner_gate_result.get("error") is not None
            else f"planner_gate_selected_{selected_planner}"
        ),
        "fallback_selected_idx": int(default_selected_index),
        "fallback_selected_source": default_selected_source,
        "planner_gate_selected_planner": selected_planner,
        "planner_gate_confidence": planner_gate_result.get("confidence"),
        "planner_gate_reasoning": planner_gate_result.get("reasoning"),
        "planner_gate_elapsed_sec": planner_gate_result.get("elapsed_sec"),
        "planner_gate_error": planner_gate_result.get("error"),
        "planner_gate_timed_out": planner_gate_result.get("timed_out"),
        "planner_gate_prompt_char_count": planner_gate_result.get("prompt_char_count"),
        "planner_gate_image_count": planner_gate_result.get("image_count"),
        "planner_gate_token_usage": planner_gate_result.get("token_usage"),
        "display_default_trajectories": bool(display_default_trajectories),
        "include_default_candidates": bool(include_default_candidates),
        "carry_previous_allowed": bool(allow_carry_previous),
        "previous_selected_source": previous_selected_source,
        "selected_score_raw": float(selected_score_raw),
        "vlm_failed": planner_gate_result.get("error") is not None,
    }
    debug["q_invoked_vlm" if q_key_prefix else "vlm_invoked"] = False
    return debug


def _scorer_debug(
    *,
    selection_result: Dict[str, object],
    planner_gate_result: Dict[str, object],
    selected_planner: str,
    default_selected_index: int,
    default_selected_source: str,
    display_default_trajectories: bool,
    include_default_candidates: bool,
    allow_carry_previous: bool,
    previous_selected_source: Optional[str],
    selected_score_raw: float,
    q_key_prefix: bool,
) -> Dict[str, object]:
    debug: Dict[str, object] = {
        "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
        "vlm_confidence": selection_result.get("vlm_confidence"),
        "vlm_reasoning": selection_result.get("vlm_reasoning"),
        "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
        "vlm_error": selection_result.get("vlm_error"),
        "vlm_candidate_count": selection_result.get("vlm_candidate_count"),
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
        "vlm_q_valid": selection_result.get("vlm_q_valid"),
        "vlm_timed_out": selection_result.get("vlm_timed_out"),
        "vlm_q_candidate_scores": selection_result.get("vlm_q_candidate_scores"),
        "vlm_q_best_candidate_index": selection_result.get("vlm_q_best_candidate_index"),
        "vlm_q_score_gap_to_carry": selection_result.get("vlm_q_score_gap_to_carry"),
        "vlm_q_score_gap_top2": selection_result.get("vlm_q_score_gap_top2"),
        "vlm_q_best_current_score": selection_result.get("vlm_q_best_current_score"),
        "vlm_q_carry_score": selection_result.get("vlm_q_carry_score"),
        "adaptive_replan_decision": selection_result.get("adaptive_replan_decision"),
        "carry_previous_valid": selection_result.get("carry_previous_valid"),
        "latency_timeline_record": selection_result.get("latency_timeline_record"),
        "execution_mode": selection_result.get("execution_mode"),
        "vlm_failed": selection_result.get("vlm_failed"),
        "fallback_selected_idx": int(default_selected_index),
        "fallback_selected_source": default_selected_source,
        "planner_gate_selected_planner": selected_planner,
        "planner_gate_confidence": planner_gate_result.get("confidence"),
        "planner_gate_reasoning": planner_gate_result.get("reasoning"),
        "planner_gate_elapsed_sec": planner_gate_result.get("elapsed_sec"),
        "planner_gate_error": planner_gate_result.get("error"),
        "planner_gate_timed_out": planner_gate_result.get("timed_out"),
        "planner_gate_prompt_char_count": planner_gate_result.get("prompt_char_count"),
        "planner_gate_image_count": planner_gate_result.get("image_count"),
        "planner_gate_token_usage": planner_gate_result.get("token_usage"),
        "display_default_trajectories": bool(display_default_trajectories),
        "include_default_candidates": bool(include_default_candidates),
        "carry_previous_allowed": bool(allow_carry_previous),
        "previous_selected_source": previous_selected_source,
        "selected_score_raw": float(selected_score_raw),
    }
    if q_key_prefix:
        debug.update(
            {
                "q_selected_idx": None,
                "q_selected_source": None,
                "q_candidate_scores": None,
                "q_carry_score": None,
                "q_best_current_score": None,
                "q_score_gap": None,
                "q_switch_margin": None,
                "q_selected_path_length": None,
                "q_best_current_path_length": None,
                "q_invoked_vlm": True,
            }
        )
    return debug


def run_learned_planner_selection(
    *,
    frame_index: int,
    camera_images: Dict[str, Any],
    info: Dict[str, object],
    vlm_selector: Any,
    scores: np.ndarray,
    learned_candidate_rows: Sequence[Dict[str, object]],
    rule_based_candidate_rows: Sequence[Dict[str, object]],
    rule_based_merge_enabled: bool,
    planner_gate_enabled: bool,
    vlm_enabled: bool,
    display_default_trajectories: bool,
    include_default_candidates: bool,
    allow_carry_previous: bool,
    previous_selected_source: Optional[str],
    learned_source_name: str,
    learned_default_source: str,
    score_fallback_key: str,
    planner_log_name: str,
    logger: Any,
    strict_learned_argmax_lookup: bool = False,
    q_key_prefix: bool = True,
) -> LearnedPlannerSelection:
    candidate_rows, selected_planner, planner_gate_result = choose_candidate_family(
        frame_index=frame_index,
        camera_images=camera_images,
        info=info,
        vlm_selector=vlm_selector,
        learned_candidate_rows=learned_candidate_rows,
        rule_based_candidate_rows=rule_based_candidate_rows,
        rule_based_merge_enabled=rule_based_merge_enabled,
        planner_gate_enabled=planner_gate_enabled,
    )

    candidate_sources = [str(row.get("source", "unknown")) for row in candidate_rows]
    logger.info(
        "%s planner gate frame=%s planner=%s confidence=%s error=%s timed_out=%s chosen_size=%d sources=%s",
        planner_log_name,
        frame_index,
        selected_planner,
        planner_gate_result.get("confidence"),
        planner_gate_result.get("error"),
        planner_gate_result.get("timed_out"),
        len(candidate_rows),
        candidate_sources,
    )

    default_selected_index, default_selected_source = default_selection_for_family(
        candidate_rows=candidate_rows,
        selected_planner=selected_planner,
        scores=scores,
        learned_source_name=learned_source_name,
        learned_default_source=learned_default_source,
        strict_learned_argmax_lookup=strict_learned_argmax_lookup,
    )

    if planner_gate_enabled:
        selected_row = dict(candidate_rows[default_selected_index])
        selected_plan = np.asarray(selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32)
        selected_idx = selected_row.get("proposal_index")
        selected_score, selected_score_raw = _selected_score_from_row(selected_row, fallback_key=score_fallback_key)
        selected_source = (
            "planner_gate_rule_based_base_policy"
            if selected_planner == "rule_based"
            else "planner_gate_learned_base_policy"
        )
        if planner_gate_result.get("error") is not None:
            selected_source = "planner_gate_failed_base_policy_fallback"
        selection_debug = _planner_gate_debug(
            planner_gate_result=planner_gate_result,
            selected_planner=selected_planner,
            default_selected_index=default_selected_index,
            default_selected_source=default_selected_source,
            display_default_trajectories=display_default_trajectories,
            include_default_candidates=include_default_candidates,
            allow_carry_previous=allow_carry_previous,
            previous_selected_source=previous_selected_source,
            selected_score_raw=selected_score_raw,
            q_key_prefix=q_key_prefix,
        )
    else:
        selection_result = vlm_selector.maybe_select(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_rows=candidate_rows,
            default_selected_index=default_selected_index,
            default_selected_source=default_selected_source,
        )
        selected_row = dict(selection_result["selected_candidate_row"])
        selected_plan = np.asarray(selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32)
        selected_idx = selected_row.get("proposal_index")
        selected_score, selected_score_raw = _selected_score_from_row(selected_row, fallback_key=score_fallback_key)
        selected_source = str(selection_result.get("selected_source", learned_default_source))
        selection_debug = _scorer_debug(
            selection_result=selection_result,
            planner_gate_result=planner_gate_result,
            selected_planner=selected_planner,
            default_selected_index=default_selected_index,
            default_selected_source=default_selected_source,
            display_default_trajectories=display_default_trajectories,
            include_default_candidates=include_default_candidates,
            allow_carry_previous=allow_carry_previous,
            previous_selected_source=previous_selected_source,
            selected_score_raw=selected_score_raw,
            q_key_prefix=q_key_prefix,
        )

    return LearnedPlannerSelection(
        selected_row=selected_row,
        selected_plan=selected_plan,
        selected_idx=None if selected_idx is None else int(selected_idx),
        selected_score=float(selected_score),
        selected_score_raw=float(selected_score_raw),
        selected_source=selected_source,
        selected_planner=selected_planner,
        candidate_rows=list(candidate_rows),
        planner_gate_result=planner_gate_result,
        default_selected_index=int(default_selected_index),
        default_selected_source=default_selected_source,
        selection_debug=selection_debug,
    )
