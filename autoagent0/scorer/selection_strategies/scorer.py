"""One-shot VLM scoring trajectory selection strategy."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from autoagent0.scorer.agent_trace import (
    PASSIVE_VERIFIER_RESULT,
    OrchestratorToolLog,
    build_or_extend_trace,
)
from autoagent0.scorer.selection_strategies.base import (
    BaseSelectionStrategy,
    LearnedPlannerSelection,
    SelectionOutcome,
)


class ScorerStrategy(BaseSelectionStrategy):
    """One-shot VLM scoring path."""

    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
    ) -> SelectionOutcome:
        sel = self.selector
        reserved_candidate_slots = (
            max(0, int(sel.rule_based_merge_cfg.topk))
            if sel.rule_based_merge_cfg.enabled and not sel.vlm_cfg.planner_gate_enabled
            else 0
        )
        learned_candidate_rows, allow_carry_previous = sel._build_learned_candidate_rows(
            proposals, scores, output_num_poses, info, reserved_candidate_slots,
        )
        rule_based_candidate_rows = sel._build_rule_based_candidate_rows(
            obs, info, info_history, privileged_info, output_num_poses,
        )

        tool_log = OrchestratorToolLog(runtime_name=sel.runtime_name)
        tool_log.reset()
        tool_log.record(
            "request_designer",
            learned_candidate_count=len(learned_candidate_rows),
            rule_based_candidate_count=len(rule_based_candidate_rows),
            combined_candidate_count=0,
        )
        selection = self._run_learned_planner_selection(
            scores=scores,
            obs=obs,
            info=info,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            allow_carry_previous=allow_carry_previous,
        )
        tool_log.record(
            "select_final_actions",
            selected_source=selection.selected_source,
            selected_planner=selection.selected_planner,
            selected_candidate_index=selection.default_selected_index,
        )
        tool_log.record("request_critique_passive", active=False)

        selection_debug = dict(selection.selection_debug)
        try:
            selection_debug["agent_trace"] = build_or_extend_trace(
                runtime_name=sel.runtime_name,
                frame_index=sel.frame_index,
                info=info,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                candidate_rows=selection.candidate_rows,
                selection=selection,
                selection_debug=selection_debug,
                planner_gate_enabled=sel.vlm_cfg.planner_gate_enabled,
                previous_verifier_feedback={},
            )
            selection_debug["agent_trace"]["verifier"] = {
                "accepted": bool(PASSIVE_VERIFIER_RESULT.accepted),
                "mode": PASSIVE_VERIFIER_RESULT.mode,
                "rejection_reason": PASSIVE_VERIFIER_RESULT.rejection_reason,
                "checks": PASSIVE_VERIFIER_RESULT.checks,
            }
            selection_debug["agent_trace"]["tool_calls"] = tool_log.to_debug_list()
        except Exception as exc:
            sel.logger.exception("Failed to attach AutoAgent0 runtime trace: %s", exc)
            selection_debug["agent_trace_error"] = str(exc)

        plan_payload = sel._build_plan_payload(
            proposals,
            scores,
            output_num_poses,
            selected_idx=None if selection.selected_idx is None else int(selection.selected_idx),
            selected_source=selection.selected_source,
            selection_debug=selection_debug,
            selected_plan_override=selection.selected_plan,
            selected_score_override=selection.selected_score,
            candidate_pool_rows=selection.candidate_rows,
        )
        return SelectionOutcome(
            plan_payload=plan_payload,
            selected_plan=selection.selected_plan,
            selected_score_raw=selection.selected_score_raw,
            selected_source=str(selection.selected_row.get("source", "vlm_selected")),
            advance_frame=True,
        )

    def _run_learned_planner_selection(
        self,
        *,
        scores: np.ndarray,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        allow_carry_previous: bool,
    ) -> LearnedPlannerSelection:
        sel = self.selector
        candidate_rows, selected_planner, planner_gate_result = self._choose_candidate_family(
            frame_index=sel.frame_index,
            camera_images=obs["rgb"],
            info=info,
            vlm_selector=sel.vlm_selector,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            rule_based_merge_enabled=sel.rule_based_merge_cfg.enabled,
            planner_gate_enabled=sel.vlm_cfg.planner_gate_enabled,
        )

        candidate_sources = [str(row.get("source", "unknown")) for row in candidate_rows]
        sel.logger.info(
            "%s planner gate frame=%s planner=%s confidence=%s error=%s timed_out=%s chosen_size=%d sources=%s",
            sel.planner_log_name,
            sel.frame_index,
            selected_planner,
            planner_gate_result.get("confidence"),
            planner_gate_result.get("error"),
            planner_gate_result.get("timed_out"),
            len(candidate_rows),
            candidate_sources,
        )

        default_selected_index, default_selected_source = self._default_selection_for_family(
            candidate_rows=candidate_rows,
            selected_planner=selected_planner,
            scores=scores,
            learned_source_name=sel.current_source_name,
            learned_default_source=sel.learned_default_source,
            strict_learned_argmax_lookup=sel.strict_learned_argmax_lookup,
        )

        if sel.vlm_cfg.planner_gate_enabled:
            selected_row = dict(candidate_rows[default_selected_index])
            selected_plan = np.asarray(selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32)
            selected_idx = selected_row.get("proposal_index")
            selected_score, selected_score_raw = self._selected_score_from_row(
                selected_row,
                fallback_key=sel.score_fallback_key,
            )
            selected_source = (
                "planner_gate_rule_based_base_policy"
                if selected_planner == "rule_based"
                else "planner_gate_learned_base_policy"
            )
            if planner_gate_result.get("error") is not None:
                selected_source = "planner_gate_failed_base_policy_fallback"
            selection_debug = self._planner_gate_debug(
                planner_gate_result=planner_gate_result,
                selected_planner=selected_planner,
                default_selected_index=default_selected_index,
                default_selected_source=default_selected_source,
                display_default_trajectories=sel.vlm_cfg.display_default_trajectories,
                include_default_candidates=sel.vlm_cfg.include_default_candidates,
                allow_carry_previous=allow_carry_previous,
                previous_selected_source=sel.previous_selected_source,
                selected_score_raw=selected_score_raw,
                q_key_prefix=sel.q_key_prefix,
            )
        else:
            selection_result = sel.vlm_selector.maybe_select(
                frame_index=sel.frame_index,
                camera_images=obs["rgb"],
                info=info,
                candidate_rows=candidate_rows,
                default_selected_index=default_selected_index,
                default_selected_source=default_selected_source,
            )
            selected_row = dict(selection_result["selected_candidate_row"])
            selected_plan = np.asarray(selected_row.get("execution_plan", selected_row["local_plan"]), dtype=np.float32)
            selected_idx = selected_row.get("proposal_index")
            selected_score, selected_score_raw = self._selected_score_from_row(
                selected_row,
                fallback_key=sel.score_fallback_key,
            )
            selected_source = str(selection_result.get("selected_source", sel.learned_default_source))
            selection_debug = self._scorer_debug(
                selection_result=selection_result,
                planner_gate_result=planner_gate_result,
                selected_planner=selected_planner,
                default_selected_index=default_selected_index,
                default_selected_source=default_selected_source,
                display_default_trajectories=sel.vlm_cfg.display_default_trajectories,
                include_default_candidates=sel.vlm_cfg.include_default_candidates,
                allow_carry_previous=allow_carry_previous,
                previous_selected_source=sel.previous_selected_source,
                selected_score_raw=selected_score_raw,
                q_key_prefix=sel.q_key_prefix,
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

    @staticmethod
    def _choose_candidate_family(
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        vlm_selector: Any,
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        rule_based_merge_enabled: bool,
        planner_gate_enabled: bool,
    ) -> tuple[list[Dict[str, object]], str, Dict[str, object]]:
        planner_gate_result = ScorerStrategy._disabled_planner_gate_result()
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

    @staticmethod
    def _selected_score_from_row(selected_row: Dict[str, object], *, fallback_key: str) -> tuple[float, float]:
        selected_score_value = selected_row.get("proposal_score")
        if selected_score_value is None:
            selected_score_value = selected_row.get(fallback_key, 0.0)
        selected_score = float(selected_score_value)

        selected_score_raw_value = selected_row.get("origin_selected_score_raw")
        if selected_score_raw_value is None:
            selected_score_raw_value = selected_score
        return selected_score, float(selected_score_raw_value)

    @staticmethod
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

    @staticmethod
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
