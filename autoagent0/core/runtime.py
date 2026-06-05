from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from autoagent0.core.designer import Designer
from autoagent0.core.planner_flow import (
    LearnedPlannerSelection,
    default_selection_for_family,
    disabled_planner_gate_result,
    run_learned_planner_selection,
)
from autoagent0.core.schemas import DesignChangeRequest
from autoagent0.core.trace import build_agent_trace
from autoagent0.core.verifier import PassiveVerifier


def _route_instruction_from_info(info: Dict[str, object]) -> str:
    for key in ("task_instruction", "route_instruction", "command"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _score_from_row(selected_row: Dict[str, object], fallback_key: str) -> tuple[float, float]:
    selected_score_value = selected_row.get("proposal_score")
    if selected_score_value is None:
        selected_score_value = selected_row.get(fallback_key, 0.0)
    selected_score = float(selected_score_value)
    selected_score_raw_value = selected_row.get("origin_selected_score_raw")
    if selected_score_raw_value is None:
        selected_score_raw_value = selected_score
    return selected_score, float(selected_score_raw_value)


def _hold_plan_like(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[0] == 0:
        return np.zeros((1, 2), dtype=np.float32)
    return np.zeros_like(plan, dtype=np.float32)


class AutoAgent0Runtime:
    """SceneSmith-style facade over the current planner selection flow.

    This class is intentionally behavior-preserving. It names the current flow
    as Orchestrator tool calls, but delegates selection to existing helpers.
    """

    def __init__(self, *, runtime_name: str, logger: Any = None):
        self.runtime_name = str(runtime_name)
        self.logger = logger
        self.designer = Designer()
        self.verifier = PassiveVerifier()
        self.previous_verifier_feedback: Dict[str, Any] = {}
        self._tool_calls: List[Dict[str, object]] = []

    def _record_tool_call(self, name: str, **metadata: object) -> None:
        self._tool_calls.append(
            {
                "name": name,
                "runtime": self.runtime_name,
                **metadata,
            }
        )

    def request_designer(
        self,
        *,
        learned_candidate_rows: Sequence[Dict[str, Any]] = (),
        rule_based_candidate_rows: Sequence[Dict[str, Any]] = (),
        combined_candidate_rows: Sequence[Dict[str, Any]] = (),
    ):
        """Normalize existing candidate rows as a Designer batch."""

        self._record_tool_call(
            "request_designer",
            learned_candidate_count=len(learned_candidate_rows),
            rule_based_candidate_count=len(rule_based_candidate_rows),
            combined_candidate_count=len(combined_candidate_rows),
        )
        return self.designer.from_existing_rows(
            learned_rows=learned_candidate_rows,
            rule_based_rows=rule_based_candidate_rows,
            combined_rows=combined_candidate_rows,
        )

    def request_critique(self, *, selected_plan: Any = None, context: Optional[Dict[str, Any]] = None):
        """Passive SceneSmith-style critique placeholder.

        The result is debug-only and must not alter selected trajectories.
        """

        self._record_tool_call("request_critique_passive", active=False)
        return self.verifier.verify(trajectory=selected_plan, context=context)

    def request_design_change(self, *, instruction: str = "") -> Dict[str, object]:
        """Future recovery hook. It is intentionally unavailable in v1."""

        self._record_tool_call(
            "request_design_change_unavailable",
            active=False,
            instruction=str(instruction),
        )
        return {
            "available": False,
            "reason": "recovery_design_change_not_implemented",
            "revised_candidates": [],
        }

    def select_final_actions_recovery_loop(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        vlm_selector: Any,
        scores: np.ndarray,
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        redesign_candidate_budget: int,
        learned_source_name: str,
        learned_default_source: str,
        score_fallback_key: str,
        planner_log_name: str,
        logger: Any,
        strict_learned_argmax_lookup: bool = False,
        fallback_mode: str = "hold",
        max_redesign_attempts: int = 1,
    ) -> LearnedPlannerSelection:
        """Run the target AutoAgent0 one-redesign recovery loop.

        This is intentionally separate from the legacy Method A/B path. The
        first critique uses the current VLM intervention mechanism on one
        default trajectory. Expanded learned + rule-based scoring is invoked
        only when that critique requests redesign.
        """

        self._tool_calls = []
        self._record_tool_call(
            "request_initial_design",
            designer="learned",
            candidate_count=len(learned_candidate_rows),
        )
        self.request_designer(
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=(),
        )
        default_selected_index, default_selected_source = default_selection_for_family(
            candidate_rows=learned_candidate_rows,
            selected_planner="learned",
            scores=scores,
            learned_source_name=learned_source_name,
            learned_default_source=learned_default_source,
            strict_learned_argmax_lookup=strict_learned_argmax_lookup,
        )
        default_row = dict(learned_candidate_rows[default_selected_index])
        default_plan = np.asarray(default_row.get("execution_plan", default_row["local_plan"]), dtype=np.float32)
        default_idx = default_row.get("proposal_index")
        default_score, default_score_raw = _score_from_row(default_row, score_fallback_key)

        self._record_tool_call(
            "request_critique",
            phase="default",
            critic="autoagent0_vlm_critic",
            candidate_count=1,
        )
        critique_result = vlm_selector.critique_autoagent0_candidate(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_row=default_row,
            stage="default",
        )
        should_redesign = critique_result.get("autoagent0_critique_rejected") is True
        critique_error = critique_result.get("autoagent0_critique_error") or critique_result.get("error")
        if critique_error is not None:
            should_redesign = False

        if not should_redesign:
            self._record_tool_call(
                "select_final_actions",
                phase="default_accepted",
                selected_source=default_selected_source,
                selected_candidate_index=default_selected_index,
            )
            selection_debug = dict(critique_result)
            selection_debug.update(
                {
                    "autoagent0_mode": "recovery_loop",
                    "autoagent0_phase": "default_accepted",
                    "autoagent0_redesign_triggered": False,
                    "autoagent0_default_critique": critique_result,
                    "autoagent0_fallback_reason": None,
                    "fallback_selected_idx": int(default_selected_index),
                    "fallback_selected_source": default_selected_source,
                    "planner_gate_selected_planner": "learned",
                }
            )
            selection = LearnedPlannerSelection(
                selected_row=default_row,
                selected_plan=default_plan,
                selected_idx=None if default_idx is None else int(default_idx),
                selected_score=default_score,
                selected_score_raw=default_score_raw,
                selected_source=default_selected_source,
                selected_planner="learned",
                candidate_rows=[default_row],
                planner_gate_result=disabled_planner_gate_result(),
                default_selected_index=0,
                default_selected_source=default_selected_source,
                selection_debug=selection_debug,
            )
            return replace(selection, selection_debug=self._attach_recovery_trace(
                frame_index=frame_index,
                info=info,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                selection=selection,
                phase="default_accepted",
            ))

        corrective_action = critique_result.get("autoagent0_critique_corrective_action")
        rejection_reason = critique_result.get("autoagent0_critique_reasoning") or "vlm_critic_requested_redesign"
        redesign_budget = max(1, int(redesign_candidate_budget))
        design_change_request = DesignChangeRequest(
            reason=str(rejection_reason),
            corrective_action=None if corrective_action is None else str(corrective_action),
            candidate_budget=redesign_budget,
            include_learned=True,
            include_rule_based=bool(rule_based_candidate_rows),
        )
        self._record_tool_call(
            "request_design_change",
            phase="default_rejected",
            reason=design_change_request.reason,
            corrective_action=design_change_request.corrective_action,
            candidate_budget=design_change_request.candidate_budget,
            include_rule_based=design_change_request.include_rule_based,
        )
        learned_budget = max(1, redesign_budget - len(rule_based_candidate_rows))
        expanded_rows = list(learned_candidate_rows[:learned_budget]) + list(rule_based_candidate_rows)
        if not expanded_rows:
            expanded_rows = [default_row]
        self._record_tool_call(
            "request_revised_design",
            designer="learned+rule_based",
            candidate_count=len(expanded_rows),
            candidate_budget=redesign_budget,
        )
        self.request_designer(
            learned_candidate_rows=learned_candidate_rows[:learned_budget],
            rule_based_candidate_rows=rule_based_candidate_rows,
            combined_candidate_rows=expanded_rows,
        )
        self._record_tool_call(
            "select_final_actions",
            scorer="autoagent0_vlm_planner",
            candidate_count=len(expanded_rows),
        )
        redesign_result = vlm_selector.score_autoagent0_candidates(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_rows=expanded_rows,
            default_selected_index=0,
            default_selected_source="autoagent0_redesign_default",
            critique_reason=design_change_request.reason,
            corrective_action=design_change_request.corrective_action,
        )
        revised_row = dict(redesign_result["selected_candidate_row"])
        revised_plan = np.asarray(revised_row.get("execution_plan", revised_row["local_plan"]), dtype=np.float32)
        revised_idx = revised_row.get("proposal_index")
        revised_score, revised_score_raw = _score_from_row(revised_row, score_fallback_key)

        self._record_tool_call(
            "request_critique",
            phase="revised",
            critic="autoagent0_vlm_critic",
            candidate_count=1,
        )
        final_critique = vlm_selector.critique_autoagent0_candidate(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_row=revised_row,
            stage="final",
            previous_feedback=design_change_request.reason,
        )
        final_rejected = final_critique.get("autoagent0_critique_rejected") is True
        final_critique_error = final_critique.get("autoagent0_critique_error") or final_critique.get("error")
        if final_critique_error is not None:
            final_rejected = False
        reached_redesign_limit = int(max_redesign_attempts) <= 1
        if final_rejected and not reached_redesign_limit:
            fallback_plan = _hold_plan_like(revised_plan) if str(fallback_mode).lower() == "hold" else default_plan
            selected_row = {
                "source": "fallback_brake_hold" if str(fallback_mode).lower() == "hold" else "fallback_learned_default",
                "proposal_index": None,
                "proposal_score": 0.0,
                "local_plan": fallback_plan,
                "execution_plan": fallback_plan.copy(),
            }
            selected_plan = fallback_plan
            selected_idx = None
            selected_score = 0.0
            selected_score_raw = 0.0
            selected_source = str(selected_row["source"])
            phase = "fallback_after_revised_rejected"
            fallback_reason = final_critique.get("autoagent0_critique_reasoning") or "final_critic_rejected_revised_candidate"
        else:
            # TODO(autoagent0): replace this threshold behavior with a combined
            # learned/rule-based scorer once the rule-based scorer design is finalized.
            selected_row = revised_row
            selected_plan = revised_plan
            selected_idx = None if revised_idx is None else int(revised_idx)
            selected_score = revised_score
            selected_score_raw = revised_score_raw
            selected_source = str(redesign_result.get("selected_source", "autoagent0_redesign_selected"))
            phase = "redesign_accepted" if not final_rejected else "redesign_selected_at_critic_limit"
            fallback_reason = None if not final_rejected else (
                "final_critic_rejected_but_max_redesign_attempts_reached_use_vlm_scorer_selection"
            )

        self._record_tool_call(
            "emit_final_actions",
            phase=phase,
            selected_source=selected_source,
            selected_candidate_index=selected_idx,
        )
        selection_debug = dict(redesign_result)
        selection_debug.update(
            {
                "autoagent0_mode": "recovery_loop",
                "autoagent0_phase": phase,
                "autoagent0_redesign_triggered": True,
                "autoagent0_default_critique": critique_result,
                "autoagent0_design_change_request": {
                    "reason": design_change_request.reason,
                    "corrective_action": design_change_request.corrective_action,
                    "candidate_budget": design_change_request.candidate_budget,
                    "include_learned": design_change_request.include_learned,
                    "include_rule_based": design_change_request.include_rule_based,
                },
                "autoagent0_redesign_request": {
                    "reason": design_change_request.reason,
                    "corrective_action": design_change_request.corrective_action,
                    "candidate_budget": design_change_request.candidate_budget,
                    "include_learned": design_change_request.include_learned,
                    "include_rule_based": design_change_request.include_rule_based,
                },
                "autoagent0_revised_candidate_count": len(expanded_rows),
                "autoagent0_revised_learned_candidate_count": sum(
                    1 for row in expanded_rows if str(row.get("source", "")).startswith(learned_source_name)
                ),
                "autoagent0_revised_rule_based_candidate_count": sum(
                    1 for row in expanded_rows if str(row.get("source", "")) == "rule_based"
                ),
                "autoagent0_final_critique": final_critique,
                "autoagent0_fallback_reason": fallback_reason,
                "autoagent0_max_redesign_attempts": int(max_redesign_attempts),
                "fallback_selected_idx": 0,
                "fallback_selected_source": "autoagent0_redesign_default",
                "planner_gate_selected_planner": "autoagent0",
            }
        )
        selection = LearnedPlannerSelection(
            selected_row=selected_row,
            selected_plan=selected_plan,
            selected_idx=selected_idx,
            selected_score=float(selected_score),
            selected_score_raw=float(selected_score_raw),
            selected_source=selected_source,
            selected_planner="autoagent0",
            candidate_rows=list(expanded_rows),
            planner_gate_result=disabled_planner_gate_result(),
            default_selected_index=0,
            default_selected_source="autoagent0_redesign_default",
            selection_debug=selection_debug,
        )
        return replace(selection, selection_debug=self._attach_recovery_trace(
            frame_index=frame_index,
            info=info,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            selection=selection,
            phase=phase,
        ))

    def select_final_actions(
        self,
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
        """Select the final action using the current behavior-preserving flow."""

        self._tool_calls = []
        self.request_designer(
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
        )
        selection = run_learned_planner_selection(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            vlm_selector=vlm_selector,
            scores=scores,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            rule_based_merge_enabled=rule_based_merge_enabled,
            planner_gate_enabled=planner_gate_enabled,
            vlm_enabled=vlm_enabled,
            display_default_trajectories=display_default_trajectories,
            include_default_candidates=include_default_candidates,
            allow_carry_previous=allow_carry_previous,
            previous_selected_source=previous_selected_source,
            learned_source_name=learned_source_name,
            learned_default_source=learned_default_source,
            score_fallback_key=score_fallback_key,
            planner_log_name=planner_log_name,
            logger=logger,
            strict_learned_argmax_lookup=strict_learned_argmax_lookup,
            q_key_prefix=q_key_prefix,
        )
        self._record_tool_call(
            "select_final_actions",
            selected_source=selection.selected_source,
            selected_planner=selection.selected_planner,
            selected_candidate_index=selection.default_selected_index,
        )
        verifier_result = self.request_critique(
            selected_plan=selection.selected_plan,
            context={"frame_index": int(frame_index), "info": info},
        )

        selection_debug = dict(selection.selection_debug)
        try:
            selection_debug["agent_trace"] = self._build_or_extend_trace(
                frame_index=frame_index,
                info=info,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                candidate_rows=selection.candidate_rows,
                selection=selection,
                selection_debug=selection_debug,
                planner_gate_enabled=planner_gate_enabled,
            )
            selection_debug["agent_trace"]["verifier"] = {
                "accepted": bool(verifier_result.accepted),
                "mode": verifier_result.mode,
                "rejection_reason": verifier_result.rejection_reason,
                "checks": verifier_result.checks,
            }
            selection_debug["agent_trace"]["tool_calls"] = list(self._tool_calls)
        except Exception as exc:
            if self.logger is not None:
                self.logger.exception("Failed to attach AutoAgent0 runtime trace: %s", exc)
            selection_debug["agent_trace_error"] = str(exc)
        self.previous_verifier_feedback = {}
        return replace(selection, selection_debug=selection_debug)

    def _build_or_extend_trace(
        self,
        *,
        frame_index: int,
        info: Dict[str, object],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        candidate_rows: Sequence[Dict[str, object]],
        selection: LearnedPlannerSelection,
        selection_debug: Dict[str, object],
        planner_gate_enabled: bool,
    ) -> Dict[str, Any]:
        existing_trace = selection_debug.get("agent_trace")
        if not isinstance(existing_trace, dict):
            existing_trace = selection.planner_gate_result.get("agent_trace")
        if isinstance(existing_trace, dict):
            trace = dict(existing_trace)
        else:
            decision_type = "planner_gate" if planner_gate_enabled else "vlm_scorer"
            confidence = _first_present(
                selection_debug.get("planner_gate_confidence"),
                selection_debug.get("vlm_confidence"),
                selection_debug.get("intervention_confidence"),
            )
            reasoning = _first_present(
                selection_debug.get("planner_gate_reasoning"),
                selection_debug.get("vlm_reasoning"),
                selection_debug.get("intervention_reasoning"),
            )
            selected_candidate_index = _first_present(
                selection_debug.get("vlm_selected_idx"),
                selection.default_selected_index,
            )
            trace = build_agent_trace(
                frame_index=frame_index,
                route_instruction=_route_instruction_from_info(info),
                info=info,
                candidate_rows=candidate_rows,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                decision_type=decision_type,
                selected_source=selection.selected_source,
                selected_planner=selection.selected_planner,
                selected_candidate_index=_optional_int(selected_candidate_index),
                confidence=_optional_float(confidence),
                reasoning=None if reasoning is None else str(reasoning),
                previous_verifier_feedback=self.previous_verifier_feedback,
            )

        trace["runtime"] = {
            "name": self.runtime_name,
            "behavior_mode": "behavior_preserving",
            "scene_smith_style": True,
        }
        return trace

    def _attach_recovery_trace(
        self,
        *,
        frame_index: int,
        info: Dict[str, object],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        selection: LearnedPlannerSelection,
        phase: str,
    ) -> Dict[str, object]:
        selection_debug = dict(selection.selection_debug)
        try:
            confidence = _first_present(
                selection_debug.get("vlm_confidence"),
                selection_debug.get("intervention_confidence"),
            )
            reasoning = _first_present(
                selection_debug.get("selected_path_reasoning"),
                selection_debug.get("vlm_reasoning"),
                selection_debug.get("intervention_reasoning"),
            )
            trace = build_agent_trace(
                frame_index=frame_index,
                route_instruction=_route_instruction_from_info(info),
                info=info,
                candidate_rows=selection.candidate_rows,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                decision_type="autoagent0_recovery_loop",
                selected_source=selection.selected_source,
                selected_planner=selection.selected_planner,
                selected_candidate_index=selection.selected_idx,
                confidence=_optional_float(confidence),
                reasoning=None if reasoning is None else str(reasoning),
                previous_verifier_feedback=self.previous_verifier_feedback,
            )
            trace["runtime"] = {
                "name": self.runtime_name,
                "behavior_mode": "agentic_recovery_loop",
                "scene_smith_style": True,
                "phase": phase,
            }
            trace["tool_calls"] = list(self._tool_calls)
            trace["critique"] = {
                "critic": "autoagent0_vlm_critic",
                "default": selection_debug.get("autoagent0_default_critique"),
                "final": selection_debug.get("autoagent0_final_critique"),
            }
            trace["design_change_request"] = selection_debug.get("autoagent0_design_change_request")
            trace["redesign_request"] = selection_debug.get("autoagent0_redesign_request")
            trace["fallback_reason"] = selection_debug.get("autoagent0_fallback_reason")
            selection_debug["agent_trace"] = trace
        except Exception as exc:
            if self.logger is not None:
                self.logger.exception("Failed to attach AutoAgent0 recovery trace: %s", exc)
            selection_debug["agent_trace_error"] = str(exc)
        self.previous_verifier_feedback = {}
        return selection_debug
