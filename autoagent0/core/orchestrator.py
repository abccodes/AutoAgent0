from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from autoagent0.core.schemas import CritiqueResult, DesignChangeRequest, FrameUncertainty
from autoagent0.core.uncertainty import (
    ROUTING_ZONE_LEAN_RULE_BASED,
    ROUTING_ZONE_NORMAL,
    ROUTING_ZONE_RULE_BASED_FALLBACK,
)


PHASE_DEFAULT_ACCEPTED = "default_accepted"
PHASE_DEFAULT_REJECTED = "default_rejected"
PHASE_REDESIGN_REJECTED_RETRY = "redesign_rejected_retry"
PHASE_REDESIGN_ACCEPTED = "redesign_accepted"
PHASE_REDESIGN_SELECTED_AT_LIMIT = "redesign_selected_at_critic_limit"
PHASE_FALLBACK_AFTER_REVISED_REJECTED = "fallback_after_revised_rejected"


@dataclass(frozen=True)
class ExpandedDesign:
    """Expanded candidate pool requested after an AutoAgent0 critique rejection."""

    rows: List[Dict[str, object]]
    learned_budget: int


@dataclass(frozen=True)
class FinalRecoveryDecision:
    """Orchestrator decision after the revised candidate critique."""

    final_rejected: bool
    continue_redesign: bool
    use_fallback: bool
    phase: str
    fallback_reason: Optional[str]


class OrchestratorToolLog:
    """Per-frame SceneSmith-style tool-call trace recorder."""

    def __init__(self, *, runtime_name: str):
        self.runtime_name = str(runtime_name)
        self._tool_calls: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._tool_calls = []

    def record(self, name: str, **metadata: object) -> None:
        self._tool_calls.append(
            {
                "name": name,
                "runtime": self.runtime_name,
                **metadata,
            }
        )

    def to_debug_list(self) -> List[Dict[str, object]]:
        return list(self._tool_calls)


class Orchestrator:
    """Pass-through wrapper around the current VLM selector API."""

    def __init__(self, selector: Any):
        self.selector = selector

    def select_trajectory(self, **kwargs: Any) -> Dict[str, Any]:
        return self.selector.maybe_select(**kwargs)

    def select_planner(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, Any]:
        return self.selector.maybe_select_planner(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
        )


def critique_requests_redesign(critique_result: Dict[str, object]) -> bool:
    """Return whether the active Critic requested redesign.

    Critic transport/parse errors intentionally preserve current behavior:
    errors do not trigger redesign because the old path treats them as a
    non-blocking critique failure.
    """

    should_redesign = critique_result.get("autoagent0_critique_rejected") is True
    critique_error = critique_result.get("autoagent0_critique_error") or critique_result.get("error")
    if critique_error is not None:
        return False
    return bool(should_redesign)


def build_design_change_request(
    critique_result: Dict[str, object],
    *,
    redesign_candidate_budget: int,
    available_rule_based_count: int,
    has_rule_based_candidates: bool,
    attempt_index: int = 1,
    frame_uncertainty: Optional[FrameUncertainty] = None,
) -> DesignChangeRequest:
    corrective_action = critique_result.get("autoagent0_critique_corrective_action")
    rejection_reason = critique_result.get("autoagent0_critique_reasoning") or "vlm_critic_requested_redesign"
    candidate_budget = max(1, int(redesign_candidate_budget))
    attempt = max(1, int(attempt_index))
    if attempt == 1:
        learned_budget = 8
        rule_based_budget = 5
        allocation_strategy = "learned8_rule5_after_default_rejection"
    elif attempt == 2:
        learned_budget = 5
        rule_based_budget = 10
        allocation_strategy = "learned5_rule10_after_revised_rejection"
    else:
        learned_budget = 3
        rule_based_budget = 12
        allocation_strategy = "recovery_heavy_at_redesign_limit"

    routing_mode = ROUTING_ZONE_NORMAL
    if frame_uncertainty is not None and has_rule_based_candidates:
        zone = frame_uncertainty.routing_zone
        if zone == ROUTING_ZONE_RULE_BASED_FALLBACK:
            learned_budget = 0
            rule_based_budget = candidate_budget
            allocation_strategy = (
                f"rule_based_only_uncertainty_intra{frame_uncertainty.intra_learned_m:.2f}"
                f"_cross{frame_uncertainty.cross_family_m:.2f}"
                f"_modes{frame_uncertainty.mode_count}"
            )
            routing_mode = ROUTING_ZONE_RULE_BASED_FALLBACK
        elif zone == ROUTING_ZONE_LEAN_RULE_BASED:
            shift = max(1, learned_budget // 2)
            learned_budget = max(0, learned_budget - shift)
            rule_based_budget = max(rule_based_budget, candidate_budget - learned_budget)
            allocation_strategy = f"{allocation_strategy}_lean_rule_based"
            routing_mode = ROUTING_ZONE_LEAN_RULE_BASED

    if not has_rule_based_candidates:
        rule_based_budget = 0
        allocation_strategy = f"{allocation_strategy}_no_rule_based_available"

    return DesignChangeRequest(
        reason=str(rejection_reason),
        corrective_action=None if corrective_action is None else str(corrective_action),
        candidate_budget=candidate_budget,
        learned_budget=learned_budget,
        rule_based_budget=rule_based_budget,
        allocation_strategy=allocation_strategy,
        include_learned=True,
        include_rule_based=bool(has_rule_based_candidates and int(available_rule_based_count) > 0),
        routing_mode=routing_mode,
    )


def build_expanded_design(
    *,
    learned_candidate_rows: Sequence[Dict[str, object]],
    rule_based_candidate_rows: Sequence[Dict[str, object]],
    default_row: Dict[str, object],
    design_change_request: DesignChangeRequest,
) -> ExpandedDesign:
    learned_budget = max(1, int(design_change_request.learned_budget))
    rule_based_budget = max(0, int(design_change_request.rule_based_budget))
    expanded_rows = list(learned_candidate_rows[:learned_budget]) + list(rule_based_candidate_rows[:rule_based_budget])
    if not expanded_rows:
        expanded_rows = [default_row]
    return ExpandedDesign(rows=expanded_rows, learned_budget=learned_budget)


def decide_final_recovery_action(
    final_critique: Dict[str, object],
    *,
    attempt_index: int,
    max_redesign_attempts: int,
) -> FinalRecoveryDecision:
    final_rejected = final_critique.get("autoagent0_critique_rejected") is True
    final_critique_error = final_critique.get("autoagent0_critique_error") or final_critique.get("error")
    if final_critique_error is not None:
        final_rejected = False

    reached_redesign_limit = int(attempt_index) >= max(1, int(max_redesign_attempts))
    if final_rejected and not reached_redesign_limit:
        fallback_reason = (
            final_critique.get("autoagent0_critique_reasoning")
            or "final_critic_rejected_revised_candidate"
        )
        return FinalRecoveryDecision(
            final_rejected=True,
            continue_redesign=True,
            use_fallback=False,
            phase=PHASE_REDESIGN_REJECTED_RETRY,
            fallback_reason=str(fallback_reason),
        )

    if final_rejected:
        return FinalRecoveryDecision(
            final_rejected=True,
            continue_redesign=False,
            use_fallback=False,
            phase=PHASE_REDESIGN_SELECTED_AT_LIMIT,
            fallback_reason="final_critic_rejected_but_max_redesign_attempts_reached_use_vlm_scorer_selection",
        )

    return FinalRecoveryDecision(
        final_rejected=False,
        continue_redesign=False,
        use_fallback=False,
        phase=PHASE_REDESIGN_ACCEPTED,
        fallback_reason=None,
    )


def normalize_corrective_action(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    direct = {
        "left": "left",
        "right": "right",
        "straight": "straight",
        "forward": "straight",
        "go straight": "straight",
        "keep straight": "straight",
        "continue straight": "straight",
        "turn left": "left",
        "go left": "left",
        "veer left": "left",
        "turn right": "right",
        "go right": "right",
        "veer right": "right",
    }
    if text in direct:
        return direct[text]
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "straight" in text or "forward" in text or "continue" in text:
        return "straight"
    return None


def coerce_candidate_scores(raw_scores: object, num_candidates: int) -> Tuple[Optional[List[float]], Optional[str]]:
    if not isinstance(raw_scores, dict):
        return None, "candidate_scores_missing"

    scores: List[Optional[float]] = [None] * num_candidates
    for raw_key, raw_value in raw_scores.items():
        try:
            idx = int(raw_key)
        except Exception:
            return None, f"candidate_score_bad_key:{raw_key}"
        if idx < 0 or idx >= num_candidates:
            return None, f"candidate_score_key_out_of_range:{idx}"
        try:
            scores[idx] = float(raw_value)
        except Exception:
            return None, f"candidate_score_bad_value:{raw_key}"

    if any(score is None for score in scores):
        missing = [idx for idx, score in enumerate(scores) if score is None]
        return None, f"candidate_scores_incomplete:{missing}"
    return [float(score) for score in scores], None


def intervention_severity_band(
    severity_score: Optional[float],
    *,
    action_threshold: float,
    high_threshold: float,
) -> Optional[str]:
    if severity_score is None:
        return None
    if severity_score >= high_threshold:
        return "high"
    if severity_score >= action_threshold:
        return "medium"
    return "low"


def coerce_intervention_decision(
    parsed: object,
) -> Tuple[Optional[bool], Optional[float], Optional[str], Optional[float], Optional[str], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, None, None, None, None, "intervention_output_invalid"

    raw_flag = parsed.get("should_intervene")
    if not isinstance(raw_flag, bool):
        return None, None, None, None, None, "intervention_flag_missing"

    raw_severity_score = parsed.get("severity_score")
    if raw_severity_score is None:
        return None, None, None, None, None, "intervention_severity_score_missing"
    try:
        severity_score = float(raw_severity_score)
    except Exception:
        return None, None, None, None, None, "intervention_severity_score_invalid"
    if not (0.0 <= severity_score <= 1.0):
        return None, None, None, None, None, f"intervention_severity_score_out_of_range:{severity_score}"

    corrective_action = normalize_corrective_action(parsed.get("corrective_action"))
    if raw_flag and corrective_action is None:
        return None, None, None, None, None, "intervention_corrective_action_missing"

    confidence = None
    if "confidence" in parsed and parsed.get("confidence") is not None:
        raw_confidence = parsed.get("confidence")
        try:
            confidence = float(raw_confidence)
        except Exception:
            return None, None, None, None, None, "intervention_confidence_invalid"

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return bool(raw_flag), severity_score, corrective_action, confidence, reasoning, None


def coerce_critique_result(parsed: object) -> CritiqueResult:
    if not isinstance(parsed, dict):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_output_invalid",
        )

    raw_accepted = parsed.get("accepted")
    if not isinstance(raw_accepted, bool):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_accepted_missing",
            raw=dict(parsed),
        )

    try:
        severity_score = float(parsed.get("severity_score"))
    except Exception:
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_severity_score_invalid",
            raw=dict(parsed),
        )
    if not (0.0 <= severity_score <= 1.0):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error=f"critique_severity_score_out_of_range:{severity_score}",
            raw=dict(parsed),
        )

    corrective_action = normalize_corrective_action(parsed.get("corrective_action")) or "straight"
    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return CritiqueResult(
                accepted=False,
                severity_score=severity_score,
                corrective_action=corrective_action,
                error="critique_confidence_invalid",
                raw=dict(parsed),
            )
    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return CritiqueResult(
        accepted=bool(raw_accepted),
        severity_score=severity_score,
        corrective_action=corrective_action,
        confidence=confidence,
        reasoning=reasoning,
        raw=dict(parsed),
    )


def select_from_vlm_scores(
    candidate_rows: Sequence[Dict[str, object]],
    vlm_scores: Sequence[float],
) -> Dict[str, object]:
    vlm_scores = [float(score) for score in vlm_scores]
    if len(vlm_scores) != len(candidate_rows):
        raise ValueError("VLM score count does not match candidate rows")

    selected_idx = int(max(range(len(vlm_scores)), key=lambda idx: vlm_scores[idx]))
    selected_row = dict(candidate_rows[selected_idx])
    row_source = str(selected_row.get("source", "current_rap"))
    if row_source == "carry_prev":
        selected_source = "vlm_selected_carry_prev"
        decision = "vlm_selected_reuse_prev"
    elif row_source.startswith("default_fallback_"):
        selected_source = "vlm_selected_default_fallback"
        decision = "vlm_selected_default_fallback"
    else:
        selected_source = "vlm_selected_current"
        decision = "vlm_selected_current"

    sorted_scores = sorted(((float(score), idx) for idx, score in enumerate(vlm_scores)), reverse=True)
    score_gap_top2 = None
    if len(sorted_scores) >= 2:
        score_gap_top2 = float(sorted_scores[0][0] - sorted_scores[1][0])

    return {
        "selected_candidate_index": selected_idx,
        "selected_candidate_row": selected_row,
        "selected_source": selected_source,
        "adaptive_replan_decision": decision,
        "vlm_q_score_gap_top2": score_gap_top2,
    }


def selected_path_reasoning(
    selected_row: Dict[str, object],
    selected_candidate_index: int,
    selected_source: str,
    vlm_scores: Optional[Sequence[float]],
    parsed_reasoning: Optional[object],
) -> str:
    if isinstance(parsed_reasoning, str) and parsed_reasoning.strip():
        return parsed_reasoning.strip()

    summary = selected_row.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "selected path"

    score_text = ""
    if vlm_scores is not None and 0 <= selected_candidate_index < len(vlm_scores):
        score_text = f" with highest q-score {float(vlm_scores[selected_candidate_index]):.3f}"
    return (
        f"Selected {selected_source} candidate {int(selected_candidate_index)}{score_text}: "
        f"{summary}."
    )
