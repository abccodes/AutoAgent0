from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CritiqueResult:
    """Structured result parsed from the active AutoAgent0 critic."""

    accepted: bool
    severity_score: float
    corrective_action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def try_parse_json(text: str) -> Optional[Dict[str, object]]:
    raw = str(text).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    blob = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if blob:
        try:
            return json.loads(blob.group(1))
        except Exception:
            pass
    return None


def empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def normalize_token_usage(token_usage: Optional[object]) -> Dict[str, int]:
    if not isinstance(token_usage, dict):
        return empty_token_usage()
    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": max(0, total_tokens),
    }


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


def coerce_vlm_candidate_scores(
    raw_scores: object,
    num_candidates: int,
) -> Tuple[Optional[List[float]], Optional[str]]:
    return coerce_candidate_scores(raw_scores, num_candidates)


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


__all__ = [
    "CritiqueResult",
    "coerce_candidate_scores",
    "coerce_critique_result",
    "coerce_intervention_decision",
    "coerce_vlm_candidate_scores",
    "empty_token_usage",
    "intervention_severity_band",
    "normalize_corrective_action",
    "normalize_token_usage",
    "select_from_vlm_scores",
    "selected_path_reasoning",
    "try_parse_json",
]
