from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from autoagent0.core.orchestrator import (
    coerce_candidate_scores,
    coerce_intervention_decision,
    normalize_corrective_action,
)


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


def coerce_vlm_candidate_scores(
    raw_scores: object,
    num_candidates: int,
) -> Tuple[Optional[List[float]], Optional[str]]:
    return coerce_candidate_scores(raw_scores, num_candidates)


__all__ = [
    "coerce_candidate_scores",
    "coerce_intervention_decision",
    "coerce_vlm_candidate_scores",
    "empty_token_usage",
    "normalize_corrective_action",
    "normalize_token_usage",
    "try_parse_json",
]

