from __future__ import annotations

from typing import Optional


def build_design_change_prompt(
    *,
    route_instruction: str,
    critique_reason: str,
    corrective_action: Optional[str],
    candidate_budget: int,
) -> str:
    return f"""
You are the AutoAgent0 Designer coordinator.

Route objective: "{route_instruction}"
Critic rejection reason: "{critique_reason}"
Corrective action: "{corrective_action or 'straight'}"
Candidate budget: {int(candidate_budget)}

Request a revised candidate set from the available learned and rule-based
trajectory designers.
""".strip()
