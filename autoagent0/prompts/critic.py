from __future__ import annotations

from typing import Dict, Optional, Sequence

from autoagent0.adapters.hugsim.context import describe_vlm_camera_inputs
from autoagent0.decision.candidates import format_candidate_text


def build_critic_prompt(
    candidate_row: Dict[str, object],
    route_instruction: str,
    *,
    task_target_hint: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = format_candidate_text([candidate_row])
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- Task target: "{task_target_hint}".
- Use the target only as route context; immediate safety still has priority.
""".strip()
    feedback_guidance = ""
    if previous_feedback:
        feedback_guidance = f"""
- Previous critic feedback: "{previous_feedback}".
- Use it only if it is still relevant in the current frame.
""".strip()

    return f"""
You are the AutoAgent0 Critic.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Candidate trajectory: one proposed action for the next short horizon

Task:
- Decide whether this trajectory is acceptable to execute now.
- Reject only when the trajectory needs redesign before execution.
- If rejected, provide one corrective action: "left", "right", or "straight".
- Return only valid JSON.

Acceptance policy:
- Accept if the trajectory is safe enough, route-consistent enough, and lane-aligned enough until the next replan.
- Reject if it is likely to cause unsafe clearance, lane departure, off-road motion, obstacle conflict, route miss, or poor recovery.
- Do not reject merely because another trajectory might be slightly smoother.
- This is a short-horizon critique, not a full-mission review.

Reasoning:
- Keep reasoning short and causal.
- Mention the visible evidence, likely consequence, and why accept/reject is appropriate.

{task_target_guidance}
{feedback_guidance}

Candidate:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "accepted": <true or false>,
  "severity_score": <float between 0.0 and 1.0>,
  "corrective_action": "<left or right or straight>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short causal explanation>"
}}
""".strip()
