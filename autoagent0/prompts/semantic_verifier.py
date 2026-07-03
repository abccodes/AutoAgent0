from __future__ import annotations

from typing import Dict, Optional, Sequence

from autoagent0.adapters.hugsim.context import describe_vlm_camera_inputs


def build_semantic_verifier_prompt(
    candidate_row: Dict[str, object],
    route_instruction: str,
    *,
    task_target_hint: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = (
        f"- candidate_{candidate_row['candidate_index']} (the colored path overlaid on the image)"
    )
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- Task target: "{task_target_hint}".
- Treat the task target as additional route context when judging on-track behavior.
""".strip()
    return f"""
You are the AutoAgent0 Semantic Verifier.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Selected trajectory: one proposed short-horizon path drawn on the front image

Important:
- The colored overlay is the proposed future path from the current ego pose.
- The vehicle has not yet executed that path.
- Judge whether the proposed path is on-track with the route objective.

Task:
- Decide whether the selected trajectory is on-track with the route objective.
- Focus on route alignment, lane consistency, and turn-direction consistency.
- Do not reject merely because another path might look smoother.
- Return only valid JSON.

Reasoning:
- Keep reasoning short and causal.
- Mention visible road/lane/intersection evidence and why the path is or is not on-track.

{task_target_guidance}

Selected trajectory:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "on_track": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short causal explanation>"
}}
""".strip()
