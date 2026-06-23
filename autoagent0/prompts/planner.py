from __future__ import annotations

from typing import Dict, Optional, Sequence

from autoagent0.adapters.hugsim.context import describe_vlm_camera_inputs
from autoagent0.decision.candidates import format_candidate_text


def build_final_action_selection_prompt(
    candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
    *,
    critique_reason: Optional[str] = None,
    corrective_action: Optional[str] = None,
    task_target_hint: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = format_candidate_text(candidate_rows)
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    candidate_index_list = ", ".join(str(row["candidate_index"]) for row in candidate_rows)
    score_schema_lines = ",\n".join(
        f'    "{row["candidate_index"]}": <float>'
        for row in candidate_rows
    )
    redesign_context = ""
    if critique_reason or corrective_action:
        redesign_context = f"""
Redesign context:
- Critic rejection reason: "{critique_reason or 'unspecified'}"
- Requested corrective action: "{corrective_action or 'straight'}"
- Treat this as short-horizon guidance, not a replacement for the route objective.
""".strip()
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- Task target: "{task_target_hint}".
- Prefer safe candidates that make progress toward the target.
""".strip()

    return f"""
You are the AutoAgent0 Planner selecting the final action after a critique.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Candidate trajectories from learned and rule-based designers

Available tool result:
- request_designer(action_generation) has produced the candidates below.

Task:
- Select exactly one candidate for execution.
- Score every candidate index: {candidate_index_list}.
- Learned and rule-based native scores are not directly comparable; judge by trajectory geometry and visual context.
- Return only valid JSON.

Selection policy:
- Prefer the safest route-consistent short-horizon action.
- Prefer lane alignment, obstacle clearance, stable progress, and realistic motion.
- If all candidates are imperfect, select the least risky candidate.
- Do not prefer a family only because it is learned or rule-based.

{redesign_context}
{task_target_guidance}

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "best_candidate_index": <int>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<short causal explanation>",
  "candidate_scores": {{
{score_schema_lines}
  }}
}}
""".strip()


def build_planner_tool_prompt(route_instruction: str) -> str:
    return f"""
You are the AutoAgent0 Planner.

Route objective: "{route_instruction}"

Use this fixed tool pattern:
1. request_designer(action_generation)
2. request_critique
3. if rejected, request_design_change
4. request_designer(action_generation)
5. select_final_actions
6. request_critique

Return only JSON tool decisions. This prompt documents the planner role; the
current runtime executes the tool sequence deterministically.
""".strip()
