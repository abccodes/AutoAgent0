from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from autoagent0.adapters.hugsim.context import describe_vlm_camera_inputs
from autoagent0.core.candidates import (
    dedupe_gate_candidates,
    family_rows_for_planner_gate,
    format_candidate_text,
    select_representative_candidate_row,
    summarize_candidate,
)


def build_scoring_prompt(
    candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
    task_target_hint: Optional[str] = None,
    intervention_corrective_action: Optional[str] = None,
    current_ego_speed_mps: Optional[float] = None,
    current_ego_accel_mps2: Optional[float] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = format_candidate_text(candidate_rows)
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    has_default_candidates = any(
        str(row.get("source", "")).startswith("default_fallback_")
        for row in candidate_rows
    )
    score_schema_lines = ",\n".join(
        f'    "{row["candidate_index"]}": <float>'
        for row in candidate_rows
    )
    candidate_index_list = ", ".join(str(row["candidate_index"]) for row in candidate_rows)
    default_candidate_guidance = ""
    if has_default_candidates:
        default_candidate_guidance = """
- Some candidates are default recovery / fallback trajectories. These represent simple options such as continuing straight or steering back toward the lane.
- The planner replans every 0.5 seconds, so the selected trajectory does NOT need to be globally perfect for the full horizon.
- Score candidates based on whether they are the best immediate action for the next 0.5 seconds.
- If a default trajectory is the safest and most directionally correct short-horizon choice, it should receive the highest score.
- Do not penalize a default trajectory just because it is simple or less committed over the long horizon.
- Favor candidates that preserve reasonable forward progress and momentum when it is safe to do so.
- Do not prefer unnecessary slowing or hesitation if a safe straight or lane-return trajectory can keep the vehicle moving correctly for the next 0.5 seconds.
""".strip()
    corrective_action_guidance = ""
    if intervention_corrective_action is not None:
        corrective_action_guidance = f"""
- A separate intervention gate suggested this short-horizon corrective action: "{intervention_corrective_action}".
- Treat that corrective action as advisory context about the immediate correction, not as a replacement for the route instruction.
- You should still score candidates against the original route instruction and the visible scene.
- The corrective action and the route instruction may differ; use the images and candidate metadata to decide what is best in this frame.
""".strip()
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- A task target marker is rendered only in the front view.
- The target marker indicates the intended goal location for the instruction: "{task_target_hint}".
- Prefer candidates that move toward that marked target when it is safe and consistent with the scene.
""".strip()
    return f"""
You are an autonomous-driving trajectory scorer performing the final action-selection stage.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Candidate trajectories: structured metadata for {len(candidate_rows)} candidates
- Optional advisory corrective context from a separate self-reflective intervention stage

Task:
- Evaluate every candidate.
- Choose the single best candidate for the next short-horizon action.
- Return one scalar score for every candidate index: {candidate_index_list}.
- Keep the response compact and return only valid JSON.

Reasoning style:
- Use concise chain-of-causation reasoning rather than scene narration.
- For the selected candidate, explain:
  1. the relevant scene evidence,
  2. the likely consequence of taking that candidate next,
  3. why that consequence is safer or more route-consistent than the alternatives.
- Reason about the next short horizon, not the full mission.

Scoring principles:
- Higher scores are better.
- Prefer candidates that stay drivable, lane-aligned, smooth, and realistic.
- Avoid nearby vehicles, obstacles, sidewalk/off-road behavior, wrong-way behavior, and unsafe lane departures.
- Maintain safe forward progress when possible.
- Follow the route instruction directly when safe:
  - if the instruction is straight, prefer continuing in the current lane/direction instead of drifting left or right,
  - if the instruction is left or right, prefer candidates that clearly begin that turn direction when safe,
  - do not deviate from the instructed direction unless safety, obstacles, or road geometry clearly require it.
- Respect lane rules and avoid unnecessary lane changes.
- If all candidates are imperfect, prefer the least risky one.
{default_candidate_guidance}
{task_target_guidance}
{corrective_action_guidance}

Important:
- The route objective remains the primary instruction.
- Advisory corrective context is secondary and should help resolve immediate risk, not replace the route objective.
- If route objective and advisory corrective context differ, reconcile them using the images and candidate metadata instead of blindly following either one.

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "best_candidate_index": <int>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<short causal explanation for why this selected candidate is best>",
  "candidate_scores": {{
{score_schema_lines}
  }},
}}
""".strip()


def build_intervention_prompt(
    baseline_candidate_row: Dict[str, object],
    route_instruction: str,
    task_target_hint: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = format_candidate_text([baseline_candidate_row])
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    multiview_guidance = ""
    if tuple(camera_order) != ("CAM_FRONT",):
        multiview_guidance = """
- Use the front image as the primary source for judging the shown trajectory's path geometry, lane alignment, and route consistency.
- Use the left, right, and back images only as supporting context for surrounding vehicles, nearby obstacles, lane occupancy, and safety conflicts.
- Do not treat side or rear clutter alone as a reason to intervene unless it indicates a real conflict relevant to the next maneuver.
- Because only the front image contains the overlaid trajectory, judge where the path goes from the front view and use the other views only to decide whether surrounding context makes intervention necessary.
""".strip()
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- A task target marker is rendered only in the front view.
- The target marker indicates the intended goal location for the instruction: "{task_target_hint}".
- Use it as route context, but do not force motion toward it when immediate safety says otherwise.
""".strip()
    return f"""
You are an autonomous-driving self-reflective intervention gate deciding whether the current baseline action should be revised before execution.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Baseline planned action: the shown baseline trajectory metadata below

Task:
- Treat the baseline trajectory as a proposed action.
- Perform counterfactual reasoning about what is likely to happen over the next short horizon if this exact baseline action is executed.
- Then decide whether revision is needed before execution.
- Return JSON only. Do not describe the images. Do not answer any other question. Do not include markdown, code fences, or extra prose.

Decision policy:
- Use counterfactual reasoning to judge whether revising the baseline is necessary before the next replan, not merely whether some alternative might be cleaner.
- Set "should_intervene" to true when the baseline action appears risky, ambiguous, instruction-inconsistent, poorly centered, too close to obstacles or lane boundaries, or likely to benefit from short-horizon correction.
- Use "should_intervene" = false when the baseline action looks safe enough to continue until the next replan, even if a different action might be slightly cleaner or more comfortable.
- Use "should_intervene" = false only for trajectories that are genuinely acceptable for the next short horizon: lane-aligned enough, route-consistent enough, and not trending toward a meaningful safety or progress problem.
- If the baseline shows noticeable drift toward a lane boundary, side clutter, a curb, vegetation, or an off-route heading, do not dismiss it as harmless merely because collision is not yet immediate. However, reserve intervention for drift or margin loss that is persistent, worsening, or likely to matter before the next replan; small recoverable deviations should usually stay below the action threshold.
- Borderline or low-margin cases may still be marked as intervention-worthy, but they should usually receive low severity only when the issue is minor enough that no corrective override is needed yet.
- In multiview mode, use extra camera views to judge surrounding safety context, not to reinterpret the path geometry shown on the front image.
{multiview_guidance}
{task_target_guidance}

Corrective action:
- If intervention is needed, provide one short-horizon corrective action.
- The corrective action must be exactly one of: "left", "right", or "straight".
- The corrective action is an advisory revision intent for the next maneuver, not a guaranteed final decision.
- If "should_intervene" is false, set "corrective_action" to "straight".

Reasoning style:
- Use concise counterfactual reasoning rather than loose description.
- Your reasoning should explicitly cover:
  1. the key scene evidence,
  2. the likely short-horizon consequence if the baseline action continues,
  3. whether that consequence is acceptable,
  4. why the corrective action is the best immediate revision when intervention is needed.
- Assign a numeric "severity_score" in [0.0, 1.0] using this guidance:
  - 0.00-0.20: effectively safe to continue; no meaningful concern before next replan.
  - 0.20-0.40: mild concern or small refinement opportunity.
  - 0.40-0.60: borderline or low-margin issue worth noting, but not necessarily strong enough to force an override.
  - 0.60-0.85: meaningful short-horizon safety, route-consistency, lane-centering, or progress concern that should revise the baseline now.
  - 0.85-1.00: concrete near-term failure mode that should revise the baseline immediately, such as likely lane departure, likely obstacle conflict, likely route miss at an intersection, or clearly unsafe clearance before the next replan.
- Use values across the range rather than snapping to boundary examples like 0.60, 0.65, or 0.85. Prefer roughly 0.05 increments and place the score at the lowest value that still matches the actual consequence.
- If "should_intervene" is false, severity_score should usually stay below 0.60, and often below 0.50 when the baseline is clearly acceptable.
- If "should_intervene" is true:
  - use roughly 0.45-0.60 for borderline cases that are worth flagging but not worth overriding,
  - use roughly 0.65-0.75 for meaningful but non-imminent corrections that should revise the baseline now,
  - use roughly 0.75-0.84 for strong non-imminent corrections with clear downside if left unchanged,
  - use 0.85+ only for imminent or clearly unacceptable trajectories.

Baseline trajectory:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "should_intervene": <true or false>,
  "severity_score": <float between 0.0 and 1.0>,
  "corrective_action": "<left or right or straight>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short causal explanation for why intervention is or is not needed>"
}}
""".strip()


def _summarize_gate_candidates(
    candidate_rows: Sequence[Dict[str, object]],
    *,
    label: str,
    limit: int = 3,
) -> str:
    if not candidate_rows:
        return f"{label}: none"

    family_rows = family_rows_for_planner_gate(candidate_rows)
    rep = select_representative_candidate_row(family_rows)
    representative_text = "none"
    if rep is not None:
        rep_summary = summarize_candidate(np.asarray(rep.get("local_plan", []), dtype=np.float32).tolist())
        representative_text = (
            f"path_length_m={rep_summary['path_length_m']} | "
            f"forward_progress_m={rep_summary['forward_progress_m']} | "
            f"end={rep_summary['end']} | "
            f"x_range=[{rep_summary['min_x']},{rep_summary['max_x']}] | "
            f"y_range=[{rep_summary['min_y']},{rep_summary['max_y']}]"
        )

    deduped_rows = dedupe_gate_candidates(family_rows, limit=limit)
    lines = [
        f"{label}:",
        f"- current_family_count={len(family_rows)}",
        f"- unique_summary_count={len(deduped_rows)}",
        f"- representative_default | {representative_text}",
    ]
    for idx, row in enumerate(deduped_rows):
        summary = row.get("summary") or summarize_candidate(np.asarray(row.get("local_plan", []), dtype=np.float32).tolist())
        lines.append(
            (
                f"- option_{idx} | "
                f"path_length_m={summary['path_length_m']} | "
                f"forward_progress_m={summary['forward_progress_m']} | "
                f"end={summary['end']} | "
                f"x_range=[{summary['min_x']},{summary['max_x']}] | "
                f"y_range=[{summary['min_y']},{summary['max_y']}]"
            )
        )
    return "\n".join(lines)


def build_planner_gate_prompt(
    *,
    learned_candidate_rows: Sequence[Dict[str, object]],
    rule_based_candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
    task_target_hint: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
    prompt_style: str = "default",
) -> str:
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    learned_family_rows = family_rows_for_planner_gate(learned_candidate_rows)
    rule_based_family_rows = family_rows_for_planner_gate(rule_based_candidate_rows)
    learned_rep = select_representative_candidate_row(learned_family_rows)
    rule_rep = select_representative_candidate_row(rule_based_family_rows)
    learned_text = _summarize_gate_candidates(learned_family_rows, label="Learned planner candidates")
    rule_text = _summarize_gate_candidates(rule_based_family_rows, label="Rule-based planner candidates")
    balancing_guidance = ""
    if learned_rep is not None and rule_rep is not None:
        learned_summary = summarize_candidate(np.asarray(learned_rep.get("local_plan", []), dtype=np.float32).tolist())
        rule_summary = summarize_candidate(np.asarray(rule_rep.get("local_plan", []), dtype=np.float32).tolist())
        learned_len = float(learned_summary.get("path_length_m", 0.0) or 0.0)
        learned_prog = float(learned_summary.get("forward_progress_m", 0.0) or 0.0)
        rule_len = float(rule_summary.get("path_length_m", 0.0) or 0.0)
        rule_prog = float(rule_summary.get("forward_progress_m", 0.0) or 0.0)
        if (learned_len <= 1.0 or learned_prog <= 0.5) and (rule_len >= 3.0 and rule_prog >= 2.0):
            balancing_guidance = f"""
- Important local context: the learned current default appears nearly stalled or extremely short-horizon (path_length_m={learned_summary['path_length_m']}, forward_progress_m={learned_summary['forward_progress_m']}), while the rule-based current default remains drivable (path_length_m={rule_summary['path_length_m']}, forward_progress_m={rule_summary['forward_progress_m']}).
- In this situation, do not treat the learned planner as preferable by default. Unless the images show that the rule-based default is clearly unsafe or off-route, prefer "rule_based" as the temporary stabilizing handoff.
""".strip()
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- A task target marker is rendered only in the front view.
- The target marker indicates the intended goal location for the instruction: "{task_target_hint}".
""".strip()
    prompt_style_norm = str(prompt_style).strip().lower()
    if prompt_style_norm == "binary_policy_compare":
        decision_principles = """
Decision principles:
- Compare only the two currently drawn policy trajectories: the green learned/base-policy default and the orange rule-based default.
- Treat this as a direct binary choice between two immediate actions, not a family ranking problem.
- Ignore planner identity, ignore candidate-count asymmetry, and ignore long-horizon style preferences.
- Prefer the trajectory that is safer, more lane-centered, more stable, and more appropriate for the next short horizon.
- If both options look plausible and the learned trajectory is not clearly better, prefer the rule-based trajectory for temporary stabilization.
- Do not penalize the rule-based trajectory just because it is shorter or more conservative if it looks cleaner or safer in the current frame.
""".strip()
    elif prompt_style_norm == "corrective_temporary":
        decision_principles = """
Decision principles:
- Choose the planner family that provides the best immediate corrective action for the next short horizon.
- Treat this as a temporary routing decision, not a final judgment of overall planner quality.
- Do not over-penalize a rule-based option just because it is shorter, more conservative, or less progress-maximizing if that conservatism appears safer, better centered, or more corrective for the immediate situation.
- Prefer the learned planner only when it is clearly superior on immediate safety, lane alignment, and route consistency.
- Prefer the rule-based planner whenever it is comparably safe and route-consistent but offers a more stable temporary corrective posture, stronger boundary respect, or cleaner short-horizon recovery.
- If the learned default appears short-horizon, hesitant, drifting, weakly centered, or not clearly better than the rule-based default, route to the rule-based planner for temporary stabilization.
- Use a conservative tie-break: when both planner families look plausible and the learned planner is not clearly better, select "rule_based".
- Focus on the next step only: which planner should temporarily take control right now?
""".strip()
    else:
        decision_principles = """
Decision principles:
- Choose the planner family whose candidate set looks more likely to be safe, route-consistent, lane-aligned, and useful for short-horizon progress.
- Prefer the learned planner when it already has good route-consistent options and no obvious need for recovery.
- Prefer the rule-based planner when the scene looks like it needs conservative recovery, stronger boundary respect, or a safer correction than the learned options provide.
- This is a short-horizon decision only; judge which planner is more appropriate for the next step, not the entire mission.
""".strip()

    if prompt_style_norm == "binary_policy_compare":
        def _rep_text(row: Optional[Dict[str, object]], fallback: str) -> str:
            if row is None:
                return fallback
            summary = summarize_candidate(np.asarray(row.get("local_plan", []), dtype=np.float32).tolist())
            return (
                f"path_length_m={summary['path_length_m']} | "
                f"forward_progress_m={summary['forward_progress_m']} | "
                f"end={summary['end']} | "
                f"x_range=[{summary['min_x']},{summary['max_x']}] | "
                f"y_range=[{summary['min_y']},{summary['max_y']}]"
            )

        comparison_block = f"""
Direct binary comparison:
- learned / green default: {_rep_text(learned_rep, 'none')}
- rule-based / orange default: {_rep_text(rule_rep, 'none')}
""".strip()
    else:
        comparison_block = f"""
Planner candidate summaries:
{learned_text}

{rule_text}
""".strip()

    return f"""
You are an autonomous-driving planner gate.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- In the front-view overlay, the learned/base-policy default trajectory is drawn in green.
- In the front-view overlay, the rule-based default trajectory is drawn in orange.
- Route objective: "{route_instruction}"
- Two planner families are available for the next short horizon:
  - learned planner
  - rule-based planner
- Planner-native candidate scores are not shown because their numeric scales are not directly comparable across planner families.
- Do not prefer a planner family just because it has more listed options; compare the quality of the current-policy representative and the family pattern.

Task:
- Decide which planner family should be trusted for the next decision step.
- Do NOT pick a trajectory index.
- Return only one planner family: "learned" or "rule_based".
- Keep the response compact and return only valid JSON.

{decision_principles}
{task_target_guidance}
{balancing_guidance}

{comparison_block}

Return ONLY valid JSON in this exact schema:
{{
  "selected_planner": "<learned or rule_based>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<short causal explanation for why this planner family is better for the next step>"
}}
""".strip()
