from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from moviepy import ImageSequenceClip


VIDEO_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
]
FRONT_CAM_NAME = "CAM_FRONT"


def _resize_for_video(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * (target_height / image.shape[0]))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _pad_row_for_video(row: np.ndarray, target_width: int) -> np.ndarray:
    if row.shape[1] == target_width:
        return row
    pad_width = target_width - row.shape[1]
    return np.pad(row, ((0, 0), (0, pad_width), (0, 0)), mode="constant")


def to_video(observations: Sequence[dict], rollout_frames: Sequence[dict], output_path: str) -> None:
    frames = []
    if not observations:
        return

    target_height = max(
        obs[cam_name].shape[0]
        for obs in observations
        for row in VIDEO_LAYOUT
        for cam_name in row
    )

    for obs in observations:
        row1 = np.concatenate(
            [_resize_for_video(obs[cam_name], target_height) for cam_name in VIDEO_LAYOUT[0]],
            axis=1,
        )
        row2 = np.concatenate(
            [_resize_for_video(obs[cam_name], target_height) for cam_name in VIDEO_LAYOUT[1]],
            axis=1,
        )
        target_width = max(row1.shape[1], row2.shape[1])
        row1 = _pad_row_for_video(row1, target_width)
        row2 = _pad_row_for_video(row2, target_width)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def _format_overlay_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _wrap_text_to_width(text: str, font: int, font_scale: float, thickness: int, max_width: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _append_wrapped_text(
    lines: list[str],
    label: str,
    text: object,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> None:
    if text is None:
        return
    label_prefix = f"{label}: "
    label_width = cv2.getTextSize(label_prefix, font, font_scale, thickness)[0][0]
    first_line_width = max(40, int(max_width) - int(label_width))
    wrapped = _wrap_text_to_width(str(text), font, font_scale, thickness, first_line_width)
    if not wrapped:
        return
    lines.append(f"{label_prefix}{wrapped[0]}")
    for continuation in wrapped[1:]:
        continuation_prefix = "  "
        continuation_width = cv2.getTextSize(continuation_prefix, font, font_scale, thickness)[0][0]
        continuation_max_width = max(40, int(max_width) - int(continuation_width))
        continuation_wrapped = _wrap_text_to_width(
            continuation,
            font,
            font_scale,
            thickness,
            continuation_max_width,
        )
        for chunk in continuation_wrapped:
            lines.append(f"{continuation_prefix}{chunk}")


def _normalize_overlay_source(selected_source: object) -> str | None:
    if selected_source is None:
        return None
    source = str(selected_source)
    if "carry_prev" in source:
        return "carry_prev"
    if source.startswith("default_fallback_"):
        return source
    return "current"


def _resolve_selected_traj_text(frame_debug: dict) -> str | None:
    candidate_sources = frame_debug.get("overlay_candidate_sources")
    if not candidate_sources:
        candidate_sources = frame_debug.get("candidate_pool_sources")
    candidate_indices = frame_debug.get("candidate_pool_proposal_indices") or []
    for rank_key in ("selected_candidate_index", "vlm_selected_idx"):
        rank_value = frame_debug.get(rank_key)
        if rank_value is None:
            continue
        try:
            rank = int(rank_value)
        except (TypeError, ValueError):
            continue
        if candidate_sources and 0 <= rank < len(candidate_sources):
            return f"#{rank}"
    selected_source = frame_debug.get("selected_source")
    selected_idx = frame_debug.get("selected_idx")
    selected_kind = _normalize_overlay_source(selected_source)
    if not candidate_sources:
        return None

    for rank, source in enumerate(candidate_sources):
        source_str = str(source)
        proposal_index = candidate_indices[rank] if rank < len(candidate_indices) else None
        is_match = False
        if selected_kind == "carry_prev":
            is_match = source_str == "carry_prev"
        elif selected_kind and selected_kind.startswith("default_fallback_"):
            is_match = source_str == selected_kind
        else:
            is_match = source_str != "carry_prev" and proposal_index == selected_idx
        if not is_match:
            continue
        return f"#{rank}"
    return None


def _format_critic_state(critique: object) -> str | None:
    if not isinstance(critique, dict):
        return None
    accepted = critique.get("autoagent0_critique_accepted")
    rejected = critique.get("autoagent0_critique_rejected")
    if accepted is True:
        state = "accepted"
    elif rejected is True or accepted is False:
        state = "rejected"
    else:
        state = "unknown"
    score = critique.get("autoagent0_critique_severity_score")
    action = critique.get("autoagent0_critique_corrective_action")
    confidence = critique.get("autoagent0_critique_confidence")
    parts = [state]
    if score is not None:
        parts.append(f"score={_format_overlay_value(score)}")
    if action is not None:
        parts.append(f"action={_format_overlay_value(action)}")
    if confidence is not None:
        parts.append(f"conf={_format_overlay_value(confidence)}")
    return " ".join(parts)


def _first_autoagent0_reasoning(frame_debug: dict) -> object:
    final_critique = frame_debug.get("autoagent0_final_critique")
    if isinstance(final_critique, dict) and final_critique.get("autoagent0_critique_reasoning"):
        return final_critique.get("autoagent0_critique_reasoning")
    default_critique = frame_debug.get("autoagent0_default_critique")
    if isinstance(default_critique, dict) and default_critique.get("autoagent0_critique_reasoning"):
        return default_critique.get("autoagent0_critique_reasoning")
    attempts = frame_debug.get("autoagent0_redesign_attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("critique_reasoning"):
                return attempt.get("critique_reasoning")
    return (
        frame_debug.get("selected_path_reasoning")
        or frame_debug.get("vlm_reasoning")
        or frame_debug.get("intervention_reasoning")
    )


def _append_autoagent0_overlay_lines(
    lines: list[str],
    frame_debug: dict,
    font: int,
    font_scale: float,
    thickness: int,
    max_text_width: int,
) -> None:
    phase = frame_debug.get("autoagent0_phase")
    mode = frame_debug.get("autoagent0_mode")
    if mode is not None:
        lines.append(f"autoagent0 mode: {_format_overlay_value(mode)}")
    if phase is not None:
        lines.append(f"autoagent0 phase: {_format_overlay_value(phase)}")

    selected_source = frame_debug.get("selected_source")
    selected_candidate_source = frame_debug.get("selected_candidate_source")
    if selected_source is not None or selected_candidate_source is not None:
        lines.append(
            "decision: "
            f"{_format_overlay_value(selected_source)}"
            f" / {_format_overlay_value(selected_candidate_source)}"
        )

    attempt_count = frame_debug.get("autoagent0_redesign_attempt_count")
    max_attempts = frame_debug.get("autoagent0_max_redesign_attempts")
    if attempt_count is not None or max_attempts is not None:
        lines.append(
            "redesign: "
            f"{_format_overlay_value(attempt_count)}"
            f"/{_format_overlay_value(max_attempts)}"
        )

    total_count = frame_debug.get("autoagent0_revised_candidate_count")
    learned_count = frame_debug.get("autoagent0_revised_learned_candidate_count")
    rule_count = frame_debug.get("autoagent0_revised_rule_based_candidate_count")
    if total_count is not None or learned_count is not None or rule_count is not None:
        lines.append(
            "candidates: "
            f"learned={_format_overlay_value(learned_count)} "
            f"rule={_format_overlay_value(rule_count)} "
            f"total={_format_overlay_value(total_count)}"
        )

    default_critic = _format_critic_state(frame_debug.get("autoagent0_default_critique"))
    if default_critic is not None:
        lines.append(f"default critic: {default_critic}")
    final_critic = _format_critic_state(frame_debug.get("autoagent0_final_critique"))
    if final_critic is not None:
        lines.append(f"final critic: {final_critic}")

    fallback_reason = frame_debug.get("autoagent0_fallback_reason")
    if fallback_reason is not None:
        lines.append(f"fallback: {_format_overlay_value(fallback_reason)}")

    _append_wrapped_text(
        lines,
        "reasoning",
        _first_autoagent0_reasoning(frame_debug),
        font,
        font_scale,
        thickness,
        max_text_width,
    )


def _build_front_overlay_lines(frame_idx: int, frame_debug: dict, run_label: str, max_text_width: int) -> list[str]:
    lines = [
        f"run: {run_label}",
        f"frame: {frame_idx}",
    ]
    latency_record = frame_debug.get("latency_timeline_record") or {}
    route_instruction = latency_record.get("route_instruction")
    if route_instruction is not None:
        lines.append(f"route: {route_instruction}")
    scoring_route = latency_record.get("scoring_route_instruction")
    if scoring_route is not None:
        lines.append(f"scoring route: {scoring_route}")
    selected_traj = _resolve_selected_traj_text(frame_debug)
    if selected_traj is not None:
        lines.append(f"selected traj: {selected_traj}")
    decision_fresh = frame_debug.get("planner_decision_fresh")
    if decision_fresh is not None:
        decision_state = "fresh" if bool(decision_fresh) else "held"
        decision_age = frame_debug.get("planner_decision_age_steps")
        if decision_age is not None:
            lines.append(f"planner decision: {decision_state} ({int(decision_age)} step old)")
        else:
            lines.append(f"planner decision: {decision_state}")
    decision_frame_index = frame_debug.get("planner_decision_frame_index")
    if decision_frame_index is not None:
        lines.append(f"planner frame: {decision_frame_index}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.39
    thickness = 1
    uses_vlm = ("vlm" in run_label) or (frame_debug.get("vlm_reasoning") is not None)
    uses_intervention = "intervention" in run_label
    uses_autoagent0 = (
        "autoagent0" in run_label
        or frame_debug.get("autoagent0_mode") is not None
        or frame_debug.get("autoagent0_phase") is not None
        or frame_debug.get("agent_trace") is not None
    )
    planner_gate_selected = frame_debug.get("planner_gate_selected_planner")
    planner_gate_confidence = frame_debug.get("planner_gate_confidence")
    planner_gate_reasoning = frame_debug.get("planner_gate_reasoning")
    planner_gate_error = frame_debug.get("planner_gate_error")
    planner_gate_timed_out = frame_debug.get("planner_gate_timed_out")
    execution_mode = frame_debug.get("execution_mode")
    uses_planner_gate = (
        planner_gate_selected is not None
        or planner_gate_reasoning is not None
        or planner_gate_error is not None
        or (isinstance(execution_mode, str) and execution_mode.startswith("planner_gate_"))
    )

    if uses_autoagent0:
        _append_autoagent0_overlay_lines(
            lines,
            frame_debug,
            font,
            font_scale,
            thickness,
            max_text_width,
        )
    elif uses_intervention:
        should_intervene = latency_record.get("intervention_should_intervene")
        lines.append(f"intervened: {_format_overlay_value(should_intervene)}")
        severity_score = latency_record.get("intervention_severity_score")
        severity_band = latency_record.get("intervention_severity_band")
        if severity_score is not None:
            lines.append(f"intervention score: {_format_overlay_value(round(float(severity_score), 3))}")
        if severity_band is not None:
            lines.append(f"intervention band: {_format_overlay_value(severity_band)}")
        confidence = latency_record.get("intervention_confidence")
        if confidence is not None:
            lines.append(f"intervention confidence: {_format_overlay_value(confidence)}")
        corrective_action = latency_record.get("intervention_corrective_action")
        if should_intervene:
            lines.append(f"corrective action: {_format_overlay_value(corrective_action)}")
        _append_wrapped_text(
            lines,
            "intervention reasoning",
            frame_debug.get("intervention_reasoning"),
            font,
            font_scale,
            thickness,
            max_text_width,
        )
        _append_wrapped_text(
            lines,
            "scorer reasoning",
            frame_debug.get("vlm_reasoning"),
            font,
            font_scale,
            thickness,
            max_text_width,
        )
    elif uses_planner_gate:
        lines.append(f"planner gate: {_format_overlay_value(planner_gate_selected)}")
        if planner_gate_confidence is not None:
            lines.append(f"planner confidence: {_format_overlay_value(planner_gate_confidence)}")
        if planner_gate_timed_out is not None:
            lines.append(f"planner timed out: {_format_overlay_value(planner_gate_timed_out)}")
        if planner_gate_error is not None:
            lines.append(f"planner error: {_format_overlay_value(planner_gate_error)}")
        _append_wrapped_text(
            lines,
            "planner reasoning",
            planner_gate_reasoning,
            font,
            font_scale,
            thickness,
            max_text_width,
        )
    elif uses_vlm:
        adaptive_decision = frame_debug.get("adaptive_replan_decision")
        if adaptive_decision is not None:
            lines.append(f"adaptive decision: {adaptive_decision}")
        q_selected_source = frame_debug.get("q_selected_source")
        q_selected_idx = frame_debug.get("q_selected_idx")
        if q_selected_source is not None or q_selected_idx is not None:
            lines.append(
                "q selection: "
                f"{_format_overlay_value(q_selected_source)}"
                f" / {_format_overlay_value(q_selected_idx)}"
            )
        _append_wrapped_text(
            lines,
            "vlm reasoning",
            frame_debug.get("vlm_reasoning"),
            font,
            font_scale,
            thickness,
            max_text_width,
        )

    return lines


def _draw_front_overlay_text(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    if not lines:
        return frame

    canvas = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.39
    thickness = 1
    line_gap = 6
    padding = 10
    origin_x = 18
    origin_y = 18

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max((size[0] for size in line_sizes), default=0)
    line_height = max((size[1] for size in line_sizes), default=0)
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap

    box_x0 = origin_x - padding
    box_y0 = origin_y - padding
    box_x1 = min(canvas.shape[1] - 1, origin_x + max_width + padding)
    box_y1 = min(canvas.shape[0] - 1, origin_y + total_height + padding)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    text_y = origin_y + line_height
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (origin_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        text_y += line_height + line_gap

    return canvas


def to_front_video(observations: Sequence[dict], rollout_frames: Sequence[dict], output_path: str, run_label: str) -> None:
    if not observations:
        return

    frames = []
    for frame_idx, obs in enumerate(observations):
        front = obs[FRONT_CAM_NAME].copy()
        frame_debug = {}
        if frame_idx < len(rollout_frames):
            frame_debug = rollout_frames[frame_idx].get("planner_debug", {}) or {}
        max_text_width = max(160, int(front.shape[1]) - 18 - 10 - 18 - 10)
        lines = _build_front_overlay_lines(frame_idx, frame_debug, run_label, max_text_width)
        frames.append(_draw_front_overlay_text(front, lines))

    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)
