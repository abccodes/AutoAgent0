#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from moviepy import ImageSequenceClip

sys.path.append(os.getcwd())

from planners.common.candidate_visuals import get_candidate_visual_style
from sim.utils.sim_utils import (
    get_camera_c2w,
    local_plan_to_front_world,
    project_world_points_to_image,
    rt2pose,
)


FRONT_CAM_NAME = "CAM_FRONT"
PLAN_VIS_FORWARD_OFFSET_M = 4.5


def format_overlay_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def wrap_text_to_width(text, font, font_scale, thickness, max_width):
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def append_wrapped_text(lines, label, text, font, font_scale, thickness, max_width):
    if text is None:
        return
    label_prefix = f"{label}: "
    label_width = cv2.getTextSize(label_prefix, font, font_scale, thickness)[0][0]
    first_line_width = max(40, int(max_width) - int(label_width))
    wrapped = wrap_text_to_width(str(text), font, font_scale, thickness, first_line_width)
    if not wrapped:
        return
    lines.append(f"{label_prefix}{wrapped[0]}")
    for continuation in wrapped[1:]:
        continuation_prefix = "  "
        continuation_width = cv2.getTextSize(continuation_prefix, font, font_scale, thickness)[0][0]
        continuation_max_width = max(40, int(max_width) - int(continuation_width))
        continuation_wrapped = wrap_text_to_width(
            continuation,
            font,
            font_scale,
            thickness,
            continuation_max_width,
        )
        for chunk in continuation_wrapped:
            lines.append(f"{continuation_prefix}{chunk}")


def normalize_overlay_source(selected_source):
    if selected_source is None:
        return None
    source = str(selected_source)
    if "carry_prev" in source:
        return "carry_prev"
    if source.startswith("default_fallback_"):
        return source
    return "current"


def resolve_selected_traj_text(frame_debug):
    candidate_sources = frame_debug.get("overlay_candidate_sources")
    if not candidate_sources:
        candidate_sources = frame_debug.get("candidate_pool_sources")
    candidate_indices = frame_debug.get("candidate_pool_proposal_indices") or []
    selected_source = frame_debug.get("selected_source")
    selected_idx = frame_debug.get("selected_idx")
    selected_kind = normalize_overlay_source(selected_source)
    if not candidate_sources:
        return None

    current_rank = 0
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
        if source_str != "carry_prev":
            current_rank += 1
        if not is_match:
            continue
        return f"#{rank}"
    return None


def build_front_overlay_lines(frame_idx, frame_debug, run_label, max_text_width):
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
    selected_traj = resolve_selected_traj_text(frame_debug)
    if selected_traj is not None:
        lines.append(f"selected traj: {selected_traj}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.39
    thickness = 1
    uses_vlm = ("vlm" in run_label) or (frame_debug.get("vlm_reasoning") is not None)
    uses_intervention = "intervention" in run_label

    if uses_intervention:
        should_intervene = latency_record.get("intervention_should_intervene")
        lines.append(f"intervened: {format_overlay_value(should_intervene)}")
        severity = latency_record.get("intervention_severity")
        if severity is not None:
            lines.append(f"intervention severity: {format_overlay_value(severity)}")
        confidence = latency_record.get("intervention_confidence")
        if confidence is not None:
            lines.append(f"intervention confidence: {format_overlay_value(confidence)}")
        corrective_action = latency_record.get("intervention_corrective_action")
        if should_intervene:
            lines.append(f"corrective action: {format_overlay_value(corrective_action)}")
        append_wrapped_text(
            lines,
            "intervention reasoning",
            frame_debug.get("intervention_reasoning"),
            font,
            font_scale,
            thickness,
            max_text_width,
        )
        append_wrapped_text(
            lines,
            "scorer reasoning",
            frame_debug.get("vlm_reasoning"),
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
                f"{format_overlay_value(q_selected_source)}"
                f" / {format_overlay_value(q_selected_idx)}"
            )
        append_wrapped_text(
            lines,
            "vlm reasoning",
            frame_debug.get("vlm_reasoning"),
            font,
            font_scale,
            thickness,
            max_text_width,
        )

    return lines


def draw_front_overlay_text(frame, lines):
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


def load_rollout_frames(scene_dir):
    data_path = os.path.join(scene_dir, "data.pkl")
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    return data[0]["frames"]


def load_infos(scene_dir):
    infos_path = os.path.join(scene_dir, "infos.pkl")
    with open(infos_path, "rb") as handle:
        return pickle.load(handle)


def load_overlay_front_frames(scene_dir):
    overlay_dir = os.path.join(scene_dir, "overlay_front")
    names = sorted(name for name in os.listdir(overlay_dir) if name.endswith(".jpg"))
    frames = []
    for name in names:
        path = os.path.join(overlay_dir, name)
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to read overlay frame {path}")
        frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return frames


def draw_candidate_label(image, points_world, intrinsic, front_c2w, label, color):
    if points_world is None or len(points_world) == 0:
        return
    pixels, valid_mask = project_world_points_to_image(points_world, intrinsic, front_c2w)
    h, w = image.shape[:2]
    for px, py, valid in reversed(list(zip(pixels[:, 0], pixels[:, 1], valid_mask))):
        if not valid:
            continue
        if 0 <= px < w and 0 <= py < h:
            cv2.putText(
                image,
                str(label),
                (int(px) + 4, int(py) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            return


def redraw_candidate_labels(frame_rgb, frame_debug, info):
    candidate_plans = frame_debug.get("overlay_candidate_plans")
    origin_pose = frame_debug.get("overlay_plan_origin_pose")
    if not candidate_plans or origin_pose is None or info is None:
        return frame_rgb

    overlay = frame_rgb.copy()
    ego_pose = rt2pose(info["ego_rot"], info["ego_pos"])
    cam_params = info["cam_params"]
    intrinsic = cam_params[FRONT_CAM_NAME]["intrinsic"]
    front_c2w = get_camera_c2w(cam_params, ego_pose, FRONT_CAM_NAME)
    origin_pose_np = np.asarray(origin_pose, dtype=np.float32)
    candidate_sources = frame_debug.get("overlay_candidate_sources") or []

    current_rank = 0
    for rank, candidate_plan in enumerate(candidate_plans):
        source = candidate_sources[rank] if rank < len(candidate_sources) else "current_rap"
        style = get_candidate_visual_style(source or "current_rap", current_rank)
        if source != "carry_prev":
            current_rank += 1
        plan_origin_front_c2w = get_camera_c2w(cam_params, origin_pose_np, FRONT_CAM_NAME)
        plan_world_line = local_plan_to_front_world(
            candidate_plan,
            plan_origin_front_c2w,
            cam_params[FRONT_CAM_NAME]["v2c"],
            include_origin=True,
            forward_offset=PLAN_VIS_FORWARD_OFFSET_M,
        )
        draw_candidate_label(
            overlay,
            plan_world_line,
            intrinsic,
            front_c2w,
            rank,
            style.color_bgr,
        )
    return overlay


def render_front_video(scene_dir, run_label, overwrite=False):
    output_path = os.path.join(scene_dir, "front.mp4")
    if os.path.exists(output_path) and not overwrite:
        return "skip_existing"

    data_path = os.path.join(scene_dir, "data.pkl")
    overlay_dir = os.path.join(scene_dir, "overlay_front")
    infos_path = os.path.join(scene_dir, "infos.pkl")
    if not os.path.exists(data_path) or not os.path.exists(infos_path) or not os.path.isdir(overlay_dir):
        return "skip_missing_inputs"

    rollout_frames = load_rollout_frames(scene_dir)
    infos = load_infos(scene_dir)
    front_frames = load_overlay_front_frames(scene_dir)
    if not front_frames:
        return "skip_no_frames"

    rendered = []
    for frame_idx, frame in enumerate(front_frames):
        frame_debug = {}
        if frame_idx < len(rollout_frames):
            frame_debug = rollout_frames[frame_idx].get("planner_debug", {}) or {}
        info = infos[frame_idx] if frame_idx < len(infos) else None
        frame = redraw_candidate_labels(frame, frame_debug, info)
        max_text_width = max(160, int(frame.shape[1]) - 18 - 10 - 18 - 10)
        lines = build_front_overlay_lines(frame_idx, frame_debug, run_label, max_text_width)
        rendered.append(draw_front_overlay_text(frame, lines))

    clip = ImageSequenceClip(rendered, fps=4)
    clip.write_videofile(output_path, logger=None)
    return "rendered"


def iter_scene_dirs(root_dir):
    for variant in sorted(os.listdir(root_dir)):
        variant_dir = os.path.join(root_dir, variant)
        if not os.path.isdir(variant_dir):
            continue
        for scene in sorted(os.listdir(variant_dir)):
            scene_dir = os.path.join(variant_dir, scene)
            if os.path.isdir(scene_dir):
                yield variant, scene_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/bigdata/aidan/outputs/benchmark/out/04_28_baselines/rap",
        help="RAP output root to scan",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate front.mp4 even if it already exists",
    )
    parser.add_argument(
        "--variant",
        default="",
        help="Optional exact variant directory name filter",
    )
    args = parser.parse_args()

    counts = {
        "rendered": 0,
        "skip_existing": 0,
        "skip_missing_inputs": 0,
        "skip_no_frames": 0,
    }

    for variant, scene_dir in iter_scene_dirs(args.root):
        if args.variant and variant != args.variant:
            continue
        status = render_front_video(scene_dir, variant, overwrite=args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}\t{scene_dir}")

    print("\nsummary")
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")


if __name__ == "__main__":
    main()
