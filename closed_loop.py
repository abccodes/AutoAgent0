import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "sim"))

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import (
    draw_projected_polyline_camera_clipped,
    draw_projected_polyline,
    get_camera_c2w,
    local_plan_to_front_world,
    project_world_points_to_image,
    resample_polyline,
    rt2pose,
    traj2control,
    traj_transform_to_global,
)
import pickle
import json
import struct
import logging
import re
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
import cv2
from moviepy import ImageSequenceClip
import time
import select
from planners.common.candidate_visuals import get_candidate_visual_style
from planners.common.vlm_env import build_prefixed_vlm_env

VIDEO_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
]
FRONT_CAM_NAME = 'CAM_FRONT'
REFERENCE_COLOR = (255, 230, 0)
PLAN_COLOR = (0, 120, 255)
REFERENCE_FORWARD_OFFSET_M = 4.0
PLAN_VIS_FORWARD_OFFSET_M = 4.5
VIS_PLAN_MIN_PATH_M = 0.5
VIS_PLAN_HOLD_FRAMES = 10**9
PLAN_REPLAN_EVERY_STEPS = 2


def _slugify_model_name(value, default='model'):
    value = '' if value is None else str(value).strip()
    if not value:
        value = default
    value = value.rstrip('/').split('/')[-1]
    if value.endswith('.ckpt') or value.endswith('.pth') or value.endswith('.pt'):
        value = os.path.splitext(value)[0]
    value = re.sub(r'[^A-Za-z0-9]+', '-', value).strip('-').lower()
    return value or default


def _resolve_output_model_slug(ad_name, planner_config):
    planner_key = 'rap' if ad_name == 'rap' else 'drivor' if ad_name == 'drivor' else ''
    if not planner_key:
        return ''

    planner_cfg = planner_config.get(planner_key, {})
    vlm_cfg = planner_cfg.get('vlm', {})
    if vlm_cfg.get('enabled', False):
        explicit_slug = vlm_cfg.get('output_model_slug', '')
        if explicit_slug:
            return _slugify_model_name(explicit_slug)
        return _slugify_model_name(vlm_cfg.get('model_id', 'vlm'))

    explicit_slug = planner_cfg.get('output_model_slug', '')
    if explicit_slug:
        return _slugify_model_name(explicit_slug)
    checkpoint = planner_cfg.get('checkpoint', '')
    return _slugify_model_name(checkpoint, default=planner_key)


def _prefix_output_dir_with_model(output_dir, model_slug):
    output_dir = str(output_dir)
    model_slug = str(model_slug or '').strip()
    if not model_slug:
        return output_dir

    parent, name = os.path.split(output_dir.rstrip(os.sep))
    if not name:
        return os.path.join(output_dir, model_slug)
    if name.startswith(f'{model_slug}_'):
        return output_dir
    return os.path.join(parent, f'{model_slug}_{name}')

def _resize_for_video(image, target_height):
    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * (target_height / image.shape[0]))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _pad_row_for_video(row, target_width):
    if row.shape[1] == target_width:
        return row
    pad_width = target_width - row.shape[1]
    return np.pad(row, ((0, 0), (0, pad_width), (0, 0)), mode='constant')


def to_video(observations, rollout_frames, output_path):
    frames = []
    if not observations:
        return

    target_height = max(
        obs[cam_name].shape[0]
        for obs in observations
        for row in VIDEO_LAYOUT
        for cam_name in row
    )

    for frame_idx, obs in enumerate(observations):
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


def _format_overlay_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _wrap_text_to_width(text, font, font_scale, thickness, max_width):
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


def _append_wrapped_text(lines, label, text, font, font_scale, thickness, max_width):
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


def _normalize_overlay_source(selected_source):
    if selected_source is None:
        return None
    source = str(selected_source)
    if "carry_prev" in source:
        return "carry_prev"
    if source.startswith("default_fallback_"):
        return source
    return "current"


def _resolve_selected_traj_text(frame_debug):
    candidate_sources = frame_debug.get("overlay_candidate_sources")
    if not candidate_sources:
        candidate_sources = frame_debug.get("candidate_pool_sources")
    candidate_indices = frame_debug.get("candidate_pool_proposal_indices") or []
    selected_source = frame_debug.get("selected_source")
    selected_idx = frame_debug.get("selected_idx")
    selected_kind = _normalize_overlay_source(selected_source)
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


def _build_front_overlay_lines(frame_idx, frame_debug, run_label, max_text_width):
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.39
    thickness = 1
    uses_vlm = ("vlm" in run_label) or (frame_debug.get("vlm_reasoning") is not None)
    uses_intervention = "intervention" in run_label

    if uses_intervention:
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


def _draw_front_overlay_text(frame, lines):
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


def to_front_video(observations, rollout_frames, output_path, run_label):
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


def write_pipe_message(pipe_path, payload_obj):
    payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pipe_path, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def read_pipe_message(pipe_path):
    with open(pipe_path, "rb") as pipe:
        header = pipe.read(8)
        if len(header) != 8:
            raise EOFError(f"Incomplete pipe header from {pipe_path}")
        payload_size = struct.unpack("<Q", header)[0]
        payload = bytearray()
        while len(payload) < payload_size:
            chunk = pipe.read(payload_size - len(payload))
            if not chunk:
                raise EOFError(f"Incomplete pipe payload from {pipe_path}")
            payload.extend(chunk)
    return pickle.loads(payload)


def write_pipe_message_file(pipe, payload_obj):
    payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        fd = pipe.fileno()
    except Exception:
        fd = None
    try:
        if fd is not None:
            try:
                import fcntl as _fcntl
                flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
            except Exception:
                flags = None
            try:
                stat_result = os.fstat(fd)
                inode = stat_result.st_ino
                mode = oct(stat_result.st_mode)
            except Exception:
                inode = None
                mode = None
        else:
            flags = None
            inode = None
            mode = None
        print(f"[write_pipe_message_file] fd={fd} inode={inode} mode={mode} flags={flags} bytes={len(payload)} t={time.time():.6f}")
    except Exception:
        print(f"[write_pipe_message_file] logging failed t={time.time():.6f}")
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()


def read_pipe_message_file(pipe):
    header = pipe.read(8)
    if len(header) != 8:
        raise EOFError("Incomplete pipe header from open pipe handle")
    payload_size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < payload_size:
        chunk = pipe.read(payload_size - len(payload))
        if not chunk:
            raise EOFError("Incomplete pipe payload from open pipe handle")
        payload.extend(chunk)
    return pickle.loads(payload)


def _select_reference_segment(reference_poses, current_c2w, lookahead=25):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    reference_xyz = reference_poses[:, :3, 3]
    current_xyz = current_c2w[:3, 3]
    nearest_idx = int(np.argmin(np.sum((reference_xyz - current_xyz[None]) ** 2, axis=1)))
    end_idx = min(len(reference_xyz), nearest_idx + lookahead)
    return np.asarray(reference_xyz[nearest_idx:end_idx], dtype=np.float32)


def _select_reference_pose_segment(reference_poses, current_c2w, lookahead=25):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)

    reference_xyz = reference_poses[:, :3, 3]
    current_xyz = current_c2w[:3, 3]
    nearest_idx = int(np.argmin(np.sum((reference_xyz - current_xyz[None]) ** 2, axis=1)))
    end_idx = min(len(reference_xyz), nearest_idx + lookahead)
    return np.asarray(reference_poses[nearest_idx:end_idx], dtype=np.float32)


def _draw_projected_points(image, pixels, valid_mask, color, radius=4):
    h, w = image.shape[:2]
    valid_indices = np.flatnonzero(valid_mask)
    for idx in valid_indices:
        px, py = pixels[idx]
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(image, (int(px), int(py)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_first_visible_segment_marker(image, pixels, valid_mask, color, radius=6):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        return

    h, w = image.shape[:2]
    rect = (0, 0, w, h)
    for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
        if end_idx != start_idx + 1:
            continue
        p0 = tuple(int(v) for v in pixels[start_idx])
        p1 = tuple(int(v) for v in pixels[end_idx])
        ok, clipped_p0, clipped_p1 = cv2.clipLine(rect, p0, p1)
        if not ok:
            continue
        anchor = clipped_p0
        cv2.circle(image, anchor, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return


def _draw_candidate_label(image, points_world, intrinsic, front_c2w, label, color):
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


def _count_in_frame(pixels, valid_mask, image_shape):
    h, w = image_shape[:2]
    if len(pixels) == 0:
        return 0
    in_frame = (
        valid_mask
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    return int(np.count_nonzero(in_frame))


def _trajectory_path_length(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())


def _build_reference_worldline(reference_poses, ground_height_fn, forward_offset=REFERENCE_FORWARD_OFFSET_M):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    reference_world = np.asarray(reference_poses[:, :3, 3], dtype=np.float32).copy()
    forward_dirs = np.asarray(reference_poses[:, :3, 2], dtype=np.float32).copy()
    forward_norm = np.linalg.norm(forward_dirs, axis=1, keepdims=True)
    forward_dirs = forward_dirs / np.clip(forward_norm, 1e-6, None)

    reference_world[:, 0] += forward_offset * forward_dirs[:, 0]
    reference_world[:, 2] += forward_offset * forward_dirs[:, 2]
    reference_world[:, 1] = np.array(
        [ground_height_fn(point[0], point[2]) for point in reference_world],
        dtype=np.float32,
    )
    return resample_polyline(reference_world, spacing=0.5)


def _resolve_reference_ground_height_fn(env):
    env_unwrapped = env.unwrapped
    if hasattr(env_unwrapped, 'scene_ground_height'):
        return env_unwrapped.scene_ground_height
    if hasattr(env_unwrapped, 'planner') and hasattr(env_unwrapped.planner, 'ground_height'):
        return env_unwrapped.planner.ground_height
    return env_unwrapped.ground_height


def _render_front_overlay(front_image, current_obs, current_info, env, plan_traj, plan_origin_pose):
    ego_pose = rt2pose(np.asarray(current_info['ego_rot']), np.asarray(current_info['ego_pos']))
    cam_params = current_info['cam_params']
    intrinsic = cam_params[FRONT_CAM_NAME]['intrinsic']
    front_c2w = get_camera_c2w(cam_params, ego_pose, FRONT_CAM_NAME, env.unwrapped.cam_rect)

    overlay = front_image.copy()

    reference_poses = _select_reference_pose_segment(env.unwrapped.ground_model[0], front_c2w, lookahead=40)
    reference_world = _build_reference_worldline(
        reference_poses,
        _resolve_reference_ground_height_fn(env),
    )
    ref_pixels, ref_valid = project_world_points_to_image(reference_world, intrinsic, front_c2w)
    draw_projected_polyline(overlay, ref_pixels, ref_valid, color=REFERENCE_COLOR, thickness=4)
    _draw_projected_points(overlay, ref_pixels, ref_valid, color=REFERENCE_COLOR, radius=4)

    def draw_candidate(candidate_plan, color, thickness, label=None):
        if candidate_plan is None or len(candidate_plan) == 0 or plan_origin_pose is None:
            return
        plan_local = np.asarray(candidate_plan, dtype=np.float32)
        plan_origin_front_c2w = get_camera_c2w(
            cam_params,
            np.asarray(plan_origin_pose, dtype=np.float32),
            FRONT_CAM_NAME,
            env.unwrapped.cam_rect,
        )
        plan_world_line = local_plan_to_front_world(
            plan_local,
            plan_origin_front_c2w,
            cam_params[FRONT_CAM_NAME]['v2c'],
            include_origin=True,
            forward_offset=PLAN_VIS_FORWARD_OFFSET_M,
        )
        plan_world_dense = resample_polyline(plan_world_line, spacing=0.08)
        draw_projected_polyline_camera_clipped(
            overlay,
            plan_world_dense,
            intrinsic,
            front_c2w,
            color=color,
            thickness=thickness,
        )
        if label is not None:
            _draw_candidate_label(
                overlay,
                plan_world_dense,
                intrinsic,
                front_c2w,
                label,
                color,
            )

    candidate_plans = current_info.get('overlay_candidate_plans')
    candidate_sources = current_info.get('overlay_candidate_sources')
    if candidate_plans:
        current_rank = 0
        for rank, candidate_plan in enumerate(candidate_plans):
            source = None if candidate_sources is None or rank >= len(candidate_sources) else candidate_sources[rank]
            style = get_candidate_visual_style(source or 'current_rap', current_rank)
            base_color = style.color_bgr
            if source != 'carry_prev':
                current_rank += 1
            draw_candidate(candidate_plan, base_color, 4 if rank == 0 else 2, label=rank)
    elif plan_traj is not None and len(plan_traj) > 0:
        draw_candidate(plan_traj, PLAN_COLOR, 4, label=0)

    return overlay


def _select_overlay_plan(plan_traj, plan_path_length, last_valid_plan, stale_frames):
    if plan_traj is not None and len(plan_traj) > 0 and plan_path_length >= VIS_PLAN_MIN_PATH_M:
        return np.asarray(plan_traj, dtype=np.float32), 0, False

    if last_valid_plan is not None and stale_frames < VIS_PLAN_HOLD_FRAMES:
        return last_valid_plan.copy(), stale_frames + 1, True

    return None, stale_frames, False


def create_gym_env(cfg, output, run_label, include_privileged_pipe=False):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    #added privileged_info for init planner work
    obs, info, privileged_info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    for pipe_path in (obs_pipe, plan_pipe):
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        os.mkfifo(pipe_path)
    # Keep both FIFOs open in RDWR mode to avoid blocking open-order deadlocks
    # between closed_loop and the planner adapter. Actual message traffic still
    # uses fresh per-message open/read/write calls below.
    obs_pipe_keepalive_fd = os.open(obs_pipe, os.O_RDWR | os.O_NONBLOCK)
    plan_pipe_keepalive_fd = os.open(plan_pipe, os.O_RDWR | os.O_NONBLOCK)
    obs_pipe_writer = os.fdopen(os.open(obs_pipe, os.O_RDWR), "wb", buffering=0)
    plan_pipe_reader = os.fdopen(os.open(plan_pipe, os.O_RDWR), "rb", buffering=0)
    # local helper for logging fd info
    try:
        import fcntl as _fcntl
    except Exception:
        _fcntl = None

    def _log_fd_info(fd, label):
        try:
            stat_result = os.fstat(fd)
            inode = stat_result.st_ino
            mode = oct(stat_result.st_mode)
        except Exception:
            inode = None
            mode = None
        try:
            if _fcntl is not None:
                flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
            else:
                flags = None
        except Exception:
            flags = None
        try:
            link = os.readlink(f"/proc/{os.getpid()}/fd/{fd}")
        except Exception:
            link = None
        print(f"[FIFO OPEN] {label} fd={fd} inode={inode} mode={mode} flags={flags} link={link} t={time.time():.6f}")

    # Log keepalive and io fds
    _log_fd_info(obs_pipe_keepalive_fd, 'obs_pipe_keepalive_fd')
    _log_fd_info(plan_pipe_keepalive_fd, 'plan_pipe_keepalive_fd')
    try:
        _log_fd_info(obs_pipe_writer.fileno(), 'obs_pipe_writer')
    except Exception:
        pass
    try:
        _log_fd_info(plan_pipe_reader.fileno(), 'plan_pipe_reader')
    except Exception:
        pass
    print('Ready for simulation')

    # Send a tiny preflight package so the planner can verify the FIFO path
    # before the first full observation payload is written.
    preflight_payload = {
        "message_type": "hugsim_preflight",
        "output_dir": output,
        "obs_pipe": obs_pipe,
        "plan_pipe": plan_pipe,
        "include_privileged_pipe": bool(include_privileged_pipe),
        "camera_count": 0 if not isinstance(obs, dict) else len(obs.get("rgb", {})),
        "timestamp": float(info.get("timestamp", 0.0)) if isinstance(info, dict) else None,
    }
    preflight_serialized = pickle.dumps(preflight_payload, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[preflight] t_before_write={time.time():.6f} bytes={len(preflight_serialized)}')
    write_pipe_message_file(obs_pipe_writer, preflight_payload)
    print(f'[preflight] t_after_write={time.time():.6f} Wrote preflight diagnostic to {obs_pipe}')

    # Pause to allow external reader to pick up preflight and for manual inspection
    # NOTE: do not attempt to read the preflight locally here — reading it
    # would consume the message so the external client could not see it.
    print(f'[preflight] sleeping 10s at t={time.time():.6f}')
    time.sleep(10.0)

    last_valid_overlay_plan = None
    last_valid_overlay_pose = None
    last_valid_overlay_plan_stale_frames = 0
    current_plan_traj = None
    current_plan_origin_pose = None
    current_topk_plans = None
    current_topk_scores = None
    current_candidate_pool_plans = None
    current_candidate_pool_scores = None
    current_candidate_pool_sources = None
    current_candidate_pool_proposal_indices = None
    current_selected_idx = None
    current_selected_source = None
    current_vlm_selected_idx = None
    current_vlm_confidence = None
    current_vlm_reasoning = None
    current_intervention_reasoning = None
    current_vlm_elapsed_sec = None
    current_vlm_error = None
    current_vlm_q_valid = None
    current_vlm_q_candidate_scores = None
    current_vlm_q_best_candidate_index = None
    current_adaptive_replan_decision = None
    current_carry_previous_valid = None
    current_latency_timeline_record = None
    current_vlm_failed = None
    current_q_selected_idx = None
    current_q_selected_source = None
    current_q_candidate_scores = None
    current_q_carry_score = None
    current_q_best_current_score = None
    current_q_score_gap = None
    current_q_invoked_vlm = None
    overlay_front_dir = os.path.join(output, 'overlay_front')
    os.makedirs(overlay_front_dir, exist_ok=True)
    for name in os.listdir(overlay_front_dir):
        if name.endswith('.jpg'):
            os.remove(os.path.join(overlay_front_dir, name))
    while not done:
        #added privileged_info for init planner work
        if obs is None or info is None or privileged_info is None:
            obs, info, privileged_info = env.reset()
        current_obs, current_info = obs, info
        infos_save.append(current_info)

        print('ego pose', current_info['ego_pos'])

        should_replan = (cnt % PLAN_REPLAN_EVERY_STEPS == 0) or (current_plan_traj is None)
        if should_replan:
            # Ensure expected camera keys exist and fill missing ones with black frames
            # Added to try to debug DrivoR related issues
            try:
                rgb = current_obs.setdefault("rgb", {})
                cam_params = current_info.get("cam_params", {}) if isinstance(current_info, dict) else {}
                expected_names = [
                    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                ]
                logger = logging.getLogger("closed_loop")
                for name in expected_names:
                    if name not in rgb or rgb.get(name) is None:
                        intr = cam_params.get(name, {}).get('intrinsic', {}) if isinstance(cam_params.get(name, {}), dict) else {}
                        W = int(intr.get('W', intr.get('width', 800)))
                        H = int(intr.get('H', intr.get('height', 450)))
                        rgb[name] = np.zeros((H, W, 3), dtype=np.uint8)
                        logger.warning("Filled missing camera %s with black frame %dx%d before sending", name, H, W)

                # Debug: summarize obs['rgb'] keys and shapes/types
                rgb_details = []
                for k, v in rgb.items():
                    if hasattr(v, 'shape'):
                        rgb_details.append(f"{k}: shape={getattr(v,'shape')} dtype={getattr(v,'dtype', None)}")
                    else:
                        rgb_details.append(f"{k}: type={type(v)}")
                logger.info("Sending obs to adapter: rgb keys=%s; details=%s", list(rgb.keys()), "; ".join(rgb_details))
            except Exception:
                logging.getLogger("closed_loop").exception("Failed to prepare current_obs before sending")
            # REFAC: default payload stays a 2-tuple. When the include flag is set,
            # send privileged_info as the third object for planners that need it.
            plan_request_payload = (
                (current_obs, current_info, privileged_info)
                if include_privileged_pipe
                else (current_obs, current_info)
            )
            write_pipe_message_file(obs_pipe_writer, plan_request_payload)
            # Old behavior kept for reference:
            # write_pipe_message_file(obs_pipe_writer, (current_obs, current_info))
            # REFAC: privileged payload form kept for reference:
            # write_pipe_message_file(obs_pipe_writer, (current_obs, current_info, privileged_info))
            plan_payload = read_pipe_message_file(plan_pipe_reader)
            current_topk_plans = None
            current_topk_scores = None
            current_candidate_pool_plans = None
            current_candidate_pool_scores = None
            current_candidate_pool_sources = None
            current_candidate_pool_proposal_indices = None
            current_selected_idx = None
            current_selected_source = None
            current_vlm_selected_idx = None
            current_vlm_confidence = None
            current_vlm_reasoning = None
            current_intervention_reasoning = None
            current_vlm_elapsed_sec = None
            current_vlm_error = None
            current_vlm_q_valid = None
            current_vlm_q_candidate_scores = None
            current_vlm_q_best_candidate_index = None
            current_adaptive_replan_decision = None
            current_carry_previous_valid = None
            current_latency_timeline_record = None
            current_vlm_failed = None
            current_q_selected_idx = None
            current_q_selected_source = None
            current_q_candidate_scores = None
            current_q_carry_score = None
            current_q_best_current_score = None
            current_q_score_gap = None
            current_q_invoked_vlm = None
            if isinstance(plan_payload, dict):
                current_plan_traj = plan_payload.get('selected_plan')
                current_topk_plans = plan_payload.get('topk_plans')
                current_topk_scores = plan_payload.get('topk_scores')
                current_candidate_pool_plans = plan_payload.get('candidate_pool_plans')
                current_candidate_pool_scores = plan_payload.get('candidate_pool_scores')
                current_candidate_pool_sources = plan_payload.get('candidate_pool_sources')
                current_candidate_pool_proposal_indices = plan_payload.get('candidate_pool_proposal_indices')
                current_selected_idx = plan_payload.get('selected_idx')
                current_selected_source = plan_payload.get('selected_source')
                current_vlm_selected_idx = plan_payload.get('vlm_selected_idx')
                current_vlm_confidence = plan_payload.get('vlm_confidence')
                current_vlm_reasoning = plan_payload.get('vlm_reasoning')
                current_intervention_reasoning = plan_payload.get('intervention_reasoning')
                current_vlm_elapsed_sec = plan_payload.get('vlm_elapsed_sec')
                current_vlm_error = plan_payload.get('vlm_error')
                current_vlm_q_valid = plan_payload.get('vlm_q_valid')
                current_vlm_q_candidate_scores = plan_payload.get('vlm_q_candidate_scores')
                current_vlm_q_best_candidate_index = plan_payload.get('vlm_q_best_candidate_index')
                current_adaptive_replan_decision = plan_payload.get('adaptive_replan_decision')
                current_carry_previous_valid = plan_payload.get('carry_previous_valid')
                current_latency_timeline_record = plan_payload.get('latency_timeline_record')
                current_vlm_failed = plan_payload.get('vlm_failed')
                current_q_selected_idx = plan_payload.get('q_selected_idx')
                current_q_selected_source = plan_payload.get('q_selected_source')
                current_q_candidate_scores = plan_payload.get('q_candidate_scores')
                current_q_carry_score = plan_payload.get('q_carry_score')
                current_q_best_current_score = plan_payload.get('q_best_current_score')
                current_q_score_gap = plan_payload.get('q_score_gap')
                current_q_invoked_vlm = plan_payload.get('q_invoked_vlm')
            else:
                current_plan_traj = plan_payload
            current_plan_origin_pose = rt2pose(
                np.asarray(current_info['ego_rot']),
                np.asarray(current_info['ego_pos']),
            )
        plan_traj = current_plan_traj

        if plan_traj is not None:
            imu_plan_traj = plan_traj[:, [1, 0]]
            imu_plan_traj[:, 1] *= -1
            global_traj = traj_transform_to_global(imu_plan_traj, current_info['ego_box'])
            current_rc = current_info.get('rc', env.unwrapped.route_completion[0])
            local_plan = np.asarray(plan_traj, dtype=np.float32)
            plan_path_length = _trajectory_path_length(local_plan)
            plan_endpoint_distance = float(np.linalg.norm(local_plan[-1])) if len(local_plan) > 0 else 0.0
            plan_min_step = float(np.linalg.norm(np.diff(local_plan, axis=0), axis=1).min()) if len(local_plan) > 1 else 0.0
            plan_max_step = float(np.linalg.norm(np.diff(local_plan, axis=0), axis=1).max()) if len(local_plan) > 1 else 0.0
            overlay_plan, last_valid_overlay_plan_stale_frames, overlay_plan_held = _select_overlay_plan(
                local_plan,
                plan_path_length,
                last_valid_overlay_plan,
                last_valid_overlay_plan_stale_frames,
            )
            overlay_plan_origin_pose = None
            if plan_path_length >= VIS_PLAN_MIN_PATH_M:
                last_valid_overlay_plan = local_plan.copy()
                last_valid_overlay_pose = current_plan_origin_pose.copy() if current_plan_origin_pose is not None else None
                overlay_plan_origin_pose = last_valid_overlay_pose
            elif overlay_plan is not None:
                overlay_plan_origin_pose = last_valid_overlay_pose
            save_data['frames'].append({
                'time_stamp': current_info['timestamp'],
                'is_key_frame': True,
                'ego_box': current_info['ego_box'],
                'ego_pos': current_info['ego_pos'],
                'ego_rot': current_info['ego_rot'],
                'obj_boxes': current_info['obj_boxes'],
                'obj_names': ['car' for _ in current_info['obj_boxes']],
                'planned_traj': {
                    'traj': global_traj,
                    'timestep': 0.5
                },
                'planner_debug': {
                    'local_plan': local_plan.tolist(),
                    'path_length_m': plan_path_length,
                    'endpoint_distance_m': plan_endpoint_distance,
                    'min_step_m': plan_min_step,
                    'max_step_m': plan_max_step,
                    'selected_idx': current_selected_idx,
                    'selected_source': current_selected_source,
                    'topk_scores': current_topk_scores,
                    'topk_count': 0 if current_topk_plans is None else int(len(current_topk_plans)),
                    'candidate_pool_scores': current_candidate_pool_scores,
                    'candidate_pool_sources': current_candidate_pool_sources,
                    'candidate_pool_proposal_indices': current_candidate_pool_proposal_indices,
                    'candidate_pool_count': 0 if current_candidate_pool_plans is None else int(len(current_candidate_pool_plans)),
                    'vlm_selected_idx': current_vlm_selected_idx,
                    'vlm_confidence': current_vlm_confidence,
                    'vlm_reasoning': current_vlm_reasoning,
                    'intervention_reasoning': current_intervention_reasoning,
                    'vlm_elapsed_sec': current_vlm_elapsed_sec,
                    'vlm_error': current_vlm_error,
                    'vlm_q_valid': current_vlm_q_valid,
                    'vlm_q_candidate_scores': current_vlm_q_candidate_scores,
                    'vlm_q_best_candidate_index': current_vlm_q_best_candidate_index,
                    'q_selected_idx': current_q_selected_idx,
                    'q_selected_source': current_q_selected_source,
                    'q_candidate_scores': current_q_candidate_scores,
                    'q_carry_score': current_q_carry_score,
                    'q_best_current_score': current_q_best_current_score,
                    'q_score_gap': current_q_score_gap,
                    'q_invoked_vlm': current_q_invoked_vlm,
                    'adaptive_replan_decision': current_adaptive_replan_decision,
                    'carry_previous_valid': current_carry_previous_valid,
                    'latency_timeline_record': current_latency_timeline_record,
                    'vlm_failed': current_vlm_failed,
                    'overlay_plan_held': bool(overlay_plan_held),
                    'overlay_plan_stale_frames': int(last_valid_overlay_plan_stale_frames),
                    'overlay_candidate_plans': None if current_candidate_pool_plans is None else [
                        np.asarray(plan, dtype=np.float32).tolist()
                        for plan in current_candidate_pool_plans
                    ],
                    'overlay_candidate_sources': None if current_candidate_pool_sources is None else [
                        str(source) for source in current_candidate_pool_sources
                    ],
                    'overlay_plan_origin_pose': None if overlay_plan_origin_pose is None else (
                        np.asarray(overlay_plan_origin_pose, dtype=np.float32).tolist()
                    ),
                },
                'collision': current_info.get('collision', False),
                'rc': current_rc
            })

            acc, steer_rate = traj2control(plan_traj, current_info)
            save_data['frames'][-1]['planner_debug']['acc_cmd'] = float(acc)
            save_data['frames'][-1]['planner_debug']['steer_rate_cmd'] = float(steer_rate)

            vis_rgb = {
                cam_name: image.copy()
                for cam_name, image in current_obs['rgb'].items()
            }
            overlay_info = dict(current_info)
            overlay_info['overlay_candidate_plans'] = current_candidate_pool_plans if current_candidate_pool_plans is not None else current_topk_plans
            overlay_info['overlay_candidate_sources'] = current_candidate_pool_sources
            vis_rgb[FRONT_CAM_NAME] = _render_front_overlay(
                current_obs['rgb'][FRONT_CAM_NAME],
                current_obs,
                overlay_info,
                env,
                overlay_plan,
                overlay_plan_origin_pose,
            )
            observations_save.append(vis_rgb)
            cv2.imwrite(
                os.path.join(overlay_front_dir, f'{len(observations_save) - 1:04d}.jpg'),
                cv2.cvtColor(vis_rgb[FRONT_CAM_NAME], cv2.COLOR_RGB2BGR),
            )

            action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            # update privileged_info after step without changing Env.step() signature
            if include_privileged_pipe:
                try:
                    privileged_info = env.unwrapped.get_agent_privileged_info()
                except Exception:
                    privileged_info = None
            cnt += 1
            done = terminated or truncated or cnt > 400

        else:
            done = True
            break

    write_pipe_message_file(obs_pipe_writer, 'Done')
    obs_pipe_writer.close()
    plan_pipe_reader.close()
    os.close(obs_pipe_keepalive_fd)
    os.close(plan_pipe_keepalive_fd)

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    try:
        to_video(
            observations_save,
            save_data['frames'],
            os.path.join(output, 'video.mp4'),
        )
    except Exception as exc:
        print(f"Skipping video export due to error: {exc}")
    try:
        to_front_video(
            observations_save,
            save_data['frames'],
            os.path.join(output, 'front.mp4'),
            run_label,
        )
    except Exception as exc:
        print(f"Skipping front video export due to error: {exc}")
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument("--planner_path", type=str, default="")
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    parser.add_argument('--include_privileged_pipe', default=False)
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    planner_path = args.planner_path
    if not planner_path:
        inferred_planner_path = os.path.join("configs", "planners", f"{args.ad}.yaml")
        if os.path.exists(inferred_planner_path):
            planner_path = inferred_planner_path
    planner_config = OmegaConf.load(planner_path) if planner_path else OmegaConf.create()
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
        {"planner": planner_config},
    )
    planner_output_suffix = args.ad
    if args.ad == 'rap' and planner_config.get('rap', {}).get('vlm', {}).get('enabled', False):
        planner_output_suffix = planner_config.get('rap', {}).get('output_suffix', 'rap_vlm')
    if args.ad == 'drivor' and planner_config.get('drivor', {}).get('vlm', {}).get('enabled', False):
        planner_output_suffix = planner_config.get('drivor', {}).get('output_suffix', 'drivor_vlm')
    if args.ad == 'rule_based' and planner_config.get('rule_based', {}).get('vlm', {}).get('enabled', False):
        planner_output_suffix = planner_config.get('rule_based', {}).get('output_suffix', 'rap_vlm')
    # planner_section = _get_planner_section(planner_config, args.ad)
    # planner_vlm_cfg = planner_section.get('vlm') or {}
    # if planner_vlm_cfg.get('enabled', False):
    #     planner_output_suffix = planner_section.get('output_suffix', f'{args.ad}_vlm')
    
    output_model_slug = _resolve_output_model_slug(args.ad, planner_config)
    cfg.base.output_dir = _prefix_output_dir_with_model(cfg.base.output_dir, output_model_slug)
    cfg.base.output_dir = cfg.base.output_dir + planner_output_suffix

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    elif args.ad == 'rap':
        ad_path = cfg.planner.rap.launch_path
    elif args.ad == "drivor":
        ad_path = cfg.planner.drivor.launch_path
    elif args.ad == "rule_based":
        ad_path = cfg.planner.rule_based.launch_path
        # rule_based_section = _get_planner_section(cfg.planner, "rule_based")
        # ad_path = rule_based_section.get(
        #     'launch_path',
        #     cfg.planner.get('drivor', {}).get('launch_path', './planners/rule_based/launch.sh'),
        # )
    else:
        raise NotImplementedError

    extra_env = {}
    if args.ad == 'rap':
        planner_python_bin = cfg.planner.rap.get('python_bin', 'python')
        rap_device = os.environ.get('RAP_DEVICE_OVERRIDE') or cfg.planner.rap.get('device', 'cuda')
        vlm_device = (
            os.environ.get('PLANNER_VLM_DEVICE_OVERRIDE')
            or os.environ.get('RAP_VLM_DEVICE_OVERRIDE')
            or cfg.planner.rap.vlm.get('device', 'auto')
        )
        extra_env = {
            'RAP_REPO_ROOT': cfg.planner.rap.get('repo_root', ''),
            'RAP_CHECKPOINT': cfg.planner.rap.get('checkpoint', ''),
            'RAP_PYTHON_BIN': planner_python_bin,
            'RAP_DEVICE': rap_device,
            'RAP_IMAGE_SCALE': cfg.planner.rap.get('image_scale', 0.4),
            'RAP_USE_SCENE_RIG_LIDAR2IMG': cfg.planner.rap.get('use_scene_rig_lidar2img', False),
            'RAP_HF_HUB_OFFLINE': cfg.planner.rap.get('hf_hub_offline', True),
            'RAP_TRANSFORMERS_OFFLINE': cfg.planner.rap.get('transformers_offline', True),
            'RAP_HF_HOME': cfg.planner.rap.get('hf_home', ''),
            'RAP_HF_HUB_CACHE': cfg.planner.rap.get('hf_hub_cache', ''),
            'RAP_TRANSFORMERS_CACHE': cfg.planner.rap.get('transformers_cache', ''),
            'RAP_NUPLAN_DEVKIT_DIR': cfg.planner.rap.get('nuplan_devkit_dir', ''),
            'RAP_BACKBONE_PATH': cfg.planner.rap.get('backbone_path', ''),
        }
        extra_env.update(
            build_prefixed_vlm_env(
                cfg.planner.rap.vlm,
                planner_python_bin=planner_python_bin,
            )
        )
        extra_env['PLANNER_VLM_DEVICE'] = vlm_device
        extra_env['RAP_VLM_DEVICE'] = vlm_device
        
    elif args.ad == "drivor":
        drivor_python_bin = cfg.planner.drivor.get('python_bin', 'python')
        drivor_device = os.environ.get('DRIVOR_DEVICE_OVERRIDE') or cfg.planner.drivor.get('device', 'cuda')
        vlm_device = (
            os.environ.get('PLANNER_VLM_DEVICE_OVERRIDE')
            or os.environ.get('DRIVOR_VLM_DEVICE_OVERRIDE')
            or cfg.planner.drivor.vlm.get('device', 'auto') if cfg.planner.drivor.get('vlm') else 'auto'
        )
        extra_env = {
            'DRIVOR_REPO_ROOT': cfg.planner.drivor.get('repo_root', ''),
            'DRIVOR_CHECKPOINT': cfg.planner.drivor.get('checkpoint', ''),
            'DRIVOR_DINO': cfg.planner.drivor.get('dino', ''),
            'DRIVOR_PYTHON_BIN': drivor_python_bin,
            'DRIVOR_DEVICE': drivor_device,
            'DRIVOR_CONFIG': cfg.planner.drivor.get('config', '')
        }
        
        # Add VLM support if configured
        if cfg.planner.drivor.get('vlm') and cfg.planner.drivor.vlm.get('enabled', False):
            extra_env.update(
                build_prefixed_vlm_env(
                    cfg.planner.drivor.vlm,
                    planner_python_bin=drivor_python_bin,
                    prefixes=("PLANNER_VLM_", "DRIVOR_VLM_"),
                )
            )
            extra_env['PLANNER_VLM_DEVICE'] = vlm_device
            extra_env['DRIVOR_VLM_DEVICE'] = vlm_device
    elif args.ad == "rule_based":
        # rule_based_cfg = _get_planner_section(cfg.planner, "rule_based")
        rule_based_python_bin = cfg.planner.rule_based.get('python_bin', 'python')
        rule_based_device = os.environ.get('RULE_BASED_DEVICE_OVERRIDE') or cfg.planner.rule_based.get('device', 'cpu')
        vlm_device = (
                os.environ.get('PLANNER_VLM_DEVICE_OVERRIDE')
                or os.environ.get('RULE_BASED_VLM_DEVICE_OVERRIDE')
                or cfg.planner.rule_based.vlm.get('device', 'auto') if cfg.planner.rule_based.get('vlm') else 'auto'
            )
        extra_env = {
            'RULE_BASED_REPO_ROOT': cfg.planner.rule_based.get('repo_root', ''),
            'RULE_BASED_PYTHON_BIN': rule_based_python_bin,
            'RULE_BASED_DEVICE': rule_based_device,
            'RULE_BASED_CONFIG': cfg.planner.rule_based.get('config', ''),
        }
        # Add VLM support if configured
        # rule_based_vlm_cfg = (cfg.planner.rule_based.get('vlm') or {}) if hasattr(cfg.planner.rule_based, 'get') else {}
        # if rule_based_vlm_cfg and rule_based_vlm_cfg.get('enabled', False):
        if cfg.planner.rule_based.get('vlm') and cfg.planner.rule_based.vlm.get('enabled', False):
            extra_env.update(
                build_prefixed_vlm_env(
                    # rule_based_vlm_cfg,
                    cfg.planner.rule_based.vlm,
                    planner_python_bin=rule_based_python_bin,
                    prefixes=("PLANNER_VLM_", "RULE_BASED_VLM_"),
                )
            )
            extra_env['PLANNER_VLM_DEVICE'] = vlm_device
            extra_env['RULE_BASED_VLM_DEVICE'] = vlm_device
    print("preparing to launch client.py")
    process = launch(ad_path, args.ad_cuda, output, extra_env=extra_env)
    print("client.py launched, waiting 10 seconds before running create_gym_env")
    time.sleep(10)
    try:
        create_gym_env(cfg, output, planner_output_suffix, args.include_privileged_pipe)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # create_gym_env(cfg, output)
