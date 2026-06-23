"""HUGSIM closed-loop simulator runner (sim-side).

Extracted from the legacy ``closed_loop.py`` so ``pipeline.py`` no longer depends
on it. Owns the simulator loop: render cameras, send the observation to the
planner subprocess over a FIFO, receive the response, turn the selected
trajectory into a control action via ``traj2control``, step the environment, and
finally export video + evaluation.

``run_closed_loop``'s ``plan_adapter`` hook lets ``pipeline.py`` convert the new
minimal ``(proposals, scores)`` planner response into a plan payload via the
pipeline-side selector.
"""
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
import logging
import time
import stat
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
import cv2
from autoagent0.adapters.hugsim.demo_tasks import (
    apply_demo_task_action_override,
    is_park_task_complete,
    is_stop_task_complete,
    summarize_demo_overlay,
)
from autoagent0.adapters.hugsim.results import (
    build_run_performance,
    prefix_output_dir_with_model,
    resolve_output_model_slug,
)
from autoagent0.adapters.hugsim.runtime import (
    raise_if_process_exited,
    read_pipe_message_file,
    write_pipe_message_file,
)
from autoagent0.adapters.hugsim.video import to_front_video, to_video
from autoagent0.config import build_prefixed_autoagent0_env
from autoagent0.adapters.hugsim.candidate_visuals import get_candidate_visual_style
from autoagent0.adapters.hugsim.task_overlay import draw_task_target_overlay
from autoagent0.vlm.vlm_env import build_prefixed_vlm_env
from autoagent0.experts.rule_based_env import build_prefixed_rule_based_env

FRONT_CAM_NAME = 'CAM_FRONT'
REFERENCE_COLOR = (255, 230, 0)
PLAN_COLOR = (0, 120, 255)
REFERENCE_FORWARD_OFFSET_M = 4.0
PLAN_VIS_FORWARD_OFFSET_M = 4.5
VIS_PLAN_MIN_PATH_M = 0.5
VIS_PLAN_HOLD_FRAMES = 10**9
PLAN_REPLAN_EVERY_STEPS = 2


def _resolve_scene_model_path(model_base, scene_name):
    direct_path = os.path.join(model_base, scene_name)
    if os.path.isfile(os.path.join(direct_path, 'cfg.yaml')):
        return direct_path

    matches = []
    for root, dirs, _files in os.walk(model_base):
        if scene_name in dirs:
            candidate = os.path.join(root, scene_name)
            if os.path.isfile(os.path.join(candidate, 'cfg.yaml')):
                matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"multiple processed scene directories found for {scene_name!r} under {model_base!r}: {matches}"
        )
    raise FileNotFoundError(
        f"processed scene directory for {scene_name!r} not found under {model_base!r}"
    )


def _parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_privileged_info(env):
    try:
        return env.unwrapped.get_agent_privileged_info()
    except Exception:
        return None


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

    task_overlay_state = draw_task_target_overlay(
        overlay,
        current_info,
        intrinsic,
        front_c2w,
        draw_status_badge=bool(current_info.get('task_active')),
    )
    return overlay, task_overlay_state


def _select_overlay_plan(plan_traj, plan_path_length, last_valid_plan, stale_frames):
    if plan_traj is not None and len(plan_traj) > 0 and plan_path_length >= VIS_PLAN_MIN_PATH_M:
        return np.asarray(plan_traj, dtype=np.float32), 0, False

    if last_valid_plan is not None and stale_frames < VIS_PLAN_HOLD_FRAMES:
        return last_valid_plan.copy(), stale_frames + 1, True

    return None, stale_frames, False


from dataclasses import dataclass, field


@dataclass
class PlanDecision:
    """Decision fields parsed from one planner response (held across non-replan frames)."""
    selected_plan: object = None
    plan_origin_pose: object = None
    topk_plans: object = None
    topk_scores: object = None
    candidate_pool_plans: object = None
    candidate_pool_scores: object = None
    candidate_pool_sources: object = None
    candidate_pool_proposal_indices: object = None
    selected_idx: object = None
    selected_source: object = None
    vlm_selected_idx: object = None
    vlm_confidence: object = None
    vlm_reasoning: object = None
    intervention_reasoning: object = None
    vlm_elapsed_sec: object = None
    vlm_error: object = None
    vlm_q_valid: object = None
    vlm_q_candidate_scores: object = None
    vlm_q_best_candidate_index: object = None
    adaptive_replan_decision: object = None
    carry_previous_valid: object = None
    latency_timeline_record: object = None
    vlm_failed: object = None
    q_selected_idx: object = None
    q_selected_source: object = None
    q_candidate_scores: object = None
    q_carry_score: object = None
    q_best_current_score: object = None
    q_score_gap: object = None
    q_invoked_vlm: object = None
    autoagent0_debug: dict = field(default_factory=dict)
    planner_decision_step: object = None
    planner_decision_frame_index: object = None
    planner_decision_timestamp: object = None


@dataclass
class SceneFifos:
    """The obs/plan FIFO pair plus keepalive descriptors for one run."""
    obs_pipe_writer: object
    plan_pipe_reader: object
    obs_pipe_keepalive_fd: int
    plan_pipe_keepalive_fd: int
    plan_response_timeout_sec: float

    def finish(self) -> None:
        write_pipe_message_file(self.obs_pipe_writer, 'Done')
        self.obs_pipe_writer.close()
        self.plan_pipe_reader.close()
        os.close(self.obs_pipe_keepalive_fd)
        os.close(self.plan_pipe_keepalive_fd)


def _open_scene_fifos(output, obs, info, include_privileged_pipe) -> SceneFifos:
    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    for pipe_path in (obs_pipe, plan_pipe):
        if os.path.exists(pipe_path):
            st = os.stat(pipe_path)
            if not stat.S_ISFIFO(st.st_mode):
                raise RuntimeError(f"Refusing to replace non-FIFO path: {pipe_path}")
            os.unlink(pipe_path)
            print(f"[FIFO RECREATE] removed stale FIFO path={pipe_path} inode={st.st_ino} t={time.time():.6f}")
        os.mkfifo(pipe_path)
        st = os.stat(pipe_path)
        print(f"[FIFO CREATE] path={pipe_path} inode={st.st_ino} mode={oct(st.st_mode)} t={time.time():.6f}")
    # Keep both FIFOs open in RDWR mode to avoid blocking open-order deadlocks
    # between the loop and the planner adapter. Actual message traffic still
    # uses fresh per-message open/read/write calls below.
    obs_pipe_keepalive_fd = os.open(obs_pipe, os.O_RDWR | os.O_NONBLOCK)
    plan_pipe_keepalive_fd = os.open(plan_pipe, os.O_RDWR | os.O_NONBLOCK)
    obs_pipe_writer = os.fdopen(os.open(obs_pipe, os.O_RDWR), "wb", buffering=0)
    plan_pipe_reader = os.fdopen(os.open(plan_pipe, os.O_RDWR), "rb", buffering=0)
    plan_response_timeout_sec = float(os.environ.get("HUGSIM_PLAN_RESPONSE_TIMEOUT_SEC", "0") or 0.0)
    print('Ready for simulation')
    preflight_payload = {
        "message_type": "hugsim_preflight",
        "output_dir": output,
        "obs_pipe": obs_pipe,
        "plan_pipe": plan_pipe,
        "include_privileged_pipe": bool(include_privileged_pipe),
        "camera_count": 0 if not isinstance(obs, dict) else len(obs.get("rgb", {})),
        "timestamp": float(info.get("timestamp", 0.0)) if isinstance(info, dict) else None,
    }
    print(f"[preflight] about to write t={time.time():.6f}")
    write_pipe_message_file(obs_pipe_writer, preflight_payload)
    print(f"[preflight] wrote t={time.time():.6f}")
    return SceneFifos(
        obs_pipe_writer=obs_pipe_writer,
        plan_pipe_reader=plan_pipe_reader,
        obs_pipe_keepalive_fd=obs_pipe_keepalive_fd,
        plan_pipe_keepalive_fd=plan_pipe_keepalive_fd,
        plan_response_timeout_sec=plan_response_timeout_sec,
    )


def _request_plan(current_obs, current_info, privileged_info, fifos, planner_process, plan_adapter, include_privileged_pipe):
    raise_if_process_exited(planner_process, "preparing planner request")
    # Ensure expected camera keys exist and fill missing ones with black frames.
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

        rgb_details = []
        for k, v in rgb.items():
            if hasattr(v, 'shape'):
                rgb_details.append(f"{k}: shape={getattr(v,'shape')} dtype={getattr(v,'dtype', None)}")
            else:
                rgb_details.append(f"{k}: type={type(v)}")
        logger.info("Sending obs to adapter: rgb keys=%s; details=%s", list(rgb.keys()), "; ".join(rgb_details))
    except Exception:
        logging.getLogger("closed_loop").exception("Failed to prepare current_obs before sending")
    plan_request_payload = (
        (current_obs, current_info, privileged_info)
        if include_privileged_pipe
        else (current_obs, current_info)
    )
    print(
        f"[plan_request] about to write t={time.time():.6f} "
        f"include_privileged_pipe={include_privileged_pipe}"
    )
    write_pipe_message_file(fifos.obs_pipe_writer, plan_request_payload)
    print(f"[plan_request] wrote t={time.time():.6f}")
    plan_payload = read_pipe_message_file(
        fifos.plan_pipe_reader,
        producer_process=planner_process,
        timeout_sec=fifos.plan_response_timeout_sec,
    )
    print(f"[plan_response] received t={time.time():.6f}")
    if plan_adapter is not None and plan_payload is not None and plan_payload != 'Done':
        plan_payload = plan_adapter(plan_payload, current_obs, current_info, privileged_info)
    return plan_payload


def _parse_plan_payload(plan_payload, cnt, current_info) -> PlanDecision:
    d = PlanDecision()
    d.planner_decision_step = cnt
    d.planner_decision_timestamp = current_info.get('timestamp')
    if isinstance(plan_payload, dict):
        d.selected_plan = plan_payload.get('selected_plan')
        d.topk_plans = plan_payload.get('topk_plans')
        d.topk_scores = plan_payload.get('topk_scores')
        d.candidate_pool_plans = plan_payload.get('candidate_pool_plans')
        d.candidate_pool_scores = plan_payload.get('candidate_pool_scores')
        d.candidate_pool_sources = plan_payload.get('candidate_pool_sources')
        d.candidate_pool_proposal_indices = plan_payload.get('candidate_pool_proposal_indices')
        d.selected_idx = plan_payload.get('selected_idx')
        d.selected_source = plan_payload.get('selected_source')
        d.vlm_selected_idx = plan_payload.get('vlm_selected_idx')
        d.vlm_confidence = plan_payload.get('vlm_confidence')
        d.vlm_reasoning = plan_payload.get('vlm_reasoning')
        d.intervention_reasoning = plan_payload.get('intervention_reasoning')
        d.vlm_elapsed_sec = plan_payload.get('vlm_elapsed_sec')
        d.vlm_error = plan_payload.get('vlm_error')
        d.vlm_q_valid = plan_payload.get('vlm_q_valid')
        d.vlm_q_candidate_scores = plan_payload.get('vlm_q_candidate_scores')
        d.vlm_q_best_candidate_index = plan_payload.get('vlm_q_best_candidate_index')
        d.adaptive_replan_decision = plan_payload.get('adaptive_replan_decision')
        d.carry_previous_valid = plan_payload.get('carry_previous_valid')
        d.latency_timeline_record = plan_payload.get('latency_timeline_record')
        d.vlm_failed = plan_payload.get('vlm_failed')
        d.q_selected_idx = plan_payload.get('q_selected_idx')
        d.q_selected_source = plan_payload.get('q_selected_source')
        d.q_candidate_scores = plan_payload.get('q_candidate_scores')
        d.q_carry_score = plan_payload.get('q_carry_score')
        d.q_best_current_score = plan_payload.get('q_best_current_score')
        d.q_score_gap = plan_payload.get('q_score_gap')
        d.q_invoked_vlm = plan_payload.get('q_invoked_vlm')
        d.autoagent0_debug = {
            key: value
            for key, value in plan_payload.items()
            if (
                key.startswith('autoagent0_')
                or key in {
                    'agent_trace',
                    'selected_candidate_index',
                    'selected_candidate_source',
                    'selected_path_reasoning',
                    'selected_proposal_index',
                }
            )
        }
        if isinstance(d.latency_timeline_record, dict):
            d.planner_decision_frame_index = d.latency_timeline_record.get('frame_index')
        else:
            d.planner_decision_frame_index = None
    else:
        d.selected_plan = plan_payload
        d.planner_decision_frame_index = None
    d.plan_origin_pose = rt2pose(
        np.asarray(current_info['ego_rot']),
        np.asarray(current_info['ego_pos']),
    )
    return d


def _record_frame(save_data, decision, current_info, *, global_traj, current_rc,
                  local_plan, plan_path_length, plan_endpoint_distance, plan_min_step, plan_max_step,
                  overlay_plan_held, overlay_plan_stale_frames, overlay_plan_origin_pose,
                  should_replan, cnt, acc, steer_rate):
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
            'selected_idx': decision.selected_idx,
            'selected_source': decision.selected_source,
            'topk_scores': decision.topk_scores,
            'topk_count': 0 if decision.topk_plans is None else int(len(decision.topk_plans)),
            'candidate_pool_scores': decision.candidate_pool_scores,
            'candidate_pool_sources': decision.candidate_pool_sources,
            'candidate_pool_proposal_indices': decision.candidate_pool_proposal_indices,
            'candidate_pool_count': 0 if decision.candidate_pool_plans is None else int(len(decision.candidate_pool_plans)),
            'vlm_selected_idx': decision.vlm_selected_idx,
            'vlm_confidence': decision.vlm_confidence,
            'vlm_reasoning': decision.vlm_reasoning,
            'intervention_reasoning': decision.intervention_reasoning,
            'vlm_elapsed_sec': decision.vlm_elapsed_sec,
            'vlm_error': decision.vlm_error,
            'vlm_q_valid': decision.vlm_q_valid,
            'vlm_q_candidate_scores': decision.vlm_q_candidate_scores,
            'vlm_q_best_candidate_index': decision.vlm_q_best_candidate_index,
            'q_selected_idx': decision.q_selected_idx,
            'q_selected_source': decision.q_selected_source,
            'q_candidate_scores': decision.q_candidate_scores,
            'q_carry_score': decision.q_carry_score,
            'q_best_current_score': decision.q_best_current_score,
            'q_score_gap': decision.q_score_gap,
            'q_invoked_vlm': decision.q_invoked_vlm,
            'adaptive_replan_decision': decision.adaptive_replan_decision,
            'execution_mode': current_info.get('execution_mode'),
            'planner_gate_selected_planner': current_info.get('planner_gate_selected_planner'),
            'planner_gate_confidence': current_info.get('planner_gate_confidence'),
            'planner_gate_reasoning': current_info.get('planner_gate_reasoning'),
            'planner_gate_elapsed_sec': current_info.get('planner_gate_elapsed_sec'),
            'planner_gate_error': current_info.get('planner_gate_error'),
            'planner_gate_timed_out': current_info.get('planner_gate_timed_out'),
            'carry_previous_valid': decision.carry_previous_valid,
            'latency_timeline_record': decision.latency_timeline_record,
            'vlm_failed': decision.vlm_failed,
            'planner_decision_fresh': bool(should_replan),
            'planner_decision_step': decision.planner_decision_step,
            'planner_decision_age_steps': None if decision.planner_decision_step is None else int(cnt - decision.planner_decision_step),
            'planner_decision_frame_index': decision.planner_decision_frame_index,
            'planner_decision_timestamp': decision.planner_decision_timestamp,
            'overlay_plan_held': bool(overlay_plan_held),
            'overlay_plan_stale_frames': int(overlay_plan_stale_frames),
            'overlay_candidate_plans': None if decision.candidate_pool_plans is None else [
                np.asarray(plan, dtype=np.float32).tolist()
                for plan in decision.candidate_pool_plans
            ],
            'overlay_candidate_sources': None if decision.candidate_pool_sources is None else [
                str(source) for source in decision.candidate_pool_sources
            ],
            'overlay_plan_origin_pose': None if overlay_plan_origin_pose is None else (
                np.asarray(overlay_plan_origin_pose, dtype=np.float32).tolist()
            ),
            'acc_cmd': float(acc),
            'steer_rate_cmd': float(steer_rate),
            **decision.autoagent0_debug,
        },
        'collision': current_info.get('collision', False),
        'rc': current_rc
    })


def _render_and_store_overlay(env, current_obs, current_info, decision, overlay_plan,
                              overlay_plan_origin_pose, observations_save, demo_overlay_records,
                              overlay_front_dir, demo_completion_reason):
    vis_rgb = {
        cam_name: image.copy()
        for cam_name, image in current_obs['rgb'].items()
    }
    overlay_info = dict(current_info)
    overlay_info['overlay_candidate_plans'] = decision.candidate_pool_plans if decision.candidate_pool_plans is not None else decision.topk_plans
    overlay_info['overlay_candidate_sources'] = decision.candidate_pool_sources
    vis_rgb[FRONT_CAM_NAME], task_overlay_state = _render_front_overlay(
        current_obs['rgb'][FRONT_CAM_NAME],
        current_obs,
        overlay_info,
        env,
        overlay_plan,
        overlay_plan_origin_pose,
    )
    demo_task_info = None
    if current_info.get('task_active'):
        demo_task_info = {
            'task_type': current_info.get('task_type'),
            'task_instruction': current_info.get('task_instruction'),
            'task_target_pose_local': current_info.get('task_target_pose_local'),
            'task_target_world': current_info.get('task_target_world'),
            'task_completion_reason': demo_completion_reason,
        }
        demo_overlay_records.append({
            'frame_idx': int(len(observations_save)),
            **task_overlay_state,
        })
    observations_save.append(vis_rgb)
    cv2.imwrite(
        os.path.join(overlay_front_dir, f'{len(observations_save) - 1:04d}.jpg'),
        cv2.cvtColor(vis_rgb[FRONT_CAM_NAME], cv2.COLOR_RGB2BGR),
    )
    return demo_task_info


def _finalize_and_evaluate(output, save_data, observations_save, infos_save, demo_overlay_records,
                           demo_task_info, demo_completion_reason, run_label, cfg, ad_name):
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
    if demo_task_info:
        final_goal_status = None
        if infos_save:
            last_info = infos_save[-1]
            if isinstance(last_info, dict):
                final_goal_status = last_info.get('task_goal_status')
                if demo_completion_reason:
                    demo_task_info['task_completion_reason'] = demo_completion_reason
        with open(os.path.join(output, 'demo_summary.json'), 'w') as wf:
            json.dump(
                summarize_demo_overlay(demo_overlay_records, demo_task_info, final_goal_status),
                wf,
                indent=2,
            )

    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    if isinstance(results, dict):
        results["performance"] = build_run_performance(
            output_dir=output,
            cfg=cfg,
            ad_name=ad_name if ad_name is not None else run_label,
            frame_count=len(save_data.get("frames", [])),
        )
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


def run_closed_loop(cfg, output, run_label, include_privileged_pipe=False, planner_process=None, plan_adapter=None, ad_name=None):
    # plan_adapter (optional): callable(raw_plan_response, current_obs, current_info,
    # privileged_info) -> plan_payload. When None (default) the value read from the
    # plan FIFO is used as-is (legacy behaviour: the planner subprocess returns the
    # full plan payload). pipeline.py passes an adapter that turns the new minimal
    # (proposals, scores) response into a plan payload via the pipeline-side selector.

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    privileged_info = _get_privileged_info(env) if include_privileged_pipe else None
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    fifos = _open_scene_fifos(output, obs, info, include_privileged_pipe)

    last_valid_overlay_plan = None
    last_valid_overlay_pose = None
    last_valid_overlay_plan_stale_frames = 0
    decision = PlanDecision()
    overlay_front_dir = os.path.join(output, 'overlay_front')
    os.makedirs(overlay_front_dir, exist_ok=True)
    for name in os.listdir(overlay_front_dir):
        if name.endswith('.jpg'):
            os.remove(os.path.join(overlay_front_dir, name))
    demo_overlay_records = []
    demo_task_info = None
    demo_completion_reason = None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
            privileged_info = _get_privileged_info(env) if include_privileged_pipe else None
        current_obs, current_info = obs, info
        infos_save.append(current_info)
        if is_stop_task_complete(current_info):
            demo_completion_reason = 'stop_reached'
            done = True
            break
        if is_park_task_complete(current_info):
            demo_completion_reason = 'park_reached'
            done = True
            break

        print('ego pose', current_info['ego_pos'])

        should_replan = (cnt % PLAN_REPLAN_EVERY_STEPS == 0) or (decision.selected_plan is None)
        if should_replan:
            plan_payload = _request_plan(
                current_obs, current_info, privileged_info, fifos,
                planner_process, plan_adapter, include_privileged_pipe,
            )
            decision = _parse_plan_payload(plan_payload, cnt, current_info)
        plan_traj = decision.selected_plan

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
                last_valid_overlay_pose = decision.plan_origin_pose.copy() if decision.plan_origin_pose is not None else None
                overlay_plan_origin_pose = last_valid_overlay_pose
            elif overlay_plan is not None:
                overlay_plan_origin_pose = last_valid_overlay_pose

            acc, steer_rate = traj2control(plan_traj, current_info)
            _record_frame(
                save_data, decision, current_info,
                global_traj=global_traj, current_rc=current_rc,
                local_plan=local_plan, plan_path_length=plan_path_length,
                plan_endpoint_distance=plan_endpoint_distance,
                plan_min_step=plan_min_step, plan_max_step=plan_max_step,
                overlay_plan_held=overlay_plan_held,
                overlay_plan_stale_frames=last_valid_overlay_plan_stale_frames,
                overlay_plan_origin_pose=overlay_plan_origin_pose,
                should_replan=should_replan, cnt=cnt, acc=acc, steer_rate=steer_rate,
            )

            demo_task_info = _render_and_store_overlay(
                env, current_obs, current_info, decision, overlay_plan, overlay_plan_origin_pose,
                observations_save, demo_overlay_records, overlay_front_dir, demo_completion_reason,
            ) or demo_task_info

            action = {'acc': acc, 'steer_rate': steer_rate}
            action = apply_demo_task_action_override(action, current_info)
            obs, reward, terminated, truncated, info = env.step(action)
            if include_privileged_pipe:
                privileged_info = _get_privileged_info(env)
            cnt += 1
            done = terminated or truncated or cnt > 400

        else:
            done = True
            break

    fifos.finish()
    _finalize_and_evaluate(
        output, save_data, observations_save, infos_save, demo_overlay_records,
        demo_task_info, demo_completion_reason, run_label, cfg, ad_name,
    )
