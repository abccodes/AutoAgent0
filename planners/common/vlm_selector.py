from __future__ import annotations

import json
import logging
import math
import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SCR

from planners.common.candidate_visuals import get_candidate_visual_style

LOG = logging.getLogger(__name__)
PLAN_DT_SEC = 0.5
PLAN_VIS_FORWARD_OFFSET_M = 4.5
PLAN_RESAMPLE_SPACING_M = 0.08
VLM_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
)


@dataclass
class VLMSelectorConfig:
    enabled: bool = False
    intervention_enabled: bool = False
    camera_mode: str = "multiview"
    intervention_camera_mode: str = ""
    scoring_camera_mode: str = ""
    backend: str = "local_transformers"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    python_bin: str = ""
    max_new_tokens: int = 300
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    enable_thinking: bool = False
    candidate_limit: int = 10
    intervention_max_new_tokens: int = 120
    timeout_sec: float = 10.0
    intervention_timeout_sec: float = 10.0
    preload_on_init: bool = True
    save_debug_artifacts: bool = True
    debug_dir_name: str = "vlm_debug"
    carry_previous_enabled: bool = True
    carry_previous_min_path_m: float = 0.5
    carry_previous_min_points: int = 2
    adaptive_replan_mode: str = "log_only"
    latency_tracking_mode: str = "full_timeline"
    q_enabled: bool = True
    q_switch_margin: float = 0.05
    q_weight_rap_score: float = 0.55
    q_weight_progress: float = 0.30
    q_weight_offcenter: float = 0.10
    q_weight_curvature: float = 0.08
    q_weight_shortplan: float = 0.18
    q_carry_score_decay: float = 0.0
    display_default_trajectories: bool = False
    include_default_candidates: bool = False


def _fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def _get_camera_matrix(intrinsic: Dict[str, float]) -> np.ndarray:
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = _fov2focal(intrinsic["fovx"], intrinsic["W"])
    K[1, 1] = _fov2focal(intrinsic["fovy"], intrinsic["H"])
    K[0, 2] = intrinsic["cx"]
    K[1, 2] = intrinsic["cy"]
    return K


def _rt2pose(r: Sequence[float], t: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = SCR.from_euler("XYZ", r, degrees=False).as_matrix().astype(np.float32)
    pose[:3, 3] = np.asarray(t, dtype=np.float32)
    return pose


def _get_camera_c2w(cam_params: Dict[str, Dict[str, np.ndarray]], ego_pose: np.ndarray, cam_name: str) -> np.ndarray:
    params = cam_params[cam_name]
    if "front2cam" in params:
        return ego_pose @ np.asarray(params["front2cam"], dtype=np.float32)

    v2front = np.asarray(cam_params["CAM_FRONT"]["v2c"], dtype=np.float32)
    v2c = np.asarray(params["v2c"], dtype=np.float32)
    c2front = v2front @ np.linalg.inv(v2c)
    return ego_pose @ c2front


def _local_plan_to_front_world(plan_traj: np.ndarray, front_c2w: np.ndarray, front_v2c: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    local_xyz = np.zeros((len(plan_traj) + 1, 3), dtype=np.float32)
    local_xyz[1:, :2] = plan_traj
    local_xyz[:, 1] += float(PLAN_VIS_FORWARD_OFFSET_M)

    camera_in_vehicle = np.linalg.inv(np.asarray(front_v2c, dtype=np.float32))[:3, 3]
    camera_in_local = np.array(
        [-camera_in_vehicle[1], camera_in_vehicle[0], camera_in_vehicle[2]],
        dtype=np.float32,
    )
    local_to_front_cam = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    points_cam = (local_to_front_cam @ (local_xyz - camera_in_local).T).T
    homogeneous = np.concatenate(
        [points_cam, np.ones((len(points_cam), 1), dtype=np.float32)],
        axis=1,
    )
    return (np.asarray(front_c2w, dtype=np.float32) @ homogeneous.T).T[:, :3].astype(np.float32)


def _resample_polyline(points_world: np.ndarray, spacing: float = PLAN_RESAMPLE_SPACING_M) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return points_world

    segment_lengths = np.linalg.norm(np.diff(points_world[:, [0, 2]], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-4:
        return points_world[:1]

    sample_distances = np.arange(0.0, total_length + spacing, spacing, dtype=np.float32)
    sample_distances[-1] = min(sample_distances[-1], total_length)

    resampled = []
    seg_idx = 0
    for dist in sample_distances:
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < dist:
            seg_idx += 1
        if seg_idx + 1 >= len(points_world):
            resampled.append(points_world[-1])
            continue
        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        denom = max(seg_end - seg_start, 1e-6)
        alpha = float((dist - seg_start) / denom)
        point = (1.0 - alpha) * points_world[seg_idx] + alpha * points_world[seg_idx + 1]
        resampled.append(point)
    return np.asarray(resampled, dtype=np.float32)


def _draw_projected_polyline_camera_clipped(
    image: np.ndarray,
    points_world: np.ndarray,
    intrinsic: Dict[str, float],
    c2w: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 3,
    near: float = 1e-3,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return image

    K = _get_camera_matrix(intrinsic)[:3, :3]
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (w2c @ homogeneous.T).T[:, :3]

    h, w = image.shape[:2]
    rect = (0, 0, w, h)

    def project_point(point_cam: np.ndarray) -> Tuple[int, int]:
        projected = K @ point_cam
        uv = projected[:2] / np.clip(projected[2], near, None)
        return tuple(np.round(uv).astype(np.int32))

    for p0, p1 in zip(camera_points[:-1], camera_points[1:]):
        z0, z1 = float(p0[2]), float(p1[2])
        if z0 <= near and z1 <= near:
            continue

        q0 = p0.copy()
        q1 = p1.copy()
        if z0 <= near:
            alpha = (near - z0) / max(z1 - z0, 1e-6)
            q0 = p0 + alpha * (p1 - p0)
            q0[2] = near
        if z1 <= near:
            alpha = (near - z1) / max(z0 - z1, 1e-6)
            q1 = p1 + alpha * (p0 - p1)
            q1[2] = near

        pixel0 = project_point(q0)
        pixel1 = project_point(q1)
        ok, clipped_p0, clipped_p1 = cv2.clipLine(rect, pixel0, pixel1)
        if not ok:
            continue
        cv2.line(image, clipped_p0, clipped_p1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return image


def _project_world_points_to_image(
    points_world: np.ndarray,
    intrinsic: Dict[str, float],
    c2w: np.ndarray,
) -> List[Tuple[int, int, bool]]:
    if len(points_world) == 0:
        return []

    K = _get_camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    valid = depth > 1e-4
    points: List[Tuple[int, int, bool]] = []
    for idx, point_cam in enumerate(camera_points):
        if not valid[idx]:
            points.append((0, 0, False))
            continue
        image_point = K[:3, :3] @ point_cam
        uv = image_point[:2] / np.clip(image_point[2], 1e-3, None)
        points.append((int(round(float(uv[0]))), int(round(float(uv[1]))), True))
    return points


def render_candidate_overlay(
    camera_image: np.ndarray,
    info: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    cam_name: str = "CAM_FRONT",
) -> np.ndarray:
    overlay = np.asarray(camera_image, dtype=np.uint8).copy()
    ego_pose = _rt2pose(info["ego_rot"], info["ego_pos"])
    cam_params = info["cam_params"]
    intrinsic = cam_params[cam_name]["intrinsic"]
    camera_c2w = _get_camera_c2w(cam_params, ego_pose, cam_name)
    camera_v2c = cam_params[cam_name]["v2c"]

    for row in candidate_rows:
        plan_local = np.asarray(row["local_plan"], dtype=np.float32)
        plan_world = _local_plan_to_front_world(plan_local, camera_c2w, camera_v2c)
        plan_world_dense = _resample_polyline(plan_world, spacing=PLAN_RESAMPLE_SPACING_M)
        color = tuple(int(v) for v in row["color_bgr"])
        _draw_projected_polyline_camera_clipped(
            overlay,
            plan_world_dense,
            intrinsic,
            camera_c2w,
            color=color,
            thickness=4 if row["candidate_rank"] == 0 else 2,
        )

        if len(plan_world_dense) > 0:
            h, w = overlay.shape[:2]
            pixels = _project_world_points_to_image(plan_world_dense, intrinsic, camera_c2w)
            label_anchor = None
            for px, py, valid in reversed(pixels):
                if valid and 0 <= px < w and 0 <= py < h:
                    label_anchor = (int(px), int(py))
                    break
            if label_anchor is not None:
                cv2.putText(
                    overlay,
                    str(row["candidate_index"]),
                    (label_anchor[0] + 4, label_anchor[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return overlay


def render_candidate_overlays(
    camera_images: Dict[str, np.ndarray],
    info: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    camera_order: Sequence[str] = VLM_CAMERA_ORDER,
) -> Dict[str, np.ndarray]:
    overlays: Dict[str, np.ndarray] = {}
    for cam_name in camera_order:
        image = camera_images.get(cam_name)
        if image is None:
            continue
        if cam_name == "CAM_FRONT":
            overlays[cam_name] = render_candidate_overlay(
                image,
                info,
                candidate_rows,
                cam_name=cam_name,
            )
        else:
            overlays[cam_name] = np.asarray(image, dtype=np.uint8).copy()
    return overlays


def summarize_candidate(
    points: List[Sequence[float]],
    current_ego_speed_mps: Optional[float] = None,
) -> Dict[str, object]:
    if not points:
        return {
            "num_points": 0,
            "start": None,
            "end": None,
            "min_x": None,
            "max_x": None,
            "min_y": None,
            "max_y": None,
            "delta_x": None,
            "delta_y": None,
            "path_length_m": None,
            "forward_progress_m": None,
            "first_step_m": None,
            "first_step_speed_mps": None,
            "avg_speed_mps": None,
            "max_step_speed_mps": None,
            "speed_delta_vs_ego_mps": None,
            "first_step_accel_vs_ego_mps2": None,
        }

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    start = [round(xs[0], 3), round(ys[0], 3)]
    end = [round(xs[-1], 3), round(ys[-1], 3)]
    points_arr = np.asarray(points, dtype=np.float32)
    step_distances = np.linalg.norm(np.diff(points_arr, axis=0), axis=1) if len(points_arr) > 1 else np.zeros((0,), dtype=np.float32)
    path_length_m = float(step_distances.sum())
    first_step_m = float(step_distances[0]) if len(step_distances) > 0 else 0.0
    first_step_speed_mps = first_step_m / PLAN_DT_SEC
    avg_speed_mps = path_length_m / (len(step_distances) * PLAN_DT_SEC) if len(step_distances) > 0 else 0.0
    max_step_speed_mps = float(step_distances.max()) / PLAN_DT_SEC if len(step_distances) > 0 else 0.0
    speed_delta_vs_ego_mps = None
    first_step_accel_vs_ego_mps2 = None
    if current_ego_speed_mps is not None:
        speed_delta_vs_ego_mps = first_step_speed_mps - float(current_ego_speed_mps)
        first_step_accel_vs_ego_mps2 = speed_delta_vs_ego_mps / PLAN_DT_SEC
    return {
        "num_points": len(points),
        "start": start,
        "end": end,
        "min_x": round(min(xs), 3),
        "max_x": round(max(xs), 3),
        "min_y": round(min(ys), 3),
        "max_y": round(max(ys), 3),
        "delta_x": round(end[0] - start[0], 3),
        "delta_y": round(end[1] - start[1], 3),
        "path_length_m": round(path_length_m, 3),
        "forward_progress_m": round(max(0.0, end[1] - start[1]), 3),
        "first_step_m": round(first_step_m, 3),
        "first_step_speed_mps": round(first_step_speed_mps, 3),
        "avg_speed_mps": round(avg_speed_mps, 3),
        "max_step_speed_mps": round(max_step_speed_mps, 3),
        "speed_delta_vs_ego_mps": None if speed_delta_vs_ego_mps is None else round(speed_delta_vs_ego_mps, 3),
        "first_step_accel_vs_ego_mps2": None if first_step_accel_vs_ego_mps2 is None else round(first_step_accel_vs_ego_mps2, 3),
    }


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def format_candidate_text(candidate_rows: Sequence[Dict[str, object]]) -> str:
    lines = []
    for row in candidate_rows:
        s = row["summary"]
        line = (
            f"- candidate_{row['candidate_index']} | color={row['color_name']} | source={row['source']} | "
        )
        line += (
            f"num_points={s['num_points']} | "
            f"start={s['start']} | end={s['end']} | "
            f"x_range=[{s['min_x']},{s['max_x']}] | y_range=[{s['min_y']},{s['max_y']}] | "
            f"delta=({s['delta_x']},{s['delta_y']}) | "
            f"path_length_m={s['path_length_m']} | "
            f"forward_progress_m={s['forward_progress_m']}"
        )
        lines.append(line)
    return "\n".join(lines)


def resolve_vlm_camera_order(camera_mode: str) -> Tuple[str, ...]:
    mode = str(camera_mode or "multiview").strip().lower()
    if mode in {"front", "front_only", "single_front"}:
        return ("CAM_FRONT",)
    return tuple(VLM_CAMERA_ORDER)


def describe_vlm_camera_inputs(camera_order: Sequence[str]) -> Tuple[str, str]:
    camera_order = tuple(camera_order)
    if camera_order == ("CAM_FRONT",):
        return (
            "A front-facing driving image.",
            "The image has the trajectory overlay.",
        )
    return (
        "Four driving images in this exact order: front, left, right, back.",
        "Only the front image has the trajectory overlay; left, right, and back are unannotated context images.",
    )


def resolve_stage_camera_order(cfg: VLMSelectorConfig, stage: str) -> Tuple[str, ...]:
    if stage == "intervention":
        camera_mode = cfg.intervention_camera_mode or cfg.camera_mode
    elif stage == "scoring":
        camera_mode = cfg.scoring_camera_mode or cfg.camera_mode
    else:
        camera_mode = cfg.camera_mode
    return resolve_vlm_camera_order(camera_mode)


def build_scoring_prompt(
    candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
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
    return f"""
You are an autonomous-driving trajectory scorer performing the final action-selection stage.

You are given:
1. {camera_line_1}
2. {camera_line_2}
3. Structured candidate trajectory metadata.
4. A high-level driving directive that must be taken seriously when scoring the candidates.
5. Optionally, advisory short-horizon corrective context from a separate reflective intervention stage.

Your job:
- Score EVERY candidate trajectory with a scalar Q-like value.
- Identify the single best candidate based on the highest generated Q-value.
- There are exactly {len(candidate_rows)} candidates in this frame.
- You MUST return one score for every candidate index: {candidate_index_list}.
- Do not omit any candidate index.
- Do not shorten the score dictionary to a partial example.
- Keep score scale internally consistent within this frame.
- Keep the response compact.
- Include a short reasoning string that specifically argues why the selected candidate is the best overall choice in this frame.
- The reasoning must justify the selected candidate relative to the alternatives using safety, route-following, lane alignment, obstacle avoidance, and motion smoothness.
- The reasoning should describe why the selected candidate is best, not just restate its index.
{default_candidate_guidance}
{corrective_action_guidance}
- Judge based on the following constraints:
  - higher scores are better,
  - staying in the drivable region,
  - maintaining lane alignment and smoothness,
  - avoiding nearby vehicles/obstacles,
  - avoiding sidewalk/off-road/wrong-way behavior,
  - smooth and realistic motion,
  - maintaining safe forward progress when possible,
  - following the route instruction directly and literally when safe,
    - if the instruction is straight, prefer trajectories that continue in the current lane/direction rather than drifting left or right,
    - if the instruction is left or right, prefer trajectories that clearly begin that turn direction when it is safe and drivable,
    - do not deviate from the instructed direction unless obstacles, lane geometry, or safety clearly require it,
  - obeying lane rules,
    - avoid unnecessary lane changes,
    - respect lane boundaries and markings such as solid vs dotted lines when they are visible,
    - only change lanes or cross lane boundaries when required for safety, obstacle avoidance, passing a blocked vehicle in the current lane, or because the road geometry clearly demands it,
  - penalize trajectories unsafe trajectories.
    - Use only what is visible in the image and candidate metadata.
    - If all candidates are imperfect, prefer the least risky one.

High-level driving directive (right,left,or straight):
{route_instruction}

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "best_candidate_index": <int>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<short explanation for why this selected candidate is best>",
  "candidate_scores": {{
{score_schema_lines}
  }},
}}
""".strip()


def build_intervention_prompt(
    baseline_candidate_row: Dict[str, object],
    route_instruction: str,
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
    return f"""
You are an autonomous-driving self-reflective safety monitor deciding whether the current planned action should be revised before execution.

You are given:
1. {camera_line_1}
2. {camera_line_2}
3. Structured metadata for that same baseline RAP-selected trajectory.
4. A route-level driving instruction.

Your job:
- Treat the shown baseline trajectory as a proposed action that may or may not need revision.
- First reason counterfactually about likely short-horizon consequences if this exact baseline trajectory is executed.
- Then decide whether the action should be revised before execution.
- Set "should_intervene" to true when the shown baseline trajectory appears risky, ambiguous, instruction-inconsistent, poorly centered, too close to obstacles or lane boundaries, or likely to benefit from short-horizon correction.
- Use "should_intervene" = false only when the shown baseline trajectory looks clearly safe, clearly lane-aligned, instruction-consistent, and comfortably clear of nearby conflicts.
- Intervene when there is a meaningful risk of obstacle collision, unsafe lane departure, wrong-way behavior, sidewalk/off-road encroachment, route mismatch, low-margin clearance, or other visible issue with the shown baseline trajectory.
- If you are uncertain whether the trajectory safely stays in-lane, clears nearby obstacles, or follows the intended route, prefer "should_intervene" = true.
- Because the planner replans frequently, it is acceptable to intervene for borderline or low-margin cases, not only catastrophic ones.
- In multiview mode, use extra camera views to judge surrounding safety context, not to reinterpret the path geometry shown on the front image.
{multiview_guidance}
- If intervention is needed, also provide a single high-level corrective action for the next short-horizon maneuver.
- The corrective action must be exactly one of: "left", "right", or "straight".
- Choose the corrective action that best describes the immediate correction a downstream scorer should consider.
- Keep the response compact.
- The reasoning should explicitly describe:
  - what is likely to happen if the baseline action is executed,
  - why that outcome is acceptable or not,
  - and why the corrective action is the best short-horizon revision when intervention is needed.

Route instruction (right,left,or straight):
{route_instruction}

Baseline RAP-selected trajectory:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "should_intervene": <true or false>,
  "corrective_action": "<left or right or straight>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<short explanation for why intervention is or is not needed>"
}}
""".strip()


def try_parse_json(text: str) -> Optional[Dict[str, object]]:
    raw = text.strip()
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


def command_to_route_instruction(command: object) -> str:
    mapping = {
        0: "right",
        1: "left",
        2: "straight",
    }
    try:
        return mapping.get(int(command), "Drive safely and choose the most reasonable trajectory.")
    except Exception:
        return "Drive safely and choose the most reasonable trajectory."


def normalize_corrective_action(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    direct = {
        "left": "left",
        "right": "right",
        "straight": "straight",
        "forward": "straight",
        "go straight": "straight",
        "keep straight": "straight",
        "continue straight": "straight",
        "turn left": "left",
        "go left": "left",
        "veer left": "left",
        "turn right": "right",
        "go right": "right",
        "veer right": "right",
    }
    if text in direct:
        return direct[text]
    if "straight" in text or "forward" in text:
        return "straight"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return None


class Qwen3TrajectorySelector:
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        requested_device = str(device).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
        )
        self.model.to(requested_device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.device = requested_device
        self._torch = torch

    def _run_inference(self, image_paths: Sequence[Path], prompt: str) -> str:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = []
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
            trimmed_ids.append(out_ids[len(in_ids):])
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def infer_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        started = time.time()
        prev_max_new_tokens = self.max_new_tokens
        try:
            if max_new_tokens is not None:
                self.max_new_tokens = int(max_new_tokens)
            raw_output = self._run_inference(image_paths, prompt)
        finally:
            if max_new_tokens is not None:
                self.max_new_tokens = prev_max_new_tokens
        parsed = try_parse_json(raw_output)
        return {
            "raw_output": raw_output,
            "parsed_output": parsed,
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
        }


class SubprocessQwen3TrajectorySelector:
    def __init__(
        self,
        python_bin: str,
        worker_script: Path,
        model_id: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> None:
        if not python_bin:
            raise ValueError("VLM python_bin is not set")
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.enable_thinking = bool(enable_thinking)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stderr_file = None
        self._ready = False

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        worker_log_path = self.worker_script.with_name("vlm_worker.stderr.log")
        self._stderr_file = worker_log_path.open("a", encoding="utf-8")
        self._proc = subprocess.Popen(
            [
                self.python_bin,
                "-B",
                str(self.worker_script),
                "--model-id",
                self.model_id,
                "--device",
                self.device,
                "--max-new-tokens",
                str(self.max_new_tokens),
                "--temperature",
                str(self.temperature),
                "--top-p",
                str(self.top_p),
                "--top-k",
                str(self.top_k),
                "--enable-thinking",
                "true" if self.enable_thinking else "false",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        self._ready = False
        return self._proc

    def _readline_with_timeout(self, proc: subprocess.Popen[str], timeout_sec: Optional[float]) -> str:
        if proc.stdout is None:
            raise RuntimeError("VLM worker stdout is unavailable")
        deadline = None if timeout_sec is None else time.monotonic() + max(float(timeout_sec), 0.0)
        stdout_fd = proc.stdout.fileno()
        while True:
            if deadline is None:
                ready, _, _ = select.select([stdout_fd], [], [])
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"VLM worker subprocess timeout after {float(timeout_sec):.3f}s")
                ready, _, _ = select.select([stdout_fd], [], [], remaining)
                if not ready:
                    continue
            if ready:
                return proc.stdout.readline()

    def preload(self, timeout_sec: Optional[float] = None) -> None:
        if self._ready and self._proc is not None and self._proc.poll() is None:
            return
        proc = self._ensure_proc()
        line = self._readline_with_timeout(proc, timeout_sec)
        if not line:
            self.close()
            raise RuntimeError("VLM worker exited before signaling readiness")
        response = json.loads(line)
        if response.get("status") == "ready":
            self._ready = True
            return
        if response.get("error"):
            self.close()
            raise RuntimeError(str(response["error"]))
        self.close()
        raise RuntimeError(f"Unexpected VLM worker preload response: {response!r}")

    def infer_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, object]:
        self.preload(timeout_sec=timeout_sec)
        proc = self._ensure_proc()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("VLM worker stdio is unavailable")

        started = time.time()
        payload = {"image_paths": [str(image_path) for image_path in image_paths], "prompt": prompt}
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        payload["temperature"] = self.temperature
        payload["top_p"] = self.top_p
        payload["top_k"] = self.top_k
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = ""
        try:
            line = self._readline_with_timeout(proc, timeout_sec)
        except TimeoutError:
            self.close()
            raise
        if not line:
            stderr = ""
            raise RuntimeError(f"VLM worker exited unexpectedly: {stderr.strip()}")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        raw_output = str(response.get("raw_output", ""))
        return {
            "raw_output": raw_output,
            "parsed_output": try_parse_json(raw_output),
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
        }

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None


def build_candidate_rows(
    candidates: Sequence[Dict[str, object]],
    current_ego_speed_mps: Optional[float] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    current_rank = 0
    for rank, candidate in enumerate(candidates):
        source = str(candidate.get("source", "current_rap"))
        style = get_candidate_visual_style(source, current_rank)
        if source != "carry_prev":
            current_rank += 1
        plan = np.asarray(candidate["local_plan"], dtype=np.float32)
        row = {
            "candidate_index": rank,
            "candidate_rank": rank,
            "proposal_index": None if candidate.get("proposal_index") is None else int(candidate["proposal_index"]),
            "source": source,
            "color_name": style.color_name,
            "color_bgr": list(style.color_bgr),
            "local_plan": plan.tolist(),
            "execution_plan": np.asarray(candidate.get("execution_plan", candidate["local_plan"]), dtype=np.float32).tolist(),
            "proposal_score": float(candidate.get("proposal_score", 0.0)),
            "rap_score": float(candidate.get("proposal_score", 0.0)),
            "origin_selected_score_raw": (
                None if candidate.get("origin_selected_score_raw") is None else float(candidate["origin_selected_score_raw"])
            ),
            "q_score": None if candidate.get("q_score") is None else float(candidate["q_score"]),
        }
        row["summary"] = summarize_candidate(row["local_plan"], current_ego_speed_mps=current_ego_speed_mps)
        rows.append(row)
    return rows


def _coerce_candidate_scores(raw_scores: object, num_candidates: int) -> Tuple[Optional[List[float]], Optional[str]]:
    if not isinstance(raw_scores, dict):
        return None, "candidate_scores_missing"

    scores: List[Optional[float]] = [None] * num_candidates
    for raw_key, raw_value in raw_scores.items():
        try:
            idx = int(raw_key)
        except Exception:
            return None, f"candidate_score_bad_key:{raw_key}"
        if idx < 0 or idx >= num_candidates:
            return None, f"candidate_score_key_out_of_range:{idx}"
        try:
            scores[idx] = float(raw_value)
        except Exception:
            return None, f"candidate_score_bad_value:{raw_key}"

    if any(score is None for score in scores):
        missing = [idx for idx, score in enumerate(scores) if score is None]
        return None, f"candidate_scores_incomplete:{missing}"
    return [float(score) for score in scores], None


def _coerce_intervention_decision(
    parsed: object,
) -> Tuple[Optional[bool], Optional[str], Optional[float], Optional[str], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, None, None, None, "intervention_output_invalid"

    raw_flag = parsed.get("should_intervene")
    if not isinstance(raw_flag, bool):
        return None, None, None, None, "intervention_flag_missing"

    corrective_action = normalize_corrective_action(parsed.get("corrective_action"))
    if raw_flag and corrective_action is None:
        return None, None, None, None, "intervention_corrective_action_missing"

    raw_confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except Exception:
        return None, None, None, None, "intervention_confidence_invalid"

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return bool(raw_flag), corrective_action, confidence, reasoning, None


def _select_from_vlm_scores(
    candidate_rows: Sequence[Dict[str, object]],
    vlm_scores: Sequence[float],
) -> Dict[str, object]:
    vlm_scores = [float(score) for score in vlm_scores]
    if len(vlm_scores) != len(candidate_rows):
        raise ValueError("VLM score count does not match candidate rows")

    selected_idx = int(max(range(len(vlm_scores)), key=lambda idx: vlm_scores[idx]))
    selected_row = dict(candidate_rows[selected_idx])
    row_source = str(selected_row.get("source", "current_rap"))
    if row_source == "carry_prev":
        selected_source = "vlm_selected_carry_prev"
        decision = "vlm_selected_reuse_prev"
    elif row_source.startswith("default_fallback_"):
        selected_source = "vlm_selected_default_fallback"
        decision = "vlm_selected_default_fallback"
    else:
        selected_source = "vlm_selected_current"
        decision = "vlm_selected_current"

    sorted_scores = sorted(((float(score), idx) for idx, score in enumerate(vlm_scores)), reverse=True)
    score_gap_top2 = None
    if len(sorted_scores) >= 2:
        score_gap_top2 = float(sorted_scores[0][0] - sorted_scores[1][0])

    return {
        "selected_candidate_index": selected_idx,
        "selected_candidate_row": selected_row,
        "selected_source": selected_source,
        "adaptive_replan_decision": decision,
        "vlm_q_score_gap_top2": score_gap_top2,
    }


def _selected_path_reasoning(
    selected_row: Dict[str, object],
    selected_candidate_index: int,
    selected_source: str,
    vlm_scores: Optional[Sequence[float]],
    parsed_reasoning: Optional[object],
) -> str:
    if isinstance(parsed_reasoning, str) and parsed_reasoning.strip():
        return parsed_reasoning.strip()

    summary = selected_row.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "selected path"

    score_text = ""
    if vlm_scores is not None and 0 <= int(selected_candidate_index) < len(vlm_scores):
        score_text = f" with highest q-score {float(vlm_scores[int(selected_candidate_index)]):.3f}"

    return (
        f"Selected {selected_source} candidate {int(selected_candidate_index)}{score_text}: "
        f"{summary}."
    )


class VLMPlanSelector:
    def __init__(self, cfg: VLMSelectorConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.debug_dir = output_dir / cfg.debug_dir_name
        self.timeline_path = self.debug_dir / "latency_timeline.jsonl"
        self.summary_path = self.debug_dir / "latency_summary.json"
        self._selector: Optional[object] = None
        self._disabled_reason: Optional[str] = None
        self._timeline_records: List[Dict[str, object]] = []

        if cfg.save_debug_artifacts:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.timeline_path.unlink(missing_ok=True)
            self.summary_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_candidates.jpg"):
                stale_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_candidates_*.jpg"):
                stale_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_gate_*.jpg"):
                stale_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_result.json"):
                stale_path.unlink(missing_ok=True)

    def _record_timeline(self, record: Dict[str, object]) -> None:
        self._timeline_records.append(record)
        if self.cfg.save_debug_artifacts:
            with self.timeline_path.open("a", encoding="utf-8") as wf:
                wf.write(json.dumps(record) + "\n")

    def _ensure_selector(self) -> Optional[object]:
        if self._selector is not None:
            return self._selector
        if self._disabled_reason is not None:
            return None
        if self.cfg.backend == "local_transformers":
            selector_factory = lambda: Qwen3TrajectorySelector(
                model_id=self.cfg.model_id,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
            )
        elif self.cfg.backend == "local_transformers_subprocess":
            selector_factory = lambda: SubprocessQwen3TrajectorySelector(
                python_bin=self.cfg.python_bin,
                worker_script=Path(__file__).with_name("vlm_worker.py"),
                model_id=self.cfg.model_id,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                enable_thinking=self.cfg.enable_thinking,
            )
        else:
            self._disabled_reason = f"unsupported_backend:{self.cfg.backend}"
            return None
        try:
            self._selector = selector_factory()
            LOG.info(
                "Initialized VLM selector backend=%s model=%s device=%s",
                self.cfg.backend,
                self.cfg.model_id,
                getattr(self._selector, "device", self.cfg.device),
            )
            return self._selector
        except Exception as exc:
            self._disabled_reason = str(exc)
            LOG.exception("Failed to initialize VLM selector, disabling VLM fallback path")
            return None

    def preload(self) -> None:
        if not self.cfg.enabled or self._disabled_reason is not None:
            return
        selector = self._ensure_selector()
        if selector is None or not self.cfg.preload_on_init:
            return
        preload_fn = getattr(selector, "preload", None)
        if not callable(preload_fn):
            return
        LOG.info(
            "Preloading VLM selector backend=%s model=%s timeout=%.3fs",
            self.cfg.backend,
            self.cfg.model_id,
            float(self.cfg.timeout_sec),
        )
        preload_fn(timeout_sec=self.cfg.timeout_sec)
        LOG.info("VLM selector preload complete model=%s", self.cfg.model_id)

    def maybe_select(
        self,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
    ) -> Dict[str, object]:
        route_instruction = command_to_route_instruction(info.get("command"))
        scoring_camera_order = resolve_stage_camera_order(self.cfg, "scoring")
        intervention_camera_order = resolve_stage_camera_order(self.cfg, "intervention")
        timestamp = float(info.get("timestamp", 0.0))
        dt_sec = 0.25
        current_ego_speed_mps = None
        current_ego_accel_mps2 = None
        try:
            current_ego_speed_mps = float(info["ego_velo"])
        except Exception:
            current_ego_speed_mps = None
        try:
            current_ego_accel_mps2 = float(info["accelerate"])
        except Exception:
            current_ego_accel_mps2 = None
        carry_previous_valid = any(row.get("source") == "carry_prev" for row in candidate_rows)
        candidate_rows = build_candidate_rows(candidate_rows, current_ego_speed_mps=current_ego_speed_mps)
        scoring_invoked = False
        intervention_invoked = False
        intervention_should_intervene = None
        intervention_corrective_action = None
        intervention_confidence = None
        intervention_reasoning = None
        intervention_elapsed_sec = 0.0
        intervention_error = None

        def _fallback_result(error: Optional[str] = None) -> Dict[str, object]:
            selected_row = dict(candidate_rows[default_selected_index])
            adaptive_replan_decision = "vlm_failed_fallback_rap"
            timeline_record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "candidate_count": len(candidate_rows),
                "carry_previous_valid": carry_previous_valid,
                "selected_source": default_selected_source,
                "selected_candidate_index": default_selected_index,
                "selected_candidate_source": selected_row.get("source"),
                "selected_proposal_index": selected_row.get("proposal_index"),
                "intervention_invoked": False,
                "intervention_should_intervene": None,
                "intervention_corrective_action": None,
                "intervention_confidence": None,
                "intervention_elapsed_sec": 0.0,
                "vlm_elapsed_sec": 0.0,
                "scoring_invoked": False,
                "latency_equivalent_steps": 0.0,
                "latency_equivalent_steps_ceil": 0,
                "adaptive_replan_decision": adaptive_replan_decision,
                "error": error,
                "q_invoked_vlm": False,
                "vlm_q_valid": False,
                "vlm_failed": True,
            }
            self._record_timeline(timeline_record)
            result = {
                "selected_candidate_row": selected_row,
                "selected_source": default_selected_source,
                "adaptive_replan_decision": adaptive_replan_decision,
                "carry_previous_valid": carry_previous_valid,
                "latency_timeline_record": timeline_record,
                "vlm_q_valid": False,
                "vlm_failed": True,
                "scoring_invoked": False,
                "intervention_invoked": False,
                "intervention_should_intervene": None,
                "intervention_corrective_action": None,
                "intervention_confidence": None,
                "intervention_reasoning": None,
                "intervention_elapsed_sec": 0.0,
            }
            if error is not None:
                result["error"] = error
            return result

        if not self.cfg.enabled:
            return _fallback_result("vlm_disabled")

        selector = self._ensure_selector()
        if selector is None:
            return _fallback_result(self._disabled_reason or "selector_unavailable")

        frame_stem = f"frame_{frame_index:04d}"
        result_path = self.debug_dir / f"{frame_stem}_result.json"
        temp_paths: List[Path] = []

        def _write_overlay_bundle(
            suffix: str,
            overlays: Dict[str, np.ndarray],
            camera_order: Sequence[str],
        ) -> List[Path]:
            image_paths: List[Path] = []
            for cam_name in camera_order:
                overlay = overlays.get(cam_name)
                if overlay is None:
                    continue
                if self.cfg.save_debug_artifacts:
                    image_path = self.debug_dir / f"{frame_stem}_{suffix}_{cam_name}.jpg"
                else:
                    image_path = self.output_dir / f"{frame_stem}_{suffix}_{cam_name}_tmp.jpg"
                    temp_paths.append(image_path)
                cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                image_paths.append(image_path)
            return image_paths

        score_overlays = render_candidate_overlays(
            camera_images,
            info,
            candidate_rows,
            camera_order=scoring_camera_order,
        )
        score_image_paths = _write_overlay_bundle("candidates", score_overlays, scoring_camera_order)

        intervention_result = None
        if self.cfg.intervention_enabled:
            intervention_invoked = True
            gate_overlays = render_candidate_overlays(
                camera_images,
                info,
                [candidate_rows[default_selected_index]],
                camera_order=intervention_camera_order,
            )
            gate_image_paths = _write_overlay_bundle("gate", gate_overlays, intervention_camera_order)
            intervention_prompt = build_intervention_prompt(
                candidate_rows[default_selected_index],
                route_instruction,
                camera_order=intervention_camera_order,
            )
            try:
                intervention_result = selector.infer_prompt(
                    image_paths=gate_image_paths,
                    prompt=intervention_prompt,
                    max_new_tokens=self.cfg.intervention_max_new_tokens,
                    timeout_sec=self.cfg.intervention_timeout_sec,
                )
            except Exception as exc:
                LOG.exception("VLM intervention gate failed, falling back to RAP argmax")
                intervention_result = {
                    "raw_output": "",
                    "parsed_output": None,
                    "elapsed_sec": 0.0,
                    "prompt": intervention_prompt,
                    "error": str(exc),
                }

            intervention_elapsed_sec = float(intervention_result.get("elapsed_sec", 0.0))
            intervention_should_intervene, intervention_corrective_action, intervention_confidence, intervention_reasoning, intervention_parse_error = (
                _coerce_intervention_decision(intervention_result.get("parsed_output"))
            )
            if intervention_elapsed_sec > self.cfg.intervention_timeout_sec:
                intervention_error = f"intervention_timeout_fallback_rap:{intervention_elapsed_sec:.3f}"
            elif intervention_parse_error is not None:
                intervention_error = intervention_parse_error
            elif intervention_result.get("error"):
                intervention_error = str(intervention_result.get("error"))

            if intervention_error is not None:
                selected_row = dict(candidate_rows[default_selected_index])
                selected_source = "gate_failed_fallback_rap"
                adaptive_replan_decision = "gate_failed_fallback_rap"
                selected_path_reasoning = (
                    intervention_reasoning.strip()
                    if isinstance(intervention_reasoning, str) and intervention_reasoning.strip()
                    else "Intervention gate failed; using baseline RAP selection."
                )
                timeline_record = {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "route_instruction": route_instruction,
                    "candidate_count": len(candidate_rows),
                    "carry_previous_valid": carry_previous_valid,
                    "carry_previous_remaining_path_m": next(
                        (path_length(np.asarray(row["local_plan"], dtype=np.float32)) for row in candidate_rows if row.get("source") == "carry_prev"),
                        0.0,
                    ),
                    "selected_source": selected_source,
                    "selected_candidate_index": default_selected_index,
                    "selected_candidate_source": selected_row.get("source"),
                    "selected_proposal_index": selected_row.get("proposal_index"),
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "vlm_elapsed_sec": 0.0,
                    "scoring_invoked": False,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "latency_equivalent_steps": 0.0,
                    "latency_equivalent_steps_ceil": 0,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "error": intervention_error,
                    "q_invoked_vlm": False,
                    "vlm_failed": True,
                }
                self._record_timeline(timeline_record)
                debug_payload = {
                    "frame_index": frame_index,
                    "route_instruction": route_instruction,
                    "default_selected_index": int(default_selected_index),
                    "default_selected_source": default_selected_source,
                    "candidate_rows": candidate_rows,
                    "intervention_result": intervention_result,
                    "scoring_result": None,
                    "scoring_invoked": False,
                    "selected_index": int(default_selected_index),
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_error": intervention_error,
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "error": intervention_error,
                    "vlm_failed": True,
                }
                if self.cfg.save_debug_artifacts:
                    result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
                if not self.cfg.save_debug_artifacts:
                    for temp_path in temp_paths:
                        temp_path.unlink(missing_ok=True)
                return {
                    "selected_candidate_row": selected_row,
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_reasoning": None,
                    "vlm_elapsed_sec": 0.0,
                    "vlm_error": intervention_error,
                    "vlm_candidate_count": len(candidate_rows),
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "vlm_failed": True,
                    "scoring_invoked": False,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                }

        scoring_route_instruction = route_instruction

        try:
            LOG.info(
                "Running VLM selection for frame=%d candidates=%d route='%s' corrective_action='%s'",
                frame_index,
                len(candidate_rows),
                scoring_route_instruction,
                intervention_corrective_action,
            )
            scoring_invoked = True
            scoring_prompt = build_scoring_prompt(
                candidate_rows,
                scoring_route_instruction,
                intervention_corrective_action=(
                    intervention_corrective_action
                    if self.cfg.intervention_enabled and intervention_should_intervene is True
                    else None
                ),
                current_ego_speed_mps=current_ego_speed_mps,
                current_ego_accel_mps2=current_ego_accel_mps2,
                camera_order=scoring_camera_order,
            )
            result = selector.infer_prompt(
                image_paths=score_image_paths,
                prompt=scoring_prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                timeout_sec=self.cfg.timeout_sec,
            )
        except Exception as exc:
            self._disabled_reason = f"selector_runtime_error:{exc}"
            close_fn = getattr(selector, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._selector = None
            LOG.exception("VLM selector inference failed, falling back to RAP argmax")
            result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": scoring_prompt if 'scoring_prompt' in locals() else "",
                "error": str(exc),
            }

        parsed = result.get("parsed_output")
        elapsed_sec = float(result.get("elapsed_sec", 0.0))
        selected_candidate_index = int(default_selected_index)
        selected_row = dict(candidate_rows[default_selected_index])
        selected_source = default_selected_source
        error = None
        vlm_confidence = None
        vlm_reasoning = None
        vlm_candidate_index = None
        vlm_q_valid = False
        vlm_q_candidate_scores = None
        vlm_q_best_candidate_index = None
        vlm_q_score_gap_to_carry = None
        vlm_q_score_gap_top2 = None
        vlm_q_best_current_score = None
        vlm_q_carry_score = None
        selected_path_reasoning = None
        vlm_timed_out = False

        if isinstance(parsed, dict):
            coerced_scores, score_error = _coerce_candidate_scores(parsed.get("candidate_scores"), len(candidate_rows))
            candidate_idx = parsed.get("best_candidate_index")
            if coerced_scores is not None:
                vlm_q_valid = True
                vlm_q_candidate_scores = [float(score) for score in coerced_scores]
                vlm_q_best_candidate_index = int(max(range(len(vlm_q_candidate_scores)), key=lambda idx: vlm_q_candidate_scores[idx]))
                if isinstance(candidate_idx, int) and 0 <= candidate_idx < len(candidate_rows):
                    vlm_candidate_index = int(candidate_idx)
                vlm_confidence = float(parsed.get("confidence", 0.0))
                vlm_reasoning = parsed.get("reasoning")
                if elapsed_sec > self.cfg.timeout_sec:
                    vlm_timed_out = True
                    error = f"selector_timeout_fallback_rap:{elapsed_sec:.3f}"
                    selected_path_reasoning = (
                        f"VLM result arrived after timeout ({elapsed_sec:.3f}s > {self.cfg.timeout_sec:.3f}s); "
                        "using RAP argmax fallback for real-time control."
                    )
                else:
                    selection = _select_from_vlm_scores(
                        candidate_rows=candidate_rows,
                        vlm_scores=coerced_scores,
                    )
                    selected_candidate_index = int(selection["selected_candidate_index"])
                    selected_row = dict(selection["selected_candidate_row"])
                    selected_source = str(selection["selected_source"])
                    vlm_q_score_gap_top2 = selection["vlm_q_score_gap_top2"]
                    selected_path_reasoning = _selected_path_reasoning(
                        selected_row=selected_row,
                        selected_candidate_index=selected_candidate_index,
                        selected_source=selected_source,
                        vlm_scores=vlm_q_candidate_scores,
                        parsed_reasoning=vlm_reasoning,
                    )
            else:
                error = score_error or "invalid_candidate_scores"
        else:
            error = (
                f"selector_timeout_budget_exceeded:{elapsed_sec:.3f}"
                if elapsed_sec > self.cfg.timeout_sec
                else str(result.get("error") or "invalid_selector_output")
            )

        if selected_path_reasoning is None:
            selected_path_reasoning = _selected_path_reasoning(
                selected_row=selected_row,
                selected_candidate_index=selected_candidate_index,
                selected_source=selected_source,
                vlm_scores=vlm_q_candidate_scores,
                parsed_reasoning=vlm_reasoning,
            )

        latency_equivalent_steps = elapsed_sec / max(dt_sec, 1e-6)
        latency_equivalent_steps_ceil = int(math.ceil(latency_equivalent_steps))
        adaptive_replan_decision = (
            "vlm_timeout_fallback_rap"
            if vlm_timed_out
            else "vlm_failed_fallback_rap"
            if not vlm_q_valid
            else (
                "vlm_selected_reuse_prev"
                if selected_row.get("source") == "carry_prev"
                else "vlm_selected_current"
            )
        )

        timeline_record = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "route_instruction": route_instruction,
            "scoring_route_instruction": scoring_route_instruction,
            "candidate_count": len(candidate_rows),
            "carry_previous_valid": carry_previous_valid,
            "carry_previous_remaining_path_m": next(
                (path_length(np.asarray(row["local_plan"], dtype=np.float32)) for row in candidate_rows if row.get("source") == "carry_prev"),
                0.0,
            ),
            "selected_source": selected_source,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_source": selected_row.get("source"),
            "selected_proposal_index": selected_row.get("proposal_index"),
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_corrective_action": intervention_corrective_action,
            "intervention_confidence": intervention_confidence,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "vlm_elapsed_sec": elapsed_sec,
            "scoring_invoked": scoring_invoked,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "latency_equivalent_steps": latency_equivalent_steps,
            "latency_equivalent_steps_ceil": latency_equivalent_steps_ceil,
            "adaptive_replan_decision": adaptive_replan_decision,
            "error": error,
            "q_invoked_vlm": True,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
        }
        self._record_timeline(timeline_record)

        debug_payload = {
            "frame_index": frame_index,
            "route_instruction": route_instruction,
            "scoring_route_instruction": scoring_route_instruction,
            "default_selected_index": int(default_selected_index),
            "default_selected_source": default_selected_source,
            "candidate_rows": candidate_rows,
            "intervention_result": intervention_result,
            "scoring_result": result,
            "scoring_invoked": scoring_invoked,
            "selected_index": int(selected_candidate_index),
            "selected_source": selected_source,
            "selected_path_reasoning": selected_path_reasoning,
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_corrective_action": intervention_corrective_action,
            "intervention_confidence": intervention_confidence,
            "intervention_reasoning": intervention_reasoning,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "intervention_error": intervention_error,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "vlm_q_best_current_score": vlm_q_best_current_score,
            "vlm_q_carry_score": vlm_q_carry_score,
            "adaptive_replan_decision": adaptive_replan_decision,
            "carry_previous_valid": carry_previous_valid,
            "latency_timeline_record": timeline_record,
            "error": error,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
        }
        if self.cfg.save_debug_artifacts:
            result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        if not self.cfg.save_debug_artifacts:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        LOG.info(
            "VLM selection frame=%d source=%s proposal=%d candidate=%s elapsed=%.3f error=%s",
            frame_index,
            selected_source,
            -1 if selected_row.get("proposal_index") is None else int(selected_row["proposal_index"]),
            "none" if vlm_q_best_candidate_index is None else str(vlm_q_best_candidate_index),
            elapsed_sec,
            error,
        )

        return {
            "selected_candidate_row": selected_row,
            "selected_source": selected_source,
            "selected_path_reasoning": selected_path_reasoning,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_reasoning": vlm_reasoning,
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_error": error,
            "vlm_candidate_count": len(candidate_rows),
            "scoring_invoked": scoring_invoked,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "vlm_q_best_current_score": vlm_q_best_current_score,
            "vlm_q_carry_score": vlm_q_carry_score,
            "adaptive_replan_decision": adaptive_replan_decision,
            "carry_previous_valid": carry_previous_valid,
            "latency_timeline_record": timeline_record,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_confidence": intervention_confidence,
            "intervention_reasoning": intervention_reasoning,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "intervention_error": intervention_error,
        }

    def finalize(self) -> None:
        if hasattr(self._selector, "close"):
            try:
                self._selector.close()
            except Exception:
                LOG.exception("Failed to close VLM selector")
        if not self._timeline_records or not self.cfg.save_debug_artifacts:
            return

        elapsed = [float(record["vlm_elapsed_sec"]) for record in self._timeline_records]
        intervention_elapsed = [float(record.get("intervention_elapsed_sec", 0.0)) for record in self._timeline_records]
        total_elapsed = [float(v) + float(g) for v, g in zip(elapsed, intervention_elapsed)]
        carry_reuse_rate = sum(
            record["adaptive_replan_decision"] in {"reuse_prev", "vlm_q_reuse_prev", "q_reuse_prev"}
            for record in self._timeline_records
        ) / len(self._timeline_records)
        switch_rate = sum(
            record["adaptive_replan_decision"] in {"switch_to_current", "vlm_q_switch_to_current", "q_switch_to_current"}
            for record in self._timeline_records
        ) / len(self._timeline_records)
        fallback_rate = sum(
            record.get("adaptive_replan_decision") in {"vlm_failed_fallback_rap", "vlm_timeout_fallback_rap", "gate_failed_fallback_rap"}
            for record in self._timeline_records
        ) / len(self._timeline_records)
        intervention_trigger_rate = sum(
            1.0
            for record in self._timeline_records
            if record.get("intervention_invoked") and record.get("intervention_should_intervene") is True
        ) / len(self._timeline_records)
        gate_skip_rate = sum(
            1.0
            for record in self._timeline_records
            if record.get("intervention_invoked") and record.get("intervention_should_intervene") is False
        ) / len(self._timeline_records)
        scoring_invoked_rate = sum(
            1.0 if record.get("scoring_invoked") else 0.0
            for record in self._timeline_records
        ) / len(self._timeline_records)
        summary = {
            "num_records": len(self._timeline_records),
            "latency_mean_sec": float(np.mean(elapsed)),
            "latency_p50_sec": float(np.percentile(elapsed, 50)),
            "latency_p95_sec": float(np.percentile(elapsed, 95)),
            "latency_max_sec": float(np.max(elapsed)),
            "intervention_latency_mean_sec": float(np.mean(intervention_elapsed)),
            "intervention_latency_p50_sec": float(np.percentile(intervention_elapsed, 50)),
            "intervention_latency_p95_sec": float(np.percentile(intervention_elapsed, 95)),
            "intervention_latency_max_sec": float(np.max(intervention_elapsed)),
            "total_vlm_latency_mean_sec": float(np.mean(total_elapsed)),
            "total_vlm_latency_p50_sec": float(np.percentile(total_elapsed, 50)),
            "total_vlm_latency_p95_sec": float(np.percentile(total_elapsed, 95)),
            "total_vlm_latency_max_sec": float(np.max(total_elapsed)),
            "latency_equivalent_steps_mean": float(
                np.mean([record["latency_equivalent_steps"] for record in self._timeline_records])
            ),
            "carry_reuse_rate": float(carry_reuse_rate),
            "switch_to_current_rate": float(switch_rate),
            "fallback_rate": float(fallback_rate),
            "intervention_trigger_rate": float(intervention_trigger_rate),
            "gate_skip_rate": float(gate_skip_rate),
            "scoring_invoked_rate": float(scoring_invoked_rate),
            "vlm_q_valid_rate": float(
                np.mean([1.0 if record.get("vlm_q_valid") else 0.0 for record in self._timeline_records])
            ),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
