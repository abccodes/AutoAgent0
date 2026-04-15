from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SCR

LOG = logging.getLogger(__name__)

TOPK_BASE_COLORS = [
    (0, 120, 255),
    (0, 180, 255),
    (0, 220, 200),
    (60, 220, 120),
    (120, 220, 60),
    (200, 220, 40),
    (255, 200, 0),
    (255, 160, 0),
    (255, 90, 0),
    (255, 0, 0),
]
PAST_TRAJECTORY_COLOR = (0, 0, 0)
TOPK_COLOR_NAMES = [
    "blue",
    "light_blue",
    "cyan",
    "green",
    "lime",
    "yellow_green",
    "amber",
    "orange",
    "orange_red",
    "red",
]
PLAN_VIS_FORWARD_OFFSET_M = 4.5
PLAN_RESAMPLE_SPACING_M = 0.08


@dataclass
class VLMSelectorConfig:
    enabled: bool = False
    intervention_mode: str = "uncertainty_only"
    backend: str = "qwen3_vl"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 300
    candidate_limit: int = 5
    timeout_sec: float = 10.0
    save_debug_artifacts: bool = True
    debug_dir_name: str = "vlm_debug"
    carry_previous_enabled: bool = True
    carry_previous_min_path_m: float = 0.5
    carry_previous_min_points: int = 2
    adaptive_replan_mode: str = "log_only"
    latency_tracking_mode: str = "full_timeline"
    q_enabled: bool = True
    q_switch_margin: float = 0.05
    q_uncertainty_margin: float = 0.03
    q_quality_floor: float = 0.18
    q_candidate_disagreement_threshold: float = 2.5
    q_weight_rap_score: float = 0.55
    q_weight_progress: float = 0.30
    q_weight_offcenter: float = 0.10
    q_weight_curvature: float = 0.08
    q_weight_shortplan: float = 0.18
    q_carry_score_decay: float = 0.0
    intervention_plan_path_floor_m: float = 1.0


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
    front_image: np.ndarray,
    info: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    front_cam_name: str = "CAM_FRONT",
) -> np.ndarray:
    overlay = np.asarray(front_image, dtype=np.uint8).copy()
    ego_pose = _rt2pose(info["ego_rot"], info["ego_pos"])
    cam_params = info["cam_params"]
    intrinsic = cam_params[front_cam_name]["intrinsic"]
    front_c2w = _get_camera_c2w(cam_params, ego_pose, front_cam_name)
    front_v2c = cam_params[front_cam_name]["v2c"]

    for row in candidate_rows:
        plan_local = np.asarray(row["local_plan"], dtype=np.float32)
        plan_world = _local_plan_to_front_world(plan_local, front_c2w, front_v2c)
        plan_world_dense = _resample_polyline(plan_world, spacing=PLAN_RESAMPLE_SPACING_M)
        color = tuple(int(v) for v in row["color_bgr"])
        _draw_projected_polyline_camera_clipped(
            overlay,
            plan_world_dense,
            intrinsic,
            front_c2w,
            color=color,
            thickness=4 if row["candidate_rank"] == 0 else 2,
        )

        if len(plan_world_dense) > 0:
            h, w = overlay.shape[:2]
            pixels = _project_world_points_to_image(plan_world_dense, intrinsic, front_c2w)
            label_anchor = None
            for px, py, valid in pixels:
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


def summarize_candidate(points: List[Sequence[float]]) -> Dict[str, object]:
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
        }

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    start = [round(xs[0], 3), round(ys[0], 3)]
    end = [round(xs[-1], 3), round(ys[-1], 3)]
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
        q_score = row.get("q_score")
        line = (
            f"- candidate_{row['candidate_index']} | color={row['color_name']} | source={row['source']} | "
            f"score={row['rap_score']:.4f} | "
            f"q_score={float(q_score):.4f} | " if q_score is not None else
            f"- candidate_{row['candidate_index']} | color={row['color_name']} | source={row['source']} | "
            f"score={row['rap_score']:.4f} | "
        )
        line += (
            f"num_points={s['num_points']} | "
            f"start={s['start']} | end={s['end']} | "
            f"x_range=[{s['min_x']},{s['max_x']}] | y_range=[{s['min_y']},{s['max_y']}] | "
            f"delta=({s['delta_x']},{s['delta_y']})"
        )
        lines.append(line)
    return "\n".join(lines)


def build_scoring_prompt(
    candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
    uncertainty_reasons: Optional[Sequence[str]] = None,
) -> str:
    candidate_text = format_candidate_text(candidate_rows)
    score_schema_lines = ",\n".join(
        f'    "{row["candidate_index"]}": <float>'
        for row in candidate_rows
    )
    candidate_index_list = ", ".join(str(row["candidate_index"]) for row in candidate_rows)
    uncertainty_text = ""
    if uncertainty_reasons:
        uncertainty_text = "Planner context:\n" + "\n".join(f"- {reason}" for reason in uncertainty_reasons)
        uncertainty_text += "\n\n"
    return f"""
You are a cautious autonomous-driving trajectory scorer.

You are given:
1. A front-facing driving image with multiple color-coded candidate trajectories overlaid.
2. Structured candidate trajectory metadata.
3. A route-level driving instruction.

Your job:
- Score EVERY candidate trajectory with a scalar Q-like value.
- Also identify the single best candidate.
- Judge based on visible scene context and trajectory plausibility.
- Prefer:
  - staying in the drivable region,
  - maintaining lane alignment,
  - avoiding nearby vehicles/obstacles,
  - avoiding sidewalk/off-road/wrong-way behavior,
  - smooth and realistic motion,
  - following the route instruction as much as possible.

Important constraints:
- Use only what is visible in the image and candidate metadata.
- Do not assume access to a hidden map or ground-truth future route.
- If multiple candidates are similar, give them similar scores.
- If all candidates are imperfect, prefer the least risky one.
- There are exactly {len(candidate_rows)} candidates in this frame.
- You MUST return one score for every candidate index: {candidate_index_list}.
- Do not omit any candidate index.
- Do not shorten the score dictionary to a partial example.
- One candidate may be "carry_prev", which means continue following the
  previously selected plan instead of switching to a fresh current proposal.
- Treat scores as relative action values:
  - higher is better,
  - score based on visible safety, lane alignment, route following, and smoothness,
  - penalize trajectories that visibly leave the road, cut across lanes, or
    look inconsistent with the scene.
- Keep score scale internally consistent within this frame.
- Keep the response compact. Do not include long explanations.

Route instruction:
{route_instruction}

{uncertainty_text}

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "candidate_scores": {{
{score_schema_lines}
  }},
  "best_candidate_index": <int>,
  "confidence": <float between 0 and 1>
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
        0: "Turn right safely and follow the road.",
        1: "Turn left safely and follow the road.",
        2: "Drive forward safely and follow the lane.",
    }
    try:
        return mapping.get(int(command), "Drive safely and choose the most reasonable trajectory.")
    except Exception:
        return "Drive safely and choose the most reasonable trajectory."


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

    def _run_inference(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            images=[image],
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

    def select(
        self,
        image_path: Path,
        candidate_rows: Sequence[Dict[str, object]],
        route_instruction: str,
        uncertainty_reasons: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        prompt = build_scoring_prompt(candidate_rows, route_instruction, uncertainty_reasons=uncertainty_reasons)
        started = time.time()
        raw_output = self._run_inference(image_path, prompt)
        parsed = try_parse_json(raw_output)
        return {
            "raw_output": raw_output,
            "parsed_output": parsed,
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
        }


def build_candidate_rows(candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    current_rank = 0
    for rank, candidate in enumerate(candidates):
        source = str(candidate.get("source", "current_rap"))
        if source == "carry_prev":
            color = PAST_TRAJECTORY_COLOR
            color_name = "black"
        else:
            color = TOPK_BASE_COLORS[min(current_rank, len(TOPK_BASE_COLORS) - 1)]
            color_name = TOPK_COLOR_NAMES[min(current_rank, len(TOPK_COLOR_NAMES) - 1)]
            current_rank += 1
        plan = np.asarray(candidate["local_plan"], dtype=np.float32)
        row = {
            "candidate_index": rank,
            "candidate_rank": rank,
            "proposal_index": None if candidate.get("proposal_index") is None else int(candidate["proposal_index"]),
            "source": source,
            "color_name": color_name,
            "color_bgr": list(color),
            "local_plan": plan.tolist(),
            "execution_plan": np.asarray(candidate.get("execution_plan", candidate["local_plan"]), dtype=np.float32).tolist(),
            "rap_score": float(candidate.get("proposal_score", 0.0)),
            "q_score": None if candidate.get("q_score") is None else float(candidate["q_score"]),
        }
        row["summary"] = summarize_candidate(row["local_plan"])
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


def _select_from_vlm_q_scores(
    candidate_rows: Sequence[Dict[str, object]],
    vlm_q_scores: Sequence[float],
    carry_switch_margin: float,
) -> Dict[str, object]:
    vlm_q_scores = [float(score) for score in vlm_q_scores]
    if len(vlm_q_scores) != len(candidate_rows):
        raise ValueError("VLM Q score count does not match candidate rows")

    carry_idx = next((idx for idx, row in enumerate(candidate_rows) if row.get("source") == "carry_prev"), None)
    current_indices = [idx for idx, row in enumerate(candidate_rows) if row.get("source") != "carry_prev"]
    if not current_indices:
        raise ValueError("No current candidates available for VLM-Q selection")

    best_current_idx = max(current_indices, key=lambda idx: vlm_q_scores[idx])
    best_current_score = float(vlm_q_scores[best_current_idx])
    carry_score = None if carry_idx is None else float(vlm_q_scores[carry_idx])

    selected_idx = int(best_current_idx)
    selected_source = "vlm_q_current"
    decision = "vlm_q_switch_to_current"
    score_gap_to_carry = None

    if carry_idx is not None and carry_score is not None:
        score_gap_to_carry = best_current_score - carry_score
        if score_gap_to_carry < float(carry_switch_margin):
            selected_idx = int(carry_idx)
            selected_source = "vlm_q_carry_prev"
            decision = "vlm_q_reuse_prev"

    sorted_scores = sorted(((float(score), idx) for idx, score in enumerate(vlm_q_scores)), reverse=True)
    score_gap_top2 = None
    if len(sorted_scores) >= 2:
        score_gap_top2 = float(sorted_scores[0][0] - sorted_scores[1][0])

    return {
        "selected_candidate_index": selected_idx,
        "selected_candidate_row": dict(candidate_rows[selected_idx]),
        "selected_source": selected_source,
        "adaptive_replan_decision": decision if carry_idx is not None else "no_valid_prev",
        "vlm_q_best_current_index": int(best_current_idx),
        "vlm_q_best_current_score": best_current_score,
        "vlm_q_carry_index": None if carry_idx is None else int(carry_idx),
        "vlm_q_carry_score": carry_score,
        "vlm_q_score_gap_to_carry": score_gap_to_carry,
        "vlm_q_score_gap_top2": score_gap_top2,
    }


class VLMPlanSelector:
    def __init__(self, cfg: VLMSelectorConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.debug_dir = output_dir / cfg.debug_dir_name
        self.timeline_path = self.debug_dir / "latency_timeline.jsonl"
        self.summary_path = self.debug_dir / "latency_summary.json"
        self._selector: Optional[Qwen3TrajectorySelector] = None
        self._disabled_reason: Optional[str] = None
        self._timeline_records: List[Dict[str, object]] = []

        if cfg.save_debug_artifacts:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.timeline_path.unlink(missing_ok=True)
            self.summary_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_candidates.jpg"):
                stale_path.unlink(missing_ok=True)
            for stale_path in self.debug_dir.glob("frame_*_result.json"):
                stale_path.unlink(missing_ok=True)

    def _record_timeline(self, record: Dict[str, object]) -> None:
        self._timeline_records.append(record)
        if self.cfg.save_debug_artifacts:
            with self.timeline_path.open("a", encoding="utf-8") as wf:
                wf.write(json.dumps(record) + "\n")

    def _ensure_selector(self) -> Optional[Qwen3TrajectorySelector]:
        if self._selector is not None:
            return self._selector
        if self._disabled_reason is not None:
            return None
        if self.cfg.backend != "qwen3_vl":
            self._disabled_reason = f"unsupported_backend:{self.cfg.backend}"
            return None
        try:
            self._selector = Qwen3TrajectorySelector(
                model_id=self.cfg.model_id,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
            )
            LOG.info(
                "Initialized VLM selector backend=%s model=%s device=%s",
                self.cfg.backend,
                self.cfg.model_id,
                self._selector.device,
            )
            return self._selector
        except Exception as exc:
            self._disabled_reason = str(exc)
            LOG.exception("Failed to initialize VLM selector, disabling VLM fallback path")
            return None

    def maybe_select(
        self,
        frame_index: int,
        front_image: np.ndarray,
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
        invoke_vlm: bool,
        uncertainty_reasons: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        route_instruction = command_to_route_instruction(info.get("command"))
        timestamp = float(info.get("timestamp", 0.0))
        dt_sec = 0.25
        carry_previous_valid = any(row.get("source") == "carry_prev" for row in candidate_rows)
        candidate_rows = build_candidate_rows(candidate_rows)

        def _fallback_result(error: Optional[str] = None) -> Dict[str, object]:
            selected_row = dict(candidate_rows[default_selected_index])
            adaptive_replan_decision = "no_valid_prev" if not carry_previous_valid else (
                "q_reuse_prev" if selected_row.get("source") == "carry_prev" else "q_switch_to_current"
            )
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
                "vlm_elapsed_sec": 0.0,
                "latency_equivalent_steps": 0.0,
                "latency_equivalent_steps_ceil": 0,
                "adaptive_replan_decision": adaptive_replan_decision,
                "error": error,
                "q_invoked_vlm": False,
                "q_uncertainty_reasons": list(uncertainty_reasons or []),
                "vlm_q_valid": False,
            }
            self._record_timeline(timeline_record)
            result = {
                "selected_candidate_row": selected_row,
                "selected_source": default_selected_source,
                "adaptive_replan_decision": adaptive_replan_decision,
                "carry_previous_valid": carry_previous_valid,
                "latency_timeline_record": timeline_record,
                "vlm_q_valid": False,
            }
            if error is not None:
                result["error"] = error
            return result

        if not invoke_vlm:
            return _fallback_result()
        if not self.cfg.enabled:
            return _fallback_result("vlm_disabled_uncertain_q_fallback")

        selector = self._ensure_selector()
        if selector is None:
            return _fallback_result(self._disabled_reason or "selector_unavailable")

        overlay = render_candidate_overlay(front_image, info, candidate_rows)
        frame_stem = f"frame_{frame_index:04d}"
        image_path = self.debug_dir / f"{frame_stem}_candidates.jpg"
        result_path = self.debug_dir / f"{frame_stem}_result.json"
        if self.cfg.save_debug_artifacts:
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            image_path = self.output_dir / f"{frame_stem}_vlm_tmp.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        try:
            LOG.info(
                "Running VLM-Q selection for frame=%d candidates=%d route='%s'",
                frame_index,
                len(candidate_rows),
                route_instruction,
            )
            result = selector.select(
                image_path=image_path,
                candidate_rows=candidate_rows,
                route_instruction=route_instruction,
                uncertainty_reasons=uncertainty_reasons,
            )
        except Exception as exc:
            LOG.exception("VLM-Q selector inference failed, falling back to fast-Q")
            result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": "",
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

        if isinstance(parsed, dict):
            coerced_scores, score_error = _coerce_candidate_scores(parsed.get("candidate_scores"), len(candidate_rows))
            candidate_idx = parsed.get("best_candidate_index")
            if coerced_scores is not None:
                selection = _select_from_vlm_q_scores(
                    candidate_rows=candidate_rows,
                    vlm_q_scores=coerced_scores,
                    carry_switch_margin=self.cfg.q_switch_margin,
                )
                vlm_q_valid = True
                vlm_q_candidate_scores = [float(score) for score in coerced_scores]
                selected_candidate_index = int(selection["selected_candidate_index"])
                selected_row = dict(selection["selected_candidate_row"])
                selected_source = str(selection["selected_source"])
                vlm_q_best_candidate_index = int(max(range(len(vlm_q_candidate_scores)), key=lambda idx: vlm_q_candidate_scores[idx]))
                vlm_q_score_gap_to_carry = selection["vlm_q_score_gap_to_carry"]
                vlm_q_score_gap_top2 = selection["vlm_q_score_gap_top2"]
                vlm_q_best_current_score = selection["vlm_q_best_current_score"]
                vlm_q_carry_score = selection["vlm_q_carry_score"]
                if isinstance(candidate_idx, int) and 0 <= candidate_idx < len(candidate_rows):
                    vlm_candidate_index = int(candidate_idx)
                vlm_confidence = float(parsed.get("confidence", 0.0))
                vlm_reasoning = parsed.get("reasoning")
                if elapsed_sec > self.cfg.timeout_sec:
                    error = f"selector_slow:{elapsed_sec:.3f}"
            else:
                error = score_error or "invalid_candidate_scores"
        else:
            error = (
                f"selector_timeout_budget_exceeded:{elapsed_sec:.3f}"
                if elapsed_sec > self.cfg.timeout_sec
                else str(result.get("error") or "invalid_selector_output")
            )

        latency_equivalent_steps = elapsed_sec / max(dt_sec, 1e-6)
        latency_equivalent_steps_ceil = int(math.ceil(latency_equivalent_steps))
        adaptive_replan_decision = "no_valid_prev"
        if carry_previous_valid:
            adaptive_replan_decision = "vlm_q_reuse_prev" if selected_row.get("source") == "carry_prev" else "vlm_q_switch_to_current"

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
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_source": selected_row.get("source"),
            "selected_proposal_index": selected_row.get("proposal_index"),
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_q_valid": vlm_q_valid,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "latency_equivalent_steps": latency_equivalent_steps,
            "latency_equivalent_steps_ceil": latency_equivalent_steps_ceil,
            "adaptive_replan_decision": adaptive_replan_decision,
            "error": error,
            "q_invoked_vlm": True,
            "q_uncertainty_reasons": list(uncertainty_reasons or []),
        }
        self._record_timeline(timeline_record)

        debug_payload = {
            "frame_index": frame_index,
            "route_instruction": route_instruction,
            "default_selected_index": int(default_selected_index),
            "default_selected_source": default_selected_source,
            "candidate_rows": candidate_rows,
            "selector_result": result,
            "selected_index": int(selected_candidate_index),
            "selected_source": selected_source,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_q_valid": vlm_q_valid,
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
        }
        if self.cfg.save_debug_artifacts:
            result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        if not self.cfg.save_debug_artifacts and image_path.exists():
            image_path.unlink(missing_ok=True)

        LOG.info(
            "VLM-Q selection frame=%d source=%s proposal=%d candidate=%s elapsed=%.3f error=%s",
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
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_reasoning": vlm_reasoning,
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_error": error,
            "vlm_candidate_count": len(candidate_rows),
            "vlm_q_valid": vlm_q_valid,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "vlm_q_best_current_score": vlm_q_best_current_score,
            "vlm_q_carry_score": vlm_q_carry_score,
            "adaptive_replan_decision": adaptive_replan_decision,
            "carry_previous_valid": carry_previous_valid,
            "latency_timeline_record": timeline_record,
        }

    def finalize(self) -> None:
        if not self._timeline_records or not self.cfg.save_debug_artifacts:
            return

        elapsed = [float(record["vlm_elapsed_sec"]) for record in self._timeline_records]
        carry_reuse_rate = sum(
            record["adaptive_replan_decision"] in {"reuse_prev", "vlm_q_reuse_prev", "q_reuse_prev"}
            for record in self._timeline_records
        ) / len(self._timeline_records)
        switch_rate = sum(
            record["adaptive_replan_decision"] in {"switch_to_current", "vlm_q_switch_to_current", "q_switch_to_current"}
            for record in self._timeline_records
        ) / len(self._timeline_records)
        fallback_rate = sum(
            record.get("selected_source", "").startswith("fallback")
            for record in self._timeline_records
        ) / len(self._timeline_records)
        summary = {
            "num_records": len(self._timeline_records),
            "latency_mean_sec": float(np.mean(elapsed)),
            "latency_p50_sec": float(np.percentile(elapsed, 50)),
            "latency_p95_sec": float(np.percentile(elapsed, 95)),
            "latency_max_sec": float(np.max(elapsed)),
            "latency_equivalent_steps_mean": float(
                np.mean([record["latency_equivalent_steps"] for record in self._timeline_records])
            ),
            "carry_reuse_rate": float(carry_reuse_rate),
            "switch_to_current_rate": float(switch_rate),
            "fallback_rate": float(fallback_rate),
            "vlm_q_valid_rate": float(
                np.mean([1.0 if record.get("vlm_q_valid") else 0.0 for record in self._timeline_records])
            ),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
