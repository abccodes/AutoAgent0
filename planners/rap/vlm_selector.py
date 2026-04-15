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
    backend: str = "qwen3_vl"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 300
    candidate_limit: int = 10
    timeout_sec: float = 10.0
    save_debug_artifacts: bool = True
    debug_dir_name: str = "vlm_debug"


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


def format_candidate_text(candidate_rows: Sequence[Dict[str, object]]) -> str:
    lines = []
    for row in candidate_rows:
        s = row["summary"]
        line = (
            f"- candidate_{row['candidate_index']} | color={row['color_name']} | source={row['source']} | "
            f"rap_score={row['rap_score']:.4f} | num_points={s['num_points']} | "
            f"start={s['start']} | end={s['end']} | "
            f"x_range=[{s['min_x']},{s['max_x']}] | y_range=[{s['min_y']},{s['max_y']}] | "
            f"delta=({s['delta_x']},{s['delta_y']})"
        )
        lines.append(line)
    return "\n".join(lines)


def build_scoring_prompt(candidate_rows: Sequence[Dict[str, object]], route_instruction: str) -> str:
    candidate_text = format_candidate_text(candidate_rows)
    return f"""
You are a cautious autonomous-driving trajectory selector.

You are given:
1. A front-facing driving image with multiple color-coded candidate trajectories overlaid.
2. Structured candidate trajectory metadata.
3. A route-level driving instruction.

Your job:
- Select the SINGLE best candidate trajectory.
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
- If multiple candidates are similar, choose the safest one.
- If all candidates are imperfect, choose the least risky one.

Route instruction:
{route_instruction}

Candidate trajectories:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "best_candidate_index": <int>,
  "best_candidate_color": "<str>",
  "confidence": <float between 0 and 1>,
  "reasoning": {{
    "safety": "<primary safety considerations>",
    "lane_alignment": "<how well it follows road geometry>",
    "confidence_justification": "<brief explanation of the confidence score>"
  }}
}}
""".strip()


def try_parse_json(text: str) -> Optional[Dict[str, object]]:
    raw = text.strip()
    for candidate in (raw,):
        try:
            return json.loads(candidate)
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

    def select(self, image_path: Path, candidate_rows: Sequence[Dict[str, object]], route_instruction: str) -> Dict[str, object]:
        prompt = build_scoring_prompt(candidate_rows, route_instruction)
        started = time.time()
        raw_output = self._run_inference(image_path, prompt)
        parsed = try_parse_json(raw_output)
        return {
            "raw_output": raw_output,
            "parsed_output": parsed,
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
        }


def build_candidate_rows(
    plans: Sequence[np.ndarray],
    proposal_indices: Sequence[int],
    proposal_scores: Sequence[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank, (plan, proposal_idx, score) in enumerate(zip(plans, proposal_indices, proposal_scores)):
        color = TOPK_BASE_COLORS[min(rank, len(TOPK_BASE_COLORS) - 1)]
        row = {
            "candidate_index": rank,
            "candidate_rank": rank,
            "proposal_index": int(proposal_idx),
            "source": "current_rap",
            "color_name": TOPK_COLOR_NAMES[min(rank, len(TOPK_COLOR_NAMES) - 1)],
            "color_bgr": list(color),
            "local_plan": np.asarray(plan, dtype=np.float32).tolist(),
            "rap_score": float(score),
        }
        row["summary"] = summarize_candidate(row["local_plan"])
        rows.append(row)
    return rows


class VLMPlanSelector:
    def __init__(self, cfg: VLMSelectorConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.debug_dir = output_dir / cfg.debug_dir_name
        self._selector: Optional[Qwen3TrajectorySelector] = None
        self._disabled_reason: Optional[str] = None

        if cfg.save_debug_artifacts:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

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
        candidate_plans: Sequence[np.ndarray],
        candidate_indices: Sequence[int],
        candidate_scores: Sequence[float],
        rap_argmax_index: int,
    ) -> Dict[str, object]:
        if not self.cfg.enabled:
            return {
                "selected_index": int(rap_argmax_index),
                "selected_source": "rap_argmax",
            }

        selector = self._ensure_selector()
        if selector is None:
            return {
                "selected_index": int(rap_argmax_index),
                "selected_source": "fallback_rap_argmax",
                "error": self._disabled_reason or "selector_unavailable",
            }

        candidate_rows = build_candidate_rows(candidate_plans, candidate_indices, candidate_scores)
        overlay = render_candidate_overlay(front_image, info, candidate_rows)
        route_instruction = command_to_route_instruction(info.get("command"))

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
                "Running VLM selection for frame=%d candidates=%d route='%s'",
                frame_index,
                len(candidate_rows),
                route_instruction,
            )
            result = selector.select(
                image_path=image_path,
                candidate_rows=candidate_rows,
                route_instruction=route_instruction,
            )
        except Exception as exc:
            LOG.exception("VLM selector inference failed, falling back to RAP argmax")
            result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": "",
                "error": str(exc),
            }
        parsed = result.get("parsed_output")
        elapsed_sec = float(result.get("elapsed_sec", 0.0))
        selected_index = int(rap_argmax_index)
        selected_source = "fallback_rap_argmax"
        error = None
        vlm_confidence = None
        vlm_reasoning = None
        vlm_candidate_index = None

        if isinstance(parsed, dict):
            candidate_idx = parsed.get("best_candidate_index")
            if isinstance(candidate_idx, int) and 0 <= candidate_idx < len(candidate_rows):
                vlm_candidate_index = int(candidate_idx)
                selected_index = int(candidate_rows[candidate_idx]["proposal_index"])
                selected_source = "vlm_qwen3"
                vlm_confidence = float(parsed.get("confidence", 0.0))
                vlm_reasoning = parsed.get("reasoning")
                if elapsed_sec > self.cfg.timeout_sec:
                    error = f"selector_slow:{elapsed_sec:.3f}"
            else:
                error = "invalid_candidate_index"
        else:
            if elapsed_sec > self.cfg.timeout_sec:
                error = f"selector_timeout_budget_exceeded:{elapsed_sec:.3f}"
            else:
                error = str(result.get("error") or "invalid_selector_output")

        debug_payload = {
            "frame_index": frame_index,
            "route_instruction": route_instruction,
            "rap_argmax_index": int(rap_argmax_index),
            "candidate_rows": candidate_rows,
            "selector_result": result,
            "selected_index": int(selected_index),
            "selected_source": selected_source,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "error": error,
        }
        if self.cfg.save_debug_artifacts:
            result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        if not self.cfg.save_debug_artifacts and image_path.exists():
            image_path.unlink(missing_ok=True)

        LOG.info(
            "VLM selection frame=%d source=%s proposal=%d candidate=%s elapsed=%.3f error=%s",
            frame_index,
            selected_source,
            int(selected_index),
            "none" if vlm_candidate_index is None else str(vlm_candidate_index),
            elapsed_sec,
            error,
        )

        return {
            "selected_index": int(selected_index),
            "selected_source": selected_source,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_reasoning": vlm_reasoning,
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_error": error,
            "vlm_candidate_count": len(candidate_rows),
        }
