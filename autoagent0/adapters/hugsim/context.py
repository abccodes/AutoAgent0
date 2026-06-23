from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from autoagent0.agent.schemas import SceneContext


VLM_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
)


def scene_context_from_hugsim_info(
    *,
    frame_index: int,
    route_instruction: str,
    info: Dict[str, Any],
) -> SceneContext:
    return SceneContext(
        frame_index=int(frame_index),
        route_instruction=str(route_instruction),
        timestamp=float(info.get("timestamp", 0.0) or 0.0),
        task_instruction=info.get("task_instruction") if isinstance(info.get("task_instruction"), str) else None,
        metadata={
            "command": info.get("command"),
            "task_type": info.get("task_type"),
        },
    )


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


def resolve_route_instruction(info: Dict[str, object]) -> str:
    task_instruction = info.get("task_instruction")
    if isinstance(task_instruction, str) and task_instruction.strip():
        return task_instruction.strip()
    return command_to_route_instruction(info.get("command"))


def describe_task_target_hint(info: Dict[str, object]) -> Optional[str]:
    task_type = str(info.get("task_type", "")).strip()
    if not task_type:
        return None
    if task_type == "park_at_target":
        return "park target"
    if task_type == "stop_at_target":
        return "stop target"
    return "goal target"


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


def resolve_stage_camera_order(cfg: object, stage: str) -> Tuple[str, ...]:
    if stage == "intervention":
        camera_mode = getattr(cfg, "intervention_camera_mode", "") or getattr(cfg, "camera_mode", "")
    elif stage == "scoring":
        camera_mode = getattr(cfg, "scoring_camera_mode", "") or getattr(cfg, "camera_mode", "")
    else:
        camera_mode = getattr(cfg, "camera_mode", "")
    return resolve_vlm_camera_order(camera_mode)


def extract_current_ego_speed_mps(info: Dict[str, object]) -> Optional[float]:
    try:
        return float(info["ego_velo"])
    except Exception:
        return None


def extract_current_ego_accel_mps2(info: Dict[str, object]) -> Optional[float]:
    try:
        return float(info["accelerate"])
    except Exception:
        return None
