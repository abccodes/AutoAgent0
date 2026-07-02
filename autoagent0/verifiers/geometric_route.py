from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from autoagent0.adapters.hugsim.context import command_to_route_instruction


RouteDirection = str  # "straight" | "left" | "right" | "unknown"

STRAIGHT_HEADING_DEG = 10.0
TURN_HEADING_DEG = 15.0
MIN_FORWARD_PROGRESS_M = 0.3


def _normalize_turn_label(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "straight" in text or text in {"go straight", "forward", "continue"}:
        return "straight"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return None


def instruction_to_turn_direction(
    *,
    route_instruction: Optional[str] = None,
    command: Optional[object] = None,
) -> Optional[str]:
    """Map route text or discrete command to a turn label when possible."""
    label = _normalize_turn_label(route_instruction)
    if label is not None:
        return label
    if command is not None:
        return _normalize_turn_label(command_to_route_instruction(command))
    return None


def classify_local_plan_endpoint_bearing(
    local_plan: np.ndarray,
    *,
    straight_heading_deg: float = STRAIGHT_HEADING_DEG,
    turn_heading_deg: float = TURN_HEADING_DEG,
    min_forward_progress_m: float = MIN_FORWARD_PROGRESS_M,
) -> Dict[str, Any]:
    """Classify a HUGSIM local plan [T, 2] using start→end endpoint bearing.

    Coordinate convention in HUGSIM local plans:
    - x: lateral, positive to the right
    - y: forward, positive ahead of ego

    Therefore ``atan2(delta_x, delta_y) > 0`` means turning right and ``< 0`` means left.
    """
    plan = np.asarray(local_plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] < 2 or len(plan) < 2:
        return {
            "geometric_direction": "unknown",
            "heading_deg": None,
            "delta_x": None,
            "delta_y": None,
            "forward_progress_m": None,
            "reason": "plan_too_short",
        }

    delta_x = float(plan[-1, 0] - plan[0, 0])
    delta_y = float(plan[-1, 1] - plan[0, 1])
    forward_progress_m = float(np.linalg.norm(plan[-1] - plan[0]))

    if delta_y < min_forward_progress_m:
        return {
            "geometric_direction": "unknown",
            "heading_deg": None,
            "delta_x": round(delta_x, 4),
            "delta_y": round(delta_y, 4),
            "forward_progress_m": round(forward_progress_m, 4),
            "reason": "insufficient_forward_progress",
        }

    heading_rad = math.atan2(delta_x, delta_y)
    heading_deg = math.degrees(heading_rad)

    if abs(heading_deg) <= straight_heading_deg:
        direction = "straight"
    elif heading_deg >= turn_heading_deg:
        direction = "right"
    elif heading_deg <= -turn_heading_deg:
        direction = "left"
    else:
        direction = "unknown"

    return {
        "geometric_direction": direction,
        "heading_deg": round(heading_deg, 3),
        "delta_x": round(delta_x, 4),
        "delta_y": round(delta_y, 4),
        "forward_progress_m": round(forward_progress_m, 4),
        "reason": None,
    }


@dataclass(frozen=True)
class GeometricRouteCheck:
    geometric_direction: str
    instruction_direction: Optional[str]
    geometric_matches_instruction: Optional[bool]
    semantic_on_track: bool
    semantic_agrees_with_geometry: Optional[bool]
    heading_deg: Optional[float]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometric_direction": self.geometric_direction,
            "instruction_direction": self.instruction_direction,
            "geometric_matches_instruction": self.geometric_matches_instruction,
            "semantic_on_track": self.semantic_on_track,
            "semantic_agrees_with_geometry": self.semantic_agrees_with_geometry,
            "heading_deg": self.heading_deg,
            "details": dict(self.details),
        }


def compare_geometric_semantic_route(
    *,
    local_plan: np.ndarray,
    route_instruction: str,
    command: Optional[object],
    semantic_on_track: bool,
) -> GeometricRouteCheck:
    """Compare endpoint geometry, route instruction, and semantic verifier output."""
    geom = classify_local_plan_endpoint_bearing(local_plan)
    instruction_direction = instruction_to_turn_direction(
        route_instruction=route_instruction,
        command=command,
    )
    geometric_direction = str(geom["geometric_direction"])

    geometric_matches_instruction: Optional[bool]
    if instruction_direction is None or geometric_direction == "unknown":
        geometric_matches_instruction = None
    else:
        geometric_matches_instruction = geometric_direction == instruction_direction

    semantic_agrees_with_geometry: Optional[bool]
    if geometric_matches_instruction is None:
        semantic_agrees_with_geometry = None
    else:
        # True when VLM on_track matches whether the plan geometry fits the command.
        semantic_agrees_with_geometry = bool(semantic_on_track) == bool(geometric_matches_instruction)

    return GeometricRouteCheck(
        geometric_direction=geometric_direction,
        instruction_direction=instruction_direction,
        geometric_matches_instruction=geometric_matches_instruction,
        semantic_on_track=bool(semantic_on_track),
        semantic_agrees_with_geometry=semantic_agrees_with_geometry,
        heading_deg=geom.get("heading_deg"),
        details={
            **geom,
            "route_instruction": route_instruction,
            "command": command,
        },
    )
