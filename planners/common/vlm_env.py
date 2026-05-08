from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

VLM_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "INTERVENTION_ENABLED": False,
    "CAMERA_MODE": "multiview",
    "INTERVENTION_CAMERA_MODE": "",
    "SCORING_CAMERA_MODE": "",
    "BACKEND": "local_transformers",
    "MODEL_ID": "Qwen/Qwen3-VL-8B-Instruct",
    "DEVICE": "auto",
    "PYTHON_BIN": "",
    "MAX_NEW_TOKENS": 300,
    "TEMPERATURE": 0.0,
    "TOP_P": 1.0,
    "TOP_K": 20,
    "ENABLE_THINKING": False,
    "CANDIDATE_LIMIT": 5,
    "INTERVENTION_MAX_NEW_TOKENS": 120,
    "TIMEOUT_SEC": 10.0,
    "INTERVENTION_TIMEOUT_SEC": 10.0,
    "INTERVENTION_ACTION_THRESHOLD": 0.6,
    "INTERVENTION_HIGH_THRESHOLD": 0.85,
    "SAVE_DEBUG_ARTIFACTS": True,
    "DEBUG_DIR_NAME": "vlm_debug",
    "CARRY_PREVIOUS_ENABLED": True,
    "CARRY_PREVIOUS_MIN_PATH_M": 0.5,
    "CARRY_PREVIOUS_MIN_POINTS": 2,
    "ADAPTIVE_REPLAN_MODE": "log_only",
    "LATENCY_TRACKING_MODE": "full_timeline",
    "Q_ENABLED": True,
    "Q_SWITCH_MARGIN": 0.05,
    "Q_WEIGHT_RAP_SCORE": 0.55,
    "Q_WEIGHT_PROGRESS": 0.30,
    "Q_WEIGHT_OFFCENTER": 0.10,
    "Q_WEIGHT_CURVATURE": 0.08,
    "Q_WEIGHT_SHORTPLAN": 0.18,
    "Q_CARRY_SCORE_DECAY": 0.0,
    "DISPLAY_DEFAULT_TRAJECTORIES": False,
    "INCLUDE_DEFAULT_CANDIDATES": False,
}

VLM_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "INTERVENTION_ENABLED": "intervention_enabled",
    "CAMERA_MODE": "camera_mode",
    "INTERVENTION_CAMERA_MODE": "intervention_camera_mode",
    "SCORING_CAMERA_MODE": "scoring_camera_mode",
    "BACKEND": "backend",
    "MODEL_ID": "model_id",
    "DEVICE": "device",
    "PYTHON_BIN": "python_bin",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "TEMPERATURE": "temperature",
    "TOP_P": "top_p",
    "TOP_K": "top_k",
    "ENABLE_THINKING": "enable_thinking",
    "CANDIDATE_LIMIT": "candidate_limit",
    "INTERVENTION_MAX_NEW_TOKENS": "intervention_max_new_tokens",
    "TIMEOUT_SEC": "timeout_sec",
    "INTERVENTION_TIMEOUT_SEC": "intervention_timeout_sec",
    "INTERVENTION_ACTION_THRESHOLD": "intervention_action_threshold",
    "INTERVENTION_HIGH_THRESHOLD": "intervention_high_threshold",
    "SAVE_DEBUG_ARTIFACTS": "save_debug_artifacts",
    "DEBUG_DIR_NAME": "debug_dir_name",
    "CARRY_PREVIOUS_ENABLED": "carry_previous_enabled",
    "CARRY_PREVIOUS_MIN_PATH_M": "carry_previous_min_path_m",
    "CARRY_PREVIOUS_MIN_POINTS": "carry_previous_min_points",
    "ADAPTIVE_REPLAN_MODE": "adaptive_replan_mode",
    "LATENCY_TRACKING_MODE": "latency_tracking_mode",
    "Q_ENABLED": "q_enabled",
    "Q_SWITCH_MARGIN": "q_switch_margin",
    "Q_WEIGHT_RAP_SCORE": "q_weight_rap_score",
    "Q_WEIGHT_PROGRESS": "q_weight_progress",
    "Q_WEIGHT_OFFCENTER": "q_weight_offcenter",
    "Q_WEIGHT_CURVATURE": "q_weight_curvature",
    "Q_WEIGHT_SHORTPLAN": "q_weight_shortplan",
    "Q_CARRY_SCORE_DECAY": "q_carry_score_decay",
    "DISPLAY_DEFAULT_TRAJECTORIES": "display_default_trajectories",
    "INCLUDE_DEFAULT_CANDIDATES": "include_default_candidates",
}


def build_prefixed_vlm_env(
    vlm_cfg: Any,
    *,
    planner_python_bin: str = "",
    prefixes: Iterable[str] = ("PLANNER_VLM_", "RAP_VLM_"),
) -> Dict[str, Any]:
    env_values: Dict[str, Any] = {}
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        value = vlm_cfg.get(field_name, default_value)
        for prefix in prefixes:
            env_values[f"{prefix}{suffix}"] = value
    return env_values


def get_prefixed_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("PLANNER_VLM_", "RAP_VLM_"),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default
