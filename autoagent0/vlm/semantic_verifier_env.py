from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

SEMANTIC_VERIFIER_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "GATE_ON_REJECT": False,
    "CAMERA_MODE": "front_only",
    "BACKEND": "local_transformers_subprocess",
    "MODEL_ID": "Qwen/Qwen3-VL-8B-Instruct",
    "DEVICE": "auto",
    "PYTHON_BIN": "",
    "MAX_NEW_TOKENS": 128,
    "TEMPERATURE": 0.0,
    "TOP_P": 1.0,
    "TOP_K": 20,
    "ENABLE_THINKING": False,
    "TIMEOUT_SEC": 60.0,
    "SAVE_DEBUG_ARTIFACTS": True,
    "DEBUG_DIR_NAME": "semantic_verifier_debug",
    "LOG_FILE_NAME": "semantic_verifier.jsonl",
    "PRELOAD_ON_INIT": True,
}

SEMANTIC_VERIFIER_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "GATE_ON_REJECT": "gate_on_reject",
    "CAMERA_MODE": "camera_mode",
    "BACKEND": "backend",
    "MODEL_ID": "model_id",
    "DEVICE": "device",
    "PYTHON_BIN": "python_bin",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "TEMPERATURE": "temperature",
    "TOP_P": "top_p",
    "TOP_K": "top_k",
    "ENABLE_THINKING": "enable_thinking",
    "TIMEOUT_SEC": "timeout_sec",
    "SAVE_DEBUG_ARTIFACTS": "save_debug_artifacts",
    "DEBUG_DIR_NAME": "debug_dir_name",
    "LOG_FILE_NAME": "log_file_name",
    "PRELOAD_ON_INIT": "preload_on_init",
}


def build_prefixed_semantic_verifier_env(
    semantic_verifier_cfg: Any,
    *,
    planner_python_bin: str = "",
    prefixes: Iterable[str] = ("PLANNER_SEMANTIC_VERIFIER_",),
) -> Dict[str, Any]:
    env_values: Dict[str, Any] = {}
    for suffix, field_name in SEMANTIC_VERIFIER_ENV_FIELD_NAMES.items():
        default_value = SEMANTIC_VERIFIER_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        value = semantic_verifier_cfg.get(field_name, default_value) if semantic_verifier_cfg else default_value
        for prefix in prefixes:
            env_values[f"{prefix}{suffix}"] = value
    return env_values


def get_prefixed_semantic_verifier_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("PLANNER_SEMANTIC_VERIFIER_",),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default
