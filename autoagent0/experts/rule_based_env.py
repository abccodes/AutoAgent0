from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple


RULE_BASED_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "REPO_ROOT": "",
    "PYTHON_BIN": "",
    "CONFIG": "",
    "DEVICE": "cpu",
    "TOPK": 3,
    "INCLUDE_PRIVILEGED_INFO": True,
    "SOURCE_NAME": "rule_based",
}


RULE_BASED_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "REPO_ROOT": "repo_root",
    "PYTHON_BIN": "python_bin",
    "CONFIG": "config_path",
    "DEVICE": "device",
    "TOPK": "topk",
    "INCLUDE_PRIVILEGED_INFO": "include_privileged_info",
    "SOURCE_NAME": "source_name",
}


def build_prefixed_rule_based_env(
    rule_based_cfg: Any,
    *,
    planner_python_bin: str = "",
    prefixes: Iterable[str] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> Dict[str, Any]:
    env_values: Dict[str, Any] = {}
    for suffix, field_name in RULE_BASED_ENV_FIELD_NAMES.items():
        default_value = RULE_BASED_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        value = rule_based_cfg.get(field_name, default_value)
        if value == default_value and suffix == "CONFIG":
            value = rule_based_cfg.get("config", default_value)
        for prefix in prefixes:
            env_values[f"{prefix}{suffix}"] = value
    return env_values


def get_prefixed_rule_based_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default
