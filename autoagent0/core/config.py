from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


AUTOAGENT0_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "MODE": "recovery_loop",
    "REDESIGN_CANDIDATE_BUDGET": 10,
    "MAX_REDESIGN_ATTEMPTS": 1,
    "FALLBACK_MODE": "hold",
}

AUTOAGENT0_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "MODE": "mode",
    "REDESIGN_CANDIDATE_BUDGET": "redesign_candidate_budget",
    "MAX_REDESIGN_ATTEMPTS": "max_redesign_attempts",
    "FALLBACK_MODE": "fallback_mode",
}


@dataclass(frozen=True)
class AutoAgent0Config:
    enabled: bool = False
    mode: str = "recovery_loop"
    redesign_candidate_budget: int = 10
    max_redesign_attempts: int = 1
    fallback_mode: str = "hold"


def _coerce_env_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


def build_prefixed_autoagent0_env(
    autoagent0_cfg: Any,
    *,
    prefixes: Iterable[str] = ("AUTOAGENT0_",),
) -> Dict[str, Any]:
    env_values: Dict[str, Any] = {}
    for suffix, field_name in AUTOAGENT0_ENV_FIELD_NAMES.items():
        default_value = AUTOAGENT0_ENV_DEFAULTS[suffix]
        value = autoagent0_cfg.get(field_name, default_value) if autoagent0_cfg else default_value
        for prefix in prefixes:
            env_values[f"{prefix}{suffix}"] = value
    return env_values


def get_prefixed_autoagent0_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("AUTOAGENT0_",),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default


def resolve_autoagent0_config(
    *,
    prefixes: Tuple[str, ...] = ("AUTOAGENT0_",),
) -> AutoAgent0Config:
    values: Dict[str, Any] = {}
    for suffix, field_name in AUTOAGENT0_ENV_FIELD_NAMES.items():
        default_value = AUTOAGENT0_ENV_DEFAULTS[suffix]
        raw_value = get_prefixed_autoagent0_env_value(suffix, default=default_value, prefixes=prefixes)
        values[field_name] = _coerce_env_value(raw_value, default_value)
    values["redesign_candidate_budget"] = max(1, int(values["redesign_candidate_budget"]))
    values["max_redesign_attempts"] = max(0, int(values["max_redesign_attempts"]))
    return AutoAgent0Config(**values)
