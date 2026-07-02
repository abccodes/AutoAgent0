from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from autoagent0.experts.rule_based_env import (
    RULE_BASED_ENV_DEFAULTS,
    RULE_BASED_ENV_FIELD_NAMES,
    get_prefixed_rule_based_env_value,
)


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleBasedMergeConfig:
    enabled: bool
    repo_root: Path
    python_bin: str
    config_path: Path
    device: str
    topk: int
    include_privileged_info: bool
    source_name: str


_SERVICE_CACHE: Dict[Tuple[str, str], Any] = {}


def _coerce_env_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


def resolve_rule_based_merge_config(
    *,
    planner_python_bin: str = "",
    prefixes: Tuple[str, ...] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> RuleBasedMergeConfig:
    values: Dict[str, Any] = {}
    for suffix, field_name in RULE_BASED_ENV_FIELD_NAMES.items():
        default_value = RULE_BASED_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        raw_value = get_prefixed_rule_based_env_value(
            suffix,
            default=default_value,
            prefixes=prefixes,
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)

    repo_root = Path(values["repo_root"]).expanduser() if values["repo_root"] else Path()
    config_path = Path(values["config_path"]).expanduser() if values["config_path"] else Path()
    topk = max(1, int(values["topk"]))
    return RuleBasedMergeConfig(
        enabled=bool(values["enabled"]),
        repo_root=repo_root,
        python_bin=str(values["python_bin"]),
        config_path=config_path,
        device=str(values["device"]),
        topk=topk,
        include_privileged_info=bool(values["include_privileged_info"]),
        source_name=str(values["source_name"] or "rule_based"),
    )


def rule_based_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def trajectory_to_proposals(selected: Any, output_num_poses: int) -> np.ndarray:
    def _traj_to_3d(traj_obj: Any) -> np.ndarray:
        states = np.asarray(traj_obj.states, dtype=np.float32)
        if states.shape[0] < output_num_poses:
            pad_len = output_num_poses - states.shape[0]
            last_state = states[-1] if len(states) > 0 else np.zeros(2, dtype=np.float32)
            pad = np.tile(last_state, (pad_len, 1)).astype(np.float32)
            states_p = np.concatenate([states, pad], axis=0)
        else:
            states_p = states[:output_num_poses]
        headings = np.zeros((output_num_poses, 1), dtype=np.float32)
        return np.concatenate([states_p[:, :2], headings], axis=1)

    if isinstance(selected, (list, tuple)) and len(selected) > 0 and isinstance(selected[0], (list, tuple)):
        proposals = [_traj_to_3d(item[0]) for item in selected]
        return np.stack(proposals, axis=0)

    return np.expand_dims(_traj_to_3d(selected), axis=0)


def trajectory_to_scores(selected: Any, debug_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if isinstance(selected, (list, tuple)) and len(selected) > 0 and isinstance(selected[0], (list, tuple)):
        scores = []
        for _, score_dict in selected:
            if score_dict is None:
                scores.append(0.0)
            else:
                scores.append(float(score_dict.get("total_score", 0.0)))
        return np.asarray(scores, dtype=np.float32)

    if debug_info is not None:
        best = debug_info.get("best_score")
        if isinstance(best, dict):
            return np.array([float(best.get("total_score", 0.0))], dtype=np.float32)

    return np.array([0.0], dtype=np.float32)


def _load_rule_based_service(cfg: RuleBasedMergeConfig):
    if not cfg.repo_root or not cfg.repo_root.is_dir():
        raise FileNotFoundError(f"rule-based repo_root missing: {cfg.repo_root}")
    if not cfg.config_path or not cfg.config_path.is_file():
        raise FileNotFoundError(f"rule-based config missing: {cfg.config_path}")

    cache_key = (str(cfg.repo_root.resolve()), str(cfg.config_path.resolve()))
    cached = _SERVICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    repo_root_str = str(cfg.repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    try:
        from privileged_planner_sd.service import PrivilegedPlannerService
    except ImportError:  # older Rule-Planner layout
        from privileged_planner.service import PrivilegedPlannerService

    with cfg.config_path.open("r", encoding="utf-8") as fh:
        planner_cfg = yaml.safe_load(fh) or {}

    service = PrivilegedPlannerService(config=planner_cfg)
    _SERVICE_CACHE[cache_key] = service
    return service


def get_rule_based_proposals_and_scores(
    cfg: RuleBasedMergeConfig,
    *,
    obs: Dict[str, Any],
    info: Dict[str, Any],
    info_history: Deque[Dict[str, Any]],
    privileged_agents: Optional[List[Dict[str, Any]]],
    output_num_poses: int,
    topk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    service = _load_rule_based_service(cfg)
    selected, planner_debug = service.process(
        obs=obs,
        info=info,
        info_history=deque(info_history, maxlen=len(info_history) or None),
        privileged_agents=privileged_agents if cfg.include_privileged_info else None,
        k=max(1, int(topk if topk is not None else cfg.topk)),
    )
    proposals = trajectory_to_proposals(selected, output_num_poses)
    scores = trajectory_to_scores(selected, planner_debug)
    return proposals, scores, planner_debug


def build_rule_based_candidate_rows(
    proposals: np.ndarray,
    scores: np.ndarray,
    *,
    output_num_poses: int,
    source_name: str,
    topk: int,
) -> List[Dict[str, object]]:
    if len(scores) == 0:
        return []

    candidate_limit = max(1, min(int(topk), int(len(scores))))
    sorted_indices = np.argsort(scores)[::-1][:candidate_limit]
    rows: List[Dict[str, object]] = []
    for idx in sorted_indices:
        full_plan = rule_based_to_hugsim_plan(proposals[idx, :output_num_poses])
        rows.append(
            {
                "source": source_name,
                "proposal_index": int(idx),
                "proposal_score": float(scores[idx]),
                "local_plan": full_plan,
                "execution_plan": full_plan.copy(),
            }
        )
    return rows
