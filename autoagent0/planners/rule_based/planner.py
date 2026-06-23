#!/usr/bin/env python3
"""Rule-based planner core: observation (+ privileged info) -> proposals + scores.

Pure-inference half of the former ``planners/rule_based/client.py``: run the
privileged rule-based planner service and return its trajectories + scores in
HUGSIM local coordinates. Unlike the learned planners, this one *requires*
``privileged_info`` (its rule engine uses ground-truth agents). Selection /
payload happen pipeline-side.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from autoagent0.planners.base import PlannerResult, PlannerService

# Rule-Planner lives in its own repo; add it to the path (env set by launch.sh).
RULE_BASED_REPO_ROOT = os.environ.get("RULE_BASED_REPO_ROOT", "")
if not RULE_BASED_REPO_ROOT:
    raise RuntimeError("RULE_BASED_REPO_ROOT must be set in environment")
sys.path.insert(0, str(Path(RULE_BASED_REPO_ROOT).resolve()))

from privileged_planner.service import PrivilegedPlannerService  # type: ignore

LOG = logging.getLogger("rule_based_adapter")

EGO_HISTORY_FRAMES = 4
DEFAULT_OUTPUT_POSES = 8
TOPK = 10


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rule_based_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def make_command_one_hot(command: int) -> np.ndarray:
    mapping = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return mapping.get(int(command), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def rule_based_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    # NAVSIM predictions: [x_forward, y_left, heading] -> HUGSIM expects [x_right, y_forward]
    right = -trajectory[:, 1]
    forward = trajectory[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def trajectory_to_proposals(selected: Any, output_num_poses: int) -> np.ndarray:
    def _traj_to_3d(traj_obj: Any) -> np.ndarray:
        states = np.asarray(traj_obj.states, dtype=np.float32)  # [T, 2]
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
        return np.stack([_traj_to_3d(tup[0]) for tup in selected], axis=0)
    return np.expand_dims(_traj_to_3d(selected), axis=0)


def trajectory_to_scores(selected: Any, debug_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if isinstance(selected, (list, tuple)) and len(selected) > 0 and isinstance(selected[0], (list, tuple)):
        scores = []
        for _, score_dict in selected:
            scores.append(0.0 if score_dict is None else float(score_dict.get("total_score", 0.0)))
        return np.asarray(scores, dtype=np.float32)
    if debug_info is not None:
        best = debug_info.get("best_score")
        if isinstance(best, dict):
            return np.array([float(best.get("total_score", 0.0))], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)


class RuleBasedPlanner(PlannerService):
    history_frames = EGO_HISTORY_FRAMES

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.planner = None
        self.output_num_poses = DEFAULT_OUTPUT_POSES

    def setup(self) -> None:
        setup_logging(self.output_dir)
        planner_config = None
        planner_config_path = os.environ.get("PLANNER_CONFIG", "").strip()
        if planner_config_path:
            try:
                import yaml
                with open(planner_config_path, "r") as f:
                    planner_config = yaml.safe_load(f)
                LOG.info("Loaded planner config from %s", planner_config_path)
            except Exception:
                LOG.exception("Failed to load PLANNER_CONFIG=%s; using None", planner_config_path)
        self.planner = PrivilegedPlannerService(config=planner_config)
        LOG.info("PrivilegedPlannerService initialized OK")
        try:
            self.output_num_poses = int(
                planner_config.get("horizon", DEFAULT_OUTPUT_POSES)
                if planner_config and isinstance(planner_config, dict)
                else DEFAULT_OUTPUT_POSES
            )
        except Exception:
            self.output_num_poses = DEFAULT_OUTPUT_POSES
        LOG.info("Rule-based ready, output_num_poses=%d", self.output_num_poses)

    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history,
        extra: Dict[str, Any],
    ) -> PlannerResult:
        selected, planner_debug = self.planner.process(
            obs=obs,
            info=info,
            info_history=info_history,
            privileged_agents=extra.get("privileged_info"),
            k=TOPK,
        )
        proposals_raw = trajectory_to_proposals(selected, self.output_num_poses)
        scores = trajectory_to_scores(selected, planner_debug)
        proposals = np.stack(
            [rule_based_to_hugsim_plan(proposals_raw[i, :self.output_num_poses]) for i in range(proposals_raw.shape[0])],
            axis=0,
        ).astype(np.float32)
        return PlannerResult(proposals=proposals, scores=np.asarray(scores, dtype=np.float32))

    def finalize(self) -> None:
        pass
