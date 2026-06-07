from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from planners.common.rule_based_provider import (
    RuleBasedMergeConfig,
    build_rule_based_candidate_rows,
    get_rule_based_proposals_and_scores,
    resolve_rule_based_merge_config,
)


@dataclass(frozen=True)
class RuleBasedExpertRef:
    """Reference to the current external Rule-Planner integration."""

    name: str = "rule_based"
    provider_module: str = "planners.common.rule_based_provider"
    client_module: str = "planners.rule_based.client"


RULE_BASED_EXPERT = RuleBasedExpertRef()


def resolve_config(*args: Any, **kwargs: Any) -> RuleBasedMergeConfig:
    return resolve_rule_based_merge_config(*args, **kwargs)


def generate_proposals_and_scores(
    cfg: RuleBasedMergeConfig,
    *,
    obs: Dict[str, Any],
    info: Dict[str, Any],
    info_history: Deque[Dict[str, Any]],
    privileged_agents: Optional[List[Dict[str, Any]]],
    output_num_poses: int,
    topk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    return get_rule_based_proposals_and_scores(
        cfg,
        obs=obs,
        info=info,
        info_history=info_history,
        privileged_agents=privileged_agents,
        output_num_poses=output_num_poses,
        topk=topk,
    )


def build_candidate_rows(*args: Any, **kwargs: Any) -> List[Dict[str, object]]:
    return build_rule_based_candidate_rows(*args, **kwargs)
