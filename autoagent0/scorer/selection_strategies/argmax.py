"""Argmax trajectory selection strategy."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from autoagent0.scorer.selection_strategies.base import (
    BaseSelectionStrategy,
    SelectionOutcome,
)


class ArgmaxStrategy(BaseSelectionStrategy):
    """Behavior-preserving non-VLM path: pick the highest-scoring proposal."""

    def _passive_uncertainty_debug(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any],
        best_idx: int,
    ) -> Optional[Dict[str, object]]:
        sel = self.selector
        cfg = sel.autoagent0_cfg
        if not (
            getattr(cfg, "enabled", False)
            and getattr(cfg, "uncertainty_enabled", False)
            and str(getattr(cfg, "uncertainty_policy_mode", "active")).lower() == "observe"
        ):
            return None

        reserved_candidate_slots = (
            max(0, int(sel.rule_based_merge_cfg.topk))
            if sel.rule_based_merge_cfg.enabled and not sel.vlm_cfg.planner_gate_enabled
            else 0
        )
        learned_candidate_rows, _ = sel._build_learned_candidate_rows(
            proposals, scores, output_num_poses, info, reserved_candidate_slots,
        )
        default_row = next(
            row
            for row in learned_candidate_rows
            if row.get("source") == sel.current_source_name
            and row.get("proposal_index") == best_idx
        )
        rule_based_candidate_rows = sel._build_rule_based_candidate_rows(
            obs, info, info_history, privileged_info, output_num_poses,
        )
        uncertainty_candidate_rows = sel._build_uncertainty_candidate_rows(proposals, scores)
        frame_uncertainty = sel.recovery_strategy._maybe_compute_frame_uncertainty(
            learned_candidate_rows=uncertainty_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            default_row=default_row,
        )
        observation = sel.recovery_strategy._uncertainty_observation_debug(
            proposals=proposals,
            scores=scores,
            rule_based_candidate_rows=rule_based_candidate_rows,
            default_row=default_row,
            frame_uncertainty=frame_uncertainty,
        )
        return {
            "autoagent0_mode": "passive_uncertainty_observe",
            "autoagent0_frame_uncertainty": sel.recovery_strategy._frame_uncertainty_debug(
                frame_uncertainty
            ),
            "autoagent0_uncertainty_observation": observation,
        }

    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
    ) -> SelectionOutcome:
        sel = self.selector
        best_idx = int(np.argmax(scores))
        selected_plan = np.asarray(proposals[best_idx], dtype=np.float32)
        selected_score = float(scores[best_idx])
        selection_debug = self._passive_uncertainty_debug(
            proposals=proposals,
            scores=scores,
            output_num_poses=output_num_poses,
            obs=obs,
            info=info,
            info_history=info_history,
            privileged_info=privileged_info,
            best_idx=best_idx,
        )
        plan_payload = sel._build_plan_payload(
            proposals,
            scores,
            output_num_poses,
            selected_idx=best_idx,
            selected_source=sel.plain_source,
            selection_debug=selection_debug,
            topk=10,
        )
        return SelectionOutcome(
            plan_payload=plan_payload,
            selected_plan=selected_plan,
            selected_score_raw=selected_score,
            selected_source=sel.current_source_name,
            advance_frame=False,
        )
