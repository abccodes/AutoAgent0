"""Pipeline-side trajectory selection for learned planners.

The planner subprocess only returns ``(proposals, scores)``. Everything that
used to live in each planner's ``client.py`` after inference — building the
candidate pool (carry-previous, top-k, rule-based merge, default fallbacks),
running the VLM / AutoAgent0 selection, and assembling the HUGSIM plan payload —
now happens here, on the pipeline (pixi) side, once for all learned planners.

``proposals`` are expected in HUGSIM local coordinates ([x_right, y_forward]),
shape ``[N, T, 2]``, so this module is planner-agnostic; per-planner labels are
passed in via the constructor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from autoagent0.adapters.hugsim.defaults import get_default_trajectories
from autoagent0.adapters.hugsim.geometry import (
    info_to_pose,
    local_plan_to_world,
    path_length,
    truncate_plan,
    world_points_to_current_local,
)
from autoagent0.decision.payloads import build_hugsim_plan_payload
from autoagent0.experts.rule_based import (
    build_rule_based_candidate_rows,
    get_rule_based_proposals_and_scores,
)

PLAN_DT_SEC = 0.5
TOPK_PROPOSALS_TO_SEND = 20


def _identity_converter(trajectory: np.ndarray) -> np.ndarray:
    # Proposals already arrive in HUGSIM coordinates, so no conversion is needed
    # when handing them to build_hugsim_plan_payload.
    return np.asarray(trajectory, dtype=np.float32)


def build_carry_plan_candidate(
    previous_plan: Optional[np.ndarray],
    previous_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_timestamp: Optional[float],
    current_info: Dict[str, object],
    vlm_cfg,
) -> Optional[Dict[str, object]]:
    if not vlm_cfg.carry_previous_enabled or previous_plan is None or previous_pose is None or previous_timestamp is None:
        return None

    current_timestamp = float(current_info.get("timestamp", previous_timestamp))
    elapsed_sec = max(0.0, current_timestamp - float(previous_timestamp))
    elapsed_pose_steps = int(round(elapsed_sec / PLAN_DT_SEC))
    if elapsed_pose_steps >= len(previous_plan):
        return None

    trimmed_plan = np.asarray(previous_plan[elapsed_pose_steps:], dtype=np.float32)
    if len(trimmed_plan) < vlm_cfg.carry_previous_min_points:
        return None

    points_world = local_plan_to_world(trimmed_plan, np.asarray(previous_pose, dtype=np.float32))
    current_local = world_points_to_current_local(points_world, info_to_pose(current_info))

    valid_mask = current_local[:, 1] > 0.0
    if not np.any(valid_mask):
        return None
    first_valid_idx = int(np.argmax(valid_mask))
    current_local = current_local[first_valid_idx:]

    if len(current_local) < vlm_cfg.carry_previous_min_points:
        return None
    if path_length(current_local) < vlm_cfg.carry_previous_min_path_m:
        return None

    return {
        "source": "carry_prev",
        "proposal_index": None,
        "proposal_score": 0.0,
        "proposal_score_norm": 0.0,
        "origin_selected_score_raw": None if previous_selected_score is None else float(previous_selected_score),
        "local_plan": current_local.astype(np.float32),
        "execution_plan": current_local.astype(np.float32),
        "carry_elapsed_sec": elapsed_sec,
        "carry_elapsed_pose_steps": elapsed_pose_steps,
    }


class LearnedPlannerSelector:
    """Stateful per-run selector. Call :meth:`select` once per frame.

    Holds the VLM selector, the AutoAgent0 runtime, the resolved configs, and the
    carry-previous state. Produces the HUGSIM plan payload the simulator consumes.
    """

    def __init__(
        self,
        *,
        vlm_selector,
        autoagent0_runtime,
        autoagent0_cfg,
        vlm_cfg,
        rule_based_merge_cfg,
        current_source_name: str = "current_rap",
        learned_default_source: str = "fallback_rap_argmax",
        plain_source: str = "rap_argmax",
        score_fallback_key: str = "rap_score",
        planner_log_name: str = "RAP",
        strict_learned_argmax_lookup: bool = True,
        q_key_prefix: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.vlm_selector = vlm_selector
        self.autoagent0_runtime = autoagent0_runtime
        self.autoagent0_cfg = autoagent0_cfg
        self.vlm_cfg = vlm_cfg
        self.rule_based_merge_cfg = rule_based_merge_cfg
        self.current_source_name = current_source_name
        self.learned_default_source = learned_default_source
        self.plain_source = plain_source
        self.score_fallback_key = score_fallback_key
        self.planner_log_name = planner_log_name
        self.strict_learned_argmax_lookup = strict_learned_argmax_lookup
        self.q_key_prefix = q_key_prefix
        self.logger = logger or logging.getLogger("planner_selection")

        self.frame_index = 0
        self.previous_selected_plan: Optional[np.ndarray] = None
        self.previous_selected_pose: Optional[np.ndarray] = None
        self.previous_selected_score: Optional[float] = None
        self.previous_selected_timestamp: Optional[float] = None
        self.previous_selected_source: Optional[str] = None

    # ------------------------------------------------------------------ payload
    def _build_plan_payload(self, proposals, scores, output_num_poses, *, topk=TOPK_PROPOSALS_TO_SEND, **kwargs) -> Dict[str, object]:
        return build_hugsim_plan_payload(
            proposals=proposals,
            scores=scores,
            output_num_poses=output_num_poses,
            plan_converter=_identity_converter,
            default_source_name=self.current_source_name,
            default_trajectory_provider=get_default_trajectories,
            topk=topk,
            **kwargs,
        )

    # --------------------------------------------------------------- candidates
    def _build_learned_candidate_rows(
        self,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        current_info: Dict[str, object],
        reserved_candidate_slots: int,
    ) -> Tuple[List[Dict[str, object]], bool]:
        vlm_cfg = self.vlm_cfg
        allow_carry_previous = not (
            self.previous_selected_source is not None
            and str(self.previous_selected_source).startswith("default_fallback_")
        )
        carry_candidate = build_carry_plan_candidate(
            previous_plan=self.previous_selected_plan if allow_carry_previous else None,
            previous_pose=self.previous_selected_pose if allow_carry_previous else None,
            previous_selected_score=self.previous_selected_score if allow_carry_previous else None,
            previous_timestamp=self.previous_selected_timestamp if allow_carry_previous else None,
            current_info=current_info,
            vlm_cfg=vlm_cfg,
        )

        sorted_indices = np.argsort(scores)[::-1]
        carry_slot_count = 1 if carry_candidate is not None else 0
        current_candidate_limit = max(
            1,
            int(vlm_cfg.candidate_limit) - carry_slot_count - max(0, int(reserved_candidate_slots)),
        )
        current_candidate_limit = min(current_candidate_limit, int(len(sorted_indices)))
        candidate_indices = sorted_indices[:current_candidate_limit]

        candidate_rows: List[Dict[str, object]] = []
        if carry_candidate is not None:
            candidate_rows.append(carry_candidate)

        for idx in candidate_indices:
            # proposals are already HUGSIM-coord and truncated to output_num_poses.
            full_plan = np.asarray(proposals[idx], dtype=np.float32)
            candidate_rows.append(
                {
                    "source": self.current_source_name,
                    "proposal_index": int(idx),
                    "proposal_score": float(scores[idx]),
                    "local_plan": full_plan,
                    "execution_plan": full_plan.copy(),
                }
            )

        if vlm_cfg.include_default_candidates:
            for default_idx, default_plan in enumerate(get_default_trajectories(output_num_poses)):
                candidate_rows.append(
                    {
                        "source": f"default_fallback_{default_idx}",
                        "proposal_index": None,
                        "proposal_score": 0.0,
                        "local_plan": default_plan,
                        "execution_plan": default_plan.copy(),
                    }
                )

        carry_row = next((row for row in candidate_rows if row.get("source") == "carry_prev"), None)
        shared_horizon = len(carry_row["local_plan"]) if carry_row is not None else output_num_poses
        shared_horizon = max(1, int(shared_horizon))
        for row in candidate_rows:
            execution_plan = np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32)
            row["execution_plan"] = execution_plan
            row["local_plan"] = truncate_plan(execution_plan, shared_horizon)

        return candidate_rows, allow_carry_previous

    def _build_rule_based_candidate_rows(
        self,
        obs,
        info,
        info_history,
        privileged_info,
        output_num_poses: int,
    ) -> List[Dict[str, object]]:
        if not self.rule_based_merge_cfg.enabled:
            return []
        try:
            rb_proposals, rb_scores, _ = get_rule_based_proposals_and_scores(
                self.rule_based_merge_cfg,
                obs=obs,
                info=info,
                info_history=info_history,
                privileged_agents=privileged_info,
                output_num_poses=output_num_poses,
                topk=self.rule_based_merge_cfg.topk,
            )
            return build_rule_based_candidate_rows(
                rb_proposals,
                rb_scores,
                output_num_poses=output_num_poses,
                source_name=self.rule_based_merge_cfg.source_name,
                topk=self.rule_based_merge_cfg.topk,
            )
        except Exception:
            self.logger.exception("Failed to append rule-based %s merge candidates", self.planner_log_name)
            return []

    # -------------------------------------------------------------------- select
    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
    ) -> Dict[str, object]:
        proposals = np.asarray(proposals, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        output_num_poses = int(proposals.shape[1]) if proposals.ndim == 3 else 0
        vlm_cfg = self.vlm_cfg

        if not vlm_cfg.enabled:
            best_idx = int(np.argmax(scores))
            selected_plan = np.asarray(proposals[best_idx], dtype=np.float32)
            selected_score = float(scores[best_idx])
            selected_score_raw = selected_score
            selected_row = {"source": self.current_source_name, "proposal_index": best_idx}
            plan_payload = self._build_plan_payload(
                proposals,
                scores,
                output_num_poses,
                selected_idx=best_idx,
                selected_source=self.plain_source,
                topk=10,
            )
        else:
            reserved_candidate_slots = (
                max(0, int(self.rule_based_merge_cfg.topk))
                if self.rule_based_merge_cfg.enabled and not vlm_cfg.planner_gate_enabled
                else 0
            )
            learned_candidate_rows, allow_carry_previous = self._build_learned_candidate_rows(
                proposals, scores, output_num_poses, info, reserved_candidate_slots,
            )
            rule_based_candidate_rows = self._build_rule_based_candidate_rows(
                obs, info, info_history, privileged_info, output_num_poses,
            )

            if self.autoagent0_cfg.enabled:
                selection = self.autoagent0_runtime.select_final_actions_recovery_loop(
                    frame_index=self.frame_index,
                    camera_images=obs["rgb"],
                    info=info,
                    vlm_selector=self.vlm_selector,
                    scores=scores,
                    learned_candidate_rows=learned_candidate_rows,
                    rule_based_candidate_rows=rule_based_candidate_rows,
                    redesign_candidate_budget=self.autoagent0_cfg.redesign_candidate_budget,
                    learned_source_name=self.current_source_name,
                    learned_default_source=self.learned_default_source,
                    score_fallback_key=self.score_fallback_key,
                    planner_log_name=self.planner_log_name,
                    logger=self.logger,
                    strict_learned_argmax_lookup=self.strict_learned_argmax_lookup,
                    fallback_mode=self.autoagent0_cfg.fallback_mode,
                    max_redesign_attempts=self.autoagent0_cfg.max_redesign_attempts,
                )
            else:
                selection = self.autoagent0_runtime.select_final_actions(
                    frame_index=self.frame_index,
                    camera_images=obs["rgb"],
                    info=info,
                    vlm_selector=self.vlm_selector,
                    scores=scores,
                    learned_candidate_rows=learned_candidate_rows,
                    rule_based_candidate_rows=rule_based_candidate_rows,
                    rule_based_merge_enabled=self.rule_based_merge_cfg.enabled,
                    planner_gate_enabled=vlm_cfg.planner_gate_enabled,
                    vlm_enabled=vlm_cfg.enabled,
                    display_default_trajectories=vlm_cfg.display_default_trajectories,
                    include_default_candidates=vlm_cfg.include_default_candidates,
                    allow_carry_previous=allow_carry_previous,
                    previous_selected_source=self.previous_selected_source,
                    learned_source_name=self.current_source_name,
                    learned_default_source=self.learned_default_source,
                    score_fallback_key=self.score_fallback_key,
                    planner_log_name=self.planner_log_name,
                    logger=self.logger,
                    strict_learned_argmax_lookup=self.strict_learned_argmax_lookup,
                    q_key_prefix=self.q_key_prefix,
                )
            selected_row = selection.selected_row
            selected_plan = selection.selected_plan
            selected_score = selection.selected_score
            selected_score_raw = selection.selected_score_raw
            self.frame_index += 1
            plan_payload = self._build_plan_payload(
                proposals,
                scores,
                output_num_poses,
                selected_idx=None if selection.selected_idx is None else int(selection.selected_idx),
                selected_source=selection.selected_source,
                selection_debug=selection.selection_debug,
                selected_plan_override=selected_plan,
                selected_score_override=selected_score,
                candidate_pool_rows=selection.candidate_rows,
            )

        self.previous_selected_plan = np.asarray(selected_plan, dtype=np.float32).copy()
        self.previous_selected_pose = info_to_pose(info)
        self.previous_selected_score = selected_score_raw
        self.previous_selected_timestamp = float(info.get("timestamp", 0.0))
        self.previous_selected_source = str(
            selected_row.get("source", self.plain_source if not vlm_cfg.enabled else "vlm_selected")
        )
        return plan_payload
