#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import struct
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from planners.common.rule_based_provider import (
    RuleBasedMergeConfig,
    build_rule_based_candidate_rows,
    get_rule_based_proposals_and_scores,
    rule_based_to_hugsim_plan,
)
from planners.common.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from planners.common.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    get_prefixed_env_value,
)


LOG = logging.getLogger("rule_based_adapter")
EGO_HISTORY_FRAMES = 4
DEFAULT_OUTPUT_POSES = 40
TOPK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based FIFO client for HUGSIM")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    return parser.parse_args()


def _coerce_env_value(raw_value, default_value):
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


def resolve_vlm_config() -> VLMSelectorConfig:
    values = {}
    rule_based_python_bin = os.environ.get("RULE_BASED_PYTHON_BIN", "")
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = rule_based_python_bin
        raw_value = get_prefixed_env_value(
            suffix,
            default=default_value,
            prefixes=("PLANNER_VLM_", "RULE_BASED_VLM_"),
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)


def resolve_rule_based_config() -> RuleBasedMergeConfig:
    repo_root = Path(os.environ["RULE_BASED_REPO_ROOT"]).expanduser().resolve()
    python_bin = os.environ.get("RULE_BASED_PYTHON_BIN", "python")
    config_path = Path(os.environ["RULE_BASED_CONFIG"]).expanduser().resolve()
    device = os.environ.get("RULE_BASED_DEVICE", "cpu")
    return RuleBasedMergeConfig(
        enabled=True,
        repo_root=repo_root,
        python_bin=python_bin,
        config_path=config_path,
        device=device,
        topk=TOPK,
        include_privileged_info=True,
        source_name="rule_based",
    )


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rule_based_client.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def read_obs_file(pipe):
    header = pipe.read(8)
    if len(header) != 8:
        raise EOFError("Incomplete pipe header from open obs pipe handle")
    payload_size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < payload_size:
        chunk = pipe.read(payload_size - len(payload))
        if not chunk:
            raise EOFError("Incomplete pipe payload from open obs pipe handle")
        payload.extend(chunk)
    return pickle.loads(payload)


def write_plan_file(pipe, plan) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()


def build_plan_payload(
    proposals: np.ndarray,
    scores: np.ndarray,
    *,
    output_num_poses: int,
    selected_idx: Optional[int],
    selected_source: str,
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int = TOPK,
) -> Dict[str, object]:
    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]
    if selected_idx is None and selected_plan_override is None:
        selected_idx = int(top_indices[0])
    else:
        selected_idx = None if selected_idx is None else int(selected_idx)

    if selected_plan_override is not None:
        selected_plan = np.asarray(selected_plan_override, dtype=np.float32)
        selected_score = float(selected_score_override) if selected_score_override is not None else None
    else:
        assert selected_idx is not None
        selected_traj = proposals[selected_idx, :output_num_poses]
        selected_plan = rule_based_to_hugsim_plan(selected_traj)
        selected_score = float(scores[selected_idx])

    if candidate_pool_rows is not None:
        candidate_pool_plans = [np.asarray(row["local_plan"], dtype=np.float32).tolist() for row in candidate_pool_rows]
        candidate_pool_execution_plans = [
            np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32).tolist()
            for row in candidate_pool_rows
        ]
        candidate_pool_scores = [float(row.get("proposal_score", 0.0)) for row in candidate_pool_rows]
        candidate_pool_q_scores = [None if row.get("q_score") is None else float(row["q_score"]) for row in candidate_pool_rows]
        candidate_pool_sources = [str(row.get("source", "rule_based")) for row in candidate_pool_rows]
        candidate_pool_proposal_indices = [
            None if row.get("proposal_index") is None else int(row["proposal_index"])
            for row in candidate_pool_rows
        ]
    else:
        candidate_pool_plans = [
            rule_based_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = ["rule_based" for _ in top_indices]
        candidate_pool_proposal_indices = [int(idx) for idx in top_indices]

    payload = {
        "selected_idx": selected_idx,
        "selected_score": selected_score,
        "selected_source": selected_source,
        "selected_plan": selected_plan,
        "topk_indices": [int(idx) for idx in top_indices],
        "topk_scores": [float(scores[idx]) for idx in top_indices],
        "topk_plans": [
            rule_based_to_hugsim_plan(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ],
        "candidate_pool_plans": candidate_pool_plans,
        "candidate_pool_execution_plans": candidate_pool_execution_plans,
        "candidate_pool_scores": candidate_pool_scores,
        "candidate_pool_q_scores": candidate_pool_q_scores,
        "candidate_pool_sources": candidate_pool_sources,
        "candidate_pool_proposal_indices": candidate_pool_proposal_indices,
    }
    if selection_debug:
        payload.update(selection_debug)
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    setup_logging(output_dir)
    cfg = resolve_rule_based_config()
    vlm_cfg = resolve_vlm_config()
    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    info_history: deque[Dict[str, object]] = deque(maxlen=EGO_HISTORY_FRAMES)
    vlm_selector = VLMPlanSelector(vlm_cfg, output_dir)
    vlm_selector.preload()
    frame_index = 0

    LOG.info(
        "Starting rule-based planner adapter repo=%s config=%s device=%s",
        cfg.repo_root,
        cfg.config_path,
        cfg.device,
    )

    LOG.info("Opening FIFO handles obs=%s plan=%s", obs_pipe, plan_pipe)
    obs_pipe_reader = os.fdopen(os.open(obs_pipe, os.O_RDWR), "rb", buffering=0)
    plan_pipe_writer = os.fdopen(os.open(plan_pipe, os.O_RDWR), "wb", buffering=0)
    LOG.info("Opened FIFO handles, entering adapter read loop")
    try:
        while True:
            try:
                LOG.info("Waiting for next observation payload on %s", obs_pipe)
                message = read_obs_file(obs_pipe_reader)
                LOG.info("Received observation payload from %s", obs_pipe)
                if message == "Done":
                    LOG.info("Received shutdown signal")
                    break
                if isinstance(message, dict) and message.get("message_type") == "hugsim_preflight":
                    LOG.info("Received preflight diagnostic, waiting for first scene payload")
                    continue

                privileged_info = None
                if isinstance(message, tuple):
                    if len(message) == 3:
                        obs, info, privileged_info = message
                    elif len(message) == 2:
                        obs, info = message
                    else:
                        raise RuntimeError(f"Unexpected rule_based message tuple length: {len(message)}")
                else:
                    raise RuntimeError(f"Unexpected rule_based message type: {type(message)}")

                info_history.append(dict(info))
                while len(info_history) < EGO_HISTORY_FRAMES:
                    info_history.appendleft(dict(info_history[0]))

                proposals, scores, planner_debug = get_rule_based_proposals_and_scores(
                    cfg,
                    obs=obs,
                    info=info,
                    info_history=info_history,
                    privileged_agents=privileged_info,
                    output_num_poses=DEFAULT_OUTPUT_POSES,
                    topk=cfg.topk,
                )
                candidate_rows = build_rule_based_candidate_rows(
                    proposals,
                    scores,
                    output_num_poses=DEFAULT_OUTPUT_POSES,
                    source_name=cfg.source_name,
                    topk=cfg.topk,
                )
                default_selected_index = int(np.argmax(scores)) if len(scores) > 0 else 0

                if not vlm_cfg.enabled:
                    selected_idx = int(np.argmax(scores))
                    selected_plan = rule_based_to_hugsim_plan(proposals[selected_idx, :DEFAULT_OUTPUT_POSES])
                    selected_score = float(scores[selected_idx])
                    selected_source = "rule_based_argmax"
                    selection_debug = {
                        "vlm_invoked": False,
                        "planner_debug": planner_debug,
                    }
                else:
                    selection_result = vlm_selector.maybe_select(
                        frame_index=frame_index,
                        camera_images=obs.get("rgb", {}),
                        info=info,
                        candidate_rows=candidate_rows,
                        default_selected_index=default_selected_index,
                        default_selected_source="rule_based_argmax",
                    )
                    frame_index += 1
                    selected_row = selection_result["selected_candidate_row"]
                    selected_plan = np.asarray(
                        selected_row.get("execution_plan", selected_row["local_plan"]),
                        dtype=np.float32,
                    )
                    selected_idx = selected_row.get("proposal_index")
                    selected_score = float(selected_row.get("proposal_score", 0.0))
                    selected_source = str(selection_result.get("selected_source", "rule_based_vlm"))
                    selection_debug = {
                        "planner_debug": planner_debug,
                        "vlm_selected_idx": selection_result.get("vlm_candidate_index"),
                        "vlm_confidence": selection_result.get("vlm_confidence"),
                        "vlm_reasoning": selection_result.get("vlm_reasoning"),
                        "vlm_elapsed_sec": selection_result.get("vlm_elapsed_sec"),
                        "vlm_error": selection_result.get("vlm_error"),
                        "scoring_invoked": selection_result.get("scoring_invoked"),
                        "intervention_invoked": selection_result.get("intervention_invoked"),
                        "intervention_should_intervene": selection_result.get("intervention_should_intervene"),
                        "intervention_severity_score": selection_result.get("intervention_severity_score"),
                        "intervention_severity_band": selection_result.get("intervention_severity_band"),
                        "intervention_corrective_action": selection_result.get("intervention_corrective_action"),
                        "intervention_confidence": selection_result.get("intervention_confidence"),
                        "intervention_reasoning": selection_result.get("intervention_reasoning"),
                        "intervention_elapsed_sec": selection_result.get("intervention_elapsed_sec"),
                        "intervention_error": selection_result.get("intervention_error"),
                        "adaptive_replan_decision": selection_result.get("adaptive_replan_decision"),
                        "carry_previous_valid": selection_result.get("carry_previous_valid"),
                        "latency_timeline_record": selection_result.get("latency_timeline_record"),
                        "vlm_failed": selection_result.get("vlm_failed"),
                        "fallback_selected_idx": int(default_selected_index),
                        "fallback_selected_source": "rule_based_argmax",
                    }

                plan_payload = build_plan_payload(
                    proposals,
                    scores,
                    output_num_poses=DEFAULT_OUTPUT_POSES,
                    selected_idx=None if selected_idx is None else int(selected_idx),
                    selected_source=selected_source,
                    selection_debug=selection_debug,
                    selected_plan_override=selected_plan,
                    selected_score_override=selected_score,
                    candidate_pool_rows=candidate_rows,
                    topk=cfg.topk,
                )
                write_plan_file(plan_pipe_writer, plan_payload)
            except Exception:
                LOG.error("Rule-based adapter loop failed")
                LOG.error(traceback.format_exc())
                try:
                    write_plan_file(plan_pipe_writer, None)
                except Exception:
                    LOG.error("Failed to notify HUGSIM about planner failure")
                return 1
    finally:
        try:
            obs_pipe_reader.close()
        except Exception:
            pass
        try:
            plan_pipe_writer.close()
        except Exception:
            pass
        try:
            vlm_selector.finalize()
        except Exception:
            LOG.exception("Error finalizing VLM selector")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
