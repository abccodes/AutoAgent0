from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from autoagent0.adapters.hugsim.context import (
    command_to_route_instruction as _aa_command_to_route_instruction,
    describe_task_target_hint as _aa_describe_task_target_hint,
    describe_vlm_camera_inputs as _aa_describe_vlm_camera_inputs,
    extract_current_ego_accel_mps2 as _aa_extract_current_ego_accel_mps2,
    extract_current_ego_speed_mps as _aa_extract_current_ego_speed_mps,
    resolve_route_instruction as _aa_resolve_route_instruction,
    resolve_stage_camera_order as _aa_resolve_stage_camera_order,
    resolve_vlm_camera_order as _aa_resolve_vlm_camera_order,
)
from autoagent0.adapters.hugsim.overlays import (
    PLANNER_GATE_LEARNED_COLOR_BGR,
    PLANNER_GATE_RULE_BASED_COLOR_BGR,
    PLAN_RESAMPLE_SPACING_M,
    PLAN_VIS_FORWARD_OFFSET_M,
    VLM_CAMERA_ORDER,
    render_candidate_overlay as _aa_render_candidate_overlay,
    render_candidate_overlays as _aa_render_candidate_overlays,
    render_planner_gate_overlays as _aa_render_planner_gate_overlays,
)
from autoagent0.scorer.candidates import (
    build_candidate_rows as _aa_build_candidate_rows,
    dedupe_gate_candidates as _aa_dedupe_gate_candidates,
    family_rows_for_planner_gate as _aa_family_rows_for_planner_gate,
    format_candidate_text as _aa_format_candidate_text,
    path_length as _aa_path_length,
    planner_gate_family_debug as _aa_planner_gate_family_debug,
    select_representative_candidate_row as _aa_select_representative_candidate_row,
    summarize_candidate as _aa_summarize_candidate,
)
from autoagent0.vlm.parsing import (
    coerce_critique_result as _aa_coerce_critique_result,
    coerce_candidate_scores as _aa_coerce_candidate_scores,
    coerce_intervention_decision as _aa_coerce_intervention_decision,
    intervention_severity_band as _aa_intervention_severity_band,
    normalize_corrective_action as _aa_normalize_corrective_action,
    select_from_vlm_scores as _aa_select_from_vlm_scores,
    selected_path_reasoning as _aa_selected_path_reasoning,
)
from autoagent0.scorer.agent_trace import build_agent_trace
from autoagent0.prompts.orchestrator import (
    build_intervention_prompt as _aa_build_intervention_prompt,
    build_planner_gate_prompt as _aa_build_planner_gate_prompt,
    build_scoring_prompt as _aa_build_scoring_prompt,
    _summarize_gate_candidates as _aa_summarize_gate_candidates,
)
from autoagent0.prompts.critic import build_critic_prompt as _aa_build_critic_prompt
from autoagent0.prompts.planner import build_final_action_selection_prompt as _aa_build_final_action_selection_prompt
from autoagent0.vlm.backends import (
    Qwen3TrajectorySelector,
    SubprocessQwen3TrajectorySelector,
)
from autoagent0.vlm.debug import append_jsonl, clear_vlm_debug_artifacts
from autoagent0.vlm.parsing import (
    empty_token_usage as _aa_empty_token_usage,
    normalize_token_usage as _aa_normalize_token_usage,
    try_parse_json as _aa_try_parse_json,
)

LOG = logging.getLogger(__name__)
PLAN_DT_SEC = 0.5


@dataclass
class VLMSelectorConfig:
    enabled: bool = False
    intervention_enabled: bool = False
    camera_mode: str = "multiview"
    intervention_camera_mode: str = ""
    scoring_camera_mode: str = ""
    backend: str = "local_transformers"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    python_bin: str = ""
    max_new_tokens: int = 300
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    enable_thinking: bool = False
    candidate_limit: int = 10
    intervention_max_new_tokens: int = 120
    timeout_sec: float = 10.0
    intervention_timeout_sec: float = 10.0
    intervention_action_threshold: float = 0.65
    intervention_high_threshold: float = 0.85
    preload_on_init: bool = True
    save_debug_artifacts: bool = True
    debug_dir_name: str = "vlm_debug"
    carry_previous_enabled: bool = True
    carry_previous_min_path_m: float = 0.5
    carry_previous_min_points: int = 2
    planner_gate_enabled: bool = False
    planner_gate_camera_mode: str = ""
    planner_gate_max_new_tokens: int = 120
    planner_gate_timeout_sec: float = 10.0
    planner_gate_default_planner: str = "learned"
    planner_gate_save_debug_artifacts: bool = True
    planner_gate_prompt_style: str = "default"
    adaptive_replan_mode: str = "log_only"
    latency_tracking_mode: str = "full_timeline"
    q_enabled: bool = True
    q_switch_margin: float = 0.05
    q_weight_rap_score: float = 0.55
    q_weight_progress: float = 0.30
    q_weight_offcenter: float = 0.10
    q_weight_curvature: float = 0.08
    q_weight_shortplan: float = 0.18
    q_carry_score_decay: float = 0.0
    display_default_trajectories: bool = False
    include_default_candidates: bool = False


# Compatibility facade: re-export shared selection helpers resolved through the
# AutoAgent0 modules so callers can import them from this single entry point.
summarize_candidate = _aa_summarize_candidate
path_length = _aa_path_length
format_candidate_text = _aa_format_candidate_text
resolve_vlm_camera_order = _aa_resolve_vlm_camera_order
describe_vlm_camera_inputs = _aa_describe_vlm_camera_inputs
resolve_stage_camera_order = _aa_resolve_stage_camera_order
build_scoring_prompt = _aa_build_scoring_prompt
build_intervention_prompt = _aa_build_intervention_prompt
_summarize_gate_candidates = _aa_summarize_gate_candidates
_planner_gate_family_debug = _aa_planner_gate_family_debug
build_planner_gate_prompt = _aa_build_planner_gate_prompt
command_to_route_instruction = _aa_command_to_route_instruction
resolve_route_instruction = _aa_resolve_route_instruction
describe_task_target_hint = _aa_describe_task_target_hint
normalize_corrective_action = _aa_normalize_corrective_action
extract_current_ego_speed_mps = _aa_extract_current_ego_speed_mps
extract_current_ego_accel_mps2 = _aa_extract_current_ego_accel_mps2
build_candidate_rows = _aa_build_candidate_rows
_select_representative_candidate_row = _aa_select_representative_candidate_row
_family_rows_for_planner_gate = _aa_family_rows_for_planner_gate
_dedupe_gate_candidates = _aa_dedupe_gate_candidates
_coerce_candidate_scores = _aa_coerce_candidate_scores
_coerce_critique_result = _aa_coerce_critique_result
_intervention_severity_band = _aa_intervention_severity_band
_coerce_intervention_decision = _aa_coerce_intervention_decision
_select_from_vlm_scores = _aa_select_from_vlm_scores
_selected_path_reasoning = _aa_selected_path_reasoning
render_candidate_overlay = _aa_render_candidate_overlay
render_candidate_overlays = _aa_render_candidate_overlays
render_planner_gate_overlays = _aa_render_planner_gate_overlays
try_parse_json = _aa_try_parse_json
_empty_token_usage = _aa_empty_token_usage
_normalize_token_usage = _aa_normalize_token_usage


class VLMPlanSelector:
    def __init__(self, cfg: VLMSelectorConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.debug_dir = output_dir / cfg.debug_dir_name
        self.timeline_path = self.debug_dir / "latency_timeline.jsonl"
        self.summary_path = self.debug_dir / "latency_summary.json"
        self._selector: Optional[object] = None
        self._disabled_reason: Optional[str] = None
        self._timeline_records: List[Dict[str, object]] = []
        self._planner_gate_records: List[Dict[str, object]] = []

        if cfg.save_debug_artifacts:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.timeline_path.unlink(missing_ok=True)
            self.summary_path.unlink(missing_ok=True)
            clear_vlm_debug_artifacts(self.debug_dir)

    def _record_timeline(self, record: Dict[str, object]) -> None:
        self._timeline_records.append(record)
        if self.cfg.save_debug_artifacts:
            append_jsonl(self.timeline_path, record)

    def _record_planner_gate(self, record: Dict[str, object]) -> None:
        self._planner_gate_records.append(record)

    def _normalized_backend(self) -> str:
        backend = str(self.cfg.backend or "").strip()
        if backend in {"subprocess_qwen3_vl", "qwen3_vl_subprocess"}:
            return "local_transformers_subprocess"
        if backend in {"qwen3_vl", "local_qwen3_vl"}:
            return "local_transformers"
        return backend

    def _ensure_selector(self) -> Optional[object]:
        if self._selector is not None:
            return self._selector
        if self._disabled_reason is not None:
            return None
        backend = self._normalized_backend()
        if backend == "local_transformers":
            selector_factory = lambda: Qwen3TrajectorySelector(
                model_id=self.cfg.model_id,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
            )
        elif backend == "local_transformers_subprocess":
            selector_factory = lambda: SubprocessQwen3TrajectorySelector(
                python_bin=self.cfg.python_bin,
                # vlm_worker.py lives in autoagent0/vlm/; this file is in autoagent0/core/.
                # Resolve the sibling-package path without importing the module (its
                # top-level `transformers` import must stay out of this process).
                worker_script=Path(__file__).resolve().parent.parent / "vlm" / "vlm_worker.py",
                model_id=self.cfg.model_id,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                enable_thinking=self.cfg.enable_thinking,
            )
        else:
            self._disabled_reason = f"unsupported_backend:{self.cfg.backend}"
            return None
        try:
            self._selector = selector_factory()
            LOG.info(
                "Initialized VLM selector backend=%s normalized_backend=%s model=%s device=%s",
                self.cfg.backend,
                backend,
                self.cfg.model_id,
                getattr(self._selector, "device", self.cfg.device),
            )
            return self._selector
        except Exception as exc:
            self._disabled_reason = str(exc)
            LOG.exception("Failed to initialize VLM selector, disabling VLM fallback path")
            return None

    def preload(self) -> None:
        if not self.cfg.enabled or self._disabled_reason is not None:
            return
        selector = self._ensure_selector()
        if selector is None or not self.cfg.preload_on_init:
            return
        preload_fn = getattr(selector, "preload", None)
        if not callable(preload_fn):
            return
        LOG.info(
            "Preloading VLM selector backend=%s normalized_backend=%s model=%s timeout=%.3fs",
            self.cfg.backend,
            self._normalized_backend(),
            self.cfg.model_id,
            float(self.cfg.timeout_sec),
        )
        preload_fn(timeout_sec=self.cfg.timeout_sec)
        LOG.info("VLM selector preload complete model=%s", self.cfg.model_id)

    def _write_stage_overlays(
        self,
        *,
        frame_stem: str,
        suffix: str,
        overlays: Dict[str, np.ndarray],
        camera_order: Sequence[str],
        temp_paths: List[Path],
    ) -> List[Path]:
        image_paths: List[Path] = []
        for cam_name in camera_order:
            overlay = overlays.get(cam_name)
            if overlay is None:
                continue
            if self.cfg.save_debug_artifacts:
                image_path = self.debug_dir / f"{frame_stem}_{suffix}_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_{suffix}_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)
        return image_paths

    def critique_autoagent0_candidate(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        candidate_row: Dict[str, object],
        stage: str,
        previous_feedback: Optional[str] = None,
    ) -> Dict[str, object]:
        route_instruction = resolve_route_instruction(info)
        task_target_hint = describe_task_target_hint(info)
        camera_order = resolve_stage_camera_order(self.cfg, "intervention")
        timestamp = float(info.get("timestamp", 0.0))
        candidate_rows = build_candidate_rows([candidate_row], current_ego_speed_mps=extract_current_ego_speed_mps(info))

        def _fallback_result(error: Optional[str]) -> Dict[str, object]:
            result = {
                "stage": stage,
                "autoagent0_critique_accepted": True,
                "autoagent0_critique_rejected": False,
                "autoagent0_critique_severity_score": 0.0,
                "autoagent0_critique_corrective_action": "straight",
                "autoagent0_critique_confidence": None,
                "autoagent0_critique_reasoning": "AutoAgent0 critic unavailable; keeping current default behavior.",
                "autoagent0_critique_error": error,
                "autoagent0_critique_elapsed_sec": 0.0,
                "autoagent0_critique_token_usage": _empty_token_usage(),
                "error": error,
            }
            timeline_record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "execution_mode": f"autoagent0_{stage}_critic_unavailable",
                "candidate_count": 1,
                "selected_source": candidate_rows[0].get("source"),
                "autoagent0_stage": stage,
                "autoagent0_tool": "request_critique",
                "autoagent0_critique_accepted": True,
                "autoagent0_critique_error": error,
                "vlm_elapsed_sec": 0.0,
                "scoring_invoked": False,
                "q_invoked_vlm": False,
                "vlm_failed": True,
                "error": error,
            }
            self._record_timeline(timeline_record)
            result["latency_timeline_record"] = timeline_record
            return result

        if not self.cfg.enabled:
            return _fallback_result("vlm_disabled")
        selector = self._ensure_selector()
        if selector is None:
            return _fallback_result(self._disabled_reason or "selector_unavailable")

        frame_stem = f"frame_{frame_index:04d}"
        temp_paths: List[Path] = []
        overlays = render_candidate_overlays(
            camera_images,
            info,
            candidate_rows,
            camera_order=camera_order,
        )
        image_paths = self._write_stage_overlays(
            frame_stem=frame_stem,
            suffix=f"autoagent0_{stage}_critic",
            overlays=overlays,
            camera_order=camera_order,
            temp_paths=temp_paths,
        )
        prompt = _aa_build_critic_prompt(
            candidate_rows[0],
            route_instruction,
            task_target_hint=task_target_hint,
            previous_feedback=previous_feedback,
            camera_order=camera_order,
        )
        try:
            inference_result = selector.infer_prompt(
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=self.cfg.intervention_max_new_tokens,
                timeout_sec=self.cfg.intervention_timeout_sec,
            )
        except Exception as exc:
            LOG.exception("AutoAgent0 critic failed")
            inference_result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": prompt,
                "error": str(exc),
                "token_usage": _empty_token_usage(),
            }

        elapsed_sec = float(inference_result.get("elapsed_sec", 0.0))
        token_usage = _normalize_token_usage(inference_result.get("token_usage"))
        critique = _coerce_critique_result(inference_result.get("parsed_output"))
        error = critique.error
        if elapsed_sec > self.cfg.intervention_timeout_sec:
            error = f"autoagent0_critic_timeout:{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))
        accepted = bool(critique.accepted) if error is None else True

        timeline_record = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "route_instruction": route_instruction,
            "execution_mode": f"autoagent0_{stage}_critic",
            "candidate_count": 1,
            "selected_source": candidate_rows[0].get("source"),
            "autoagent0_stage": stage,
            "autoagent0_tool": "request_critique",
            "autoagent0_critique_accepted": accepted,
            "autoagent0_critique_severity_score": critique.severity_score,
            "autoagent0_critique_corrective_action": critique.corrective_action,
            "autoagent0_critique_confidence": critique.confidence,
            "autoagent0_critique_elapsed_sec": elapsed_sec,
            "autoagent0_critique_prompt_tokens": token_usage["prompt_tokens"],
            "autoagent0_critique_completion_tokens": token_usage["completion_tokens"],
            "autoagent0_critique_total_tokens": token_usage["total_tokens"],
            "vlm_elapsed_sec": elapsed_sec,
            "scoring_invoked": False,
            "q_invoked_vlm": True,
            "vlm_failed": error is not None,
            "error": error,
        }
        self._record_timeline(timeline_record)
        result = {
            "stage": stage,
            "autoagent0_critique_accepted": accepted,
            "autoagent0_critique_rejected": not accepted,
            "autoagent0_critique_severity_score": critique.severity_score,
            "autoagent0_critique_corrective_action": critique.corrective_action,
            "autoagent0_critique_confidence": critique.confidence,
            "autoagent0_critique_reasoning": critique.reasoning,
            "autoagent0_critique_error": error,
            "autoagent0_critique_elapsed_sec": elapsed_sec,
            "autoagent0_critique_token_usage": token_usage,
            "autoagent0_critique_result": inference_result,
            "latency_timeline_record": timeline_record,
            "error": error,
        }
        if self.cfg.save_debug_artifacts:
            result_path = self.debug_dir / f"{frame_stem}_autoagent0_{stage}_critique.json"
            result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        else:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
        return result

    def score_autoagent0_candidates(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
        critique_reason: Optional[str] = None,
        corrective_action: Optional[str] = None,
        stage: str = "revised",
    ) -> Dict[str, object]:
        route_instruction = resolve_route_instruction(info)
        task_target_hint = describe_task_target_hint(info)
        camera_order = resolve_stage_camera_order(self.cfg, "scoring")
        timestamp = float(info.get("timestamp", 0.0))
        candidate_rows = build_candidate_rows(candidate_rows, current_ego_speed_mps=extract_current_ego_speed_mps(info))

        def _fallback_result(error: Optional[str]) -> Dict[str, object]:
            selected_row = dict(candidate_rows[default_selected_index])
            timeline_record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "execution_mode": f"autoagent0_{stage}_scoring_unavailable",
                "candidate_count": len(candidate_rows),
                "selected_source": default_selected_source,
                "selected_candidate_index": default_selected_index,
                "autoagent0_stage": stage,
                "autoagent0_tool": "select_final_actions",
                "vlm_elapsed_sec": 0.0,
                "scoring_invoked": False,
                "q_invoked_vlm": False,
                "vlm_failed": True,
                "error": error,
            }
            self._record_timeline(timeline_record)
            return {
                "selected_candidate_row": selected_row,
                "selected_source": default_selected_source,
                "selected_path_reasoning": "AutoAgent0 scorer unavailable; using default candidate.",
                "execution_mode": f"autoagent0_{stage}_scoring_unavailable",
                "vlm_candidate_index": None,
                "vlm_confidence": None,
                "vlm_reasoning": None,
                "vlm_elapsed_sec": 0.0,
                "vlm_error": error,
                "vlm_candidate_count": len(candidate_rows),
                "vlm_q_valid": False,
                "vlm_q_candidate_scores": None,
                "vlm_q_best_candidate_index": None,
                "vlm_q_score_gap_top2": None,
                "adaptive_replan_decision": "autoagent0_scorer_failed_default",
                "latency_timeline_record": timeline_record,
                "vlm_failed": True,
                "scoring_invoked": False,
                "scoring_token_usage": _empty_token_usage(),
                "error": error,
            }

        if not candidate_rows:
            raise ValueError("AutoAgent0 scoring requires at least one candidate")
        if not self.cfg.enabled:
            return _fallback_result("vlm_disabled")
        selector = self._ensure_selector()
        if selector is None:
            return _fallback_result(self._disabled_reason or "selector_unavailable")

        frame_stem = f"frame_{frame_index:04d}"
        temp_paths: List[Path] = []
        overlays = render_candidate_overlays(
            camera_images,
            info,
            candidate_rows,
            camera_order=camera_order,
        )
        image_paths = self._write_stage_overlays(
            frame_stem=frame_stem,
            suffix=f"autoagent0_{stage}_candidates",
            overlays=overlays,
            camera_order=camera_order,
            temp_paths=temp_paths,
        )
        prompt = _aa_build_final_action_selection_prompt(
            candidate_rows,
            route_instruction,
            critique_reason=critique_reason,
            corrective_action=corrective_action,
            task_target_hint=task_target_hint,
            camera_order=camera_order,
        )
        try:
            inference_result = selector.infer_prompt(
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                timeout_sec=self.cfg.timeout_sec,
            )
        except Exception as exc:
            self._disabled_reason = f"selector_runtime_error:{exc}"
            LOG.exception("AutoAgent0 scorer failed")
            inference_result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": prompt,
                "error": str(exc),
                "token_usage": _empty_token_usage(),
            }

        parsed = inference_result.get("parsed_output")
        elapsed_sec = float(inference_result.get("elapsed_sec", 0.0))
        token_usage = _normalize_token_usage(inference_result.get("token_usage"))
        selected_candidate_index = int(default_selected_index)
        selected_row = dict(candidate_rows[default_selected_index])
        selected_source = default_selected_source
        vlm_confidence = None
        vlm_reasoning = None
        vlm_q_valid = False
        vlm_q_candidate_scores = None
        vlm_q_best_candidate_index = None
        vlm_q_score_gap_top2 = None
        error = None
        if isinstance(parsed, dict):
            coerced_scores, score_error = _coerce_candidate_scores(parsed.get("candidate_scores"), len(candidate_rows))
            if coerced_scores is None:
                error = score_error or "invalid_candidate_scores"
            elif elapsed_sec > self.cfg.timeout_sec:
                error = f"autoagent0_scorer_timeout:{elapsed_sec:.3f}"
            else:
                vlm_q_valid = True
                vlm_q_candidate_scores = [float(score) for score in coerced_scores]
                vlm_q_best_candidate_index = int(max(range(len(vlm_q_candidate_scores)), key=lambda idx: vlm_q_candidate_scores[idx]))
                selection = _select_from_vlm_scores(candidate_rows=candidate_rows, vlm_scores=coerced_scores)
                selected_candidate_index = int(selection["selected_candidate_index"])
                selected_row = dict(selection["selected_candidate_row"])
                selected_source = str(selection["selected_source"])
                vlm_q_score_gap_top2 = selection["vlm_q_score_gap_top2"]
                vlm_confidence = float(parsed.get("confidence", 0.0))
                vlm_reasoning = parsed.get("reasoning")
        else:
            error = (
                f"autoagent0_scorer_timeout:{elapsed_sec:.3f}"
                if elapsed_sec > self.cfg.timeout_sec
                else str(inference_result.get("error") or "invalid_selector_output")
            )

        selected_path_reasoning = _selected_path_reasoning(
            selected_row=selected_row,
            selected_candidate_index=selected_candidate_index,
            selected_source=selected_source,
            vlm_scores=vlm_q_candidate_scores,
            parsed_reasoning=vlm_reasoning,
        )
        timeline_record = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "route_instruction": route_instruction,
            "execution_mode": f"autoagent0_{stage}_scoring",
            "candidate_count": len(candidate_rows),
            "selected_source": selected_source,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_source": selected_row.get("source"),
            "selected_proposal_index": selected_row.get("proposal_index"),
            "autoagent0_stage": stage,
            "autoagent0_tool": "select_final_actions",
            "vlm_elapsed_sec": elapsed_sec,
            "scoring_prompt_tokens": token_usage["prompt_tokens"],
            "scoring_completion_tokens": token_usage["completion_tokens"],
            "scoring_total_tokens": token_usage["total_tokens"],
            "scoring_invoked": True,
            "vlm_q_valid": vlm_q_valid,
            "vlm_failed": error is not None,
            "error": error,
        }
        self._record_timeline(timeline_record)
        result = {
            "selected_candidate_row": selected_row,
            "selected_source": selected_source,
            "selected_path_reasoning": selected_path_reasoning,
            "execution_mode": f"autoagent0_{stage}_scoring",
            "vlm_candidate_index": selected_candidate_index if error is None else None,
            "vlm_confidence": vlm_confidence,
            "vlm_reasoning": vlm_reasoning,
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_error": error,
            "vlm_candidate_count": len(candidate_rows),
            "vlm_q_valid": vlm_q_valid,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "adaptive_replan_decision": f"autoagent0_{stage}_scoring",
            "latency_timeline_record": timeline_record,
            "vlm_failed": error is not None,
            "scoring_invoked": True,
            "scoring_token_usage": token_usage,
            "scoring_result": inference_result,
            "error": error,
        }
        if self.cfg.save_debug_artifacts:
            result_path = self.debug_dir / f"{frame_stem}_autoagent0_{stage}_scoring.json"
            result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        else:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
        return result

    def maybe_select(
        self,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
        force_scoring: bool = False,
        intervention_corrective_action_override: Optional[str] = None,
        execution_mode_label: Optional[str] = None,
    ) -> Dict[str, object]:
        route_instruction = resolve_route_instruction(info)
        task_target_hint = describe_task_target_hint(info)
        scoring_camera_order = resolve_stage_camera_order(self.cfg, "scoring")
        intervention_camera_order = resolve_stage_camera_order(self.cfg, "intervention")
        timestamp = float(info.get("timestamp", 0.0))
        dt_sec = 0.25
        current_ego_speed_mps = None
        current_ego_accel_mps2 = None
        try:
            current_ego_speed_mps = float(info["ego_velo"])
        except Exception:
            current_ego_speed_mps = None
        try:
            current_ego_accel_mps2 = float(info["accelerate"])
        except Exception:
            current_ego_accel_mps2 = None
        carry_previous_valid = any(row.get("source") == "carry_prev" for row in candidate_rows)
        candidate_rows = build_candidate_rows(candidate_rows, current_ego_speed_mps=current_ego_speed_mps)
        scoring_invoked = False
        intervention_invoked = False
        intervention_should_intervene = None
        intervention_severity_score = None
        intervention_severity_band = None
        intervention_corrective_action = None
        intervention_confidence = None
        intervention_reasoning = None
        intervention_elapsed_sec = 0.0
        intervention_error = None
        intervention_token_usage = _empty_token_usage()

        def _fallback_result(error: Optional[str] = None) -> Dict[str, object]:
            selected_row = dict(candidate_rows[default_selected_index])
            adaptive_replan_decision = "vlm_failed_fallback_rap"
            execution_mode = "base_policy_vlm_unavailable"
            timeline_record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "execution_mode": execution_mode,
                "candidate_count": len(candidate_rows),
                "carry_previous_valid": carry_previous_valid,
                "selected_source": default_selected_source,
                "selected_candidate_index": default_selected_index,
                "selected_candidate_source": selected_row.get("source"),
                "selected_proposal_index": selected_row.get("proposal_index"),
                "intervention_invoked": False,
                "intervention_should_intervene": None,
                "intervention_severity_score": None,
                "intervention_severity_band": None,
                "intervention_corrective_action": None,
                "intervention_confidence": None,
                "intervention_elapsed_sec": 0.0,
                "intervention_prompt_tokens": 0,
                "intervention_completion_tokens": 0,
                "intervention_total_tokens": 0,
                "vlm_elapsed_sec": 0.0,
                "scoring_prompt_tokens": 0,
                "scoring_completion_tokens": 0,
                "scoring_total_tokens": 0,
                "scoring_invoked": False,
                "latency_equivalent_steps": 0.0,
                "latency_equivalent_steps_ceil": 0,
                "adaptive_replan_decision": adaptive_replan_decision,
                "error": error,
                "q_invoked_vlm": False,
                "vlm_q_valid": False,
                "vlm_failed": True,
            }
            self._record_timeline(timeline_record)
            result = {
                "selected_candidate_row": selected_row,
                "selected_source": default_selected_source,
                "adaptive_replan_decision": adaptive_replan_decision,
                "execution_mode": execution_mode,
                "carry_previous_valid": carry_previous_valid,
                "latency_timeline_record": timeline_record,
                "vlm_q_valid": False,
                "vlm_failed": True,
                "scoring_invoked": False,
                "intervention_invoked": False,
                "intervention_should_intervene": None,
                "intervention_severity_score": None,
                "intervention_severity_band": None,
                "intervention_corrective_action": None,
                "intervention_confidence": None,
                "intervention_reasoning": None,
                "intervention_elapsed_sec": 0.0,
                "intervention_token_usage": _empty_token_usage(),
                "scoring_token_usage": _empty_token_usage(),
            }
            if error is not None:
                result["error"] = error
            return result

        if not self.cfg.enabled:
            return _fallback_result("vlm_disabled")

        selector = self._ensure_selector()
        if selector is None:
            return _fallback_result(self._disabled_reason or "selector_unavailable")

        frame_stem = f"frame_{frame_index:04d}"
        result_path = self.debug_dir / f"{frame_stem}_result.json"
        temp_paths: List[Path] = []

        def _write_overlay_bundle(
            suffix: str,
            overlays: Dict[str, np.ndarray],
            camera_order: Sequence[str],
        ) -> List[Path]:
            image_paths: List[Path] = []
            for cam_name in camera_order:
                overlay = overlays.get(cam_name)
                if overlay is None:
                    continue
                if self.cfg.save_debug_artifacts:
                    image_path = self.debug_dir / f"{frame_stem}_{suffix}_{cam_name}.jpg"
                else:
                    image_path = self.output_dir / f"{frame_stem}_{suffix}_{cam_name}_tmp.jpg"
                    temp_paths.append(image_path)
                cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                image_paths.append(image_path)
            return image_paths

        score_overlays = render_candidate_overlays(
            camera_images,
            info,
            candidate_rows,
            camera_order=scoring_camera_order,
        )
        score_image_paths = _write_overlay_bundle("candidates", score_overlays, scoring_camera_order)

        intervention_result = None
        if self.cfg.intervention_enabled and not force_scoring:
            intervention_invoked = True
            gate_overlays = render_candidate_overlays(
                camera_images,
                info,
                [candidate_rows[default_selected_index]],
                camera_order=intervention_camera_order,
            )
            gate_image_paths = _write_overlay_bundle("gate", gate_overlays, intervention_camera_order)
            intervention_prompt = build_intervention_prompt(
                candidate_rows[default_selected_index],
                route_instruction,
                task_target_hint=task_target_hint,
                camera_order=intervention_camera_order,
            )
            try:
                intervention_result = selector.infer_prompt(
                    image_paths=gate_image_paths,
                    prompt=intervention_prompt,
                    max_new_tokens=self.cfg.intervention_max_new_tokens,
                    timeout_sec=self.cfg.intervention_timeout_sec,
                )
            except Exception as exc:
                LOG.exception("VLM intervention gate failed, falling back to RAP argmax")
                intervention_result = {
                    "raw_output": "",
                    "parsed_output": None,
                    "elapsed_sec": 0.0,
                    "prompt": intervention_prompt,
                    "error": str(exc),
                    "token_usage": _empty_token_usage(),
                }

            intervention_elapsed_sec = float(intervention_result.get("elapsed_sec", 0.0))
            intervention_token_usage = _normalize_token_usage(intervention_result.get("token_usage"))
            intervention_should_intervene, intervention_severity_score, intervention_corrective_action, intervention_confidence, intervention_reasoning, intervention_parse_error = (
                _coerce_intervention_decision(intervention_result.get("parsed_output"))
            )
            intervention_severity_band = _intervention_severity_band(
                intervention_severity_score,
                action_threshold=float(self.cfg.intervention_action_threshold),
                high_threshold=float(self.cfg.intervention_high_threshold),
            )
            if intervention_elapsed_sec > self.cfg.intervention_timeout_sec:
                intervention_error = f"intervention_timeout_fallback_rap:{intervention_elapsed_sec:.3f}"
            elif intervention_parse_error is not None:
                intervention_error = intervention_parse_error
            elif intervention_result.get("error"):
                intervention_error = str(intervention_result.get("error"))

            if intervention_error is not None:
                selected_row = dict(candidate_rows[default_selected_index])
                selected_source = "gate_failed_fallback_rap"
                adaptive_replan_decision = "gate_failed_fallback_rap"
                execution_mode = "gate_failed_base_policy_fallback"
                selected_path_reasoning = (
                    intervention_reasoning.strip()
                    if isinstance(intervention_reasoning, str) and intervention_reasoning.strip()
                    else "Intervention gate failed; using baseline RAP selection."
                )
                timeline_record = {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "route_instruction": route_instruction,
                    "execution_mode": execution_mode,
                    "candidate_count": len(candidate_rows),
                    "carry_previous_valid": carry_previous_valid,
                    "carry_previous_remaining_path_m": next(
                        (path_length(np.asarray(row["local_plan"], dtype=np.float32)) for row in candidate_rows if row.get("source") == "carry_prev"),
                        0.0,
                    ),
                    "selected_source": selected_source,
                    "selected_candidate_index": default_selected_index,
                    "selected_candidate_source": selected_row.get("source"),
                    "selected_proposal_index": selected_row.get("proposal_index"),
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_prompt_tokens": intervention_token_usage["prompt_tokens"],
                    "intervention_completion_tokens": intervention_token_usage["completion_tokens"],
                    "intervention_total_tokens": intervention_token_usage["total_tokens"],
                    "vlm_elapsed_sec": 0.0,
                    "scoring_prompt_tokens": 0,
                    "scoring_completion_tokens": 0,
                    "scoring_total_tokens": 0,
                    "scoring_invoked": False,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "latency_equivalent_steps": 0.0,
                    "latency_equivalent_steps_ceil": 0,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "error": intervention_error,
                    "q_invoked_vlm": False,
                    "vlm_failed": True,
                }
                self._record_timeline(timeline_record)
                debug_payload = {
                    "frame_index": frame_index,
                    "route_instruction": route_instruction,
                    "default_selected_index": int(default_selected_index),
                    "default_selected_source": default_selected_source,
                    "candidate_rows": candidate_rows,
                    "intervention_result": intervention_result,
                    "scoring_result": None,
                    "scoring_invoked": False,
                    "selected_index": int(default_selected_index),
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "execution_mode": execution_mode,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_error": intervention_error,
                    "intervention_token_usage": intervention_token_usage,
                    "scoring_token_usage": _empty_token_usage(),
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "error": intervention_error,
                    "vlm_failed": True,
                    "agent_trace": build_agent_trace(
                        frame_index=frame_index,
                        route_instruction=route_instruction,
                        info=info,
                        candidate_rows=candidate_rows,
                        decision_type="intervention_gate",
                        selected_source=selected_source,
                        selected_candidate_index=int(default_selected_index),
                        confidence=intervention_confidence,
                        reasoning=selected_path_reasoning,
                    ),
                }
                if self.cfg.save_debug_artifacts:
                    result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
                if not self.cfg.save_debug_artifacts:
                    for temp_path in temp_paths:
                        temp_path.unlink(missing_ok=True)
                return {
                    "selected_candidate_row": selected_row,
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "execution_mode": execution_mode,
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_reasoning": None,
                    "vlm_elapsed_sec": 0.0,
                    "vlm_error": intervention_error,
                    "vlm_candidate_count": len(candidate_rows),
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "vlm_failed": True,
                    "scoring_invoked": False,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_token_usage": intervention_token_usage,
                    "scoring_token_usage": _empty_token_usage(),
                }

            if intervention_should_intervene is False:
                selected_row = dict(candidate_rows[default_selected_index])
                selected_source = default_selected_source
                adaptive_replan_decision = "gate_no_intervention_use_rap"
                execution_mode = "base_policy_no_intervention"
                selected_path_reasoning = (
                    intervention_reasoning.strip()
                    if isinstance(intervention_reasoning, str) and intervention_reasoning.strip()
                    else "Intervention gate judged the baseline trajectory sufficient; keeping base policy."
                )
                timeline_record = {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "route_instruction": route_instruction,
                    "execution_mode": execution_mode,
                    "candidate_count": len(candidate_rows),
                    "carry_previous_valid": carry_previous_valid,
                    "carry_previous_remaining_path_m": next(
                        (path_length(np.asarray(row["local_plan"], dtype=np.float32)) for row in candidate_rows if row.get("source") == "carry_prev"),
                        0.0,
                    ),
                    "selected_source": selected_source,
                    "selected_candidate_index": default_selected_index,
                    "selected_candidate_source": selected_row.get("source"),
                    "selected_proposal_index": selected_row.get("proposal_index"),
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_prompt_tokens": intervention_token_usage["prompt_tokens"],
                    "intervention_completion_tokens": intervention_token_usage["completion_tokens"],
                    "intervention_total_tokens": intervention_token_usage["total_tokens"],
                    "vlm_elapsed_sec": 0.0,
                    "scoring_prompt_tokens": 0,
                    "scoring_completion_tokens": 0,
                    "scoring_total_tokens": 0,
                    "scoring_invoked": False,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "latency_equivalent_steps": 0.0,
                    "latency_equivalent_steps_ceil": 0,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "error": None,
                    "q_invoked_vlm": False,
                    "vlm_failed": False,
                }
                self._record_timeline(timeline_record)
                debug_payload = {
                    "frame_index": frame_index,
                    "route_instruction": route_instruction,
                    "default_selected_index": int(default_selected_index),
                    "default_selected_source": default_selected_source,
                    "candidate_rows": candidate_rows,
                    "intervention_result": intervention_result,
                    "scoring_result": None,
                    "scoring_invoked": False,
                    "selected_index": int(default_selected_index),
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "execution_mode": execution_mode,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_error": None,
                    "intervention_token_usage": intervention_token_usage,
                    "scoring_token_usage": _empty_token_usage(),
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "error": None,
                    "vlm_failed": False,
                    "agent_trace": build_agent_trace(
                        frame_index=frame_index,
                        route_instruction=route_instruction,
                        info=info,
                        candidate_rows=candidate_rows,
                        decision_type="intervention_gate",
                        selected_source=selected_source,
                        selected_candidate_index=int(default_selected_index),
                        confidence=intervention_confidence,
                        reasoning=selected_path_reasoning,
                    ),
                }
                if self.cfg.save_debug_artifacts:
                    result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
                if not self.cfg.save_debug_artifacts:
                    for temp_path in temp_paths:
                        temp_path.unlink(missing_ok=True)
                return {
                    "selected_candidate_row": selected_row,
                    "selected_source": selected_source,
                    "selected_path_reasoning": selected_path_reasoning,
                    "execution_mode": execution_mode,
                    "vlm_candidate_index": None,
                    "vlm_confidence": None,
                    "vlm_reasoning": None,
                    "vlm_elapsed_sec": 0.0,
                    "vlm_error": None,
                    "vlm_candidate_count": len(candidate_rows),
                    "vlm_q_valid": False,
                    "vlm_timed_out": False,
                    "vlm_q_candidate_scores": None,
                    "vlm_q_best_candidate_index": None,
                    "vlm_q_score_gap_to_carry": None,
                    "vlm_q_score_gap_top2": None,
                    "vlm_q_best_current_score": None,
                    "vlm_q_carry_score": None,
                    "adaptive_replan_decision": adaptive_replan_decision,
                    "carry_previous_valid": carry_previous_valid,
                    "latency_timeline_record": timeline_record,
                    "vlm_failed": False,
                    "scoring_invoked": False,
                    "intervention_invoked": True,
                    "intervention_should_intervene": intervention_should_intervene,
                    "intervention_severity_score": intervention_severity_score,
                    "intervention_severity_band": intervention_severity_band,
                    "intervention_corrective_action": intervention_corrective_action,
                    "intervention_confidence": intervention_confidence,
                    "intervention_reasoning": intervention_reasoning,
                    "intervention_elapsed_sec": intervention_elapsed_sec,
                    "intervention_error": None,
                    "intervention_token_usage": intervention_token_usage,
                    "scoring_token_usage": _empty_token_usage(),
                }

        scoring_route_instruction = route_instruction
        intervention_action_for_scoring = (
            intervention_corrective_action_override
            if intervention_corrective_action_override is not None
            else (
                intervention_corrective_action
                if self.cfg.intervention_enabled
                and intervention_should_intervene is True
                and intervention_severity_score is not None
                and intervention_severity_score >= float(self.cfg.intervention_action_threshold)
                else None
            )
        )

        try:
            LOG.info(
                "Running VLM selection for frame=%d candidates=%d route='%s' corrective_action='%s' intervention_score='%.3f' intervention_band='%s'",
                frame_index,
                len(candidate_rows),
                scoring_route_instruction,
                intervention_action_for_scoring,
                -1.0 if intervention_severity_score is None else intervention_severity_score,
                intervention_severity_band,
            )
            scoring_invoked = True
            scoring_prompt = build_scoring_prompt(
                candidate_rows,
                scoring_route_instruction,
                task_target_hint=task_target_hint,
                intervention_corrective_action=(
                    intervention_action_for_scoring
                ),
                current_ego_speed_mps=current_ego_speed_mps,
                current_ego_accel_mps2=current_ego_accel_mps2,
                camera_order=scoring_camera_order,
            )
            result = selector.infer_prompt(
                image_paths=score_image_paths,
                prompt=scoring_prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                timeout_sec=self.cfg.timeout_sec,
            )
        except Exception as exc:
            self._disabled_reason = f"selector_runtime_error:{exc}"
            close_fn = getattr(selector, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._selector = None
            LOG.exception("VLM selector inference failed, falling back to RAP argmax")
            result = {
                "raw_output": "",
                "parsed_output": None,
                "elapsed_sec": 0.0,
                "prompt": scoring_prompt if 'scoring_prompt' in locals() else "",
                "error": str(exc),
                "token_usage": _empty_token_usage(),
            }

        parsed = result.get("parsed_output")
        elapsed_sec = float(result.get("elapsed_sec", 0.0))
        scoring_token_usage = _normalize_token_usage(result.get("token_usage"))
        selected_candidate_index = int(default_selected_index)
        selected_row = dict(candidate_rows[default_selected_index])
        selected_source = default_selected_source
        error = None
        vlm_confidence = None
        vlm_reasoning = None
        vlm_candidate_index = None
        vlm_q_valid = False
        vlm_q_candidate_scores = None
        vlm_q_best_candidate_index = None
        vlm_q_score_gap_to_carry = None
        vlm_q_score_gap_top2 = None
        vlm_q_best_current_score = None
        vlm_q_carry_score = None
        selected_path_reasoning = None
        vlm_timed_out = False

        if isinstance(parsed, dict):
            coerced_scores, score_error = _coerce_candidate_scores(parsed.get("candidate_scores"), len(candidate_rows))
            candidate_idx = parsed.get("best_candidate_index")
            if coerced_scores is not None:
                vlm_q_valid = True
                vlm_q_candidate_scores = [float(score) for score in coerced_scores]
                vlm_q_best_candidate_index = int(max(range(len(vlm_q_candidate_scores)), key=lambda idx: vlm_q_candidate_scores[idx]))
                if isinstance(candidate_idx, int) and 0 <= candidate_idx < len(candidate_rows):
                    vlm_candidate_index = int(candidate_idx)
                vlm_confidence = float(parsed.get("confidence", 0.0))
                vlm_reasoning = parsed.get("reasoning")
                if elapsed_sec > self.cfg.timeout_sec:
                    vlm_timed_out = True
                    error = f"selector_timeout_fallback_rap:{elapsed_sec:.3f}"
                    selected_path_reasoning = (
                        f"VLM result arrived after timeout ({elapsed_sec:.3f}s > {self.cfg.timeout_sec:.3f}s); "
                        "using RAP argmax fallback for real-time control."
                    )
                else:
                    selection = _select_from_vlm_scores(
                        candidate_rows=candidate_rows,
                        vlm_scores=coerced_scores,
                    )
                    selected_candidate_index = int(selection["selected_candidate_index"])
                    selected_row = dict(selection["selected_candidate_row"])
                    selected_source = str(selection["selected_source"])
                    vlm_q_score_gap_top2 = selection["vlm_q_score_gap_top2"]
                    selected_path_reasoning = _selected_path_reasoning(
                        selected_row=selected_row,
                        selected_candidate_index=selected_candidate_index,
                        selected_source=selected_source,
                        vlm_scores=vlm_q_candidate_scores,
                        parsed_reasoning=vlm_reasoning,
                    )
            else:
                error = score_error or "invalid_candidate_scores"
        else:
            error = (
                f"selector_timeout_budget_exceeded:{elapsed_sec:.3f}"
                if elapsed_sec > self.cfg.timeout_sec
                else str(result.get("error") or "invalid_selector_output")
            )

        if selected_path_reasoning is None:
            selected_path_reasoning = _selected_path_reasoning(
                selected_row=selected_row,
                selected_candidate_index=selected_candidate_index,
                selected_source=selected_source,
                vlm_scores=vlm_q_candidate_scores,
                parsed_reasoning=vlm_reasoning,
            )

        latency_equivalent_steps = elapsed_sec / max(dt_sec, 1e-6)
        latency_equivalent_steps_ceil = int(math.ceil(latency_equivalent_steps))
        adaptive_replan_decision = (
            "vlm_timeout_fallback_rap"
            if vlm_timed_out
            else "vlm_failed_fallback_rap"
            if not vlm_q_valid
            else (
                "vlm_selected_reuse_prev"
                if selected_row.get("source") == "carry_prev"
                else "vlm_selected_current"
            )
        )
        execution_mode = execution_mode_label or "intervention_triggered_scoring"

        timeline_record = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "route_instruction": route_instruction,
            "scoring_route_instruction": scoring_route_instruction,
            "execution_mode": execution_mode,
            "candidate_count": len(candidate_rows),
            "carry_previous_valid": carry_previous_valid,
            "carry_previous_remaining_path_m": next(
                (path_length(np.asarray(row["local_plan"], dtype=np.float32)) for row in candidate_rows if row.get("source") == "carry_prev"),
                0.0,
            ),
            "selected_source": selected_source,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_source": selected_row.get("source"),
            "selected_proposal_index": selected_row.get("proposal_index"),
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_severity_score": intervention_severity_score,
            "intervention_severity_band": intervention_severity_band,
            "intervention_corrective_action": intervention_corrective_action,
            "intervention_confidence": intervention_confidence,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "intervention_prompt_tokens": intervention_token_usage["prompt_tokens"],
            "intervention_completion_tokens": intervention_token_usage["completion_tokens"],
            "intervention_total_tokens": intervention_token_usage["total_tokens"],
            "vlm_elapsed_sec": elapsed_sec,
            "scoring_prompt_tokens": scoring_token_usage["prompt_tokens"],
            "scoring_completion_tokens": scoring_token_usage["completion_tokens"],
            "scoring_total_tokens": scoring_token_usage["total_tokens"],
            "scoring_invoked": scoring_invoked,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "latency_equivalent_steps": latency_equivalent_steps,
            "latency_equivalent_steps_ceil": latency_equivalent_steps_ceil,
            "adaptive_replan_decision": adaptive_replan_decision,
            "error": error,
            "q_invoked_vlm": True,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
        }
        self._record_timeline(timeline_record)

        debug_payload = {
            "frame_index": frame_index,
            "route_instruction": route_instruction,
            "scoring_route_instruction": scoring_route_instruction,
            "default_selected_index": int(default_selected_index),
            "default_selected_source": default_selected_source,
            "candidate_rows": candidate_rows,
            "intervention_result": intervention_result,
            "scoring_result": result,
            "scoring_invoked": scoring_invoked,
            "selected_index": int(selected_candidate_index),
            "selected_source": selected_source,
            "selected_path_reasoning": selected_path_reasoning,
            "execution_mode": execution_mode,
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_severity_score": intervention_severity_score,
            "intervention_severity_band": intervention_severity_band,
            "intervention_corrective_action": intervention_corrective_action,
            "intervention_confidence": intervention_confidence,
            "intervention_reasoning": intervention_reasoning,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "intervention_error": intervention_error,
            "intervention_token_usage": intervention_token_usage,
            "scoring_token_usage": scoring_token_usage,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "vlm_q_best_current_score": vlm_q_best_current_score,
            "vlm_q_carry_score": vlm_q_carry_score,
            "adaptive_replan_decision": adaptive_replan_decision,
            "carry_previous_valid": carry_previous_valid,
            "latency_timeline_record": timeline_record,
            "error": error,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
            "agent_trace": build_agent_trace(
                frame_index=frame_index,
                route_instruction=route_instruction,
                info=info,
                candidate_rows=candidate_rows,
                decision_type="trajectory_scorer",
                selected_source=selected_source,
                selected_candidate_index=int(selected_candidate_index),
                confidence=vlm_confidence,
                reasoning=selected_path_reasoning,
            ),
        }
        if self.cfg.save_debug_artifacts:
            result_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        if not self.cfg.save_debug_artifacts:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        LOG.info(
            "VLM selection frame=%d source=%s proposal=%d candidate=%s elapsed=%.3f error=%s",
            frame_index,
            selected_source,
            -1 if selected_row.get("proposal_index") is None else int(selected_row["proposal_index"]),
            "none" if vlm_q_best_candidate_index is None else str(vlm_q_best_candidate_index),
            elapsed_sec,
            error,
        )

        return {
            "selected_candidate_row": selected_row,
            "selected_source": selected_source,
            "execution_mode": execution_mode,
            "selected_path_reasoning": selected_path_reasoning,
            "vlm_candidate_index": vlm_candidate_index,
            "vlm_confidence": vlm_confidence,
            "vlm_reasoning": vlm_reasoning,
            "vlm_elapsed_sec": elapsed_sec,
            "vlm_error": error,
            "vlm_candidate_count": len(candidate_rows),
            "scoring_invoked": scoring_invoked,
            "vlm_q_valid": vlm_q_valid,
            "vlm_timed_out": vlm_timed_out,
            "vlm_q_candidate_scores": vlm_q_candidate_scores,
            "vlm_q_best_candidate_index": vlm_q_best_candidate_index,
            "vlm_q_score_gap_to_carry": vlm_q_score_gap_to_carry,
            "vlm_q_score_gap_top2": vlm_q_score_gap_top2,
            "vlm_q_best_current_score": vlm_q_best_current_score,
            "vlm_q_carry_score": vlm_q_carry_score,
            "adaptive_replan_decision": adaptive_replan_decision,
            "carry_previous_valid": carry_previous_valid,
            "latency_timeline_record": timeline_record,
            "vlm_failed": (not vlm_q_valid) or vlm_timed_out,
            "intervention_invoked": intervention_invoked,
            "intervention_should_intervene": intervention_should_intervene,
            "intervention_severity_score": intervention_severity_score,
            "intervention_severity_band": intervention_severity_band,
            "intervention_corrective_action": intervention_corrective_action,
            "intervention_confidence": intervention_confidence,
            "intervention_reasoning": intervention_reasoning,
            "intervention_elapsed_sec": intervention_elapsed_sec,
            "intervention_error": intervention_error,
            "intervention_token_usage": intervention_token_usage,
            "scoring_token_usage": scoring_token_usage,
        }

    def maybe_select_planner(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        route_instruction = resolve_route_instruction(info)
        task_target_hint = describe_task_target_hint(info)
        default_planner = str(self.cfg.planner_gate_default_planner or "learned").strip().lower()
        if default_planner not in {"learned", "rule_based"}:
            default_planner = "learned"

        if not self.cfg.enabled or not self.cfg.planner_gate_enabled:
            return {
                "selected_planner": default_planner,
                "confidence": None,
                "reasoning": "planner_gate_disabled",
                "elapsed_sec": 0.0,
                "error": "planner_gate_disabled",
                "timed_out": False,
                "prompt_char_count": 0,
                "image_count": 0,
                "token_usage": _empty_token_usage(),
            }

        if not learned_candidate_rows:
            return {
                "selected_planner": "rule_based" if rule_based_candidate_rows else default_planner,
                "confidence": None,
                "reasoning": "missing_learned_candidates",
                "elapsed_sec": 0.0,
                "error": "missing_learned_candidates",
                "timed_out": False,
                "prompt_char_count": 0,
                "image_count": 0,
                "token_usage": _empty_token_usage(),
            }
        if not rule_based_candidate_rows:
            return {
                "selected_planner": default_planner,
                "confidence": None,
                "reasoning": "missing_rule_based_candidates",
                "elapsed_sec": 0.0,
                "error": "missing_rule_based_candidates",
                "timed_out": False,
                "prompt_char_count": 0,
                "image_count": 0,
                "token_usage": _empty_token_usage(),
            }

        selector = self._ensure_selector()
        if selector is None:
            return {
                "selected_planner": default_planner,
                "confidence": None,
                "reasoning": "selector_unavailable",
                "elapsed_sec": 0.0,
                "error": self._disabled_reason or "selector_unavailable",
                "timed_out": False,
                "prompt_char_count": 0,
                "image_count": 0,
                "token_usage": _empty_token_usage(),
            }

        camera_order = resolve_vlm_camera_order(self.cfg.planner_gate_camera_mode or self.cfg.camera_mode)
        overlays = render_planner_gate_overlays(
            camera_images,
            info,
            learned_candidate_rows,
            rule_based_candidate_rows,
            camera_order=camera_order,
        )
        frame_stem = f"frame_{frame_index:04d}"
        image_paths: List[Path] = []
        temp_paths: List[Path] = []
        for cam_name in camera_order:
            overlay = overlays.get(cam_name)
            if overlay is None:
                continue
            if self.cfg.planner_gate_save_debug_artifacts and self.cfg.save_debug_artifacts:
                image_path = self.debug_dir / f"{frame_stem}_planner_gate_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_planner_gate_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)

        prompt = build_planner_gate_prompt(
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            route_instruction=route_instruction,
            task_target_hint=task_target_hint,
            camera_order=camera_order,
            prompt_style=self.cfg.planner_gate_prompt_style,
        )
        elapsed_sec = 0.0
        parsed = None
        error = None
        token_usage = _empty_token_usage()
        try:
            result = selector.infer_prompt(
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=self.cfg.planner_gate_max_new_tokens,
                timeout_sec=self.cfg.planner_gate_timeout_sec,
            )
            elapsed_sec = float(result.get("elapsed_sec", 0.0))
            parsed = result.get("parsed_output")
            token_usage = _normalize_token_usage(result.get("token_usage"))
            if not isinstance(parsed, dict):
                error = str(result.get("error") or "invalid_planner_gate_output")
        except Exception as exc:
            LOG.exception("Planner gate inference failed, falling back to %s", default_planner)
            error = str(exc)
            result = None

        selected_planner = default_planner
        confidence = None
        reasoning = None
        timed_out = elapsed_sec > float(self.cfg.planner_gate_timeout_sec)
        if isinstance(parsed, dict):
            raw_planner = str(parsed.get("selected_planner", "")).strip().lower()
            if raw_planner in {"learned", "rule_based"}:
                selected_planner = raw_planner
            else:
                error = error or f"invalid_planner_choice:{raw_planner}"
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = None
            reasoning = parsed.get("reasoning")
        if timed_out:
            error = error or f"planner_gate_timeout:{elapsed_sec:.3f}"
            selected_planner = default_planner

        planner_gate_record = {
            "frame_index": frame_index,
            "selected_planner": selected_planner,
            "execution_mode": (
                "planner_gate_failed_base_policy_fallback"
                if error is not None
                else f"planner_gate_selected_{selected_planner}"
            ),
            "elapsed_sec": elapsed_sec,
            "timed_out": timed_out,
            "error": error,
            "prompt_char_count": len(prompt),
            "image_count": len(image_paths),
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
        }
        self._record_planner_gate(planner_gate_record)

        if self.cfg.planner_gate_save_debug_artifacts and self.cfg.save_debug_artifacts:
            learned_debug = _planner_gate_family_debug(learned_candidate_rows)
            rule_based_debug = _planner_gate_family_debug(rule_based_candidate_rows)
            gate_payload = {
                "frame_index": frame_index,
                "route_instruction": route_instruction,
                "default_planner": default_planner,
                "selected_planner": selected_planner,
                "execution_mode": planner_gate_record["execution_mode"],
                "confidence": confidence,
                "reasoning": reasoning,
                "elapsed_sec": elapsed_sec,
                "timed_out": timed_out,
                "error": error,
                "prompt_char_count": len(prompt),
                "image_count": len(image_paths),
                "token_usage": token_usage,
                "planner_gate_prompt_style": self.cfg.planner_gate_prompt_style,
                "learned_candidate_count": len(learned_candidate_rows),
                "rule_based_candidate_count": len(rule_based_candidate_rows),
                "learned_candidates": learned_debug,
                "rule_based_candidates": rule_based_debug,
                "learned_default_candidate": learned_debug[0] if learned_debug else None,
                "rule_based_default_candidate": rule_based_debug[0] if rule_based_debug else None,
                "agent_trace": build_agent_trace(
                    frame_index=frame_index,
                    route_instruction=route_instruction,
                    info=info,
                    learned_candidate_rows=learned_candidate_rows,
                    rule_based_candidate_rows=rule_based_candidate_rows,
                    decision_type="planner_gate",
                    selected_planner=selected_planner,
                    confidence=confidence,
                    reasoning=reasoning,
                ),
            }
            gate_result_path = self.debug_dir / f"{frame_stem}_planner_gate_result.json"
            gate_result_path.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

        return {
            "selected_planner": selected_planner,
            "confidence": confidence,
            "reasoning": reasoning,
            "elapsed_sec": elapsed_sec,
            "error": error,
            "timed_out": timed_out,
            "prompt_char_count": len(prompt),
            "image_count": len(image_paths),
            "token_usage": token_usage,
        }

    def finalize(self) -> None:
        if hasattr(self._selector, "close"):
            try:
                self._selector.close()
            except Exception:
                LOG.exception("Failed to close VLM selector")
        if not self.cfg.save_debug_artifacts:
            return
        if not self._timeline_records and not self._planner_gate_records:
            return

        def _sum_token_usage(records: Sequence[Dict[str, object]], prefix: str) -> Dict[str, int]:
            prompt = sum(int(record.get(f"{prefix}_prompt_tokens", 0) or 0) for record in records)
            completion = sum(int(record.get(f"{prefix}_completion_tokens", 0) or 0) for record in records)
            total = sum(int(record.get(f"{prefix}_total_tokens", 0) or 0) for record in records)
            return {
                "prompt_tokens_total": int(prompt),
                "completion_tokens_total": int(completion),
                "total_tokens_total": int(total if total > 0 else prompt + completion),
            }

        def _planner_gate_token_usage(records: Sequence[Dict[str, object]]) -> Dict[str, int]:
            prompt = sum(int(record.get("prompt_tokens", 0) or 0) for record in records)
            completion = sum(int(record.get("completion_tokens", 0) or 0) for record in records)
            total = sum(int(record.get("total_tokens", 0) or 0) for record in records)
            return {
                "prompt_tokens_total": int(prompt),
                "completion_tokens_total": int(completion),
                "total_tokens_total": int(total if total > 0 else prompt + completion),
            }

        def _safe_latency_summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
            if not values:
                return {
                    f"{prefix}_mean_sec": 0.0,
                    f"{prefix}_p50_sec": 0.0,
                    f"{prefix}_p95_sec": 0.0,
                    f"{prefix}_max_sec": 0.0,
                }
            return {
                f"{prefix}_mean_sec": float(np.mean(values)),
                f"{prefix}_p50_sec": float(np.percentile(values, 50)),
                f"{prefix}_p95_sec": float(np.percentile(values, 95)),
                f"{prefix}_max_sec": float(np.max(values)),
            }

        def _records_for_mode(records: Sequence[Dict[str, object]], mode: str) -> List[Dict[str, object]]:
            return [record for record in records if str(record.get("execution_mode", "")) == mode]

        def _branch_summary(
            *,
            records: Sequence[Dict[str, object]],
            latency_field: str,
            token_prefix: Optional[str],
        ) -> Dict[str, object]:
            values = [float(record.get(latency_field, 0.0) or 0.0) for record in records]
            summary = {
                "count": int(len(records)),
                **_safe_latency_summary(values, "latency"),
            }
            if token_prefix is not None:
                summary["tokens"] = _sum_token_usage(records, token_prefix)
            return summary

        record_count = len(self._timeline_records)
        elapsed = [float(record["vlm_elapsed_sec"]) for record in self._timeline_records]
        intervention_elapsed = [float(record.get("intervention_elapsed_sec", 0.0)) for record in self._timeline_records]
        total_elapsed = [float(v) + float(g) for v, g in zip(elapsed, intervention_elapsed)]
        planner_gate_elapsed = [float(record.get("elapsed_sec", 0.0)) for record in self._planner_gate_records]
        carry_reuse_rate = (
            sum(
                record["adaptive_replan_decision"] in {"reuse_prev", "vlm_q_reuse_prev", "q_reuse_prev"}
                for record in self._timeline_records
            ) / record_count
            if record_count else 0.0
        )
        switch_rate = (
            sum(
                record["adaptive_replan_decision"] in {"switch_to_current", "vlm_q_switch_to_current", "q_switch_to_current"}
                for record in self._timeline_records
            ) / record_count
            if record_count else 0.0
        )
        fallback_rate = (
            sum(
                record.get("adaptive_replan_decision") in {"vlm_failed_fallback_rap", "vlm_timeout_fallback_rap", "gate_failed_fallback_rap"}
                for record in self._timeline_records
            ) / record_count
            if record_count else 0.0
        )
        intervention_trigger_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked") and record.get("intervention_should_intervene") is True
            ) / record_count
            if record_count else 0.0
        )
        intervention_scores = [
            float(record["intervention_severity_score"])
            for record in self._timeline_records
            if record.get("intervention_severity_score") is not None
        ]
        intervention_low_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked") and record.get("intervention_severity_band") == "low"
            ) / record_count
            if record_count else 0.0
        )
        intervention_medium_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked") and record.get("intervention_severity_band") == "medium"
            ) / record_count
            if record_count else 0.0
        )
        intervention_high_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked") and record.get("intervention_severity_band") == "high"
            ) / record_count
            if record_count else 0.0
        )
        intervention_action_applied_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked")
                and record.get("intervention_should_intervene") is True
                and record.get("intervention_severity_score") is not None
                and float(record.get("intervention_severity_score")) >= float(self.cfg.intervention_action_threshold)
            ) / record_count
            if record_count else 0.0
        )
        gate_skip_rate = (
            sum(
                1.0
                for record in self._timeline_records
                if record.get("intervention_invoked") and record.get("intervention_should_intervene") is False
            ) / record_count
            if record_count else 0.0
        )
        scoring_invoked_rate = (
            sum(
                1.0 if record.get("scoring_invoked") else 0.0
                for record in self._timeline_records
            ) / record_count
            if record_count else 0.0
        )
        intervention_invoked_count = sum(1 for record in self._timeline_records if record.get("intervention_invoked"))
        intervention_timeout_count = sum(
            1
            for record in self._timeline_records
            if str(record.get("error") or "").startswith("intervention_timeout")
        )
        intervention_invalid_count = sum(
            1
            for record in self._timeline_records
            if str(record.get("error") or "") == "intervention_output_invalid"
        )
        intervention_error_count = sum(
            1
            for record in self._timeline_records
            if record.get("intervention_invoked") and record.get("error") is not None
        )
        intervention_valid_count = max(0, intervention_invoked_count - intervention_error_count)
        scoring_invoked_count = sum(1 for record in self._timeline_records if record.get("scoring_invoked"))
        scoring_timeout_count = sum(1 for record in self._timeline_records if record.get("vlm_timed_out"))
        scoring_error_count = sum(
            1
            for record in self._timeline_records
            if record.get("scoring_invoked") and record.get("error") is not None
        )
        scoring_valid_count = max(0, sum(1 for record in self._timeline_records if record.get("vlm_q_valid")) - scoring_timeout_count)
        planner_gate_invoked_count = len(self._planner_gate_records)
        planner_gate_timeout_count = sum(1 for record in self._planner_gate_records if record.get("timed_out"))
        planner_gate_error_count = sum(1 for record in self._planner_gate_records if record.get("error") is not None)
        planner_gate_valid_count = sum(1 for record in self._planner_gate_records if record.get("error") is None)
        base_policy_no_intervention_records = _records_for_mode(self._timeline_records, "base_policy_no_intervention")
        gate_failed_base_policy_records = _records_for_mode(self._timeline_records, "gate_failed_base_policy_fallback")
        intervention_scoring_records = _records_for_mode(self._timeline_records, "intervention_triggered_scoring")
        planner_gate_learned_records = _records_for_mode(self._planner_gate_records, "planner_gate_selected_learned")
        planner_gate_rule_based_records = _records_for_mode(self._planner_gate_records, "planner_gate_selected_rule_based")
        planner_gate_failed_records = _records_for_mode(self._planner_gate_records, "planner_gate_failed_base_policy_fallback")
        scoring_token_usage = _sum_token_usage(self._timeline_records, "scoring")
        intervention_token_usage = _sum_token_usage(self._timeline_records, "intervention")
        planner_gate_token_usage = _planner_gate_token_usage(self._planner_gate_records)
        total_prompt_tokens = (
            scoring_token_usage["prompt_tokens_total"]
            + intervention_token_usage["prompt_tokens_total"]
            + planner_gate_token_usage["prompt_tokens_total"]
        )
        total_completion_tokens = (
            scoring_token_usage["completion_tokens_total"]
            + intervention_token_usage["completion_tokens_total"]
            + planner_gate_token_usage["completion_tokens_total"]
        )
        total_tokens = (
            scoring_token_usage["total_tokens_total"]
            + intervention_token_usage["total_tokens_total"]
            + planner_gate_token_usage["total_tokens_total"]
        )
        summary = {
            "num_records": record_count,
            "planner_gate_records": planner_gate_invoked_count,
            **_safe_latency_summary(elapsed, "latency"),
            **_safe_latency_summary(intervention_elapsed, "intervention_latency"),
            **_safe_latency_summary(total_elapsed, "total_vlm_latency"),
            **_safe_latency_summary(planner_gate_elapsed, "planner_gate_latency"),
            "latency_equivalent_steps_mean": float(
                np.mean([record["latency_equivalent_steps"] for record in self._timeline_records])
            ) if record_count else 0.0,
            "carry_reuse_rate": float(carry_reuse_rate),
            "switch_to_current_rate": float(switch_rate),
            "fallback_rate": float(fallback_rate),
            "intervention_trigger_rate": float(intervention_trigger_rate),
            "intervention_low_rate": float(intervention_low_rate),
            "intervention_medium_rate": float(intervention_medium_rate),
            "intervention_high_rate": float(intervention_high_rate),
            "intervention_action_applied_rate": float(intervention_action_applied_rate),
            "intervention_action_threshold": float(self.cfg.intervention_action_threshold),
            "intervention_high_threshold": float(self.cfg.intervention_high_threshold),
            "intervention_severity_score_mean": float(np.mean(intervention_scores)) if intervention_scores else 0.0,
            "intervention_severity_score_p50": float(np.percentile(intervention_scores, 50)) if intervention_scores else 0.0,
            "intervention_severity_score_p95": float(np.percentile(intervention_scores, 95)) if intervention_scores else 0.0,
            "gate_skip_rate": float(gate_skip_rate),
            "scoring_invoked_rate": float(scoring_invoked_rate),
            "vlm_q_valid_rate": float(
                np.mean([1.0 if record.get("vlm_q_valid") else 0.0 for record in self._timeline_records])
            ) if record_count else 0.0,
            "counts": {
                "intervention_invoked": int(intervention_invoked_count),
                "intervention_valid": int(intervention_valid_count),
                "intervention_invalid": int(intervention_invalid_count),
                "intervention_timeout": int(intervention_timeout_count),
                "intervention_error": int(intervention_error_count),
                "scoring_invoked": int(scoring_invoked_count),
                "scoring_valid": int(scoring_valid_count),
                "scoring_timeout": int(scoring_timeout_count),
                "scoring_error": int(scoring_error_count),
                "planner_gate_invoked": int(planner_gate_invoked_count),
                "planner_gate_valid": int(planner_gate_valid_count),
                "planner_gate_timeout": int(planner_gate_timeout_count),
                "planner_gate_error": int(planner_gate_error_count),
                "base_policy_no_intervention": int(len(base_policy_no_intervention_records)),
                "gate_failed_base_policy_fallback": int(len(gate_failed_base_policy_records)),
                "intervention_triggered_scoring": int(len(intervention_scoring_records)),
                "planner_gate_selected_learned": int(len(planner_gate_learned_records)),
                "planner_gate_selected_rule_based": int(len(planner_gate_rule_based_records)),
                "planner_gate_failed_base_policy_fallback": int(len(planner_gate_failed_records)),
            },
            "branch_breakdown": {
                "solo_or_merge": {
                    "base_policy_no_intervention": _branch_summary(
                        records=base_policy_no_intervention_records,
                        latency_field="intervention_elapsed_sec",
                        token_prefix="intervention",
                    ),
                    "gate_failed_base_policy_fallback": _branch_summary(
                        records=gate_failed_base_policy_records,
                        latency_field="intervention_elapsed_sec",
                        token_prefix="intervention",
                    ),
                    "intervention_triggered_scoring": {
                        **_safe_latency_summary(
                            [
                                float(record.get("intervention_elapsed_sec", 0.0) or 0.0) + float(record.get("vlm_elapsed_sec", 0.0) or 0.0)
                                for record in intervention_scoring_records
                            ],
                            "latency",
                        ),
                        "count": int(len(intervention_scoring_records)),
                        "tokens": {
                            "intervention": _sum_token_usage(intervention_scoring_records, "intervention"),
                            "scoring": _sum_token_usage(intervention_scoring_records, "scoring"),
                        },
                    },
                },
                "planner_gate": {
                    "selected_learned": {
                        **_safe_latency_summary(
                            [float(record.get("elapsed_sec", 0.0) or 0.0) for record in planner_gate_learned_records],
                            "latency",
                        ),
                        "count": int(len(planner_gate_learned_records)),
                        "tokens": _planner_gate_token_usage(planner_gate_learned_records),
                    },
                    "selected_rule_based": {
                        **_safe_latency_summary(
                            [float(record.get("elapsed_sec", 0.0) or 0.0) for record in planner_gate_rule_based_records],
                            "latency",
                        ),
                        "count": int(len(planner_gate_rule_based_records)),
                        "tokens": _planner_gate_token_usage(planner_gate_rule_based_records),
                    },
                    "failed_base_policy_fallback": {
                        **_safe_latency_summary(
                            [float(record.get("elapsed_sec", 0.0) or 0.0) for record in planner_gate_failed_records],
                            "latency",
                        ),
                        "count": int(len(planner_gate_failed_records)),
                        "tokens": _planner_gate_token_usage(planner_gate_failed_records),
                    },
                },
            },
            "tokens": {
                "prompt_tokens_total": int(total_prompt_tokens),
                "completion_tokens_total": int(total_completion_tokens),
                "total_tokens_total": int(total_tokens),
                "by_stage": {
                    "intervention": intervention_token_usage,
                    "scoring": scoring_token_usage,
                    "planner_gate": planner_gate_token_usage,
                },
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
