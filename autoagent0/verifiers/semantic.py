from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from autoagent0.adapters.hugsim.context import resolve_route_instruction
from autoagent0.scorer.agent_schemas import SemanticVerifierResult
from autoagent0.scorer.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from autoagent0.verifiers.geometric_route import compare_geometric_semantic_route
from autoagent0.vlm.debug import append_jsonl
from autoagent0.vlm.semantic_verifier_env import (
    SEMANTIC_VERIFIER_ENV_DEFAULTS,
    SEMANTIC_VERIFIER_ENV_FIELD_NAMES,
    build_prefixed_semantic_verifier_env,
    get_prefixed_semantic_verifier_env_value,
)


LOG = logging.getLogger(__name__)

# Backwards-compatible alias used by verifiers.__init__.
SemanticVerificationResult = SemanticVerifierResult


def _coerce_env_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


@dataclass(frozen=True)
class SemanticVerifierConfig:
    enabled: bool = False
    gate_on_reject: bool = False
    camera_mode: str = "front_only"
    backend: str = "local_transformers_subprocess"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    python_bin: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    enable_thinking: bool = False
    timeout_sec: float = 60.0
    save_debug_artifacts: bool = True
    debug_dir_name: str = "semantic_verifier_debug"
    log_file_name: str = "semantic_verifier.jsonl"
    preload_on_init: bool = True


def resolve_semantic_verifier_config(
    *,
    prefixes: tuple[str, ...] = ("PLANNER_SEMANTIC_VERIFIER_",),
) -> SemanticVerifierConfig:
    values: Dict[str, Any] = {}
    for suffix, field_name in SEMANTIC_VERIFIER_ENV_FIELD_NAMES.items():
        default_value = SEMANTIC_VERIFIER_ENV_DEFAULTS[suffix]
        raw_value = get_prefixed_semantic_verifier_env_value(
            suffix,
            default=default_value,
            prefixes=prefixes,
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return SemanticVerifierConfig(**values)


def _semantic_config_to_vlm_config(cfg: SemanticVerifierConfig) -> VLMSelectorConfig:
    return VLMSelectorConfig(
        enabled=cfg.enabled,
        camera_mode=cfg.camera_mode,
        scoring_camera_mode=cfg.camera_mode,
        intervention_camera_mode=cfg.camera_mode,
        backend=cfg.backend,
        model_id=cfg.model_id,
        device=cfg.device,
        python_bin=cfg.python_bin,
        max_new_tokens=cfg.max_new_tokens,
        intervention_max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        enable_thinking=cfg.enable_thinking,
        timeout_sec=cfg.timeout_sec,
        intervention_timeout_sec=cfg.timeout_sec,
        preload_on_init=cfg.preload_on_init,
        save_debug_artifacts=cfg.save_debug_artifacts,
        debug_dir_name=cfg.debug_dir_name,
    )


class SemanticVerifier:
    """VLM-based post-selection route/track verifier."""

    def __init__(self, cfg: SemanticVerifierConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self._log_path = self.output_dir / cfg.log_file_name
        self._text_log_path = self.output_dir / "semantic_verifier.log"
        self._vlm_selector = VLMPlanSelector(
            _semantic_config_to_vlm_config(cfg),
            self.output_dir,
        )
        self.last_result: Optional[SemanticVerifierResult] = None
        self.last_debug: Dict[str, Any] = {}
        self._configure_file_logging()

    def _configure_file_logging(self) -> None:
        if not self.cfg.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path.unlink(missing_ok=True)
        self._text_log_path.unlink(missing_ok=True)

        handler = logging.FileHandler(self._text_log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
        handler.setLevel(logging.INFO)
        if not any(
            isinstance(existing, logging.FileHandler)
            and getattr(existing, "baseFilename", "") == handler.baseFilename
            for existing in LOG.handlers
        ):
            LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)

    def _wall_clock_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_verification_event(
        self,
        *,
        frame_index: int,
        sim_timestamp: float,
        result: SemanticVerifierResult,
        feedback: Dict[str, Any],
        mode: str,
    ) -> None:
        record = {
            "wall_clock_utc": self._wall_clock_timestamp(),
            "sim_timestamp": float(sim_timestamp),
            "frame_index": int(frame_index),
            "mode": mode,
            "accepted": result.accepted,
            "on_track": result.accepted,
            "confidence": result.confidence,
            "rejection_reason": result.rejection_reason,
            "route_instruction": result.checks.get("route_instruction"),
            "command": result.checks.get("command"),
            "selected_source": result.checks.get("selected_source"),
            "elapsed_sec": result.checks.get("elapsed_sec"),
            "error": result.checks.get("error"),
            "geometric_direction": result.checks.get("geometric_direction"),
            "instruction_direction": result.checks.get("instruction_direction"),
            "geometric_matches_instruction": result.checks.get("geometric_matches_instruction"),
            "semantic_agrees_with_geometry": result.checks.get("semantic_agrees_with_geometry"),
            "heading_deg": result.checks.get("heading_deg"),
            "feedback": feedback,
            "checks": dict(result.checks),
        }
        append_jsonl(self._log_path, record)
        LOG.info(
            "semantic_verifier frame=%s sim_t=%.3f route=%s on_track=%s geom=%s instr=%s "
            "geom_match=%s semantic_geom_agree=%s heading_deg=%s mode=%s",
            frame_index,
            sim_timestamp,
            record.get("route_instruction"),
            result.accepted,
            record.get("geometric_direction"),
            record.get("instruction_direction"),
            record.get("geometric_matches_instruction"),
            record.get("semantic_agrees_with_geometry"),
            record.get("heading_deg"),
            mode,
        )
        if result.rejection_reason:
            LOG.info("semantic_verifier reasoning: %s", result.rejection_reason)

    def preload(self) -> None:
        if not self.cfg.enabled:
            return
        self._vlm_selector.preload()

    def verify(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        local_plan: np.ndarray,
        selected_source: Optional[str] = None,
        previous_feedback: Optional[str] = None,
    ) -> SemanticVerifierResult:
        route_instruction = resolve_route_instruction(info)
        candidate_row = {
            "source": str(selected_source or "selected"),
            "local_plan": np.asarray(local_plan, dtype=np.float32),
            "execution_plan": np.asarray(local_plan, dtype=np.float32),
            "proposal_score": 0.0,
            "proposal_index": None,
        }
        debug = self._vlm_selector.verify_semantic_track(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_row=candidate_row,
            previous_feedback=previous_feedback,
        )
        self.last_debug = dict(debug)
        semantic_on_track = bool(debug.get("semantic_verifier_accepted", True))
        geometric_check = compare_geometric_semantic_route(
            local_plan=np.asarray(local_plan, dtype=np.float32),
            route_instruction=route_instruction,
            command=info.get("command"),
            semantic_on_track=semantic_on_track,
        )
        geometric_payload = geometric_check.to_dict()
        result = SemanticVerifierResult(
            accepted=semantic_on_track,
            mode="semantic_vlm",
            rejection_reason=(
                None
                if semantic_on_track
                else str(debug.get("semantic_verifier_reasoning") or debug.get("semantic_verifier_error") or "off_track")
            ),
            confidence=(
                None
                if debug.get("semantic_verifier_confidence") is None
                else float(debug.get("semantic_verifier_confidence"))
            ),
            checks={
                "route_instruction": route_instruction,
                "command": info.get("command"),
                "task_instruction": info.get("task_instruction"),
                "on_track": semantic_on_track,
                "selected_source": selected_source,
                "elapsed_sec": debug.get("semantic_verifier_elapsed_sec"),
                "error": debug.get("semantic_verifier_error"),
                "vlm_raw": debug.get("semantic_verifier_raw"),
                **geometric_payload,
            },
        )
        self.last_result = result
        return result

    def feedback_for_orchestrator(self, result: SemanticVerifierResult) -> Dict[str, Any]:
        return {
            "accepted": result.accepted,
            "on_track": result.accepted,
            "rejection_reason": result.rejection_reason,
            "confidence": result.confidence,
            "route_instruction": result.checks.get("route_instruction"),
            "mode": "feedback_only" if not self.cfg.gate_on_reject else "active_gate",
            "wall_clock_utc": self._wall_clock_timestamp(),
            "checks": dict(result.checks),
        }

    @classmethod
    def from_planner_config(
        cls,
        planner_cfg: Dict[str, Any],
        output_dir: Path,
        *,
        planner_python_bin: str = "",
    ) -> "SemanticVerifier":
        semantic_cfg = planner_cfg.get("semantic_verifier", {}) or {}
        env_values = build_prefixed_semantic_verifier_env(
            semantic_cfg,
            planner_python_bin=planner_python_bin,
        )
        for key, value in env_values.items():
            os.environ[key] = str(value)
        device_override = (
            os.environ.get("PLANNER_SEMANTIC_VERIFIER_DEVICE_OVERRIDE")
            or semantic_cfg.get("device", "auto")
        )
        os.environ["PLANNER_SEMANTIC_VERIFIER_DEVICE"] = str(device_override)
        return cls(resolve_semantic_verifier_config(), Path(output_dir))


def apply_semantic_verifier_debug(decision: Any, semantic_debug: Dict[str, Any]) -> None:
    """Merge semantic verifier debug fields into a plan decision payload."""
    if not semantic_debug:
        return
    decision.autoagent0_debug = dict(decision.autoagent0_debug or {})
    for key, value in semantic_debug.items():
        if key.startswith("semantic_verifier_") or key in {"route_instruction"}:
            decision.autoagent0_debug[key] = value


@dataclass(frozen=True)
class SemanticVerifierStepOutcome:
    """Result of one post-selection semantic verification step."""

    decision: Any
    feedback: Optional[Dict[str, Any]] = None


def apply_semantic_verifier_to_decision(
    semantic_verifier: Optional[SemanticVerifier],
    *,
    decision: Any,
    camera_images: Dict[str, np.ndarray],
    current_info: Dict[str, object],
    frame_index: int,
    previous_feedback: Optional[str] = None,
    recover_decision_fn: Optional[Callable[[Any], Any]] = None,
) -> SemanticVerifierStepOutcome:
    """Run semantic verification on the selected plan and optionally recover it.

    This owns the semantic verifier policy:
    - call the VLM verifier
    - attach debug metadata to the decision
    - optionally replace the decision when ``gate_on_reject`` is enabled

    The closed loop should only decide *when* to call this helper and how to
    wire recovery via ``recover_decision_fn``.
    """
    if semantic_verifier is None or not semantic_verifier.cfg.enabled:
        return SemanticVerifierStepOutcome(decision=decision, feedback=None)

    local_plan = decision.selected_plan
    if local_plan is None or len(np.asarray(local_plan)) == 0:
        return SemanticVerifierStepOutcome(decision=decision, feedback=None)

    result = semantic_verifier.verify(
        frame_index=frame_index,
        camera_images=camera_images,
        info=current_info,
        local_plan=np.asarray(local_plan, dtype=np.float32),
        selected_source=decision.selected_source,
        previous_feedback=previous_feedback,
    )
    semantic_debug = dict(semantic_verifier.last_debug)
    apply_semantic_verifier_debug(decision, semantic_debug)

    feedback = semantic_verifier.feedback_for_orchestrator(result)
    mode = "feedback_only" if not semantic_verifier.cfg.gate_on_reject else "active_gate"
    semantic_verifier.log_verification_event(
        frame_index=frame_index,
        sim_timestamp=float(current_info.get("timestamp", 0.0) or 0.0),
        result=result,
        feedback=feedback,
        mode=mode,
    )
    apply_semantic_verifier_debug(decision, {
        **semantic_debug,
        "semantic_verifier_mode": mode,
        "semantic_verifier_wall_clock_utc": feedback.get("wall_clock_utc"),
    })

    if result.accepted:
        return SemanticVerifierStepOutcome(decision=decision, feedback=feedback)

    if not semantic_verifier.cfg.gate_on_reject:
        LOG.warning(
            "semantic verifier rejected frame=%s route=%s reason=%s (feedback_only; plan unchanged)",
            frame_index,
            feedback.get("route_instruction"),
            result.rejection_reason,
        )
        return SemanticVerifierStepOutcome(decision=decision, feedback=feedback)

    LOG.warning(
        "semantic verifier rejected frame=%s route=%s reason=%s (active_gate)",
        frame_index,
        feedback.get("route_instruction"),
        result.rejection_reason,
    )

    if recover_decision_fn is None:
        LOG.warning("semantic verifier rejected but no recovery hook is configured; keeping plan")
        return SemanticVerifierStepOutcome(decision=decision, feedback=feedback)

    recovered_decision = recover_decision_fn(decision)
    if recovered_decision.selected_plan is not decision.selected_plan:
        apply_semantic_verifier_debug(recovered_decision, {
            **semantic_debug,
            "semantic_verifier_recovery_applied": True,
        })
    return SemanticVerifierStepOutcome(decision=recovered_decision, feedback=feedback)
