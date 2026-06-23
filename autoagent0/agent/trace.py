from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from autoagent0.agent.designer import summarize_sources
from autoagent0.agent.schemas import AgentStepTrace, OrchestratorDecision, SceneContext
from autoagent0.agent.verifier import PassiveVerifier


TRACE_SCHEMA_VERSION = "autoagent0.trace.v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_agent_trace(
    *,
    frame_index: int,
    route_instruction: str,
    info: Optional[Dict[str, Any]] = None,
    candidate_rows: Sequence[Dict[str, Any]] = (),
    learned_candidate_rows: Sequence[Dict[str, Any]] = (),
    rule_based_candidate_rows: Sequence[Dict[str, Any]] = (),
    decision_type: str,
    selected_source: Optional[str] = None,
    selected_planner: Optional[str] = None,
    selected_candidate_index: Optional[int] = None,
    confidence: Optional[float] = None,
    reasoning: Optional[str] = None,
    previous_verifier_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info = info or {}
    all_rows = list(candidate_rows) or list(learned_candidate_rows) + list(rule_based_candidate_rows)
    scene = SceneContext(
        frame_index=int(frame_index),
        route_instruction=str(route_instruction),
        timestamp=_safe_float(info.get("timestamp", 0.0)),
        task_instruction=info.get("task_instruction") if isinstance(info.get("task_instruction"), str) else None,
        metadata={
            "command": info.get("command"),
            "task_type": info.get("task_type"),
        },
    )
    designer = {
        "candidate_count": len(all_rows),
        "learned_candidate_count": len(learned_candidate_rows),
        "rule_based_candidate_count": len(rule_based_candidate_rows),
        "candidate_sources": summarize_sources(all_rows),
    }
    decision = OrchestratorDecision(
        decision_type=decision_type,
        selected_source=selected_source,
        selected_planner=selected_planner,
        selected_candidate_index=selected_candidate_index,
        confidence=confidence,
        reasoning=reasoning,
    )
    trace = AgentStepTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        scene=scene,
        designer=designer,
        orchestrator=decision,
        verifier=PassiveVerifier().verify(),
        previous_verifier_feedback=previous_verifier_feedback,
    )
    return trace.to_debug_dict()

