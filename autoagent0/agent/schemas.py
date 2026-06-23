from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SceneContext:
    """Normalized scene metadata exposed to AutoAgent0 components."""

    frame_index: int
    route_instruction: str
    timestamp: float = 0.0
    task_instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryCandidate:
    """Normalized view of a candidate row produced by an expert backend."""

    source: str
    local_plan: Any
    proposal_index: Optional[int] = None
    proposal_score: Optional[float] = None
    candidate_index: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignBatch:
    """Candidates generated for one control step."""

    learned: List[TrajectoryCandidate] = field(default_factory=list)
    rule_based: List[TrajectoryCandidate] = field(default_factory=list)
    combined: List[TrajectoryCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class OrchestratorDecision:
    """Structured orchestrator decision for trace/debug output."""

    decision_type: str
    selected_source: Optional[str] = None
    selected_planner: Optional[str] = None
    selected_candidate_index: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerToolCall:
    """SceneSmith-style planner tool call for trace/debug output."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CritiqueResult:
    """Structured result from the active AutoAgent0 critic."""

    accepted: bool
    severity_score: float
    corrective_action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignChangeRequest:
    """Request issued after critic rejection to expand/revise candidates."""

    reason: str
    corrective_action: Optional[str]
    candidate_budget: int
    learned_budget: int
    rule_based_budget: int
    allocation_strategy: str
    include_learned: bool = True
    include_rule_based: bool = True


@dataclass(frozen=True)
class FinalActionSelection:
    """Final selected action metadata for the active AutoAgent0 path."""

    selected_source: str
    selected_planner: str
    selected_candidate_index: Optional[int] = None
    fallback_selected: bool = False
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class VerifierResult:
    """Verifier result. Phase 1 is passive and always accepts."""

    accepted: bool
    mode: str = "passive"
    rejection_reason: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStepTrace:
    """Debug-only trace joining designer, orchestrator, and verifier state."""

    schema_version: str
    scene: SceneContext
    designer: Dict[str, Any]
    orchestrator: OrchestratorDecision
    verifier: VerifierResult
    previous_verifier_feedback: Optional[Dict[str, Any]] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene": {
                "frame_index": self.scene.frame_index,
                "route_instruction": self.scene.route_instruction,
                "timestamp": self.scene.timestamp,
                "task_instruction": self.scene.task_instruction,
                "metadata": self.scene.metadata,
            },
            "designer": self.designer,
            "orchestrator": {
                "decision_type": self.orchestrator.decision_type,
                "selected_source": self.orchestrator.selected_source,
                "selected_planner": self.orchestrator.selected_planner,
                "selected_candidate_index": self.orchestrator.selected_candidate_index,
                "confidence": self.orchestrator.confidence,
                "reasoning": self.orchestrator.reasoning,
                "metadata": self.orchestrator.metadata,
            },
            "verifier": {
                "accepted": self.verifier.accepted,
                "mode": self.verifier.mode,
                "rejection_reason": self.verifier.rejection_reason,
                "checks": self.verifier.checks,
            },
            "previous_verifier_feedback": self.previous_verifier_feedback or {},
        }
