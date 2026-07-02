"""Trajectory verifiers for the closed loop.

A verifier scores the selected trajectory (local plan ``[T, 2]``) before it is
turned into a control action. The loop rejects trajectories whose score falls
below a threshold and recovers by regenerating the plan with a different planner.
"""
from autoagent0.verifiers.base import BaseVerifier
from autoagent0.verifiers.dummy import PDMSVerifier, VerificationResult

from autoagent0.verifiers.semantic import (
    SemanticVerifier,
    SemanticVerificationResult,
    SemanticVerifierResult,
    SemanticVerifierStepOutcome,
    apply_semantic_verifier_debug,
    apply_semantic_verifier_to_decision,
)
from autoagent0.verifiers.geometric_route import (
    GeometricRouteCheck,
    classify_local_plan_endpoint_bearing,
    compare_geometric_semantic_route,
    instruction_to_turn_direction,
)

__all__ = [
    "BaseVerifier",
    "PDMSVerifier",
    "SemanticVerifier",
    "SemanticVerificationResult",
    "SemanticVerifierResult",
    "SemanticVerifierStepOutcome",
    "VerificationResult",
    "GeometricRouteCheck",
    "classify_local_plan_endpoint_bearing",
    "compare_geometric_semantic_route",
    "instruction_to_turn_direction",
    "apply_semantic_verifier_debug",
    "apply_semantic_verifier_to_decision",
]
