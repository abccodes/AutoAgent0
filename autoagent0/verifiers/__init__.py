"""Trajectory verifiers for the closed loop.

A verifier scores the selected trajectory (local plan ``[T, 2]``) before it is
turned into a control action. The loop rejects trajectories whose score falls
below a threshold and recovers by regenerating the plan with a different planner.
"""
from autoagent0.verifiers.base import BaseVerifier
from autoagent0.verifiers.dummy import TTCVerifier

__all__ = ["BaseVerifier", "TTCVerifier"]
