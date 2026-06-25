"""Action verifiers for the closed loop.

A verifier inspects a finalized control action (the ``{'acc', 'steer_rate'}`` dict
passed to ``env.step``) and returns a scalar score. The loop rejects actions whose
score falls below a threshold and falls back to recovery action selection.
"""
from autoagent0.verifiers.base import BaseVerifier
from autoagent0.verifiers.dummy import DummyVerifier

__all__ = ["BaseVerifier", "DummyVerifier"]
