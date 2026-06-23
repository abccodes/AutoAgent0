from __future__ import annotations

from typing import Any, Dict, Optional

from autoagent0.agent.schemas import VerifierResult


class PassiveVerifier:
    """Phase-1 verifier stub. It records the boundary without changing control."""

    def verify(self, trajectory: Any = None, context: Optional[Dict[str, Any]] = None) -> VerifierResult:
        return VerifierResult(
            accepted=True,
            mode="passive",
            rejection_reason=None,
            checks={},
        )

