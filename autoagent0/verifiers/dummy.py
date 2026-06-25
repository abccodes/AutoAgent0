"""A placeholder verifier that always approves the trajectory."""
from typing import Any, Optional

from autoagent0.verifiers.base import BaseVerifier


class DummyVerifier(BaseVerifier):
    """Always returns a high score, so the loop never triggers recovery.

    Stand-in until a real trajectory verifier is implemented. It ignores the
    trajectory contents and unconditionally approves it.
    """

    HIGH_SCORE = 1.0

    def score(self, trajectory: Any, current_info: Optional[Any] = None) -> float:
        return self.HIGH_SCORE
