"""A placeholder verifier that always approves the action."""
from typing import Any, Mapping, Optional

from autoagent0.verifiers.base import BaseVerifier


class DummyVerifier(BaseVerifier):
    """Always returns a high score, so the loop never triggers recovery.

    This is a stand-in until a real verifier is implemented. It ignores the action
    contents and unconditionally approves it.
    """

    HIGH_SCORE = 1.0

    def score(self, action: Mapping[str, Any], current_info: Optional[Mapping[str, Any]] = None) -> float:
        return self.HIGH_SCORE
