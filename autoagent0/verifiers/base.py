"""Base class for action verifiers."""
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class BaseVerifier(ABC):
    """Scores a control action before it is applied to the environment.

    Subclasses implement :meth:`score`, which receives the finalized action dict
    (same format as the one passed to ``env.step``, i.e. ``{'acc', 'steer_rate'}``)
    and returns a scalar score where higher means "safer / more trustworthy". The
    closed loop treats a score below its threshold as a rejected action that must be
    replaced by a recovery action.
    """

    @abstractmethod
    def score(self, action: Mapping[str, Any], current_info: Optional[Mapping[str, Any]] = None) -> float:
        """Return a scalar score for ``action`` (higher is better).

        Args:
            action: The finalized control action, e.g. ``{'acc': ..., 'steer_rate': ...}``.
            current_info: Optional environment info for the current step, in case the
                verifier needs context beyond the action itself.
        """
        raise NotImplementedError
