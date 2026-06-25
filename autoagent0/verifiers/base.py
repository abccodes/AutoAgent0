"""Base class for trajectory verifiers."""
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseVerifier(ABC):
    """Scores a planned trajectory before it is turned into a control action.

    Subclasses implement :meth:`score`, which receives the selected trajectory
    (the local plan, shape ``[T, 2]``) and returns a scalar score where higher
    means "safer / more trustworthy". The closed loop treats a score below its
    threshold as a rejected trajectory that must be replaced via recovery (e.g.
    regenerating with a different planner).
    """

    @abstractmethod
    def score(self, trajectory: Any, current_info: Optional[Any] = None) -> float:
        """Return a scalar score for ``trajectory`` (higher is better).

        Args:
            trajectory: The selected local plan, shape ``[T, 2]``.
            current_info: Optional environment info for the current step, in case
                the verifier needs context beyond the trajectory itself.
        """
        raise NotImplementedError
