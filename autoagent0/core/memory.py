from __future__ import annotations

from typing import Any, Dict, List


class NoOpMemory:
    """Memory boundary for future verifier/recovery patterns."""

    def recall(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def record(self, event: Dict[str, Any]) -> None:
        return None

