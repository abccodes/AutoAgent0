from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearnedExpertRef:
    """Reference to an existing learned planner backend."""

    name: str
    client_module: str


RAP_EXPERT = LearnedExpertRef(name="rap", client_module="planners.rap.client")
DRIVOR_EXPERT = LearnedExpertRef(name="drivor", client_module="planners.drivor.client")

