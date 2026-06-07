from __future__ import annotations

from typing import Any, Dict


def selected_plan_from_payload(plan_payload: Dict[str, Any]) -> Any:
    """Return the selected trajectory from an existing HUGSIM plan payload."""

    return plan_payload.get("selected_plan")


def attach_agent_trace(plan_payload: Dict[str, Any], agent_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Attach trace metadata without altering the selected trajectory fields."""

    updated = dict(plan_payload)
    updated["agent_trace"] = agent_trace
    return updated

