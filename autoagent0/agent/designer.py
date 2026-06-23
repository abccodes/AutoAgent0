from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from autoagent0.agent.schemas import DesignBatch, TrajectoryCandidate


def candidate_row_to_trajectory(row: Dict[str, Any]) -> TrajectoryCandidate:
    """Convert the existing HUGSIM candidate-row dict into an AutoAgent0 view."""

    proposal_index = row.get("proposal_index")
    proposal_score = row.get("proposal_score")
    return TrajectoryCandidate(
        source=str(row.get("source") or "unknown"),
        local_plan=row.get("local_plan"),
        proposal_index=int(proposal_index) if isinstance(proposal_index, int) else proposal_index,
        proposal_score=float(proposal_score) if isinstance(proposal_score, (int, float)) else None,
        candidate_index=row.get("candidate_index"),
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"source", "local_plan", "proposal_index", "proposal_score", "candidate_index"}
        },
    )


def candidate_rows_to_trajectories(rows: Sequence[Dict[str, Any]]) -> List[TrajectoryCandidate]:
    return [candidate_row_to_trajectory(row) for row in rows]


def summarize_sources(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("source") or "unknown") for row in rows))


class Designer:
    """Behavior-preserving wrapper around existing candidate construction."""

    def from_existing_rows(
        self,
        *,
        learned_rows: Sequence[Dict[str, Any]] = (),
        rule_based_rows: Sequence[Dict[str, Any]] = (),
        combined_rows: Sequence[Dict[str, Any]] = (),
    ) -> DesignBatch:
        return DesignBatch(
            learned=candidate_rows_to_trajectories(learned_rows),
            rule_based=candidate_rows_to_trajectories(rule_based_rows),
            combined=candidate_rows_to_trajectories(combined_rows),
        )

