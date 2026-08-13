"""Recover uncertainty features that were retained in historical HUGSIM traces."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _probabilities(scores: np.ndarray) -> np.ndarray:
    std = float(np.std(scores))
    logits = np.zeros_like(scores) if std <= 1e-12 else (scores - np.mean(scores)) / std
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / max(float(np.sum(weights)), 1e-12)


def score_features(values: Any) -> Dict[str, Optional[float]]:
    """Match the current monitor's score-only features using saved top-k scores."""

    unavailable = {
        "score_entropy": None,
        "effective_candidate_count": None,
        "score_margin_normalized": None,
    }
    try:
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return unavailable
    scores = scores[np.isfinite(scores)]
    if len(scores) < 2:
        return unavailable
    weights = _probabilities(scores)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    order = np.sort(scores)[::-1]
    return {
        "score_entropy": entropy / max(math.log(len(scores)), 1e-12),
        "effective_candidate_count": float(1.0 / np.sum(weights**2)),
        "score_margin_normalized": float(
            (order[0] - order[1]) / max(float(np.std(scores)), 1e-12)
        ),
    }


def _aligned_plans(plans: Sequence[Any]) -> Optional[np.ndarray]:
    arrays = []
    for plan in plans:
        try:
            value = np.asarray(plan, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if value.ndim != 2 or value.shape[1] < 2 or len(value) == 0:
            return None
        arrays.append(value[:, :2])
    if not arrays:
        return None
    horizon = min(len(value) for value in arrays)
    return np.stack([value[:horizon] for value in arrays])


def post_selection_features(debug: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute trajectory features on the saved post-selection pool.

    These fields are diagnostics only: accepted frames usually saved just the selected
    candidate, so this pool is not the raw planner proposal distribution.
    """

    names = {
        "post_dispersion_m": None,
        "post_pairwise_mean_m": None,
        "post_pairwise_p90_m": None,
        "post_endpoint_mean_m": None,
        "post_primitive_distance_m": None,
    }
    plans_raw = debug.get("overlay_candidate_plans")
    scores_raw = debug.get("candidate_pool_scores")
    sources = debug.get("overlay_candidate_sources") or debug.get("candidate_pool_sources")
    if not isinstance(plans_raw, list) or not isinstance(scores_raw, list) or not isinstance(sources, list):
        return names
    if not (len(plans_raw) == len(scores_raw) == len(sources)):
        return names
    learned_indices = [idx for idx, source in enumerate(sources) if source != "rule_based"]
    learned = _aligned_plans([plans_raw[idx] for idx in learned_indices])
    if learned is not None and len(learned) >= 2:
        scores = np.asarray([scores_raw[idx] for idx in learned_indices], dtype=np.float64)
        weights = _probabilities(scores)
        mean_plan = np.einsum("n,ntd->td", weights, learned)
        variance = np.einsum(
            "n,nt->t", weights, np.sum((learned - mean_plan[None]) ** 2, axis=2)
        )
        pair_mask = np.triu(np.ones((len(learned), len(learned)), dtype=bool), k=1)
        pairwise = np.mean(
            np.linalg.norm(learned[:, None] - learned[None, :], axis=3), axis=2
        )[pair_mask]
        endpoints = np.linalg.norm(
            learned[:, None, -1] - learned[None, :, -1], axis=2
        )[pair_mask]
        names.update(
            {
                "post_dispersion_m": float(np.mean(np.sqrt(np.maximum(variance, 0.0)))),
                "post_pairwise_mean_m": float(np.mean(pairwise)),
                "post_pairwise_p90_m": float(np.percentile(pairwise, 90)),
                "post_endpoint_mean_m": float(np.mean(endpoints)),
            }
        )
    rule_indices = [idx for idx, source in enumerate(sources) if source == "rule_based"]
    if learned_indices and rule_indices:
        best_idx = max(learned_indices, key=lambda idx: float(scores_raw[idx]))
        best = _aligned_plans([plans_raw[best_idx]])
        rules = _aligned_plans([plans_raw[idx] for idx in rule_indices])
        if best is not None and rules is not None:
            horizon = min(best.shape[1], rules.shape[1])
            distance = np.mean(
                np.linalg.norm(rules[:, :horizon] - best[0, None, :horizon], axis=2),
                axis=1,
            )
            names["post_primitive_distance_m"] = float(np.min(distance))
    return names


def raw_observation_features(debug: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract current-style geometry from the passive observation schema."""

    names: Dict[str, Optional[float]] = {
        "raw_candidate_count": None,
        "raw_dispersion_m": None,
        "raw_dispersion_normalized": None,
        "raw_pairwise_mean_m": None,
        "raw_pairwise_p90_m": None,
        "raw_endpoint_mean_m": None,
        "raw_score_entropy": None,
        "raw_effective_candidate_count": None,
        "raw_score_margin_normalized": None,
        "raw_primitive_distance_m": None,
    }
    observation = debug.get("autoagent0_uncertainty_observation")
    if not isinstance(observation, dict):
        return names
    learned = _aligned_plans(observation.get("learned_proposals") or [])
    try:
        scores = np.asarray(observation.get("learned_scores"), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return names
    if learned is None or len(learned) != len(scores):
        return names
    names["raw_candidate_count"] = float(len(learned))
    score_values = score_features(scores)
    names.update({f"raw_{key}": value for key, value in score_values.items()})
    if len(learned) >= 2:
        weights = _probabilities(scores)
        mean_plan = np.einsum("n,ntd->td", weights, learned)
        variance = np.einsum(
            "n,nt->t", weights, np.sum((learned - mean_plan[None]) ** 2, axis=2)
        )
        pair_mask = np.triu(np.ones((len(learned), len(learned)), dtype=bool), k=1)
        pairwise = np.mean(
            np.linalg.norm(learned[:, None] - learned[None, :], axis=3), axis=2
        )[pair_mask]
        endpoints = np.linalg.norm(
            learned[:, None, -1] - learned[None, :, -1], axis=2
        )[pair_mask]
        dispersion = float(np.mean(np.sqrt(np.maximum(variance, 0.0))))
        travel = float(np.mean(np.linalg.norm(learned[:, -1], axis=1)))
        names.update(
            {
                "raw_dispersion_m": dispersion,
                "raw_dispersion_normalized": dispersion / max(travel, 1e-6),
                "raw_pairwise_mean_m": float(np.mean(pairwise)),
                "raw_pairwise_p90_m": float(np.percentile(pairwise, 90)),
                "raw_endpoint_mean_m": float(np.mean(endpoints)),
            }
        )
    rules = _aligned_plans(observation.get("rule_based_proposals") or [])
    if rules is not None and len(learned):
        best = learned[int(np.argmax(scores))]
        horizon = min(len(best), rules.shape[1])
        distances = np.mean(
            np.linalg.norm(rules[:, :horizon] - best[None, :horizon], axis=2), axis=1
        )
        names["raw_primitive_distance_m"] = float(np.min(distances))
    return names


def temporal_change(
    local_plan: Any,
    origin_pose: Any,
    previous_world: Optional[np.ndarray],
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Compare selected plans after mapping the previous plan into the current ego frame."""

    try:
        plan = np.asarray(local_plan, dtype=np.float64)
        pose = np.asarray(origin_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None
    if plan.ndim != 2 or plan.shape[1] < 2 or pose.shape != (4, 4) or len(plan) == 0:
        return None, None
    homogeneous = np.column_stack(
        [plan[:, :2], np.zeros(len(plan), dtype=np.float64), np.ones(len(plan))]
    )
    current_world = (pose @ homogeneous.T).T[:, :3]
    if previous_world is None:
        return None, current_world
    previous_h = np.column_stack([previous_world, np.ones(len(previous_world))])
    previous_local = (np.linalg.inv(pose) @ previous_h.T).T[:, :2]
    horizon = min(len(plan), len(previous_local))
    change = float(np.mean(np.linalg.norm(plan[:horizon, :2] - previous_local[:horizon], axis=1)))
    return change, current_world
