"""ROC analysis and joint threshold grid search for uncertainty calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricROC:
    name: str
    auc: float
    threshold_at_target_recall: Optional[float]
    precision_at_target: Optional[float]
    recall_at_target: Optional[float]
    fraction_flagged_at_target: Optional[float]
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]


def _compute_roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep thresholds high→low, return (fpr, tpr, thresholds) including endpoints."""

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)
    if total_pos == 0 or total_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # Keep only points where the next score is different (avoid duplicates from ties).
    distinct = np.concatenate([np.diff(s) != 0, [True]])
    fpr = fp[distinct] / total_neg
    tpr = tp[distinct] / total_pos
    thresholds = s[distinct]
    # Prepend (0, 0) at +inf and append (1, 1) at -inf for proper trapezoidal AUC.
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[np.inf], thresholds])
    return fpr, tpr, thresholds


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


def metric_roc(
    df: pd.DataFrame,
    *,
    metric: str,
    label_col: str = "future_unsafe",
    target_recall: float = 0.8,
) -> MetricROC:
    sub = df[[metric, label_col]].dropna()
    if sub.empty:
        return MetricROC(metric, float("nan"), None, None, None, None, [], [], [])

    scores = sub[metric].to_numpy(dtype=np.float64)
    labels = sub[label_col].to_numpy(dtype=np.int64)
    fpr, tpr, thresholds = _compute_roc(scores, labels)
    auc = _auc(fpr, tpr)

    target_threshold: Optional[float] = None
    precision_at_target: Optional[float] = None
    recall_at_target: Optional[float] = None
    fraction_flagged: Optional[float] = None

    hits = np.where(tpr >= target_recall)[0]
    if hits.size:
        idx = int(hits[0])
        t = float(thresholds[idx])
        target_threshold = t if np.isfinite(t) else float(scores.min())
        flagged = scores >= target_threshold
        tp = int(np.sum(flagged & (labels == 1)))
        fp = int(np.sum(flagged & (labels == 0)))
        denom = tp + fp
        precision_at_target = float(tp / denom) if denom else 0.0
        total_pos = int(labels.sum())
        recall_at_target = float(tp / total_pos) if total_pos else 0.0
        fraction_flagged = float(flagged.mean())

    return MetricROC(
        name=metric,
        auc=auc,
        threshold_at_target_recall=target_threshold,
        precision_at_target=precision_at_target,
        recall_at_target=recall_at_target,
        fraction_flagged_at_target=fraction_flagged,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
    )


@dataclass
class GridResult:
    t_intra: float
    t_cross: float
    mode_count_high: int
    fraction_lean_or_fallback: float
    precision: float
    recall: float
    f1: float


def _classify_zone(
    intra: np.ndarray,
    cross: np.ndarray,
    modes: np.ndarray,
    *,
    t_intra: float,
    t_cross: float,
    mode_count_high: int,
) -> np.ndarray:
    intra_high = intra >= t_intra
    cross_high = cross >= t_cross
    modes_fallback = modes >= int(mode_count_high)
    modes_lean = (modes >= 2) & (~modes_fallback)
    fallback = (intra_high & cross_high) | modes_fallback
    lean = (intra_high | cross_high | modes_lean) & (~fallback)
    return (fallback | lean).astype(np.int64)


def joint_grid_search(
    df: pd.DataFrame,
    *,
    t_intra_grid: Sequence[float],
    t_cross_grid: Sequence[float],
    mode_count_high_grid: Sequence[int] = (2, 3),
    label_col: str = "future_unsafe",
    target_recall: float = 0.8,
) -> Tuple[Optional[GridResult], List[GridResult]]:
    sub = df[["intra_m", "cross_m", "mode_count", label_col]].dropna()
    if sub.empty:
        return None, []
    intra = sub["intra_m"].to_numpy(dtype=np.float64)
    cross = sub["cross_m"].to_numpy(dtype=np.float64)
    modes = sub["mode_count"].to_numpy(dtype=np.int64)
    labels = sub[label_col].to_numpy(dtype=np.int64)
    total_pos = int(labels.sum())

    results: List[GridResult] = []
    for ti in t_intra_grid:
        for tc in t_cross_grid:
            for mh in mode_count_high_grid:
                flagged = _classify_zone(
                    intra, cross, modes,
                    t_intra=float(ti), t_cross=float(tc), mode_count_high=int(mh),
                )
                tp = int(np.sum((flagged == 1) & (labels == 1)))
                fp = int(np.sum((flagged == 1) & (labels == 0)))
                denom = tp + fp
                precision = float(tp / denom) if denom else 0.0
                recall = float(tp / total_pos) if total_pos else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                fraction_flagged = float(flagged.mean())
                results.append(
                    GridResult(
                        t_intra=float(ti),
                        t_cross=float(tc),
                        mode_count_high=int(mh),
                        fraction_lean_or_fallback=fraction_flagged,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                    )
                )

    feasible = [r for r in results if r.recall >= target_recall]
    if feasible:
        best = max(feasible, key=lambda r: (r.precision, -r.fraction_lean_or_fallback))
    else:
        best = max(results, key=lambda r: r.f1) if results else None
    return best, results


def per_metric_grid(values: np.ndarray, points: int = 11) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    pct = np.linspace(5, 95, points)
    grid = np.percentile(finite, pct).tolist()
    seen = set()
    out: List[float] = []
    for v in grid:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out
