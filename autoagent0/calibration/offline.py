"""Leakage-safe offline evaluation for historical HUGSIM uncertainty traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold


LEGACY_FEATURES = ("intra_m", "cross_m", "max_silhouette", "mode_count")
RECOVERABLE_FEATURES = LEGACY_FEATURES + (
    "score_entropy",
    "effective_candidate_count",
    "score_margin_normalized",
    "temporal_selected_change_m",
)
POST_SELECTION_FEATURES = (
    "post_dispersion_m",
    "post_pairwise_mean_m",
    "post_pairwise_p90_m",
    "post_endpoint_mean_m",
    "post_primitive_distance_m",
)
RAW_PROPOSAL_FEATURES = (
    "raw_dispersion_m",
    "raw_dispersion_normalized",
    "raw_pairwise_mean_m",
    "raw_pairwise_p90_m",
    "raw_endpoint_mean_m",
    "raw_score_entropy",
    "raw_effective_candidate_count",
    "raw_score_margin_normalized",
    "raw_primitive_distance_m",
)


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray
    train_groups: Tuple[str, ...]
    test_groups: Tuple[str, ...]


def derive_future_outcomes(df: pd.DataFrame, *, horizon_steps: int) -> pd.DataFrame:
    """Re-label a loaded corpus from current unsafe frames without crossing routes."""

    output = df.copy()
    output["future_unsafe"] = 0
    output["steps_to_unsafe"] = np.nan
    horizon = max(1, int(horizon_steps))
    for _run_id, indices in output.groupby("run_id", sort=False).groups.items():
        ordered = np.asarray(sorted(indices, key=lambda idx: int(output.at[idx, "frame_idx"])))
        unsafe = output.loc[ordered, "unsafe_now"].to_numpy(dtype=bool)
        labels = np.zeros(len(ordered), dtype=np.int64)
        steps = np.full(len(ordered), np.nan)
        for position in range(len(ordered)):
            offsets = np.flatnonzero(unsafe[position : position + horizon])
            if len(offsets):
                labels[position] = 1
                steps[position] = float(offsets[0])
        output.loc[ordered, "future_unsafe"] = labels
        output.loc[ordered, "steps_to_unsafe"] = steps
    return output


def event_metrics(
    df: pd.DataFrame,
    flagged: np.ndarray,
    *,
    lookback_steps: int,
    frame_dt_sec: float = 0.5,
    merge_gap_steps: int = 5,
) -> Tuple[Dict[str, float | None], List[Dict[str, Any]]]:
    """Score alerts against unsafe episodes using only pre-onset warnings."""

    alerts = np.asarray(flagged, dtype=bool)
    if len(alerts) != len(df):
        raise ValueError("event alert count must match corpus frame count")
    lookback = max(1, int(lookback_steps))
    event_rows: List[Dict[str, Any]] = []
    false_alert_episodes = 0
    false_alert_frames = 0
    eligible_false_frames = 0
    total_duration_sec = 0.0
    for run_id, indices in df.groupby("run_id", sort=False).groups.items():
        ordered = np.asarray(sorted(indices, key=lambda idx: int(df.at[idx, "frame_idx"])))
        unsafe = df.loc[ordered, "unsafe_now"].to_numpy(dtype=bool)
        run_alerts = alerts[ordered]
        timestamps = (
            df.loc[ordered, "timestamp_sec"].to_numpy(dtype=np.float64)
            if "timestamp_sec" in df.columns else np.zeros(0, dtype=np.float64)
        )
        finite_timestamps = timestamps[np.isfinite(timestamps)]
        if len(finite_timestamps) >= 2:
            deltas = np.diff(finite_timestamps)
            positive_deltas = deltas[deltas > 0]
            final_step = float(np.median(positive_deltas)) if len(positive_deltas) else frame_dt_sec
            total_duration_sec += float(finite_timestamps[-1] - finite_timestamps[0] + final_step)
        else:
            total_duration_sec += len(ordered) * float(frame_dt_sec)
        unsafe_positions = np.flatnonzero(unsafe)
        event_bounds: List[Tuple[int, int]] = []
        if len(unsafe_positions):
            start = previous = int(unsafe_positions[0])
            for position in unsafe_positions[1:]:
                position = int(position)
                if position - previous - 1 > max(0, int(merge_gap_steps)):
                    event_bounds.append((start, previous))
                    start = position
                previous = position
            event_bounds.append((start, previous))
        covered_alerts = np.zeros(len(ordered), dtype=bool)
        event_occupied = np.zeros(len(ordered), dtype=bool)
        for event_index, (onset, event_end) in enumerate(event_bounds):
            window_start = max(0, int(onset) - lookback)
            warning_positions = np.flatnonzero(run_alerts[window_start:onset]) + window_start
            covered_alerts[window_start:onset] = True
            event_occupied[onset : event_end + 1] = True
            detected = len(warning_positions) > 0
            first_warning = int(warning_positions[0]) if detected else None
            event_rows.append(
                {
                    "run_id": run_id,
                    "scene_group": str(df.at[ordered[onset], "scene_group"]),
                    "event_index": event_index,
                    "onset_frame_idx": int(df.at[ordered[onset], "frame_idx"]),
                    "detected": detected,
                    "first_warning_frame_idx": (
                        None if first_warning is None else int(df.at[ordered[first_warning], "frame_idx"])
                    ),
                    "lead_steps": None if first_warning is None else int(onset - first_warning),
                }
            )
        eligible_false = ~covered_alerts & ~event_occupied
        false = run_alerts & eligible_false
        false_starts = false & ~np.concatenate([[False], false[:-1]])
        false_alert_episodes += int(np.sum(false_starts))
        false_alert_frames += int(np.sum(false))
        eligible_false_frames += int(np.sum(eligible_false))
    detected_rows = [row for row in event_rows if row["detected"]]
    leads = [int(row["lead_steps"]) for row in detected_rows]
    event_count = len(event_rows)
    detected_count = len(detected_rows)
    duration_minutes = total_duration_sec / 60.0
    precision = detected_count / max(detected_count + false_alert_episodes, 1)
    return {
        "event_count": float(event_count),
        "detected_event_count": float(detected_count),
        "event_recall": detected_count / max(event_count, 1),
        "event_precision": precision,
        "false_alert_episodes": float(false_alert_episodes),
        "false_alerts_per_minute": false_alert_episodes / max(duration_minutes, 1e-12),
        "false_alert_frame_rate": false_alert_frames / max(eligible_false_frames, 1),
        "median_first_warning_lead_steps": float(np.median(leads)) if leads else None,
    }, event_rows


def make_group_folds(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str = "scene_group",
    n_splits: int = 5,
    seed: int = 17,
) -> List[Fold]:
    """Stratify labels while keeping all variants of a NuScenes scene together."""

    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("offline benchmark needs at least two scene groups")
    split_count = min(max(2, int(n_splits)), len(unique_groups))
    labels = df[label_col].to_numpy(dtype=np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=split_count, shuffle=True, random_state=int(seed)
    )
    folds: List[Fold] = []
    for train, test in splitter.split(np.zeros(len(df)), labels, groups):
        train_groups = tuple(sorted(set(groups[train])))
        test_groups = tuple(sorted(set(groups[test])))
        if set(train_groups) & set(test_groups):
            raise AssertionError("scene group leaked across train and test")
        folds.append(Fold(train, test, train_groups, test_groups))
    return folds


def binary_metrics(
    labels: np.ndarray,
    flagged: np.ndarray,
    *,
    steps_to_unsafe: np.ndarray | None = None,
) -> Dict[str, float | None]:
    labels = np.asarray(labels, dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)
    tp = int(np.sum(labels & flagged))
    fp = int(np.sum(~labels & flagged))
    positives = int(np.sum(labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    base_rate = float(np.mean(labels))
    result: Dict[str, float | None] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": float(np.mean(flagged)),
        "lift": precision / base_rate if base_rate > 0 else None,
        "true_positives": float(tp),
        "false_positives": float(fp),
    }
    if steps_to_unsafe is not None:
        steps = np.asarray(steps_to_unsafe, dtype=np.float64)
        valid = labels & flagged & np.isfinite(steps)
        result["median_lead_steps"] = float(np.median(steps[valid])) if np.any(valid) else None
    return result


def _score_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_coverage: float,
    max_coverage: float,
) -> float:
    candidates = np.unique(
        np.quantile(scores, np.linspace(1.0 - max_coverage, 1.0 - min_coverage, 31))
    )
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(labels, scores >= threshold)
        feasible.append((float(metrics["f1"]), float(metrics["precision"]), float(threshold)))
    return max(feasible)[2]


def _summarize_folds(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "precision", "recall", "f1", "coverage", "lift", "median_lead_steps",
        "auroc", "average_precision",
    )
    summary: Dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[f"fold_{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"fold_{key}_std"] = float(np.std(values)) if values else None
    return summary


def evaluate_fixed_policies(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    label_col: str,
) -> Dict[str, Dict[str, Any]]:
    policies = fixed_policy_flags(df)
    labels = df[label_col].to_numpy(dtype=np.int64)
    lead = df["steps_to_unsafe"].to_numpy(dtype=np.float64)
    output: Dict[str, Dict[str, Any]] = {}
    for name, flagged in policies.items():
        fold_rows = [
            binary_metrics(labels[fold.test], flagged[fold.test], steps_to_unsafe=lead[fold.test])
            for fold in folds
        ]
        output[name] = {
            **binary_metrics(labels, flagged, steps_to_unsafe=lead),
            **_summarize_folds(fold_rows),
            "folds": fold_rows,
        }
    return output


def fixed_policy_flags(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return the historical fixed routing decisions for ablation reporting."""

    intra = df["intra_m"].to_numpy(dtype=np.float64)
    cross = df["cross_m"].to_numpy(dtype=np.float64)
    modes = df["mode_count"].to_numpy(dtype=np.int64)
    quadrant = (intra <= 0.20) & (cross <= 2.40)
    return {
        "fixed_quadrant_only": quadrant,
        "legacy_runtime_fallback": quadrant | (modes >= 3),
        "legacy_runtime_active": quadrant | (intra <= 0.20) | (cross <= 2.40) | (modes >= 2),
    }


def evaluate_quadrant_cv(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    label_col: str,
    min_coverage: float,
    max_coverage: float,
) -> Dict[str, Any]:
    labels = df[label_col].to_numpy(dtype=np.int64)
    intra = df["intra_m"].to_numpy(dtype=np.float64)
    cross = df["cross_m"].to_numpy(dtype=np.float64)
    lead = df["steps_to_unsafe"].to_numpy(dtype=np.float64)
    predictions = np.zeros(len(df), dtype=bool)
    fold_rows = []
    for fold_index, fold in enumerate(folds):
        candidates = []
        for q_intra in np.linspace(0.05, 0.50, 10):
            t_intra = float(np.quantile(intra[fold.train], q_intra))
            for q_cross in np.linspace(0.05, 0.50, 10):
                t_cross = float(np.quantile(cross[fold.train], q_cross))
                train_flagged = (intra[fold.train] < t_intra) & (cross[fold.train] < t_cross)
                coverage = float(np.mean(train_flagged))
                if min_coverage <= coverage <= max_coverage:
                    metrics = binary_metrics(labels[fold.train], train_flagged)
                    candidates.append(
                        (float(metrics["f1"]), float(metrics["precision"]), t_intra, t_cross)
                    )
        if not candidates:
            raise ValueError("no quadrant threshold met the requested coverage range")
        _, _, t_intra, t_cross = max(candidates)
        test_flagged = (intra[fold.test] < t_intra) & (cross[fold.test] < t_cross)
        predictions[fold.test] = test_flagged
        fold_rows.append(
            {
                "fold": fold_index,
                "t_intra": t_intra,
                "t_cross": t_cross,
                **binary_metrics(
                    labels[fold.test], test_flagged, steps_to_unsafe=lead[fold.test]
                ),
            }
        )
    return {
        **binary_metrics(labels, predictions, steps_to_unsafe=lead),
        **_summarize_folds(fold_rows),
        "folds": fold_rows,
        "predictions": predictions,
    }


def evaluate_logistic_cv(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    name: str,
    features: Sequence[str],
    label_col: str,
    min_coverage: float,
    max_coverage: float,
) -> Dict[str, Any]:
    labels = df[label_col].to_numpy(dtype=np.int64)
    lead = df["steps_to_unsafe"].to_numpy(dtype=np.float64)
    probabilities = np.zeros(len(df), dtype=np.float64)
    predictions = np.zeros(len(df), dtype=bool)
    fold_rows = []
    for fold_index, fold in enumerate(folds):
        transformer = ColumnTransformer(
            [("features", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), list(features))]
        )
        model = Pipeline([
            ("prepare", transformer),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)),
        ])
        model.fit(df.iloc[fold.train], labels[fold.train])
        train_scores = model.predict_proba(df.iloc[fold.train])[:, 1]
        threshold = _select_threshold(
            labels[fold.train], train_scores,
            min_coverage=min_coverage, max_coverage=max_coverage,
        )
        test_scores = model.predict_proba(df.iloc[fold.test])[:, 1]
        test_flagged = test_scores >= threshold
        probabilities[fold.test] = test_scores
        predictions[fold.test] = test_flagged
        fold_rows.append(
            {
                "fold": fold_index,
                "threshold": threshold,
                **binary_metrics(
                    labels[fold.test], test_flagged, steps_to_unsafe=lead[fold.test]
                ),
                **_score_metrics(labels[fold.test], test_scores),
            }
        )
    return {
        "name": name,
        "features": list(features),
        **binary_metrics(labels, predictions, steps_to_unsafe=lead),
        **_score_metrics(labels, probabilities),
        **_summarize_folds(fold_rows),
        "folds": fold_rows,
        "predictions": predictions,
        "probabilities": probabilities,
    }
