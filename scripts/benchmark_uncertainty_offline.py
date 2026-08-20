#!/usr/bin/env python3
"""Run route-grouped offline uncertainty evaluation on historical HUGSIM runs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from autoagent0.calibration.loader import load_corpus  # noqa: E402
from autoagent0.calibration.offline import (  # noqa: E402
    LEGACY_FEATURES,
    POST_SELECTION_FEATURES,
    RAW_PROPOSAL_FEATURES,
    RECOVERABLE_FEATURES,
    binary_metrics,
    derive_future_outcomes,
    event_metrics,
    evaluate_fixed_policies,
    evaluate_logistic_cv,
    evaluate_quadrant_cv,
    fixed_policy_flags,
    make_group_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--max-coverage", type=float, default=0.15)
    parser.add_argument("--event-horizons", type=int, nargs="+", default=(5, 10, 20))
    parser.add_argument("--frame-dt-sec", type=float, default=0.25)
    parser.add_argument("--event-merge-gap-steps", type=int, default=5)
    return parser.parse_args()


def _serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def _format_optional(value: Any, format_spec: str, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:{format_spec}}{suffix}"


def _analyzer_provenance() -> Dict[str, Any]:
    source_paths = [
        "autoagent0/calibration/features.py",
        "autoagent0/calibration/labels.py",
        "autoagent0/calibration/loader.py",
        "autoagent0/calibration/offline.py",
        "scripts/benchmark_uncertainty_offline.py",
    ]
    digest = hashlib.sha256()
    for relative_path in source_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update((Path(REPO_ROOT) / relative_path).read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "source_files": source_paths,
    }


def render_report(
    df: pd.DataFrame,
    *,
    evaluated_df: pd.DataFrame,
    args: argparse.Namespace,
    legacy_fixed: Dict[str, Dict[str, Any]],
    fixed: Dict[str, Dict[str, Any]],
    quadrant: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    event_results: Dict[str, Dict[str, Any]],
    horizon_rows: List[Dict[str, Any]],
    scene_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = ["# HUGSIM offline uncertainty benchmark", ""]
    lines.extend([
        f"- Frames: **{len(df)}**",
        f"- Predictive cohort: **{len(evaluated_df)} currently-safe frames**",
        f"- Routes: **{df['run_id'].nunique()}**",
        f"- Scene groups: **{df['scene_group'].nunique()}** (all variants kept in one fold)",
        f"- Label horizon: **{args.horizon_steps} frames**",
        "- Primary target: future evaluator plan-risk (NC/DAC), not observed physical contact",
        "- Alert rates use recorded simulator timestamps (CLI frame duration is only a fallback)",
        f"- CV: **{args.folds} folds**, seed **{args.seed}**",
        f"- Operating coverage selected on training folds: **{args.min_coverage:.0%}-{args.max_coverage:.0%}**",
        "",
        "## Label prevalence",
        "",
        "| label | positive rate |",
        "|---|---:|",
    ])
    for label in ("future_unsafe", "future_collision", "future_nc_failure", "future_dac_failure"):
        lines.append(f"| {label} | {df[label].mean():.3f} |")
    lines.extend([
        "",
        f"Primary evaluation excludes the **{int(df['unsafe_now'].sum())} frames whose current plan fails NC/DAC or whose executed action records collision**. "
        f"The future-unsafe rate in the remaining cohort is **{evaluated_df['future_unsafe'].mean():.3f}**.",
    ])
    historical_nc = df[(df["nc_failure_now"] == 1) & df["nc_failure_type"].isna()]
    object_overlap = int(historical_nc["nc_object_overlap_evidence"].eq(True).sum())
    background_only = int(historical_nc["nc_object_overlap_evidence"].eq(False).sum())
    lines.extend([
        "",
        "## NC provenance",
        "",
        f"The corpus has **{len(historical_nc)} NC-failing frames** without exact failure-type metadata. "
        f"Object-box replay finds overlap evidence in **{object_overlap}**; the other **{background_only}** "
        "have no object overlap and therefore imply a static-background NC failure. An overlap is evidence, "
        "not an exact cause, because static geometry may fail earlier in the same plan.",
        "New evaluator outputs record the exact first `nc_failure_type` and `nc_failure_step`.",
    ])
    lines.extend(["", "## Feature availability", "", "| feature | available |", "|---|---:|"])
    for feature in RECOVERABLE_FEATURES + RAW_PROPOSAL_FEATURES + POST_SELECTION_FEATURES:
        lines.append(f"| {feature} | {df[feature].notna().mean():.1%} |")
    lines.extend([
        "",
        "Post-selection trajectory features are diagnostic only and are excluded from predictive models. "
        "Raw-proposal models are evaluated only when the proposal telemetry is available.",
        "",
        "## Fixed legacy policies",
        "",
        "| policy | all-frame precision | predictive precision | predictive recall | F1 | coverage | lift | median lead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, result in fixed.items():
        lead = result.get("median_lead_steps")
        lead_text = "n/a" if lead is None else f"{lead:.1f}"
        lines.append(
            f"| {name} | {legacy_fixed[name]['precision']:.3f} | {result['precision']:.3f} | {result['recall']:.3f} | "
            f"{result['f1']:.3f} | {result['coverage']:.3f} | {result['lift']:.2f}x | {lead_text} |"
        )
    lines.extend([
        "",
        "The all-frame column reproduces the historical plan-risk label semantics. Predictive columns require the current frame to be safe, so lead time is greater than zero.",
    ])
    lines.extend([
        "",
        "## Held-out grouped CV",
        "",
        "Thresholds and quadrant boundaries are chosen on training folds only.",
        "",
        "| model | precision | recall | F1 | coverage | lift | OOF AUROC | fold AUROC mean+/-std | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    cv_results = {"trained_low_quadrant": quadrant, **models}
    for name, result in cv_results.items():
        auroc = result.get("auroc")
        ap = result.get("average_precision")
        fold_auc = result.get("fold_auroc_mean")
        fold_auc_std = result.get("fold_auroc_std")
        fold_auc_text = "n/a" if fold_auc is None else f"{fold_auc:.3f}+/-{fold_auc_std:.3f}"
        lines.append(
            f"| {name} | {result['precision']:.3f} | {result['recall']:.3f} | "
            f"{result['f1']:.3f} | {result['coverage']:.3f} | {result['lift']:.2f}x | "
            f"{'n/a' if auroc is None else f'{auroc:.3f}'} | {fold_auc_text} | "
            f"{'n/a' if ap is None else f'{ap:.3f}'} |"
        )
    quadrant_fixed = fixed["fixed_quadrant_only"]
    fallback_fixed = fixed["legacy_runtime_fallback"]
    legacy_model = models["legacy_lr"]
    recoverable_model = models["recoverable_lr"]
    lines.extend([
        "",
        "## Event-level prediction",
        "",
        f"Unsafe frames separated by at most {args.event_merge_gap_steps} safe frames count as one event. "
        "Alerts must occur before onset within the label horizon.",
        "",
        "| policy | events detected | event recall | episode precision | false alerts/min | false-frame burden | median first-warning lead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, result in event_results.items():
        lead = result.get("median_first_warning_lead_steps")
        lines.append(
            f"| {name} | {int(result['detected_event_count'])}/{int(result['event_count'])} | "
            f"{result['event_recall']:.3f} | {result['event_precision']:.3f} | "
            f"{result['false_alerts_per_minute']:.2f} | {result['false_alert_frame_rate']:.1%} | "
            f"{'n/a' if lead is None else f'{lead:.1f}'} |"
        )
    lines.extend([
        "",
        "## Horizon sweep",
        "",
        "| horizon | policy | frame precision | frame recall | event recall | false alerts/min |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    selected_horizon_policies = {"fixed_quadrant_only", "trained_low_quadrant", "recoverable_lr"}
    for row in horizon_rows:
        if row["policy"] not in selected_horizon_policies:
            continue
        lines.append(
            f"| {row['horizon_steps']} | {row['policy']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['event_recall']:.3f} | {row['false_alerts_per_minute']:.2f} |"
        )
    fixed_scenes = [row for row in scene_rows if row["policy"] == "fixed_quadrant_only"]
    detected_by_scene = sorted(
        (int(row["detected_event_count"]) for row in fixed_scenes), reverse=True
    )
    top_three_detected = sum(detected_by_scene[:3])
    all_detected = sum(detected_by_scene)
    lines.extend([
        "",
        "## Findings",
        "",
        f"- The fixed low-intra/low-cross quadrant gives **{_format_optional(quadrant_fixed['lift'], '.2f', 'x')} lift** "
        f"at **{_format_optional(quadrant_fixed['coverage'], '.1%')} coverage**, with median lead "
        f"**{_format_optional(quadrant_fixed['median_lead_steps'], '.1f')} frames**.",
        f"- Adding the mode-count fallback lowers predictive precision from **{quadrant_fixed['precision']:.3f}** "
        f"to **{fallback_fixed['precision']:.3f}**.",
        f"- The recoverable feature model has AUROC **{_format_optional(recoverable_model['auroc'], '.3f')}** versus "
        f"**{_format_optional(legacy_model['auroc'], '.3f')}** for legacy features; saved score and temporal features do not improve this corpus.",
        f"- Recoverable-model fold AUROC is **{_format_optional(recoverable_model['fold_auroc_mean'], '.3f')}+/-"
        f"{_format_optional(recoverable_model['fold_auroc_std'], '.3f')}**, indicating substantial scene-family variation.",
        f"- The three most responsive scene families account for **{top_three_detected}/{all_detected}** "
        "fixed-quadrant event detections, so the signal is concentrated rather than broadly generalizing.",
    ])
    interpretation_limits = [
        "",
        "## Interpretation limits",
        "",
        "- Historical `nc` is a counterfactual collision check over the saved multi-step plan; it is not physical contact at that frame.",
        "- Static evaluator NC uses `scene.ply`, whose export is broader than the opacity-filtered point set used for physical simulator collision. Background-only NC may therefore be conservative evaluator risk rather than contact risk.",
        "- The target is future evaluator plan-risk, not the current AgenticDriving `coverage_rescue` target.",
        "- Results support feature screening and experiment design; they do not replace a held-out live A/B run.",
    ]
    if df["execution_outcome_available"].mean() >= 0.95:
        interpretation_limits.append(
            "- This corpus uses the current post-step execution-outcome contract, including terminal collision and route-departure state."
        )
    else:
        interpretation_limits.append(
            "- Most frames predate post-step outcome capture, so terminal collisions and route departures may be absent."
        )
    if df["raw_candidate_count"].notna().mean() >= 0.95:
        interpretation_limits.append(
            "- Full raw proposal telemetry is available, so raw-proposal models use the planner distribution rather than a post-selection pool."
        )
    else:
        interpretation_limits.append(
            "- Saved trajectory pools are post-selection and cannot reproduce the current proposal geometry exactly."
        )
    lines.extend(interpretation_limits)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not 0 < args.min_coverage <= args.max_coverage < 1:
        raise ValueError("coverage bounds must satisfy 0 < min <= max < 1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_corpus(args.runs, horizon_steps=args.horizon_steps)
    if df.empty:
        print("[offline] no uncertainty frames found", file=sys.stderr)
        return 1
    df["corpus_index"] = np.arange(len(df), dtype=np.int64)
    df = derive_future_outcomes(df, horizon_steps=args.horizon_steps)
    legacy_folds = make_group_folds(
        df, label_col="future_unsafe", n_splits=args.folds, seed=args.seed
    )
    legacy_fixed = evaluate_fixed_policies(df, legacy_folds, label_col="future_unsafe")
    evaluated_df = df[df["unsafe_now"] == 0].reset_index(drop=True)
    folds = make_group_folds(
        evaluated_df, label_col="future_unsafe", n_splits=args.folds, seed=args.seed
    )
    fixed = evaluate_fixed_policies(evaluated_df, folds, label_col="future_unsafe")
    quadrant = evaluate_quadrant_cv(
        evaluated_df, folds, label_col="future_unsafe",
        min_coverage=args.min_coverage, max_coverage=args.max_coverage,
    )
    feature_sets = {
        "legacy_no_mode_lr": LEGACY_FEATURES[:-1],
        "legacy_lr": LEGACY_FEATURES,
        "recoverable_lr": RECOVERABLE_FEATURES,
    }
    if min(df[feature].notna().mean() for feature in RAW_PROPOSAL_FEATURES) >= 0.90:
        feature_sets["raw_proposal_lr"] = RAW_PROPOSAL_FEATURES
    models = {
        name: evaluate_logistic_cv(
            evaluated_df, folds, name=name, features=features, label_col="future_unsafe",
            min_coverage=args.min_coverage, max_coverage=args.max_coverage,
        )
        for name, features in feature_sets.items()
    }
    prediction_arrays = fixed_policy_flags(df)
    for name, result in {"trained_low_quadrant": quadrant, **models}.items():
        full = np.zeros(len(df), dtype=bool)
        full[evaluated_df["corpus_index"].to_numpy(dtype=np.int64)] = result["predictions"]
        prediction_arrays[name] = full

    event_results = {}
    event_rows = []
    for name, flagged in prediction_arrays.items():
        metrics, rows = event_metrics(
            df, flagged, lookback_steps=args.horizon_steps,
            frame_dt_sec=args.frame_dt_sec, merge_gap_steps=args.event_merge_gap_steps,
        )
        event_results[name] = metrics
        event_rows.extend({"policy": name, **row} for row in rows)

    horizon_rows = []
    for horizon in sorted(set(max(1, int(value)) for value in args.event_horizons)):
        horizon_df = derive_future_outcomes(df, horizon_steps=horizon)
        horizon_eval = horizon_df[horizon_df["unsafe_now"] == 0].reset_index(drop=True)
        horizon_folds = make_group_folds(
            horizon_eval, label_col="future_unsafe", n_splits=args.folds, seed=args.seed
        )
        horizon_quadrant = evaluate_quadrant_cv(
            horizon_eval, horizon_folds, label_col="future_unsafe",
            min_coverage=args.min_coverage, max_coverage=args.max_coverage,
        )
        horizon_models = {
            name: evaluate_logistic_cv(
                horizon_eval, horizon_folds, name=name, features=features,
                label_col="future_unsafe", min_coverage=args.min_coverage,
                max_coverage=args.max_coverage,
            )
            for name, features in feature_sets.items()
        }
        horizon_predictions = fixed_policy_flags(horizon_df)
        for name, result in {"trained_low_quadrant": horizon_quadrant, **horizon_models}.items():
            full = np.zeros(len(horizon_df), dtype=bool)
            full[horizon_eval["corpus_index"].to_numpy(dtype=np.int64)] = result["predictions"]
            horizon_predictions[name] = full
        for name, flagged in horizon_predictions.items():
            cohort_flagged = flagged[horizon_eval["corpus_index"].to_numpy(dtype=np.int64)]
            frame_result = binary_metrics(
                horizon_eval["future_unsafe"].to_numpy(dtype=np.int64), cohort_flagged,
                steps_to_unsafe=horizon_eval["steps_to_unsafe"].to_numpy(dtype=np.float64),
            )
            events, _ = event_metrics(
                horizon_df, flagged, lookback_steps=horizon,
                frame_dt_sec=args.frame_dt_sec, merge_gap_steps=args.event_merge_gap_steps,
            )
            horizon_rows.append({"horizon_steps": horizon, "policy": name, **frame_result, **events})

    scene_rows = []
    for scene_group, scene_df in df.groupby("scene_group", sort=True):
        scene_indices = scene_df["corpus_index"].to_numpy(dtype=np.int64)
        scene_safe = scene_df["unsafe_now"].to_numpy(dtype=np.int64) == 0
        for name, flagged in prediction_arrays.items():
            frame_result = binary_metrics(
                scene_df.loc[scene_safe, "future_unsafe"].to_numpy(dtype=np.int64),
                flagged[scene_indices][scene_safe],
            )
            events, _ = event_metrics(
                scene_df.reset_index(drop=True), flagged[scene_indices],
                lookback_steps=args.horizon_steps, frame_dt_sec=args.frame_dt_sec,
                merge_gap_steps=args.event_merge_gap_steps,
            )
            scene_rows.append({"scene_group": scene_group, "policy": name, **frame_result, **events})

    review_columns = [
        "run_id", "scene_group", "frame_idx", "timestamp_sec", "future_unsafe", "steps_to_unsafe",
        *RECOVERABLE_FEATURES,
    ]
    review_rows = []
    for name in ("fixed_quadrant_only", "trained_low_quadrant", "recoverable_lr"):
        flags = prediction_arrays[name][evaluated_df["corpus_index"].to_numpy(dtype=np.int64)]
        labels = evaluated_df["future_unsafe"].to_numpy(dtype=bool)
        for category, mask in {
            "true_positive": flags & labels,
            "false_positive": flags & ~labels,
            "false_negative": ~flags & labels,
        }.items():
            cases = evaluated_df.loc[mask, review_columns].head(100).copy()
            cases.insert(0, "category", category)
            cases.insert(0, "policy", name)
            review_rows.append(cases)
    predictions = evaluated_df[["run_id", "scene_group", "frame_idx", "future_unsafe"]].copy()
    predictions["trained_low_quadrant"] = quadrant["predictions"]
    for name, result in models.items():
        predictions[f"{name}_flagged"] = result["predictions"]
        predictions[f"{name}_risk"] = result["probabilities"]
    predictions.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    df.to_parquet(out_dir / "corpus_enriched.parquet", index=False)
    pd.DataFrame(horizon_rows).to_csv(out_dir / "horizon_sweep.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(out_dir / "scene_breakdown.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out_dir / "event_review.csv", index=False)
    pd.concat(review_rows, ignore_index=True).to_csv(out_dir / "frame_review_cases.csv", index=False)
    payload = {
        "provenance": {
            **_analyzer_provenance(),
            "run_roots": args.runs,
            "horizon_steps": args.horizon_steps,
            "folds": args.folds,
            "seed": args.seed,
            "min_coverage": args.min_coverage,
            "max_coverage": args.max_coverage,
            "event_horizons": args.event_horizons,
            "frame_dt_sec": args.frame_dt_sec,
            "event_merge_gap_steps": args.event_merge_gap_steps,
            "label_semantics": {
                "nc": "saved multi-step plan intersects evaluator background or obstacle geometry",
                "collision": "post-step physical collision when execution_outcome is available; pre-step legacy fallback otherwise",
                "historical_limitation": "terminal post-step outcomes were not saved in the legacy corpus",
            },
        },
        "dataset": {
            "frames": len(df),
            "predictive_cohort_frames": len(evaluated_df),
            "predictive_cohort_positive_rate": float(evaluated_df["future_unsafe"].mean()),
            "routes": int(df["run_id"].nunique()),
            "scene_groups": int(df["scene_group"].nunique()),
        },
        "legacy_all_frame_fixed_policies": legacy_fixed,
        "predictive_fixed_policies": fixed,
        "trained_low_quadrant": _serializable(quadrant),
        "models": {name: _serializable(result) for name, result in models.items()},
        "event_metrics": event_results,
        "fold_groups": [
            {"train": list(fold.train_groups), "test": list(fold.test_groups)} for fold in folds
        ],
    }
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "benchmark.md").write_text(
        render_report(
            df, evaluated_df=evaluated_df, args=args, legacy_fixed=legacy_fixed,
            fixed=fixed, quadrant=quadrant, models=models, event_results=event_results,
            horizon_rows=horizon_rows, scene_rows=scene_rows,
        )
    )
    print(f"[offline] wrote benchmark to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
