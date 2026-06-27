#!/usr/bin/env python3
"""Offline calibration of AutoAgent0 uncertainty thresholds.

Usage:
    python scripts/calibrate_uncertainty.py \
        --runs /path/to/run_root \
        --horizon-steps 20 \
        --out-dir /path/to/out_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from autoagent0.calibration.loader import load_corpus  # noqa: E402
from autoagent0.calibration.sweep import (  # noqa: E402
    GridResult,
    MetricROC,
    joint_grid_search,
    metric_roc,
    per_metric_grid,
)


CURRENT_DEFAULTS = {
    "t_intra": 1.5,
    "t_cross": 2.0,
    "mode_count_high": 3,
    "silhouette_min": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run-root paths or globs")
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--grid-points", type=int, default=11)
    return parser.parse_args()


def write_histograms(df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, default in (("intra_m", 1.5), ("cross_m", 2.0), ("max_silhouette", 0.25)):
        series = df[metric].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series.to_numpy(), bins=60, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(default, color="firebrick", linestyle="--", label=f"current default = {default}")
        ax.set_xlabel(metric)
        ax.set_ylabel("frame count")
        ax.set_title(f"{metric} distribution across corpus")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"distribution_{metric}.png"), dpi=120)
        plt.close(fig)


def write_roc(roc: MetricROC, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not roc.fpr or not roc.tpr:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc.fpr, roc.tpr, color="steelblue", label=f"{roc.name} (AUC={roc.auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC: {roc.name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"roc_{roc.name}.png"), dpi=120)
    plt.close(fig)


def render_markdown(
    df: pd.DataFrame,
    rocs: Dict[str, MetricROC],
    best: GridResult | None,
    horizon_steps: int,
    target_recall: float,
) -> str:
    lines: List[str] = []
    lines.append("# Uncertainty calibration report\n")
    lines.append(f"- Corpus frames: **{len(df)}**")
    lines.append(f"- Unique runs: **{df['run_id'].nunique() if 'run_id' in df.columns else 0}**")
    lines.append(f"- Safety label: future_unsafe (collision OR nc<1 OR dac<1 within next {horizon_steps} steps)")
    lines.append(f"- Positive rate: **{df['future_unsafe'].mean():.4f}**")
    lines.append(f"- Target recall: {target_recall}\n")

    lines.append("## Distribution percentiles\n")
    pct_table = []
    for metric in ("intra_m", "cross_m", "max_silhouette"):
        series = df[metric].dropna()
        if series.empty:
            continue
        pcts = np.percentile(series.to_numpy(), [25, 50, 75, 90, 95, 99]).tolist()
        pct_table.append((metric, pcts))
    if pct_table:
        lines.append("| metric | p25 | p50 | p75 | p90 | p95 | p99 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for metric, pcts in pct_table:
            lines.append(f"| {metric} | " + " | ".join(f"{v:.3f}" for v in pcts) + " |")
        lines.append("")

    lines.append("## Per-metric ROC vs. future_unsafe\n")
    lines.append("| metric | AUC | threshold @ recall>=target | precision | recall | fraction flagged |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("intra_m", "cross_m", "max_silhouette"):
        roc = rocs.get(name)
        if roc is None:
            continue
        thr = "n/a" if roc.threshold_at_target_recall is None else f"{roc.threshold_at_target_recall:.3f}"
        prec = "n/a" if roc.precision_at_target is None else f"{roc.precision_at_target:.3f}"
        rec = "n/a" if roc.recall_at_target is None else f"{roc.recall_at_target:.3f}"
        frac = "n/a" if roc.fraction_flagged_at_target is None else f"{roc.fraction_flagged_at_target:.3f}"
        lines.append(f"| {name} | {roc.auc:.3f} | {thr} | {prec} | {rec} | {frac} |")
    lines.append("")

    lines.append("## Joint grid search\n")
    if best is None:
        lines.append("Grid search did not find a feasible configuration.")
    else:
        lines.append(f"Best at recall >= {target_recall}:")
        lines.append(f"- `t_intra={best.t_intra:.3f}`")
        lines.append(f"- `t_cross={best.t_cross:.3f}`")
        lines.append(f"- `mode_count_high={best.mode_count_high}`")
        lines.append(f"- precision={best.precision:.3f}, recall={best.recall:.3f}, f1={best.f1:.3f}")
        lines.append(f"- fraction routed to lean/fallback={best.fraction_lean_or_fallback:.3f}")
        lines.append("\nCurrent defaults: " + ", ".join(f"`{k}={v}`" for k, v in CURRENT_DEFAULTS.items()))
    return "\n".join(lines) + "\n"


def render_diff(best: GridResult | None) -> str:
    if best is None:
        return "# no feasible grid result; review report before changing config\n"
    return (
        "--- a/autoagent0/config.py\n"
        "+++ b/autoagent0/config.py\n"
        "@@\n"
        f'-    "UNCERTAINTY_T_INTRA": {CURRENT_DEFAULTS["t_intra"]},\n'
        f'+    "UNCERTAINTY_T_INTRA": {best.t_intra:.3f},\n'
        f'-    "UNCERTAINTY_T_CROSS": {CURRENT_DEFAULTS["t_cross"]},\n'
        f'+    "UNCERTAINTY_T_CROSS": {best.t_cross:.3f},\n'
        f'-    "UNCERTAINTY_MODE_COUNT_HIGH": {CURRENT_DEFAULTS["mode_count_high"]},\n'
        f'+    "UNCERTAINTY_MODE_COUNT_HIGH": {best.mode_count_high},\n'
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_corpus(args.runs, horizon_steps=args.horizon_steps)
    if df.empty:
        print(f"[calibrate] no frames loaded from {args.runs}", file=sys.stderr)
        with open(os.path.join(args.out_dir, "calibration_report.md"), "w") as fh:
            fh.write("# Uncertainty calibration report\n\nNo frames found.\n")
        return 1

    df.to_parquet(os.path.join(args.out_dir, "corpus.parquet"), index=False)

    write_histograms(df, args.out_dir)

    rocs: Dict[str, MetricROC] = {}
    for metric in ("intra_m", "cross_m", "max_silhouette"):
        roc = metric_roc(df, metric=metric, target_recall=args.target_recall)
        rocs[metric] = roc
        write_roc(roc, args.out_dir)

    intra_grid = per_metric_grid(df["intra_m"].dropna().to_numpy(), points=args.grid_points)
    cross_grid = per_metric_grid(df["cross_m"].dropna().to_numpy(), points=args.grid_points)
    if not intra_grid:
        intra_grid = [CURRENT_DEFAULTS["t_intra"]]
    if not cross_grid:
        cross_grid = [CURRENT_DEFAULTS["t_cross"]]

    best, grid = joint_grid_search(
        df,
        t_intra_grid=intra_grid,
        t_cross_grid=cross_grid,
        mode_count_high_grid=(2, 3),
        target_recall=args.target_recall,
    )

    report_json = {
        "horizon_steps": args.horizon_steps,
        "target_recall": args.target_recall,
        "n_frames": int(len(df)),
        "n_runs": int(df["run_id"].nunique()),
        "positive_rate": float(df["future_unsafe"].mean()),
        "current_defaults": CURRENT_DEFAULTS,
        "per_metric_roc": {
            name: {
                "auc": roc.auc,
                "threshold_at_target_recall": roc.threshold_at_target_recall,
                "precision": roc.precision_at_target,
                "recall": roc.recall_at_target,
                "fraction_flagged": roc.fraction_flagged_at_target,
            }
            for name, roc in rocs.items()
        },
        "best_grid_result": asdict(best) if best else None,
        "grid_search": [asdict(g) for g in grid],
        "intra_grid": intra_grid,
        "cross_grid": cross_grid,
    }
    with open(os.path.join(args.out_dir, "calibration_report.json"), "w") as fh:
        json.dump(report_json, fh, indent=2)

    with open(os.path.join(args.out_dir, "calibration_report.md"), "w") as fh:
        fh.write(render_markdown(df, rocs, best, args.horizon_steps, args.target_recall))

    with open(os.path.join(args.out_dir, "recommended_config.diff"), "w") as fh:
        fh.write(render_diff(best))

    print(f"[calibrate] wrote report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
