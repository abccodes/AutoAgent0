#!/usr/bin/env python3
"""Aggregate PDMS + related metrics across the three A/B configs.

Reads eval.json from each config's run root and writes:
  - ab_results.csv           per-(scene, config) row
  - ab_summary.md            markdown table + per-scene delta table
  - ab_summary.json          machine-readable summary
  - ab_pdms_deltas.png       per-scene PDMS delta bars (critic_only vs baseline, full vs critic_only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

CONFIGS = ("baseline", "critic_only", "full")
DEFAULT_RUN_ROOT = "/bigdata/aidan/outputs/benchmark/out/baselines/drivor_autoagent0/nuscenes/full"
METRIC_KEYS = ("pdms", "nc", "dac", "ttc", "c", "rc", "hdscore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date-tag", required=True, help="e.g. 20260710")
    p.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    p.add_argument("--out-dir", default=None, help="default: {run_root}/_ab_summary_{date_tag}")
    return p.parse_args()


def load_scene(eval_path: Path) -> Dict[str, float]:
    try:
        d = json.loads(eval_path.read_text())
    except Exception as exc:
        return {"error": str(exc)}
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    perf = d.get("performance") or {}
    if isinstance(perf, dict):
        fc = perf.get("frame_count")
        if isinstance(fc, int):
            out["frame_count"] = float(fc)
    return out


def find_scene_dirs(config_root: Path) -> Dict[str, Path]:
    scenes: Dict[str, Path] = {}
    if not config_root.exists():
        return scenes
    for sub in sorted(config_root.iterdir()):
        if not sub.is_dir():
            continue
        eval_json = sub / "eval.json"
        if eval_json.exists():
            scenes[sub.name] = eval_json
    return scenes


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    tag = args.date_tag
    out_dir = Path(args.out_dir) if args.out_dir else run_root / f"_ab_summary_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each config
    per_config: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cfg in CONFIGS:
        cfg_root = run_root / f"ab-{tag}-{cfg}"
        scenes = find_scene_dirs(cfg_root)
        per_config[cfg] = {name: load_scene(ep) for name, ep in scenes.items()}
        print(f"[{cfg}] found {len(scenes)} scenes at {cfg_root}")

    # Union of scenes across configs (report missing per config)
    all_scenes = sorted(set().union(*(set(v.keys()) for v in per_config.values())))

    # CSV
    csv_path = out_dir / "ab_results.csv"
    with open(csv_path, "w") as fh:
        fh.write("scene,config,{}\n".format(",".join(METRIC_KEYS) + ",frame_count"))
        for scene in all_scenes:
            for cfg in CONFIGS:
                row = per_config[cfg].get(scene, {})
                vals = [f"{row.get(k, ''):.6f}" if k in row else "" for k in METRIC_KEYS]
                fc = row.get("frame_count", "")
                fh.write(f"{scene},{cfg},{','.join(vals)},{fc}\n")
    print(f"wrote {csv_path}")

    # Aggregate means across scenes (only include scenes present in all 3 configs for fair comparison)
    complete = [s for s in all_scenes if all(s in per_config[cfg] and "pdms" in per_config[cfg][s] for cfg in CONFIGS)]
    partial = [s for s in all_scenes if s not in complete]
    print(f"scenes complete across all 3 configs: {len(complete)}  (partial/missing: {len(partial)})")

    summary: Dict[str, Dict[str, float]] = {}
    for cfg in CONFIGS:
        agg = {}
        for k in METRIC_KEYS:
            vals = [per_config[cfg][s][k] for s in complete if k in per_config[cfg].get(s, {})]
            if vals:
                agg[k] = sum(vals) / len(vals)
                agg[f"{k}_min"] = min(vals)
                agg[f"{k}_max"] = max(vals)
        # scene-level rates: NC/DAC failure = fraction of scenes with metric<1
        for k in ("nc", "dac"):
            vals = [per_config[cfg][s].get(k) for s in complete]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                agg[f"{k}_failure_rate"] = sum(1 for v in vals if v < 1.0) / len(vals)
        summary[cfg] = agg

    # Markdown report
    md_path = out_dir / "ab_summary.md"
    lines: List[str] = []
    lines.append(f"# A/B test summary — date tag `{tag}`\n")
    lines.append(f"- scenes complete across all 3 configs: **{len(complete)}**")
    if partial:
        lines.append(f"- partial/missing (excluded from means): {len(partial)}")
        for s in partial:
            missing = [c for c in CONFIGS if s not in per_config[c]]
            lines.append(f"  - `{s}` missing in: {', '.join(missing)}")
    lines.append("")

    lines.append("## Mean metrics across complete scenes\n")
    lines.append("| metric | baseline | critic_only | full | Δ(full - baseline) | Δ(full - critic_only) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in METRIC_KEYS:
        b = summary["baseline"].get(k)
        c = summary["critic_only"].get(k)
        f = summary["full"].get(k)
        if b is None or c is None or f is None:
            continue
        d_full_base = f - b
        d_full_crit = f - c
        lines.append(f"| {k} | {b:.4f} | {c:.4f} | {f:.4f} | {d_full_base:+.4f} | {d_full_crit:+.4f} |")
    lines.append("")

    lines.append("## Scene-level failure rates (fraction of scenes with metric < 1.0)\n")
    lines.append("| metric | baseline | critic_only | full |")
    lines.append("|---|---:|---:|---:|")
    for k in ("nc", "dac"):
        fr_key = f"{k}_failure_rate"
        b = summary["baseline"].get(fr_key)
        c = summary["critic_only"].get(fr_key)
        f = summary["full"].get(fr_key)
        if b is None:
            continue
        lines.append(f"| {k} failure rate | {b:.3f} | {c:.3f} | {f:.3f} |")
    lines.append("")

    lines.append("## Per-scene PDMS\n")
    lines.append("| scene | baseline | critic_only | full | Δ(full - baseline) |")
    lines.append("|---|---:|---:|---:|---:|")
    for scene in complete:
        b = per_config["baseline"][scene].get("pdms")
        c = per_config["critic_only"][scene].get("pdms")
        f = per_config["full"][scene].get("pdms")
        if b is None or c is None or f is None:
            continue
        lines.append(f"| {scene} | {b:.3f} | {c:.3f} | {f:.3f} | {f - b:+.3f} |")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")

    # JSON dump
    json_path = out_dir / "ab_summary.json"
    with open(json_path, "w") as fh:
        json.dump({
            "date_tag": tag,
            "run_root": str(run_root),
            "complete_scenes": complete,
            "partial_scenes": partial,
            "summary": summary,
            "per_scene": {c: per_config[c] for c in CONFIGS},
        }, fh, indent=2)
    print(f"wrote {json_path}")

    # Bar chart of per-scene PDMS deltas
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        if complete:
            deltas_full_base = [per_config["full"][s]["pdms"] - per_config["baseline"][s]["pdms"] for s in complete]
            deltas_full_crit = [per_config["full"][s]["pdms"] - per_config["critic_only"][s]["pdms"] for s in complete]
            x = np.arange(len(complete))
            w = 0.4
            fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(complete)), 5))
            ax.bar(x - w/2, deltas_full_base, w, label="full − baseline", color="steelblue")
            ax.bar(x + w/2, deltas_full_crit, w, label="full − critic_only", color="firebrick")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace("scene-", "") for s in complete], rotation=75, fontsize=8)
            ax.set_ylabel("Δ PDMS", fontsize=10)
            ax.set_title(f"Per-scene PDMS deltas — A/B {tag}", fontsize=11)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "ab_pdms_deltas.png", dpi=140)
            plt.close(fig)
            print(f"wrote {out_dir / 'ab_pdms_deltas.png'}")
    except Exception as exc:
        print(f"skipping chart (matplotlib unavailable): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
