#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(__file__)
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from baseline_registry import load_registry, resolve_baseline_output_root


SCAN_ROOTS = [
    "/bigdata/aidan/outputs/benchmark/out/04_28_baselines",
    "/bigdata/aidan/outputs/benchmark/out/05_04_26",
    "/bigdata/aidan/outputs/benchmark/out/05_06_26",
    "/bigdata/aidan/outputs/benchmark/out/05_06_26_v2",
    "/bigdata/aidan/outputs/benchmark/out/05_22",
    "/bigdata/aidan/outputs/benchmark/out/05_26_26",
    "/bigdata/aidan/outputs/benchmark/out/waymo_",
    "/bigdata/aidan/outputs/benchmark/out/kitti360_",
]


def count_eval_files(root: str) -> int:
    total = 0
    for current_root, _, files in os.walk(root):
        if "eval.json" in files:
            total += 1
    return total


def planned_entries(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    for item in registry.get("historical_roots", []):
        run_type = "archive" if item.get("status") != "canonical" else "canonical"
        target_root = resolve_baseline_output_root(
            baseline_id=item["baseline_id"],
            dataset=item["dataset"],
            suite=item["suite"],
            run_type=run_type,
            archive_reason=item.get("archive_reason"),
            registry=registry,
        )
        entries.append(
            {
                "source_root": item["source_root"],
                "baseline_id": item["baseline_id"],
                "dataset": item["dataset"],
                "suite": item["suite"],
                "status": item["status"],
                "archive_reason": item.get("archive_reason"),
                "correctness_note": item.get("correctness_note"),
                "target_root": target_root,
                "eval_count": count_eval_files(item["source_root"]) if os.path.isdir(item["source_root"]) else 0,
                "exists": os.path.isdir(item["source_root"]),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    args = parser.parse_args()

    registry = load_registry()
    report = {
        "scan_roots": [
            {
                "root": root,
                "exists": os.path.isdir(root),
                "eval_count": count_eval_files(root) if os.path.isdir(root) else 0,
            }
            for root in SCAN_ROOTS
        ],
        "planned_migrations": planned_entries(registry),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Historical scan roots:")
    for item in report["scan_roots"]:
        print(f"- {item['root']}: exists={item['exists']} eval_count={item['eval_count']}")
    print("\nPlanned migrations:")
    for item in report["planned_migrations"]:
        print(
            f"- {item['source_root']} -> {item['target_root']} "
            f"(status={item['status']} eval_count={item['eval_count']} exists={item['exists']})"
        )


if __name__ == "__main__":
    main()
