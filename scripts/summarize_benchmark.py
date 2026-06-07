#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HUGSIM benchmark eval.json files")
    parser.add_argument("--root", required=True, help="Directory containing per-scenario output folders")
    return parser.parse_args()


def read_records(root: Path) -> List[Dict[str, object]]:
    records = []
    for eval_path in sorted(root.glob("*/eval.json")):
        run_dir = eval_path.parent
        scene_name, _, mode = run_dir.name.partition("_")
        if not mode:
            mode = "unknown"
        difficulty = mode.split("_", 1)[0]
        with open(eval_path, "r") as fh:
            metrics = json.load(fh)
        records.append(
            {
                "run_dir": run_dir.name,
                "scene_name": scene_name,
                "mode": mode,
                "difficulty": difficulty,
                "pdms": float(metrics.get("pdms", 0.0)),
                "rc": float(metrics.get("rc", 0.0)),
                "hdscore": float(metrics.get("hdscore", 0.0)),
            }
        )
    return records


def summarize(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(record["difficulty"], []).append(record)

    summary: Dict[str, Dict[str, float]] = {}
    for difficulty, items in grouped.items():
        count = len(items)
        summary[difficulty] = {
            "count": count,
            "pdms": sum(item["pdms"] for item in items) / count,
            "rc": sum(item["rc"] for item in items) / count,
            "hdscore": sum(item["hdscore"] for item in items) / count,
        }
    return summary


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    records = read_records(root)
    if not records:
        raise SystemExit(f"No eval.json files found under {root}")

    print("Per-scenario results")
    for record in records:
        print(
            f"{record['run_dir']}: pdms={record['pdms']:.4f} "
            f"rc={record['rc']:.4f} hdscore={record['hdscore']:.4f}"
        )

    print("\nBy difficulty")
    for difficulty, metrics in summarize(records).items():
        print(
            f"{difficulty}: count={metrics['count']} pdms={metrics['pdms']:.4f} "
            f"rc={metrics['rc']:.4f} hdscore={metrics['hdscore']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
