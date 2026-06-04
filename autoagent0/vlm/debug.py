from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def clear_vlm_debug_artifacts(debug_dir: Path) -> None:
    patterns = [
        "frame_*_candidates.jpg",
        "frame_*_candidates_*.jpg",
        "frame_*_gate_*.jpg",
        "frame_*_planner_gate_*.jpg",
        "frame_*_result.json",
        "frame_*_planner_gate.json",
    ]
    for pattern in patterns:
        for stale_path in debug_dir.glob(pattern):
            stale_path.unlink(missing_ok=True)


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(record) + "\n")


def write_debug_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def unlink_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


__all__ = [
    "append_jsonl",
    "clear_vlm_debug_artifacts",
    "unlink_paths",
    "write_debug_json",
]

