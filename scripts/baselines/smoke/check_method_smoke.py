#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
COMMON_DIR = THIS_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from baseline_registry import get_baseline_entry, resolve_scene_output_path  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


VLM_METHOD_FAMILIES = {
    "solo_learned",
    "vlm_intervention",
    "choice_a_rule_merge",
    "choice_b_rule_gate",
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_json(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def _metric(eval_data: Dict[str, Any], key: str) -> str:
    value = eval_data.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def check_method(
    *,
    baseline_id: str,
    dataset: str,
    suite: str,
    run_id: str,
    scenario_path: str,
) -> Dict[str, str]:
    scenario = OmegaConf.load(scenario_path)
    output_dir = Path(
        resolve_scene_output_path(
            baseline_id=baseline_id,
            dataset=dataset,
            suite=suite,
            scene_name=str(scenario.scene_name),
            mode=str(scenario.mode),
            run_type="debug",
            run_variant=run_id,
        )
    )
    entry = get_baseline_entry(baseline_id)
    method_family = str(entry.get("method_family", ""))
    eval_path = output_dir / "eval.json"
    output_txt = output_dir / "output.txt"
    issues: List[str] = []
    evidence: List[str] = []

    eval_data = _load_json(eval_path)
    if eval_data is None:
        issues.append("missing_eval")
        eval_data = {}
    if not output_txt.exists():
        issues.append("missing_output_txt")

    if method_family in VLM_METHOD_FAMILIES:
        latency_path = output_dir / "vlm_debug" / "latency_summary.json"
        if latency_path.exists():
            evidence.append("latency")
        else:
            issues.append("missing_latency_summary")

    if method_family == "choice_a_rule_merge":
        frame_path = _first_json(output_dir / "vlm_debug", "frame_*_result.json")
        frame_data = _load_json(frame_path) if frame_path else None
        if not frame_data:
            issues.append("missing_choice_a_frame_debug")
        else:
            if "agent_trace" not in frame_data:
                issues.append("missing_agent_trace")
            evidence.append(str(frame_data.get("execution_mode", "choice_a_debug")))

    if method_family == "choice_b_rule_gate":
        gate_path = _first_json(output_dir / "vlm_debug", "frame_*_planner_gate_result.json")
        gate_data = _load_json(gate_path) if gate_path else None
        if not gate_data:
            issues.append("missing_planner_gate_debug")
        else:
            if "agent_trace" not in gate_data:
                issues.append("missing_agent_trace")
            evidence.append(str(gate_data.get("execution_mode", "planner_gate_debug")))

    if method_family == "rule_based":
        evidence.append("rule_based_eval")

    state = "PASS" if not issues else "FAIL:" + ",".join(issues)
    return {
        "method": baseline_id,
        "dataset": dataset,
        "state": state,
        "pdms": _metric(eval_data, "pdms"),
        "rc": _metric(eval_data, "rc"),
        "evidence": ";".join(evidence) if evidence else "NA",
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nuscenes", "waymo", "kitti360"])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--methods", required=True, help="space-separated baseline IDs")
    args = parser.parse_args()

    rows = [
        check_method(
            baseline_id=method,
            dataset=args.dataset,
            suite=args.suite,
            run_id=args.run_id,
            scenario_path=args.scenario_path,
        )
        for method in args.methods.split()
    ]

    headers = ["method", "dataset", "state", "pdms", "rc", "evidence", "output_dir"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))

    return 0 if all(row["state"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

