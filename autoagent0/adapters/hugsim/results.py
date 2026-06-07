from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict


def slugify_model_name(value: Any, default: str = "model") -> str:
    value = "" if value is None else str(value).strip()
    if not value:
        value = default
    value = value.rstrip("/").split("/")[-1]
    if value.endswith(".ckpt") or value.endswith(".pth") or value.endswith(".pt"):
        value = os.path.splitext(value)[0]
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or default


def resolve_output_model_slug(ad_name: str, planner_config: Any) -> str:
    planner_key = "rap" if ad_name == "rap" else "drivor" if ad_name == "drivor" else ""
    if not planner_key:
        return ""

    planner_cfg = planner_config.get(planner_key, {})
    vlm_cfg = planner_cfg.get("vlm", {})
    if vlm_cfg.get("enabled", False):
        explicit_slug = vlm_cfg.get("output_model_slug", "")
        if explicit_slug:
            if str(explicit_slug).strip().lower() in {"none", "disable", "disabled", "off"}:
                return ""
            return slugify_model_name(explicit_slug)
        return slugify_model_name(vlm_cfg.get("model_id", "vlm"))

    explicit_slug = planner_cfg.get("output_model_slug", "")
    if explicit_slug:
        if str(explicit_slug).strip().lower() in {"none", "disable", "disabled", "off"}:
            return ""
        return slugify_model_name(explicit_slug)
    checkpoint = planner_cfg.get("checkpoint", "")
    return slugify_model_name(checkpoint, default=planner_key)


def prefix_output_dir_with_model(output_dir: Any, model_slug: Any) -> str:
    output_dir = str(output_dir)
    model_slug = str(model_slug or "").strip()
    if not model_slug:
        return output_dir

    parent, name = os.path.split(output_dir.rstrip(os.sep))
    if not name:
        return os.path.join(output_dir, model_slug)
    if name.startswith(f"{model_slug}_"):
        return output_dir
    return os.path.join(parent, f"{model_slug}_{name}")


def safe_get_planner_cfg(cfg: Any, ad_name: str) -> Dict[str, Any]:
    try:
        planner_cfg = cfg.planner.get(ad_name, {})
        return planner_cfg or {}
    except Exception:
        return {}


def empty_performance_summary() -> Dict[str, object]:
    return {
        "num_records": 0,
        "planner_gate_records": 0,
        "latency_mean_sec": 0.0,
        "latency_p50_sec": 0.0,
        "latency_p95_sec": 0.0,
        "latency_max_sec": 0.0,
        "intervention_latency_mean_sec": 0.0,
        "intervention_latency_p50_sec": 0.0,
        "intervention_latency_p95_sec": 0.0,
        "intervention_latency_max_sec": 0.0,
        "total_vlm_latency_mean_sec": 0.0,
        "total_vlm_latency_p50_sec": 0.0,
        "total_vlm_latency_p95_sec": 0.0,
        "total_vlm_latency_max_sec": 0.0,
        "planner_gate_latency_mean_sec": 0.0,
        "planner_gate_latency_p50_sec": 0.0,
        "planner_gate_latency_p95_sec": 0.0,
        "planner_gate_latency_max_sec": 0.0,
        "latency_equivalent_steps_mean": 0.0,
        "carry_reuse_rate": 0.0,
        "switch_to_current_rate": 0.0,
        "fallback_rate": 0.0,
        "intervention_trigger_rate": 0.0,
        "intervention_low_rate": 0.0,
        "intervention_medium_rate": 0.0,
        "intervention_high_rate": 0.0,
        "intervention_action_applied_rate": 0.0,
        "intervention_action_threshold": 0.0,
        "intervention_high_threshold": 0.0,
        "intervention_severity_score_mean": 0.0,
        "intervention_severity_score_p50": 0.0,
        "intervention_severity_score_p95": 0.0,
        "gate_skip_rate": 0.0,
        "scoring_invoked_rate": 0.0,
        "vlm_q_valid_rate": 0.0,
        "counts": {
            "intervention_invoked": 0,
            "intervention_valid": 0,
            "intervention_invalid": 0,
            "intervention_timeout": 0,
            "intervention_error": 0,
            "scoring_invoked": 0,
            "scoring_valid": 0,
            "scoring_timeout": 0,
            "scoring_error": 0,
            "planner_gate_invoked": 0,
            "planner_gate_valid": 0,
            "planner_gate_timeout": 0,
            "planner_gate_error": 0,
            "base_policy_no_intervention": 0,
            "gate_failed_base_policy_fallback": 0,
            "intervention_triggered_scoring": 0,
            "planner_gate_selected_learned": 0,
            "planner_gate_selected_rule_based": 0,
            "planner_gate_failed_base_policy_fallback": 0,
        },
        "branch_breakdown": {
            "solo_or_merge": {
                "base_policy_no_intervention": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "total_tokens_total": 0,
                    },
                },
                "gate_failed_base_policy_fallback": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "total_tokens_total": 0,
                    },
                },
                "intervention_triggered_scoring": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "intervention": {
                            "prompt_tokens_total": 0,
                            "completion_tokens_total": 0,
                            "total_tokens_total": 0,
                        },
                        "scoring": {
                            "prompt_tokens_total": 0,
                            "completion_tokens_total": 0,
                            "total_tokens_total": 0,
                        },
                    },
                },
            },
            "planner_gate": {
                "selected_learned": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "total_tokens_total": 0,
                    },
                },
                "selected_rule_based": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "total_tokens_total": 0,
                    },
                },
                "failed_base_policy_fallback": {
                    "count": 0,
                    "latency_mean_sec": 0.0,
                    "latency_p50_sec": 0.0,
                    "latency_p95_sec": 0.0,
                    "latency_max_sec": 0.0,
                    "tokens": {
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "total_tokens_total": 0,
                    },
                },
            },
        },
        "tokens": {
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "by_stage": {
                "intervention": {"prompt_tokens_total": 0, "completion_tokens_total": 0, "total_tokens_total": 0},
                "scoring": {"prompt_tokens_total": 0, "completion_tokens_total": 0, "total_tokens_total": 0},
                "planner_gate": {"prompt_tokens_total": 0, "completion_tokens_total": 0, "total_tokens_total": 0},
            },
        },
    }


def build_run_performance(output_dir: str, cfg: Any, ad_name: str, frame_count: int) -> Dict[str, object]:
    planner_cfg = safe_get_planner_cfg(cfg, ad_name)
    vlm_cfg = planner_cfg.get("vlm", {}) if planner_cfg else {}
    summary_path = os.path.join(output_dir, "vlm_debug", "latency_summary.json")
    summary = empty_performance_summary()
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as rf:
                loaded = json.load(rf)
            if isinstance(loaded, dict):
                summary.update({k: v for k, v in loaded.items() if k not in {"counts", "tokens", "branch_breakdown"}})
                if isinstance(loaded.get("counts"), dict):
                    summary["counts"].update(loaded["counts"])
                if isinstance(loaded.get("branch_breakdown"), dict):
                    summary["branch_breakdown"] = loaded["branch_breakdown"]
                if isinstance(loaded.get("tokens"), dict):
                    summary_tokens = summary["tokens"]
                    summary_tokens.update({k: v for k, v in loaded["tokens"].items() if k != "by_stage"})
                    if isinstance(loaded["tokens"].get("by_stage"), dict):
                        for stage_name, stage_usage in loaded["tokens"]["by_stage"].items():
                            if stage_name in summary_tokens["by_stage"] and isinstance(stage_usage, dict):
                                summary_tokens["by_stage"][stage_name].update(stage_usage)
        except Exception:
            logging.exception("Failed to load latency summary from %s", summary_path)

    return {
        "frame_count": int(frame_count),
        "planner_backend": str(ad_name),
        "vlm_enabled": bool(vlm_cfg.get("enabled", False)) if vlm_cfg else False,
        "intervention_enabled": bool(vlm_cfg.get("intervention_enabled", False)) if vlm_cfg else False,
        "planner_gate_enabled": bool(vlm_cfg.get("planner_gate_enabled", False)) if vlm_cfg else False,
        "latency_tracking_mode": str(vlm_cfg.get("latency_tracking_mode", "")) if vlm_cfg else "",
        "counts": summary.get("counts", {}),
        "tokens": summary.get("tokens", {}),
        "branch_breakdown": summary.get("branch_breakdown", {}),
        "summary": summary,
    }
