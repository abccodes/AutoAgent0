#!/usr/bin/env python3
import os
from typing import Any, Dict

from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REGISTRY_PATH = os.path.join(REPO_ROOT, "configs", "baselines", "registry.yaml")


def load_registry(registry_path: str = REGISTRY_PATH) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(registry_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid registry at {registry_path}")
    return cfg


def get_baseline_entry(baseline_id: str, registry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    registry = registry or load_registry()
    baselines = registry.get("baselines", {})
    if baseline_id not in baselines:
        raise KeyError(f"unknown baseline_id: {baseline_id}")
    return baselines[baseline_id]


def get_dataset_entry(baseline_id: str, dataset: str, registry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    entry = get_baseline_entry(baseline_id, registry)
    datasets = entry.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"baseline {baseline_id} does not support dataset {dataset}")
    return datasets[dataset]


def resolve_baseline_output_root(
    baseline_id: str,
    dataset: str,
    suite: str,
    run_variant: str | None = None,
    run_type: str = "canonical",
    archive_reason: str | None = None,
    registry: Dict[str, Any] | None = None,
) -> str:
    registry = registry or load_registry()
    roots = registry.get("roots", {})
    dataset_entry = get_dataset_entry(baseline_id, dataset, registry)
    run_variant = run_variant or get_baseline_entry(baseline_id, registry).get("run_variant", "default")

    if run_type == "canonical":
        base_root = roots["canonical"]
        return os.path.join(base_root, baseline_id, dataset, suite, run_variant)
    if run_type == "debug":
        base_root = roots["debug"]
        return os.path.join(base_root, baseline_id, dataset, "current")
    if run_type == "archive":
        base_root = roots["archive"]
        archive_reason = archive_reason or "historical_unverified"
        return os.path.join(base_root, baseline_id, dataset, suite, archive_reason, run_variant)

    raise ValueError(f"unsupported run_type: {run_type}")


def resolve_scene_output_path(
    baseline_id: str,
    dataset: str,
    suite: str,
    scene_name: str,
    mode: str,
    run_variant: str | None = None,
    run_type: str = "canonical",
    archive_reason: str | None = None,
    registry: Dict[str, Any] | None = None,
) -> str:
    root = resolve_baseline_output_root(
        baseline_id=baseline_id,
        dataset=dataset,
        suite=suite,
        run_variant=run_variant,
        run_type=run_type,
        archive_reason=archive_reason,
        registry=registry,
    )
    return os.path.join(root, f"{scene_name}_{mode}")


def infer_scene_set_file(dataset: str, suite: str) -> str:
    if suite == "3easy":
        if dataset == "waymo":
            return "configs/benchmark/scene_sets/waymo_3easy.txt"
        if dataset == "kitti360":
            return "configs/benchmark/scene_sets/kitti360_3easy.txt"
    raise KeyError(f"no scene set file registered for dataset={dataset} suite={suite}")
