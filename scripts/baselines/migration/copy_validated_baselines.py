#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INVENTORY_PATH = os.path.join(REPO_ROOT, "configs", "baselines", "validated_runs.yaml")
REGISTRY_PATH = os.path.join(REPO_ROOT, "configs", "baselines", "registry.yaml")

COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from baseline_registry import resolve_baseline_output_root  # noqa: E402


def copy_tree_contents(src: str, dst: str, dry_run: bool, move: bool) -> None:
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    if dry_run:
        return
    if move:
        parent = os.path.dirname(dst)
        os.makedirs(parent, exist_ok=True)
        if os.path.exists(dst):
            raise FileExistsError(f"destination already exists: {dst}")
        shutil.move(src, dst)
        return

    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(src):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(dst, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()

    inventory = OmegaConf.load(INVENTORY_PATH)
    registry = OmegaConf.to_container(OmegaConf.load(REGISTRY_PATH), resolve=True)

    items = list(inventory.get("validated_runs", []))
    if args.include_archive:
        items.extend(inventory.get("archive_only_runs", []))

    for item in items:
        run_type = "archive" if item.get("status") == "archive_only" else "canonical"
        archive_reason = "known_wrong_semantics" if run_type == "archive" else None
        dst = resolve_baseline_output_root(
            baseline_id=str(item.baseline_id),
            dataset=str(item.dataset),
            suite=str(item.suite),
            run_type=run_type,
            archive_reason=archive_reason,
            registry=registry,
        )
        src = str(item.output_root)
        print(f"{src} -> {dst}")
        copy_tree_contents(src, dst, args.dry_run, args.move)


if __name__ == "__main__":
    main()
