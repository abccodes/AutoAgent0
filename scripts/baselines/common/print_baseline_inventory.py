#!/usr/bin/env python3
import os
import sys

from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INVENTORY_PATH = os.path.join(REPO_ROOT, "configs", "baselines", "validated_runs.yaml")


def main() -> None:
    cfg = OmegaConf.load(INVENTORY_PATH)

    canonical = []
    historical = []
    for item in cfg.get("validated_runs", []):
        status = str(item.get("status", ""))
        if status == "validated_canonical":
            canonical.append(item)
        else:
            historical.append(item)

    print("Validated canonical baselines:")
    for item in canonical:
        expected = item.get("scene_count_expected", "?")
        completed = item.get("scene_count_completed", "?")
        print(
            f"- {item.baseline_id} {item.dataset}/{item.suite}: "
            f"{completed}/{expected} "
            f"status={item.status} root={item.output_root}"
        )

    print("\nValidated historical baselines:")
    for item in historical:
        expected = item.get("scene_count_expected", "?")
        completed = item.get("scene_count_completed", "?")
        print(
            f"- {item.baseline_id} {item.dataset}/{item.suite}: "
            f"{completed}/{expected} "
            f"status={item.status} root={item.output_root}"
        )

    print("\nPending baselines:")
    for item in cfg.get("pending_baselines", []):
        print(f"- {item.baseline_id} {item.dataset}/{item.suite}: status={item.status}")


if __name__ == "__main__":
    main()
