#!/usr/bin/env python3
import argparse
import os
import re

from omegaconf import OmegaConf


def slugify_model_name(value: str, default: str = "model") -> str:
    value = "" if value is None else str(value).strip()
    if not value:
        value = default
    value = value.rstrip("/").split("/")[-1]
    if value.endswith((".ckpt", ".pth", ".pt")):
        value = os.path.splitext(value)[0]
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or default


def resolve_output_model_slug(ad_name, planner_config):
    planner_key = "rap" if ad_name == "rap" else "drivor" if ad_name == "drivor" else ""
    if not planner_key:
        return ""

    planner_cfg = planner_config.get(planner_key, {})
    vlm_cfg = planner_cfg.get("vlm", {})
    if vlm_cfg.get("enabled", False):
        explicit_slug = vlm_cfg.get("output_model_slug", "")
        if explicit_slug:
            return slugify_model_name(explicit_slug)
        return slugify_model_name(vlm_cfg.get("model_id", "vlm"))

    explicit_slug = planner_cfg.get("output_model_slug", "")
    if explicit_slug:
        return slugify_model_name(explicit_slug)
    checkpoint = planner_cfg.get("checkpoint", "")
    return slugify_model_name(checkpoint, default=planner_key)


def prefix_output_dir_with_model(output_dir, model_slug):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_name", required=True)
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--planner_path")
    args = parser.parse_args()

    ad_name = "drivor" if args.planner_name.startswith("drivor") else "rap"
    planner_path = args.planner_path
    if not planner_path:
        inferred = os.path.join("configs", "planners", f"{ad_name}.yaml")
        planner_path = inferred if os.path.exists(inferred) else ""

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    planner_config = OmegaConf.load(planner_path) if planner_path else OmegaConf.create()

    planner_output_suffix = ad_name
    if ad_name == "rap" and planner_config.get("rap", {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get("rap", {}).get("output_suffix", "rap_vlm")
    if ad_name == "drivor" and planner_config.get("drivor", {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get("drivor", {}).get("output_suffix", "drivor_vlm")

    output_model_slug = resolve_output_model_slug(ad_name, planner_config)
    output_dir = prefix_output_dir_with_model(base_config.output_dir, output_model_slug)
    output_dir = output_dir + planner_output_suffix
    output = os.path.join(output_dir, f"{scenario_config.scene_name}_{scenario_config.mode}")
    print(output)


if __name__ == "__main__":
    main()
