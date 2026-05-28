#!/usr/bin/env python3
import argparse
import os
import re

from omegaconf import OmegaConf


DEFAULT_BASE_PATHS = {
    "nuscenes": "configs/sim/nuscenes_base_local.yaml",
    "waymo": "configs/sim/waymo_base_local.yaml",
    "kitti360": "configs/sim/kitti360_base_local.yaml",
}


def default_planner_path(planner_name: str) -> str:
    if planner_name.startswith("rule_based"):
        return os.path.join("configs", "planners", "rule_based_local_aidan.yaml")
    return os.path.join("configs", "planners", f"{planner_name}.yaml")


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


def resolve_ad_name(planner_name: str) -> str:
    if planner_name.startswith("drivor"):
        return "drivor"
    if planner_name.startswith("rule_based"):
        return "rule_based"
    if planner_name.startswith("rap"):
        return "rap"
    raise ValueError(f"unsupported planner_name: {planner_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_name", required=True)
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--base_path")
    parser.add_argument("--planner_path")
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    data_type = str(scenario_config.get("data_type", "nuscenes")).strip().lower()
    base_path = args.base_path or DEFAULT_BASE_PATHS.get(data_type)
    if not base_path:
        raise ValueError(f"unsupported data_type: {data_type}")

    ad_name = resolve_ad_name(args.planner_name)
    planner_path = args.planner_path
    if not planner_path:
        inferred = default_planner_path(args.planner_name)
        planner_path = inferred if os.path.exists(inferred) else ""

    base_config = OmegaConf.load(base_path)
    planner_config = OmegaConf.load(planner_path) if planner_path else OmegaConf.create()

    planner_output_suffix = ad_name
    if ad_name == "rap" and planner_config.get("rap", {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get("rap", {}).get("output_suffix", "rap_vlm")
    if ad_name == "drivor" and planner_config.get("drivor", {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get("drivor", {}).get("output_suffix", "drivor_vlm")
    if ad_name == "rule_based" and planner_config.get("rule_based", {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get("rule_based", {}).get("output_suffix", "rule_based_vlm")

    output_model_slug = resolve_output_model_slug(ad_name, planner_config)
    output_dir = prefix_output_dir_with_model(base_config.output_dir, output_model_slug)
    output_dir = output_dir + planner_output_suffix
    output = os.path.join(output_dir, f"{scenario_config.scene_name}_{scenario_config.mode}")
    print(output)


if __name__ == "__main__":
    main()
