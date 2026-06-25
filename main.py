#!/usr/bin/env python3
"""Top-level closed-loop orchestrator entry point.

Wires the three layers together: launches the requested planner subprocess
(``autoagent0/planners``), builds the pipeline-side selector
(``autoagent0/scorer``), and drives the simulator loop
(``autoagent0/orchestration/loop``).

Each planner subprocess only runs inference and returns ``(proposals, scores)``
over the plan FIFO. All candidate selection (carry-previous, top-k, rule-based
merge, VLM / AutoAgent0 reasoning) and plan payload construction happen here on
the pixi side via ``LearnedPlannerSelector``, injected into the simulator loop
through its ``plan_adapter`` hook. Supports ``--ad rap|drivor|rule_based``.
"""
import logging
import os
import sys
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "sim"))

from autoagent0.orchestration.loop import (
    run_closed_loop,
    _parse_boolish,
    _resolve_scene_model_path,
)
from sim.utils.launch_ad import launch, check_alive

from autoagent0.adapters.hugsim.results import (
    prefix_output_dir_with_model,
    resolve_output_model_slug,
)
from autoagent0.config import build_prefixed_autoagent0_env, resolve_autoagent0_config
from autoagent0.scorer.planner_selection import LearnedPlannerSelector
from autoagent0.scorer.vlm_selector import VLMPlanSelector, VLMSelectorConfig
from autoagent0.experts.rule_based import (
    get_rule_based_proposals_and_scores,
    resolve_rule_based_merge_config,
)
from autoagent0.experts.rule_based_provider import rule_based_to_hugsim_plan
from autoagent0.experts.rule_based_env import build_prefixed_rule_based_env
from autoagent0.vlm.vlm_env import (
    VLM_ENV_DEFAULTS,
    VLM_ENV_FIELD_NAMES,
    build_prefixed_vlm_env,
    get_prefixed_env_value,
)

# Per-planner selection labels (mirror the legacy clients' arguments). All three
# now share LearnedPlannerSelector.
PLANNER_SPECS = {
    "rap": {
        "runtime_name": "rap",
        "current_source_name": "current_rap",
        "learned_default_source": "fallback_rap_argmax",
        "plain_source": "rap_argmax",
        "score_fallback_key": "rap_score",
        "planner_log_name": "RAP",
        "strict_learned_argmax_lookup": True,
        "q_key_prefix": True,
        "supports_rule_merge": True,
        "always_privileged": False,
    },
    "drivor": {
        "runtime_name": "drivor",
        "current_source_name": "current_drivor",
        "learned_default_source": "drivor_argmax",
        "plain_source": "drivor_argmax",
        "score_fallback_key": "proposal_score",
        "planner_log_name": "DrivoR",
        "strict_learned_argmax_lookup": False,
        "q_key_prefix": False,
        "supports_rule_merge": True,
        "always_privileged": False,
    },
    "rule_based": {
        "runtime_name": "rule_based",
        "current_source_name": "rule_based_planner",
        "learned_default_source": "rule_based_argmax",
        "plain_source": "rule_based_argmax",
        "score_fallback_key": "proposal_score",
        "planner_log_name": "RuleBased",
        "strict_learned_argmax_lookup": False,
        "q_key_prefix": False,
        "supports_rule_merge": False,  # the rule-based planner does not self-merge
        "always_privileged": True,     # its core inference needs privileged_info
    },
}


def _coerce_env_value(raw_value, default_value):
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


def _resolve_vlm_config_from_env() -> VLMSelectorConfig:
    """Build VLMSelectorConfig from the PLANNER_VLM_ env vars set below."""
    values = {}
    for suffix, field_name in VLM_ENV_FIELD_NAMES.items():
        default_value = VLM_ENV_DEFAULTS[suffix]
        raw_value = get_prefixed_env_value(suffix, default=default_value, prefixes=("PLANNER_VLM_",))
        values[field_name] = _coerce_env_value(raw_value, default_value)
    return VLMSelectorConfig(**values)


def _build_launch_env(cfg, ad: str) -> dict:
    """Env for the inference subprocess (planner-specific *_REPO_ROOT etc.)."""
    pc = cfg.planner[ad]
    python_bin = pc.get("python_bin", "python")
    if ad == "rap":
        device = os.environ.get("RAP_DEVICE_OVERRIDE") or pc.get("device", "cuda")
        return {
            "RAP_REPO_ROOT": pc.get("repo_root", ""),
            "RAP_CHECKPOINT": pc.get("checkpoint", ""),
            "RAP_PYTHON_BIN": python_bin,
            "RAP_DEVICE": device,
            "RAP_IMAGE_SCALE": pc.get("image_scale", 0.4),
            "RAP_USE_SCENE_RIG_LIDAR2IMG": pc.get("use_scene_rig_lidar2img", False),
            "RAP_HF_HUB_OFFLINE": pc.get("hf_hub_offline", True),
            "RAP_TRANSFORMERS_OFFLINE": pc.get("transformers_offline", True),
            "RAP_HF_HOME": pc.get("hf_home", ""),
            "RAP_HF_HUB_CACHE": pc.get("hf_hub_cache", ""),
            "RAP_TRANSFORMERS_CACHE": pc.get("transformers_cache", ""),
            "RAP_NUPLAN_DEVKIT_DIR": pc.get("nuplan_devkit_dir", ""),
            "RAP_BACKBONE_PATH": pc.get("backbone_path", ""),
        }
    if ad == "drivor":
        device = os.environ.get("DRIVOR_DEVICE_OVERRIDE") or pc.get("device", "cuda")
        return {
            "DRIVOR_REPO_ROOT": pc.get("repo_root", ""),
            "DRIVOR_CHECKPOINT": pc.get("checkpoint", ""),
            "DRIVOR_DINO": pc.get("dino", ""),
            "DRIVOR_PYTHON_BIN": python_bin,
            "DRIVOR_DEVICE": device,
            "DRIVOR_CONFIG": pc.get("config", ""),
        }
    if ad == "rule_based":
        device = os.environ.get("RULE_BASED_DEVICE_OVERRIDE") or pc.get("device", "cpu")
        return {
            "RULE_BASED_REPO_ROOT": pc.get("repo_root", ""),
            "RULE_BASED_PYTHON_BIN": python_bin,
            "RULE_BASED_DEVICE": device,
            "RULE_BASED_CONFIG": pc.get("config", ""),
        }
    raise NotImplementedError(ad)


def _build_selector(cfg, output, ad: str):
    """Resolve VLM / rule-merge / AutoAgent0 configs in-process (via PLANNER_*
    env, reusing the existing resolvers) and build the pipeline-side selector.
    Returns (selector, vlm_selector, rule_based_merge_cfg)."""
    spec = PLANNER_SPECS[ad]
    pc = cfg.planner[ad]
    python_bin = pc.get("python_bin", "python")

    # VLM env -> os.environ -> resolve (unified PLANNER_VLM_ prefix).
    vlm_env = build_prefixed_vlm_env(pc.get("vlm", {}), planner_python_bin=python_bin, prefixes=("PLANNER_VLM_",))
    for key, value in vlm_env.items():
        os.environ[key] = str(value)
    vlm_device = (
        os.environ.get("PLANNER_VLM_DEVICE_OVERRIDE")
        or pc.get("vlm", {}).get("device", "auto")
    )
    os.environ["PLANNER_VLM_DEVICE"] = str(vlm_device)

    if spec["supports_rule_merge"]:
        rb_env = build_prefixed_rule_based_env(
            pc.get("rule_based_merge", {}), planner_python_bin=python_bin, prefixes=("PLANNER_RULE_BASED_",),
        )
        for key, value in rb_env.items():
            os.environ[key] = str(value)

    autoagent0_env = build_prefixed_autoagent0_env(pc.get("autoagent0", {}))
    for key, value in autoagent0_env.items():
        os.environ[key] = str(value)

    vlm_cfg = _resolve_vlm_config_from_env()
    rule_based_merge_cfg = resolve_rule_based_merge_config(
        planner_python_bin=python_bin, prefixes=("PLANNER_RULE_BASED_",),
    )
    autoagent0_cfg = resolve_autoagent0_config()

    vlm_selector = VLMPlanSelector(vlm_cfg, Path(output))
    vlm_selector.preload()
    logging.info(
        "pipeline %s selector: vlm_enabled=%s autoagent0_enabled=%s rule_merge_enabled=%s",
        ad, vlm_cfg.enabled, autoagent0_cfg.enabled, rule_based_merge_cfg.enabled,
    )

    selector = LearnedPlannerSelector(
        vlm_selector=vlm_selector,
        runtime_name=spec["runtime_name"],
        autoagent0_cfg=autoagent0_cfg,
        vlm_cfg=vlm_cfg,
        rule_based_merge_cfg=rule_based_merge_cfg,
        current_source_name=spec["current_source_name"],
        learned_default_source=spec["learned_default_source"],
        plain_source=spec["plain_source"],
        score_fallback_key=spec["score_fallback_key"],
        planner_log_name=spec["planner_log_name"],
        strict_learned_argmax_lookup=spec["strict_learned_argmax_lookup"],
        q_key_prefix=spec["q_key_prefix"],
        logger=logging.getLogger(f"{ad}_selection"),
    )
    return selector, vlm_selector, rule_based_merge_cfg


def _build_recover_plan(selector, rule_based_merge_cfg):
    """Build the loop's recovery hook (full pipeline on rule-based proposals).

    When the closed loop's verifier rejects the learned trajectory, this regenerates
    proposals with the rule-based planner and runs them through the SAME selection
    pipeline as the normal path -- candidate pool + VLM scorer / agentic recovery
    loop -- via ``selector.select``. Returns a HUGSIM plan payload, or None if the
    rule-based planner yields nothing (the loop then keeps the learned plan).
    Defensive: any failure degrades to None rather than raising.

    Only wired when VLM is enabled; with VLM disabled the loop does not recover.
    """

    def recover_plan(obs, info, info_history, privileged_info, output_num_poses):
        try:
            # rule-based trajectory
            proposals, scores, _ = get_rule_based_proposals_and_scores(
                rule_based_merge_cfg,
                obs=obs,
                info=info,
                info_history=deque(info_history, maxlen=len(info_history) or None),
                privileged_agents=privileged_info,
                output_num_poses=output_num_poses,
                topk=rule_based_merge_cfg.topk,
            )
            if scores is None or len(scores) == 0:
                return None
            proposals_hugsim = np.stack(
                [
                    rule_based_to_hugsim_plan(np.asarray(proposals[idx])[:output_num_poses])
                    for idx in range(len(proposals))
                ],
                axis=0,
            ).astype(np.float32)
            # memory agent: select actions recovery loop
            return selector.select(
                proposals=proposals_hugsim,
                scores=scores,
                obs=obs,
                info=info,
                info_history=info_history,
                privileged_info=privileged_info,
            )
        except Exception:
            logging.getLogger("closed_loop").exception("rule-based recovery failed")
            return None

    return recover_plan


if __name__ == "__main__":
    parser = ArgumentParser(description="New-architecture closed-loop pipeline")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument("--planner_path", type=str, default="")
    parser.add_argument("--ad", default="rap", choices=sorted(PLANNER_SPECS))
    parser.add_argument("--ad_cuda", default="1")
    parser.add_argument("--include_privileged_pipe", default=False)
    args = parser.parse_args()

    ad = args.ad
    spec = PLANNER_SPECS[ad]

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    planner_path = args.planner_path
    if not planner_path:
        inferred_planner_path = os.path.join("configs", "planners", f"{ad}.yaml")
        if os.path.exists(inferred_planner_path):
            planner_path = inferred_planner_path
    planner_config = OmegaConf.load(planner_path) if planner_path else OmegaConf.create()
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
        {"planner": planner_config},
    )

    planner_output_suffix = ad
    if planner_config.get(ad, {}).get("vlm", {}).get("enabled", False):
        planner_output_suffix = planner_config.get(ad, {}).get("output_suffix", f"{ad}_vlm")
    output_root_override = os.environ.get("BENCHMARK_OUTPUT_ROOT_OVERRIDE", "").strip()
    if output_root_override:
        cfg.base.output_dir = output_root_override
    else:
        output_model_slug = resolve_output_model_slug(ad, planner_config)
        cfg.base.output_dir = prefix_output_dir_with_model(cfg.base.output_dir, output_model_slug)
        cfg.base.output_dir = cfg.base.output_dir + planner_output_suffix

    model_path = _resolve_scene_model_path(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, "cfg.yaml"))
    cfg.update(model_config)
    cfg.model_path = model_path

    # Absolute path so the planner subprocess (whose launch.sh may `cd` into its
    # own repo) resolves the FIFO directory identically to this process.
    output = os.path.abspath(os.path.join(cfg.base.output_dir, cfg.scenario.scene_name + "_" + cfg.scenario.mode))
    os.makedirs(output, exist_ok=True)

    # Always use the new inference-only launch scripts under autoagent0/planners/.
    # (cfg.planner.<ad>.launch_path is the legacy ./planners/<ad>/launch.sh and is
    # intentionally ignored here.)
    ad_path = os.path.join(".", "autoagent0", "planners", ad, "launch.sh")
    extra_env = _build_launch_env(cfg, ad)

    selector, vlm_selector, rule_based_merge_cfg = _build_selector(cfg, output, ad)
    # Recovery runs the rule-based proposals through the full selection pipeline
    # (incl. the agentic recovery loop). Only meaningful with VLM enabled; with VLM
    # disabled there is no recovery and the loop keeps the learned plan.
    recover_plan = (
        _build_recover_plan(selector, rule_based_merge_cfg)
        if selector.vlm_cfg.enabled
        else None
    )

    info_history = deque(maxlen=4)

    def plan_adapter(raw_response, current_obs, current_info, privileged_info):
        proposals, scores = raw_response
        info_history.append(dict(current_info))
        return selector.select(
            proposals=proposals,
            scores=scores,
            obs=current_obs,
            info=current_info,
            info_history=list(info_history),
            privileged_info=privileged_info,
        )

    include_privileged_pipe = _parse_boolish(args.include_privileged_pipe)
    if spec["always_privileged"]:
        include_privileged_pipe = True
    elif spec["supports_rule_merge"]:
        rb_merge_cfg = cfg.planner[ad].get("rule_based_merge", {})
        if rb_merge_cfg.get("enabled", False) and rb_merge_cfg.get("include_privileged_info", True):
            include_privileged_pipe = True

    process = launch(ad_path, args.ad_cuda, output, extra_env=extra_env)
    try:
        run_closed_loop(
            cfg,
            output,
            planner_output_suffix,
            include_privileged_pipe=include_privileged_pipe,
            planner_process=process,
            plan_adapter=plan_adapter,
            ad_name=ad,
            recover_plan=recover_plan,
        )
        check_alive(process)
    except Exception:
        import traceback
        traceback.print_exc()
        process.kill()
    finally:
        try:
            vlm_selector.finalize()
        except Exception:
            logging.getLogger("pipeline").exception("Error finalizing VLM selector")
