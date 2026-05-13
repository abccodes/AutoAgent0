#!/usr/bin/env python3
"""
Minimal test harness to verify HUGSIM payload (obs/info) and privileged agent info.

Runs 2 timesteps of the env, extracts obs/info/privileged data, and writes to JSON.
No external planner or FIFO setup required.

Usage:
  python payload_test.py \
    --scenario_path configs/scene.yaml \
    --base_path configs/base.yaml \
    --camera_path configs/camera.yaml \
    --kinematic_path configs/kinematic.yaml \
    [--output_dir /path/to/output]
"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "sim"))

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, rt2pose
from omegaconf import OmegaConf
import json
import numpy as np
from typing import Any, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _serialize_for_json(obj: Any, max_items: int = 100) -> Any:
    """
    Recursively serialize numpy arrays, tensors, and other non-JSON-serializable types.
    Truncate large arrays to summaries.
    """
    if isinstance(obj, np.ndarray):
        if obj.size > max_items:
            return {
                "_type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "size": int(obj.size),
                "sample": obj.flat[:min(10, obj.size)].tolist(),
                "truncated": True,
            }
        else:
            return {
                "_type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.tolist(),
            }
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif hasattr(obj, "cpu") and hasattr(obj, "numpy"):  # torch tensor
        try:
            arr = obj.detach().cpu().numpy()
            return _serialize_for_json(arr, max_items)
        except Exception as e:
            logger.warning(f"Failed to convert torch tensor: {e}")
            return f"<torch.Tensor: {obj.shape}>"
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v, max_items) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item, max_items) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def run_payload_test(cfg: OmegaConf, output_dir: str) -> None:
    """
    Run minimal 2-step env loop and dump obs/info/privileged data to JSON.
    
    Args:
        cfg: OmegaConf configuration.
        output_dir: Directory to write JSON outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Creating HUGSim environment...")
    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output_dir)
    
    frames_data = []
    
    try:
        # Frame 0: Reset
        logger.info("Resetting environment...")
        obs0, info0, priv0 = env.reset()
        # obs0, info0 = env.reset()
        
        # # Get privileged info at frame 0
        # priv0 = env.unwrapped.get_agent_privileged_info()
        
        frame0_data = {
            "frame_idx": 0,
            "obs": _serialize_for_json(obs0),
            "info": _serialize_for_json(info0),
            "privileged_agents": _serialize_for_json(priv0),
        }
        frames_data.append(frame0_data)
        logger.info(f"Frame 0 captured. Ego pos: {info0.get('ego_pos', 'N/A')}")
        
        # Frame 1: Step
        logger.info("Taking a step...")
        # Minimal action: zero acceleration and steering rate
        action = {'acc': 0.0, 'steer_rate': 0.0}
        obs1, reward, terminated, truncated, info1 = env.step(action)
        # fetch privileged info separately to remain gym-compliant
        try:
            priv1 = env.unwrapped.get_agent_privileged_info()
        except Exception:
            priv1 = None
        frame1_data = {
            "frame_idx": 1,
            "obs": _serialize_for_json(obs1),
            "info": _serialize_for_json(info1),
            "privileged_agents": _serialize_for_json(priv1),
            "reward": float(reward) if reward is not None else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        frames_data.append(frame1_data)
        logger.info(f"Frame 1 captured. Ego pos: {info1.get('ego_pos', 'N/A')}")
        
    except Exception as e:
        logger.exception(f"Error during test loop: {e}")
        raise
    finally:
        env.close()
    
    # Write output JSON
    output_json_path = os.path.join(output_dir, 'payload_test_output.json')
    logger.info(f"Writing results to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(frames_data, f, indent=2)
    
    logger.info(f"✓ Payload test complete. Output written to {output_json_path}")
    
    # Also write individual frame JSONs for convenience
    for frame_data in frames_data:
        frame_idx = frame_data['frame_idx']
        frame_json_path = os.path.join(output_dir, f'payload_test_frame{frame_idx}.json')
        with open(frame_json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)
        logger.info(f"  → {frame_json_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="HUGSIM payload and privileged agent info test")
    parser.add_argument("--scenario_path", type=str, required=True, 
                        help="Path to scenario config YAML")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to base config YAML")
    parser.add_argument("--camera_path", type=str, required=True,
                        help="Path to camera config YAML")
    parser.add_argument("--kinematic_path", type=str, required=True,
                        help="Path to kinematic config YAML")
    parser.add_argument("--output_dir", type=str, default="outputs/payload_test",
                        help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    # Load and merge configs
    logger.info("Loading configs...")
    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
    )
    
    # Load model config
    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config_path = os.path.join(model_path, 'cfg.yaml')
    if os.path.exists(model_config_path):
        logger.info(f"Loading model config from {model_config_path}")
        model_config = OmegaConf.load(model_config_path)
        cfg.update(model_config)
    else:
        logger.warning(f"Model config not found at {model_config_path}; skipping")
    # Ensure cfg has `model_path` (required by the environment loader)
    if 'model_path' not in cfg:
        logger.info(f"Setting cfg.model_path to {model_path}")
        cfg.model_path = model_path
    
    # Run test
    run_payload_test(cfg, args.output_dir)
