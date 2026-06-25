#!/usr/bin/env bash
set -e

cd /home/jiageng/AutoAgent0

SCENARIO_PATH="/home/jiageng/AutoAgent0/released_assets/scenarios/nuscenes/scene-0383-easy-00.yaml"

# Same invocation as test.sh, but drives the new-architecture pipeline.py:
# the RAP subprocess only does inference (returns proposals+scores) and all
# selection happens pipeline-side.
CUDA_VISIBLE_DEVICES=2 pixi run python main.py \
  --scenario_path "$SCENARIO_PATH" \
  --base_path "/home/jiageng/AutoAgent0/configs/sim/nuscenes_base_local.yaml" \
  --camera_path "/home/jiageng/AutoAgent0/configs/sim/nuscenes_camera.yaml" \
  --kinematic_path "/home/jiageng/AutoAgent0/configs/sim/kinematic.yaml" \
  --ad_cuda 4 \
  --ad rap
