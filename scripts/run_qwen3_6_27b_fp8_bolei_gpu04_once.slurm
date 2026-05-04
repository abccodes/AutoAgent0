#!/bin/bash
#SBATCH --job-name=hugsim-qwen3627bfp8-once
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu05
#SBATCH --nodelist=bolei-gpu04
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
SCENARIO_PATH="${SCENARIO_PATH:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_debug_qwen3_6_27b_fp8.yaml}"
PLANNER_PATH="${PLANNER_PATH:-configs/planners/rap_vlm_qwen3_6_27b_fp8.yaml}"
SETUP_VLM_ENV="${SETUP_VLM_ENV:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"

mkdir -p /bigdata/aidan/outputs/slurm
cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "scenario=${SCENARIO_PATH}"
echo "base=${BASE_PATH}"
echo "planner_path=${PLANNER_PATH}"
echo "model_id=${MODEL_ID}"
echo "partition=gpu05"
echo "nodelist=bolei-gpu04"
echo "gpu_count=1"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "setup_vlm_env=${SETUP_VLM_ENV}"

export PLANNER_NAME="rap_vlm"
export BASE_PATH
export PLANNER_PATH
export SETUP_VLM_ENV
export MODEL_ID

bash scripts/run_single_scene.slurm
