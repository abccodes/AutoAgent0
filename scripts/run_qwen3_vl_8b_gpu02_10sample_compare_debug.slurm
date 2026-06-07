#!/bin/bash
#SBATCH --job-name=hugsim-qwen3vl8b-10cmp
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_debug.yaml}"
PLANNER_PATH="${PLANNER_PATH:-configs/planners/rap_vlm_intervention_compare_multiview.yaml}"
SETUP_VLM_ENV="${SETUP_VLM_ENV:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
SCENES="${SCENES:-configs/benchmark/nuscenes_all_variants/scene-0013-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0064-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0038-medium-00.yaml configs/benchmark/nuscenes_all_variants/scene-0062-medium-00.yaml configs/benchmark/nuscenes_all_variants/scene-0418-medium-00.yaml configs/benchmark/nuscenes_all_variants/scene-0930-medium-01.yaml configs/benchmark/nuscenes_all_variants/scene-0138-hard-00.yaml configs/benchmark/nuscenes_all_variants/scene-0383-hard-00.yaml configs/benchmark/nuscenes_all_variants/scene-0051-extreme-00.yaml configs/benchmark/nuscenes_all_variants/scene-0411-extreme-00.yaml}"

mkdir -p /bigdata/aidan/outputs/slurm
cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "base=${BASE_PATH}"
echo "planner_path=${PLANNER_PATH}"
echo "model_id=${MODEL_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "partition=gpu02"
echo "scene_count=$(wc -w <<< "${SCENES}")"
echo "scenes=${SCENES}"

export BASE_PATH
export PLANNER_PATH
export MODEL_ID

if [[ "${SETUP_VLM_ENV}" == "1" ]]; then
    echo "preparing VLM env for ${MODEL_ID}"
    export INSTALL_VLLM=0
    bash scripts/setup_vlm_env.sh
fi

for scene in ${SCENES}; do
    echo "running scene=${scene}"
    export PLANNER_NAME="rap_vlm"
    export SCENARIO_PATH="${scene}"
    export SETUP_VLM_ENV=0
    bash scripts/run_single_scene.slurm
done
