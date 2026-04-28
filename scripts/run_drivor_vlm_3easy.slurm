#!/bin/bash
#SBATCH --job-name=hugsim-drivor-vlm-3easy
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_drivor.yaml}"
PLANNER_PATH="${PLANNER_PATH:-configs/planners/drivor_vlm.yaml}"
SCENES="${SCENES:-configs/benchmark/nuscenes_all_variants/scene-0010-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0013-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml}"
DRIVOR_DEVICE_OVERRIDE="${DRIVOR_DEVICE_OVERRIDE:-}"
PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-}"

mkdir -p /bigdata/aidan/outputs/slurm
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "base=${BASE_PATH}"
echo "planner_path=${PLANNER_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "scenes=${SCENES}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
    DRIVOR_DEVICE_OVERRIDE="${DRIVOR_DEVICE_OVERRIDE:-cuda:0}"
    PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-cuda:1}"
fi

echo "drivor_device_override=${DRIVOR_DEVICE_OVERRIDE:-unset}"
echo "planner_vlm_device_override=${PLANNER_VLM_DEVICE_OVERRIDE:-unset}"

export BASE_PATH
export PLANNER_PATH
export DRIVOR_DEVICE_OVERRIDE
export PLANNER_VLM_DEVICE_OVERRIDE

for scene in ${SCENES}; do
    echo "running scene=${scene}"
    export PLANNER_NAME="drivor"
    export SCENARIO_PATH="${scene}"
    bash scripts/run_single_scene.slurm
done
