#!/bin/bash
#SBATCH --job-name=hugsim-single
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PLANNER_NAME="${PLANNER_NAME:-rap}"
SCENARIO_PATH="${SCENARIO_PATH:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}"
BASE_PATH="${BASE_PATH:-}"
CAMERA_PATH="${CAMERA_PATH:-}"
SIM_CUDA="${SIM_CUDA:-inherit}"
AD_CUDA="${AD_CUDA:-inherit}"
SETUP_VLM_ENV="${SETUP_VLM_ENV:-0}"
RAP_DEVICE_OVERRIDE="${RAP_DEVICE_OVERRIDE:-}"
DRIVOR_DEVICE_OVERRIDE="${DRIVOR_DEVICE_OVERRIDE:-}"
PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-}"

mkdir -p /bigdata/aidan/outputs/slurm
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

echo "job_id=${SLURM_JOB_ID:-none}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "planner=${PLANNER_NAME}"
echo "scenario=${SCENARIO_PATH}"
echo "base=${BASE_PATH}"
echo "camera=${CAMERA_PATH}"
echo "sim_cuda=${SIM_CUDA}"
echo "ad_cuda=${AD_CUDA}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "home=${HOME}"
echo "xdg_cache_home=${XDG_CACHE_HOME}"
echo "mplconfigdir=${MPLCONFIGDIR}"

case "${PLANNER_NAME}" in
    rap_vlm)
        if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
            RAP_DEVICE_OVERRIDE="${RAP_DEVICE_OVERRIDE:-cuda:0}"
            PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-cuda:1}"
        else
            RAP_DEVICE_OVERRIDE="${RAP_DEVICE_OVERRIDE:-cuda:0}"
            if [[ "${MODEL_ID:-}" == *"Qwen/Qwen3.6-"* ]]; then
                PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-auto}"
            else
                PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-cuda:0}"
            fi
        fi
        ;;
    drivor_vlm)
        if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
            DRIVOR_DEVICE_OVERRIDE="${DRIVOR_DEVICE_OVERRIDE:-cuda:0}"
            PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-cuda:1}"
        else
            DRIVOR_DEVICE_OVERRIDE="${DRIVOR_DEVICE_OVERRIDE:-cuda:0}"
            PLANNER_VLM_DEVICE_OVERRIDE="${PLANNER_VLM_DEVICE_OVERRIDE:-cuda:0}"
        fi
        ;;
esac

echo "rap_device_override=${RAP_DEVICE_OVERRIDE:-unset}"
echo "drivor_device_override=${DRIVOR_DEVICE_OVERRIDE:-unset}"
echo "planner_vlm_device_override=${PLANNER_VLM_DEVICE_OVERRIDE:-unset}"

if [[ "${SETUP_VLM_ENV}" == "1" && "${PLANNER_NAME}" == "rap_vlm" ]]; then
    if [[ "${MODEL_ID:-}" == *"Qwen/Qwen3.6-"* ]]; then
        export VLM_ENV_DIR="${VLM_ENV_DIR:-/bigdata/aidan/.home/envs/vlm_qwen36}"
        export VLM_ENV_REUSE_KEY="${VLM_ENV_REUSE_KEY:-qwen3.6-cu121-torch251}"
        echo "vlm_env_dir=${VLM_ENV_DIR}"
        echo "vlm_env_reuse_key=${VLM_ENV_REUSE_KEY}"
    fi
    bash scripts/setup_vlm_env.sh
fi

if [[ "${PLANNER_NAME}" == "rap_vlm" && "${MODEL_ID:-}" == *"Qwen/Qwen3.6-"* ]]; then
    export VLM_ENV_DIR="${VLM_ENV_DIR:-/bigdata/aidan/.home/envs/vlm_qwen36}"
    export PLANNER_VLM_PYTHON_BIN="${PLANNER_VLM_PYTHON_BIN:-${VLM_ENV_DIR}/bin/python}"
    echo "planner_vlm_python_bin=${PLANNER_VLM_PYTHON_BIN}"
fi

export RAP_DEVICE_OVERRIDE
export DRIVOR_DEVICE_OVERRIDE
export PLANNER_VLM_DEVICE_OVERRIDE

BASE_PATH="${BASE_PATH}" \
CAMERA_PATH="${CAMERA_PATH}" \
bash scripts/baselines/common/run_single_nuscenes_scene.sh "${PLANNER_NAME}" "${SCENARIO_PATH}" "${SIM_CUDA}" "${AD_CUDA}"
