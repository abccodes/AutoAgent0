#!/bin/bash
#SBATCH --job-name=hugsim-drivor-vlm-variants-3easy-debug
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION_NAME="${SLURM_JOB_PARTITION:-gpu02}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_drivor_debug.yaml}"
SCENES="${SCENES:-configs/benchmark/nuscenes_all_variants/scene-0010-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0013-easy-00.yaml configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml}"

mkdir -p /bigdata/aidan/outputs/slurm
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "partition=${PARTITION_NAME}"
echo "base=${BASE_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "scenes=${SCENES}"

GPU_PAIRS=(
    "cuda:0 cuda:1"
    "cuda:2 cuda:3"
    "cuda:4 cuda:5"
)

run_variant() {
    local planner_path="$1"

    echo "=== planner_path=${planner_path} base_path=${BASE_PATH} ==="
    export PLANNER_PATH="${planner_path}"

    local idx=0
    local pids=()
    for scene in ${SCENES}; do
        local pair="${GPU_PAIRS[$idx]}"
        local drivor_device
        local vlm_device
        drivor_device="$(echo "${pair}" | awk '{print $1}')"
        vlm_device="$(echo "${pair}" | awk '{print $2}')"
        echo "--- scene=${scene}"
        (
            export DRIVOR_DEVICE_OVERRIDE="${drivor_device}"
            export PLANNER_VLM_DEVICE_OVERRIDE="${vlm_device}"
            BASE_PATH="${BASE_PATH}" \
            bash scripts/run_single_nuscenes_scene.sh drivor "${scene}" inherit inherit
        ) &
        pids+=($!)
        idx=$((idx + 1))
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "variant failed: ${planner_path}" >&2
        exit 1
    fi
}

run_variant "configs/planners/drivor_vlm_intervention.yaml"
run_variant "configs/planners/drivor_vlm_default_trajectory_intervention.yaml"
run_variant "configs/planners/drivor_vlm.yaml"
run_variant "configs/planners/drivor_vlm_default_trajectory_config.yaml"
