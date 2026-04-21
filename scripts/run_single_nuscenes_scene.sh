#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "usage: $0 <planner:{rap|rap_vlm}> [scenario_yaml] [sim_cuda|inherit] [ad_cuda|inherit]" >&2
    exit 2
fi

PLANNER_NAME="${1:?missing planner name}"
SCENARIO_PATH="${2:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}"
DEFAULT_CUDA_ID="0"
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]; then
    DEFAULT_CUDA_ID="inherit"
fi
SIM_CUDA="${3:-${DEFAULT_CUDA_ID}}"
AD_CUDA="${4:-${DEFAULT_CUDA_ID}}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local.yaml}"
CAMERA_PATH="${CAMERA_PATH:-configs/sim/nuscenes_camera.yaml}"
KINEMATIC_PATH="${KINEMATIC_PATH:-configs/sim/kinematic.yaml}"
PLANNER_PATH="${PLANNER_PATH:-configs/planners/${PLANNER_NAME}.yaml}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"

if [[ ! -f "${SCENARIO_PATH}" ]]; then
    echo "missing scenario config: ${SCENARIO_PATH}" >&2
    exit 1
fi

if [[ ! -f "${BASE_PATH}" ]]; then
    echo "missing base config: ${BASE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PLANNER_PATH}" ]]; then
    echo "missing planner config: ${PLANNER_PATH}" >&2
    exit 1
fi

if [[ ! -x "${HUGSIM_PYTHON_BIN}" ]]; then
    echo "missing HUGSIM python: ${HUGSIM_PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p /bigdata/aidan/outputs/benchmark/out

HUGSIM_ENV_ROOT="$(cd "$(dirname "${HUGSIM_PYTHON_BIN}")/.." && pwd)"
TORCH_LIB_DIR="${HUGSIM_ENV_ROOT}/lib/python3.11/site-packages/torch/lib"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${HUGSIM_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "planner=${PLANNER_NAME}"
echo "scenario=${SCENARIO_PATH}"
echo "base=${BASE_PATH}"
echo "planner_config=${PLANNER_PATH}"
echo "hugsim_python=${HUGSIM_PYTHON_BIN}"
echo "sim_cuda=${SIM_CUDA} ad_cuda=${AD_CUDA}"

if [[ "${SIM_CUDA}" == "inherit" ]]; then
    "${HUGSIM_PYTHON_BIN}" closed_loop.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad rap \
        --ad_cuda "${AD_CUDA}"
else
    CUDA_VISIBLE_DEVICES="${SIM_CUDA}" \
    "${HUGSIM_PYTHON_BIN}" closed_loop.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad rap \
        --ad_cuda "${AD_CUDA}"
fi
