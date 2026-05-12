#!/usr/bin/env bash
# Payload test harness for HUGSIM env verification
# Runs 2 timesteps and dumps obs/info/privileged agent data to JSON
#
# Usage:
#   ./payload_test.sh [scenario_yaml] [sim_cuda|inherit]
#
# Examples:
#   ./payload_test.sh configs/benchmark/nuscenes/scene-0383-easy-00.yaml
#   ./payload_test.sh configs/benchmark/waymo/demo_scene.yaml 0
#   ./payload_test.sh                                           # uses default scene

set -euo pipefail

# Configuration
SCENARIO_PATH="${1:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}"
DEFAULT_CUDA_ID="0"
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]; then
    DEFAULT_CUDA_ID="inherit"
fi
SIM_CUDA="${2:-${DEFAULT_CUDA_ID}}"

BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base.yaml}"
CAMERA_PATH="${CAMERA_PATH:-configs/sim/nuscenes_camera.yaml}"
KINEMATIC_PATH="${KINEMATIC_PATH:-configs/sim/kinematic.yaml}"

# Python environment
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/payload_test}"

# Validate config files
if [[ ! -f "${SCENARIO_PATH}" ]]; then
    echo "ERROR: missing scenario config: ${SCENARIO_PATH}" >&2
    exit 1
fi

if [[ ! -f "${BASE_PATH}" ]]; then
    echo "ERROR: missing base config: ${BASE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${CAMERA_PATH}" ]]; then
    echo "ERROR: missing camera config: ${CAMERA_PATH}" >&2
    exit 1
fi

if [[ ! -f "${KINEMATIC_PATH}" ]]; then
    echo "ERROR: missing kinematic config: ${KINEMATIC_PATH}" >&2
    exit 1
fi

# Setup environment
export PYTHONPATH="${PWD}:${PWD}/sim${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

HUGSIM_ENV_ROOT="$(cd "$(dirname "${HUGSIM_PYTHON_BIN}")/.." 2>/dev/null && pwd || echo /usr/local/cuda)"
TORCH_LIB_DIR="${HUGSIM_ENV_ROOT}/lib/python3.11/site-packages/torch/lib"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${HUGSIM_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Log info
echo "=========================================="
echo "HUGSIM Payload Test Harness"
echo "=========================================="
echo "Scenario:     ${SCENARIO_PATH}"
echo "Base Config:  ${BASE_PATH}"
echo "Camera Config: ${CAMERA_PATH}"
echo "Kinematic Config: ${KINEMATIC_PATH}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Python:       ${HUGSIM_PYTHON_BIN}"
echo "CUDA Device:  ${SIM_CUDA}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run test
if [[ "${SIM_CUDA}" == "inherit" ]]; then
    echo "Running payload test (CUDA_VISIBLE_DEVICES inherited)..."
    "${HUGSIM_PYTHON_BIN}" planners/rule_based/payload_test.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --output_dir "${OUTPUT_DIR}"
else
    echo "Running payload test (CUDA_VISIBLE_DEVICES=${SIM_CUDA})..."
    CUDA_VISIBLE_DEVICES="${SIM_CUDA}" \
    "${HUGSIM_PYTHON_BIN}" planners/rule_based/payload_test.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --output_dir "${OUTPUT_DIR}"
fi

EXIT_CODE=$?
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Test completed successfully"
    echo "Output files in: ${OUTPUT_DIR}"
    ls -lh "${OUTPUT_DIR}"/payload_test*.json 2>/dev/null || true
else
    echo "✗ Test failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
