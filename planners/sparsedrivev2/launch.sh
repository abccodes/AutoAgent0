#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${1:?missing cuda id}"
OUTPUT_DIR="${2:?missing output dir}"
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
HUGSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${CUDA_ID}" != "inherit" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
fi

# Debug: surface key env and paths for troubleshooting torch/LD issues
echo "LAUNCH DEBUG: CUDA_ID=${CUDA_ID} OUTPUT_DIR=${OUTPUT_DIR}"
echo "LAUNCH DEBUG: initial LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "LAUNCH DEBUG: (SparseDriveV2 env vars will be shown after defaults are applied)"
# Provided by HUGSIM closed_loop.py via extra_env
: "${SPARSEDRIVE_PYTHON_BIN:=python}"
: "${SPARSEDRIVE_REPO_ROOT:?SPARSEDRIVE_REPO_ROOT is not set}"
: "${SPARSEDRIVE_CHECKPOINT:?SPARSEDRIVE_CHECKPOINT is not set}"
: "${SPARSEDRIVE_DEVICE:=cuda}"
: "${SPARSEDRIVE_CONFIG:?SPARSEDRIVE_CONFIG is not set}"

# Optional: if you add config composition
# : "${SPARSEDRIVE_CONFIG_DIR:?SPARSEDRIVE_CONFIG_DIR is not set}"
# : "${SPARSEDRIVE_EXPERIMENT:?SPARSEDRIVE_EXPERIMENT is not set}"

export PYTHONPATH="${HUGSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure the Sparsedrive env's torch CUDA libs resolve before system/HUGSIM copies.
echo "LAUNCH DEBUG: SPARSEDRIVE_PYTHON_BIN=${SPARSEDRIVE_PYTHON_BIN:-unset}"
SPARSEDRIVE_ENV_ROOT="$(cd "$(dirname "${SPARSEDRIVE_PYTHON_BIN}")/.." && pwd)"
echo "LAUNCH DEBUG: SPARSEDRIVE_ENV_ROOT=${SPARSEDRIVE_ENV_ROOT}"
SPARSEDRIVE_TORCH_LIB_DIR="${SPARSEDRIVE_ENV_ROOT}/lib/python3.8/site-packages/torch/lib"
if [[ -d "${SPARSEDRIVE_TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SPARSEDRIVE_TORCH_LIB_DIR}:${SPARSEDRIVE_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "LAUNCH DEBUG: SPARSEDRIVE_PYTHON_BIN=${SPARSEDRIVE_PYTHON_BIN:-unset}"
echo "LAUNCH DEBUG: SPARSEDRIVE_REPO_ROOT=${SPARSEDRIVE_REPO_ROOT:-unset}"
echo "LAUNCH DEBUG: SPARSEDRIVE_CHECKPOINT=${SPARSEDRIVE_CHECKPOINT:-unset}"

echo "LAUNCH DEBUG: SPARSEDRIVE_TORCH_LIB_DIR=${SPARSEDRIVE_TORCH_LIB_DIR} (exists=$( [[ -d \"${SPARSEDRIVE_TORCH_LIB_DIR}\" ]] && echo true || echo false ))"
echo "LAUNCH DEBUG: LD_LIBRARY_PATH after adding drivor torch lib=${LD_LIBRARY_PATH:-unset}"

# Prevent the DrivoR env from accidentally loading HUGSIM pixi torch libs.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  CLEANED_LD_LIBRARY_PATH="$(python3 - <<'PY'
import os
entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
filtered = [entry for entry in entries if "/HUGSIM/.pixi/" not in entry]
print(":".join(filtered))
PY
)"
  if [[ -n "${CLEANED_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${CLEANED_LD_LIBRARY_PATH}"
  else
    unset LD_LIBRARY_PATH
  fi
fi

echo "LAUNCH DEBUG: LD_LIBRARY_PATH after cleaning HUGSIM entries=${LD_LIBRARY_PATH:-unset}"

echo "LAUNCH DEBUG: Environment summary (CUDA/LD/PYTHONPATH):"
env | grep -E 'CUDA|LD_LIBRARY_PATH|PYTHONPATH' || true

cd "${SPARSEDRIVE_REPO_ROOT}"

echo "LAUNCH DEBUG: exec -> ${SPARSEDRIVE_PYTHON_BIN} ${SCRIPT_DIR}/client.py --output ${OUTPUT_DIR}"
exec "${SPARSEDRIVE_PYTHON_BIN}" "${SCRIPT_DIR}/client.py" --output "${OUTPUT_DIR}"
