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

# Provided by HUGSIM closed_loop.py via extra_env
: "${SPARSEDRIVE_PYTHON_BIN:=python}"
: "${SPARSEDRIVE_REPO_ROOT:?SPARSEDRIVE_REPO_ROOT is not set}"
: "${SPARSEDRIVE_CHECKPOINT:?SPARSEDRIVE_CHECKPOINT is not set}"
: "${SPARSEDRIVE_DEVICE:=cuda}"

echo "LAUNCH DEBUG: CUDA_ID=${CUDA_ID} OUTPUT_DIR=${OUTPUT_DIR}"
echo "LAUNCH DEBUG: SPARSEDRIVE_PYTHON_BIN=${SPARSEDRIVE_PYTHON_BIN}"
echo "LAUNCH DEBUG: SPARSEDRIVE_REPO_ROOT=${SPARSEDRIVE_REPO_ROOT}"
echo "LAUNCH DEBUG: SPARSEDRIVE_CHECKPOINT=${SPARSEDRIVE_CHECKPOINT}"

# Build sanitized env for SparseDrive child process only.
SPARSEDRIVE_ENV_ROOT="$(cd "$(dirname "${SPARSEDRIVE_PYTHON_BIN}")/.." && pwd)"
SPARSEDRIVE_PY_VER="$("${SPARSEDRIVE_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SPARSEDRIVE_TORCH_LIB_DIR="${SPARSEDRIVE_ENV_ROOT}/lib/python${SPARSEDRIVE_PY_VER}/site-packages/torch/lib"

# Keep HUGSIM import path for planners/common, but sanitize all conflicting runtime vars.
export PYTHONPATH="${HUGSIM_ROOT}"

# Remove variables known to cause cross-env contamination.
unset PIP_PREFIX || true
unset PIP_TARGET || true
unset PYTHONHOME || true

# Start with conda env libs only.
if [[ -d "${SPARSEDRIVE_TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SPARSEDRIVE_TORCH_LIB_DIR}:${SPARSEDRIVE_ENV_ROOT}/lib"
else
  export LD_LIBRARY_PATH="${SPARSEDRIVE_ENV_ROOT}/lib"
fi

# Ensure conda env bin is first.
export PATH="${SPARSEDRIVE_ENV_ROOT}/bin:${PATH}"

echo "LAUNCH DEBUG: SPARSEDRIVE_ENV_ROOT=${SPARSEDRIVE_ENV_ROOT}"
echo "LAUNCH DEBUG: SPARSEDRIVE_PY_VER=${SPARSEDRIVE_PY_VER}"
echo "LAUNCH DEBUG: SPARSEDRIVE_TORCH_LIB_DIR=${SPARSEDRIVE_TORCH_LIB_DIR} (exists=$( [[ -d "${SPARSEDRIVE_TORCH_LIB_DIR}" ]] && echo true || echo false ))"
echo "LAUNCH DEBUG: PYTHONPATH=${PYTHONPATH:-unset}"
echo "LAUNCH DEBUG: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "LAUNCH DEBUG: PATH(head)=$(echo "${PATH}" | cut -d: -f1-3)"

cd "${SPARSEDRIVE_REPO_ROOT}"

echo "LAUNCH DEBUG: exec -> ${SPARSEDRIVE_PYTHON_BIN} ${SCRIPT_DIR}/client.py --output ${OUTPUT_DIR}"
exec "${SPARSEDRIVE_PYTHON_BIN}" "${SCRIPT_DIR}/client.py" --output "${OUTPUT_DIR}"