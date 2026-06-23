#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${1:?missing cuda id}"
OUTPUT_DIR="${2:?missing output dir}"
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
HUGSIM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${CUDA_ID}" != "inherit" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
fi

# Provided by pipeline.py via extra_env
: "${DRIVOR_PYTHON_BIN:=python}"
: "${DRIVOR_REPO_ROOT:?DRIVOR_REPO_ROOT is not set}"
: "${DRIVOR_CHECKPOINT:?DRIVOR_CHECKPOINT is not set}"
: "${DRIVOR_DEVICE:=cuda}"
: "${DRIVOR_DINO:?DRIVOR_DINO is not set}"
: "${DRIVOR_CONFIG:?DRIVOR_CONFIG is not set}"

export PYTHONPATH="${HUGSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure the DrivoR env's torch CUDA libs resolve before system/HUGSIM copies.
DRIVOR_ENV_ROOT="$(cd "$(dirname "${DRIVOR_PYTHON_BIN}")/.." && pwd)"
DRIVOR_TORCH_LIB_DIR="${DRIVOR_ENV_ROOT}/lib/python3.8/site-packages/torch/lib"
if [[ -d "${DRIVOR_TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${DRIVOR_TORCH_LIB_DIR}:${DRIVOR_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

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

cd "${DRIVOR_REPO_ROOT}"

exec "${DRIVOR_PYTHON_BIN}" -m autoagent0.planners.drivor --output "${OUTPUT_DIR}"
