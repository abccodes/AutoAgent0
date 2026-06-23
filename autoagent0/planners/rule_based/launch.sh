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
: "${RULE_BASED_PYTHON_BIN:=python}"
: "${RULE_BASED_REPO_ROOT:?RULE_BASED_REPO_ROOT is not set}"
: "${RULE_BASED_DEVICE:=cuda}"
: "${RULE_BASED_CONFIG:?RULE_BASED_CONFIG is not set}"

export PYTHONPATH="${HUGSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure the rule-based env can resolve packaged shared libs before system copies.
RULE_BASED_ENV_ROOT="$(cd "$(dirname "${RULE_BASED_PYTHON_BIN}")/.." && pwd)"
RULE_BASED_TORCH_LIB_DIR="${RULE_BASED_TORCH_LIB_DIR:-}"
if [[ -n "${RULE_BASED_TORCH_LIB_DIR}" && -d "${RULE_BASED_TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${RULE_BASED_TORCH_LIB_DIR}:${RULE_BASED_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Prevent the rule-based env from accidentally loading HUGSIM pixi torch libs.
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

export PLANNER_CONFIG="${RULE_BASED_CONFIG}"

cd "${RULE_BASED_REPO_ROOT}"

exec "${RULE_BASED_PYTHON_BIN}" -m autoagent0.planners.rule_based --output "${OUTPUT_DIR}"
