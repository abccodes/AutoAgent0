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

# Debug: echo invocation and key environment values so runtime can be inspected
echo "[rule_based/launch.sh] PID $$ invoked"
echo "  CUDA_ID=${CUDA_ID}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  EXPECTED_OBS_PIPE=${OUTPUT_DIR}/obs_pipe"


# Provided by HUGSIM closed_loop.py via extra_env
: "${RULE_BASED_PYTHON_BIN:=python}"
: "${RULE_BASED_REPO_ROOT:?RULE_BASED_REPO_ROOT is not set}"
: "${RULE_BASED_DEVICE:=cuda}"
: "${RULE_BASED_CONFIG:?RULE_BASED_CONFIG is not set}"

echo "  RULE_BASED_PYTHON_BIN=${RULE_BASED_PYTHON_BIN}"
echo "  RULE_BASED_REPO_ROOT=${RULE_BASED_REPO_ROOT}"
echo "  RULE_BASED_DEVICE=${RULE_BASED_DEVICE}"
echo "  RULE_BASED_CONFIG=${RULE_BASED_CONFIG}"
echo "  AD_CUDA=${AD_CUDA:-<unset>}"
#gpt recommended this for some reason, but I don't see a reason to deviate from the format that DrivoR has.
# export RULE_BASED_PYTHON_BIN="${RULE_BASED_PYTHON_BIN:-python}"
# export RULE_BASED_REPO_ROOT="${RULE_BASED_REPO_ROOT:?RULE_BASED_REPO_ROOT is not set}"
# export RULE_BASED_DEVICE="${RULE_BASED_DEVICE:-cuda}"
# export RULE_BASED_CONFIG="${RULE_BASED_CONFIG:?RULE_BASED_CONFIG is not set}"

export PYTHONPATH="${HUGSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure the rule-based env can resolve any packaged shared libraries before the
# system copies. This is optional and only applies when a torch lib directory is present.
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
#original code
# cd "${DRIVOR_REPO_ROOT}"
export PLANNER_CONFIG="${RULE_BASED_CONFIG}"

cd "${RULE_BASED_REPO_ROOT}"

echo "Changing to repo: $(pwd)"
echo "Launching client: ${RULE_BASED_PYTHON_BIN} ${SCRIPT_DIR}/client.py --output ${OUTPUT_DIR}"
exec "${RULE_BASED_PYTHON_BIN}" "${SCRIPT_DIR}/client.py" --output "${OUTPUT_DIR}"
