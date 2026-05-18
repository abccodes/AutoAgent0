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

: "${RULE_BASED_PYTHON_BIN:=python}"
: "${RULE_BASED_REPO_ROOT:?RULE_BASED_REPO_ROOT is not set}"
: "${RULE_BASED_DEVICE:=cpu}"
: "${RULE_BASED_CONFIG:?RULE_BASED_CONFIG is not set}"

export PYTHONPATH="${HUGSIM_ROOT}:${HUGSIM_ROOT}/sim${PYTHONPATH:+:${PYTHONPATH}}"
cd "${HUGSIM_ROOT}"

exec "${RULE_BASED_PYTHON_BIN}" "${SCRIPT_DIR}/client.py" --output "${OUTPUT_DIR}"
