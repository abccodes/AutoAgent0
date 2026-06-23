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

PYTHON_BIN="${RAP_PYTHON_BIN:-python}"
HF_HOME_DIR="${RAP_HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE_DIR="${RAP_HF_HUB_CACHE:-${HF_HOME_DIR}/hub}"
TRANSFORMERS_CACHE_DIR="${RAP_TRANSFORMERS_CACHE:-${HF_HUB_CACHE_DIR}}"
NUPLAN_DEVKIT_DIR="${RAP_NUPLAN_DEVKIT_DIR:-/bigdata/jason/drivor_evaluation/DrivoR/nuplan-devkit}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}"
export HF_HUB_OFFLINE="${RAP_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${RAP_TRANSFORMERS_OFFLINE:-1}"

if [[ -d "${NUPLAN_DEVKIT_DIR}" ]]; then
    export PYTHONPATH="${NUPLAN_DEVKIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export PYTHONPATH="${HUGSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Prepend the RAP repo so its navsim/agents/rap_dino takes priority over any
# other navsim installation on PYTHONPATH (e.g. a global PYTHONPATH entry).
if [[ -n "${RAP_REPO_ROOT:-}" && -d "${RAP_REPO_ROOT}" ]]; then
    export PYTHONPATH="${RAP_REPO_ROOT}:${PYTHONPATH}"
fi

# Prevent the RAP env from accidentally loading HUGSIM pixi torch libs.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    OLD_IFS="${IFS}"
    IFS=':'
    CLEANED_ENTRIES=()
    for entry in ${LD_LIBRARY_PATH}; do
        if [[ "${entry}" != *"/HUGSIM/.pixi/"* ]]; then
            CLEANED_ENTRIES+=("${entry}")
        fi
    done
    IFS="${OLD_IFS}"
    CLEANED_LD_LIBRARY_PATH="$(printf '%s:' "${CLEANED_ENTRIES[@]}")"
    CLEANED_LD_LIBRARY_PATH="${CLEANED_LD_LIBRARY_PATH%:}"
    if [[ -n "${CLEANED_LD_LIBRARY_PATH}" ]]; then
        export LD_LIBRARY_PATH="${CLEANED_LD_LIBRARY_PATH}"
    else
        unset LD_LIBRARY_PATH
    fi
fi

exec "${PYTHON_BIN}" -m autoagent0.planners.rap --output "${OUTPUT_DIR}"
