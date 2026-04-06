#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${1:?missing cuda id}"
OUTPUT_DIR="${2:?missing output dir}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

PYTHON_BIN="${RAP_PYTHON_BIN:-python}"
HF_HOME_DIR="${RAP_HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE_DIR="${RAP_HF_HUB_CACHE:-${HF_HOME_DIR}/hub}"
TRANSFORMERS_CACHE_DIR="${RAP_TRANSFORMERS_CACHE:-${HF_HUB_CACHE_DIR}}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}"
export HF_HUB_OFFLINE="${RAP_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${RAP_TRANSFORMERS_OFFLINE:-1}"

# Prevent the RAP env from accidentally loading HUGSIM pixi torch libs.
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

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/client.py" --output "${OUTPUT_DIR}"
