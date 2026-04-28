#!/usr/bin/env bash
set -euo pipefail

VLM_ENV_DIR="${VLM_ENV_DIR:-/bigdata/aidan/.home/envs/vlm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HF_HOME_DIR="${HF_HOME_DIR:-/bigdata/aidan/models/hf}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME_DIR}/hub}"

if [[ -z "${BOOTSTRAP_PYTHON_BIN}" ]]; then
    for candidate in \
        /bigdata/aidan/models/hf/cli/venv/bin/python \
        /bigdata/aidan/.home/envs/hf/bin/python \
        "${PYTHON_BIN}"
    do
        if [[ -x "${candidate}" ]]; then
            BOOTSTRAP_PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

echo "vlm_env_dir=${VLM_ENV_DIR}"
echo "python_bin=${PYTHON_BIN}"
echo "bootstrap_python_bin=${BOOTSTRAP_PYTHON_BIN}"
echo "model_id=${MODEL_ID}"
echo "hf_home=${HF_HOME_DIR}"

mkdir -p "${HF_HUB_CACHE_DIR}"

if [[ -d "${VLM_ENV_DIR}" && ! -x "${VLM_ENV_DIR}/bin/pip" ]]; then
    rm -rf "${VLM_ENV_DIR}"
fi

if [[ ! -x "${VLM_ENV_DIR}/bin/python" || ! -x "${VLM_ENV_DIR}/bin/pip" ]]; then
    if "${PYTHON_BIN}" -m venv "${VLM_ENV_DIR}"; then
        :
    else
        "${BOOTSTRAP_PYTHON_BIN}" -m pip install --upgrade pip virtualenv
        "${BOOTSTRAP_PYTHON_BIN}" -m virtualenv -p "${PYTHON_BIN}" "${VLM_ENV_DIR}"
    fi
fi

"${VLM_ENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VLM_ENV_DIR}/bin/pip" install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url "${TORCH_INDEX_URL}"
"${VLM_ENV_DIR}/bin/pip" install \
    "transformers>=4.57.0" \
    accelerate \
    sentencepiece \
    pillow \
    huggingface_hub

HF_HOME="${HF_HOME_DIR}" \
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}" \
MODEL_ID="${MODEL_ID}" \
"${VLM_ENV_DIR}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_files_only=False,
)
print(f"downloaded {model_id} into {cache_dir}")
PY
