#!/usr/bin/env bash
set -euo pipefail

VLM_ENV_DIR="${VLM_ENV_DIR:-/bigdata/aidan/.home/envs/vlm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HF_HOME_DIR="${HF_HOME_DIR:-/bigdata/aidan/models/hf}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME_DIR}/hub}"
TMPDIR="${TMPDIR:-/tmp}"
VLM_ENV_REUSE_KEY="${VLM_ENV_REUSE_KEY:-default}"
READY_MARKER_BASENAME=".setup_ready"
READY_MARKER_PATH="${VLM_ENV_DIR}/${READY_MARKER_BASENAME}.${VLM_ENV_REUSE_KEY}"

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
echo "tmpdir=${TMPDIR}"
echo "vlm_env_reuse_key=${VLM_ENV_REUSE_KEY}"

mkdir -p "${HF_HUB_CACHE_DIR}"
mkdir -p "${TMPDIR}"

export TMPDIR
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export PIP_NO_CACHE_DIR=1

if [[ -d "${VLM_ENV_DIR}" && ! -x "${VLM_ENV_DIR}/bin/pip" ]]; then
    rm -rf "${VLM_ENV_DIR}"
fi

if [[ -x "${VLM_ENV_DIR}/bin/python" ]]; then
    if ! "${VLM_ENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "broken_vlm_env_detected=1"
        rm -rf "${VLM_ENV_DIR}"
    fi
fi

if [[ -f "${READY_MARKER_PATH}" && -x "${VLM_ENV_DIR}/bin/python" && -x "${VLM_ENV_DIR}/bin/pip" ]]; then
    echo "reusing_existing_vlm_env=1"
else
    echo "reusing_existing_vlm_env=0"
    rm -f "${VLM_ENV_DIR}/${READY_MARKER_BASENAME}."* 2>/dev/null || true

    if [[ ! -x "${VLM_ENV_DIR}/bin/python" || ! -x "${VLM_ENV_DIR}/bin/pip" ]]; then
        if "${PYTHON_BIN}" -m venv "${VLM_ENV_DIR}"; then
            :
        else
            "${BOOTSTRAP_PYTHON_BIN}" -m pip install --upgrade pip virtualenv
            "${BOOTSTRAP_PYTHON_BIN}" -m virtualenv -p "${PYTHON_BIN}" "${VLM_ENV_DIR}"
        fi
    fi

    "${VLM_ENV_DIR}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel
    "${VLM_ENV_DIR}/bin/pip" install \
        --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
    "${VLM_ENV_DIR}/bin/pip" install \
        --no-cache-dir \
        --upgrade \
        "transformers[serving]" \
        accelerate \
        einops \
        ninja \
        sentencepiece \
        pillow \
        huggingface_hub \
        openai

    # Qwen3.6 fast path in transformers requires both FLA kernels and causal-conv1d.
    # The packaged flash-linear-attention path supports PyTorch >= 2.5.
    "${VLM_ENV_DIR}/bin/pip" uninstall -y fla-core flash-linear-attention causal-conv1d || true
    "${VLM_ENV_DIR}/bin/pip" install \
        --no-cache-dir \
        --upgrade \
        flash-linear-attention \
        causal-conv1d
fi

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

touch "${READY_MARKER_PATH}"
echo "vlm_env_ready_marker=${READY_MARKER_PATH}"
