#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"
VLM_ENV_DIR="${VLM_ENV_DIR:-/bigdata/aidan/.home/envs/vlm}"
HF_HOME_DIR="${HF_HOME_DIR:-/bigdata/aidan/models/hf}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME_DIR}/hub}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "model_id=${MODEL_ID}"
echo "vlm_env_dir=${VLM_ENV_DIR}"
echo "hf_home=${HF_HOME_DIR}"
echo "hf_hub_cache=${HF_HUB_CACHE_DIR}"

MODEL_ID="${MODEL_ID}" \
VLM_ENV_DIR="${VLM_ENV_DIR}" \
HF_HOME_DIR="${HF_HOME_DIR}" \
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
BOOTSTRAP_PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN}" \
TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
bash scripts/setup_vlm_env.sh
