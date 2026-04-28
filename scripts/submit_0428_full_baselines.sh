#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
MAX_JOBS="${MAX_JOBS:-0}"
GPU_BUDGET="${GPU_BUDGET:-8}"

cd "${REPO_ROOT}"

PARTITION="${PARTITION}" SCENARIO_DIR="${SCENARIO_DIR}" MAX_JOBS="${MAX_JOBS}" GPU_BUDGET="${GPU_BUDGET}" \
    bash scripts/submit_0428_drivor_full.sh
PARTITION="${PARTITION}" SCENARIO_DIR="${SCENARIO_DIR}" MAX_JOBS="${MAX_JOBS}" GPU_BUDGET="${GPU_BUDGET}" \
    bash scripts/submit_0428_rap_full.sh
