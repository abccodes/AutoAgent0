#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
SBATCH_WAIT="${SBATCH_WAIT:-1}"

cd "${REPO_ROOT}"

PARTITION="${PARTITION}" SCENARIO_DIR="${SCENARIO_DIR}" SBATCH_WAIT="${SBATCH_WAIT}" \
    bash scripts/submit_0428_drivor_full.sh
PARTITION="${PARTITION}" SCENARIO_DIR="${SCENARIO_DIR}" SBATCH_WAIT="${SBATCH_WAIT}" \
    bash scripts/submit_0428_rap_full.sh
