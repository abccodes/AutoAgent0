#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_drivor.yaml}"
MAX_JOBS="${MAX_JOBS:-0}"

PLANNERS=(
    "configs/planners/drivor_vlm_intervention.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention.yaml"
    "configs/planners/drivor_vlm.yaml"
    "configs/planners/drivor_vlm_default_trajectory_config.yaml"
)

cd "${REPO_ROOT}"

for planner_path in "${PLANNERS[@]}"; do
    planner_tag="$(basename "${planner_path}" .yaml)"
    echo "submitting planner=${planner_tag} scenario_dir=${SCENARIO_DIR}"
    PLANNER_PATH="${planner_path}" \
    BASE_PATH="${BASE_PATH}" \
    SUBMIT_JOB_LABEL="${planner_tag}" \
    bash scripts/submit_nuscenes_batch.sh drivor_vlm "${SCENARIO_DIR}" "${PARTITION}" "${MAX_JOBS}"
done
