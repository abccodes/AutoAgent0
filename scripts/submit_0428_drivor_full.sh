#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0428_drivor.yaml}"
MAX_JOBS="${MAX_JOBS:-0}"
GPU_BUDGET="${GPU_BUDGET:-0}"

declare -a PLANNER_NAMES=(
    "drivor"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
)

declare -a PLANNER_PATHS=(
    "configs/planners/drivor.yaml"
    "configs/planners/drivor_vlm_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_0428.yaml"
    "configs/planners/drivor_vlm_intervention_1cam_0428.yaml"
    "configs/planners/drivor_vlm_intervention_4cam_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention_1cam_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention_4cam_0428.yaml"
)

declare -a JOB_LABELS=(
    "drivor"
    "drivor_vlm"
    "drivor_vlm_default_trajectory"
    "drivor_intervention_vlm_1_cam"
    "drivor_intervention_vlm_4_cam"
    "drivor_intervention_vlm_default_trajectories_1_cam"
    "drivor_intervention_vlm_default_trajectories_4_cam"
)

cd "${REPO_ROOT}"

for idx in "${!PLANNER_NAMES[@]}"; do
    planner_name="${PLANNER_NAMES[$idx]}"
    planner_path="${PLANNER_PATHS[$idx]}"
    job_label="${JOB_LABELS[$idx]}"
    echo "submitting planner=${job_label} scenario_dir=${SCENARIO_DIR}"
    PLANNER_PATH="${planner_path}" \
    BASE_PATH="${BASE_PATH}" \
    SUBMIT_JOB_LABEL="${job_label}" \
    SUBMIT_RUN_PREFIX="hugsim-0428" \
    GPU_BUDGET="${GPU_BUDGET}" \
    bash scripts/submit_nuscenes_batch.sh "${planner_name}" "${SCENARIO_DIR}" "${PARTITION}" "${MAX_JOBS}"
done
