#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SCENE_A="${SCENE_A:-scene-0010-easy-00.yaml}"
SCENE_B="${SCENE_B:-scene-0013-easy-00.yaml}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_drivor_debug.yaml}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/bigdata/aidan/outputs/slurm}"

mkdir -p "${SLURM_LOG_DIR}"

SCENES_FILE="$(mktemp /tmp/drivor_vlm_debug_2_easy.XXXXXX.scenes)"
trap 'rm -f "${SCENES_FILE}"' EXIT

printf '%s/%s\n%s/%s\n' "${SCENARIO_DIR}" "${SCENE_A}" "${SCENARIO_DIR}" "${SCENE_B}" > "${SCENES_FILE}"

run_variant() {
  local planner_path="$1"
  local planner_name="$2"
  local job_name="$3"

  echo "Submitting ${job_name} with scenes:"
  cat "${SCENES_FILE}"

  sbatch --wait \
    --partition="${PARTITION}" \
    --time="${SBATCH_TIME}" \
    --job-name="${job_name}" \
    --output="${SLURM_LOG_DIR}/${job_name}-%A_%a.out" \
    --array=0-1%2 \
    --gres=gpu:2 \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",BASE_PATH="${BASE_PATH}",PLANNER_PATH="${planner_path}",PLANNER_NAME="${planner_name}",SCENES_FILE="${SCENES_FILE}" \
    scripts/run_scene_array_item.slurm
}

cd "${REPO_ROOT}"

run_variant "configs/planners/drivor_vlm_0428.yaml" "drivor_vlm" "hugsim-drivor-vlm-debug2"
run_variant "configs/planners/drivor_vlm_intervention_1cam_0428.yaml" "drivor_vlm" "hugsim-drivor-int1-debug2"
