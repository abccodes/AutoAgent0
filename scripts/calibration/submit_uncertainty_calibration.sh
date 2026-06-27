#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0522_full.yaml}"
MAX_SCENES="${MAX_SCENES:-0}"
GPU_WORKERS="${GPU_WORKERS:-6}"
PLANNER_FAMILY="${PLANNER_FAMILY:-drivor}"
BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE:-canonical}"
RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-calibration-$(date +%Y%m%d)}"
CALIBRATION_HORIZON_STEPS="${CALIBRATION_HORIZON_STEPS:-20}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "${PLANNER_FAMILY}" in
    drivor|rap|both) ;;
    *)
        echo "unsupported PLANNER_FAMILY=${PLANNER_FAMILY}" >&2
        exit 1
        ;;
esac

scene_count="$(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | wc -l | tr -d ' ')"
echo "partition=${PARTITION}"
echo "sbatch_time=${SBATCH_TIME}"
echo "gpu_workers=${GPU_WORKERS}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scene_count=${scene_count}"
echo "max_scenes=${MAX_SCENES}"
echo "planner_family=${PLANNER_FAMILY}"
echo "run_variant=${RUN_VARIANT_TAG}"
echo "base_path=${BASE_PATH}"
echo "calibration_horizon_steps=${CALIBRATION_HORIZON_STEPS}"
echo "skip_analysis=${SKIP_ANALYSIS}"

sbatch_cmd=(
  sbatch
  --partition="${PARTITION}"
  --time="${SBATCH_TIME}"
  --gres="gpu:${GPU_WORKERS}"
  --export=ALL,SCENARIO_DIR="${SCENARIO_DIR}",BASE_PATH="${BASE_PATH}",MAX_SCENES="${MAX_SCENES}",GPU_WORKERS="${GPU_WORKERS}",PLANNER_FAMILY="${PLANNER_FAMILY}",BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE}",RUN_VARIANT_TAG="${RUN_VARIANT_TAG}",CALIBRATION_HORIZON_STEPS="${CALIBRATION_HORIZON_STEPS}",SKIP_ANALYSIS="${SKIP_ANALYSIS}"
  scripts/calibration/run_uncertainty_calibration.slurm
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'dry_run_command='
  printf '%q ' "${sbatch_cmd[@]}"
  printf '\n'
  exit 0
fi

"${sbatch_cmd[@]}"
