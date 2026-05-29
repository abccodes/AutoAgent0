#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0522_full.yaml}"
MAX_SCENES="${MAX_SCENES:-0}"
METHOD_FILTER="${METHOD_FILTER:-all}"
RUN_TYPE="${RUN_TYPE:-canonical}"
DRY_RUN="${DRY_RUN:-0}"

case "${RUN_TYPE}" in
  canonical|debug) ;;
  *)
    echo "unsupported RUN_TYPE=${RUN_TYPE}" >&2
    exit 1
    ;;
esac

gpu_budget="6"
methods="drivor_impl_a drivor_impl_b rap_impl_a rap_impl_b"
case "${METHOD_FILTER}" in
  all)
    gpu_budget="6 (1 + 1 + 2 + 2)"
    ;;
  rap_only)
    gpu_budget="4 (2 + 2)"
    methods="rap_impl_a rap_impl_b"
    ;;
  drivor_only)
    gpu_budget="2 (1 + 1)"
    methods="drivor_impl_a drivor_impl_b"
    ;;
  drivor_impl_a|drivor_impl_b)
    gpu_budget="1"
    methods="${METHOD_FILTER}"
    ;;
  rap_impl_a|rap_impl_b)
    gpu_budget="2"
    methods="${METHOD_FILTER}"
    ;;
  *)
    echo "unsupported METHOD_FILTER=${METHOD_FILTER}" >&2
    exit 1
    ;;
esac

scene_count="$(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | wc -l | tr -d ' ')"
echo "partition=${PARTITION}"
echo "sbatch_time=${SBATCH_TIME}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scene_count=${scene_count}"
echo "base_path=${BASE_PATH}"
echo "max_scenes=${MAX_SCENES}"
echo "method_filter=${METHOD_FILTER}"
echo "methods=${methods}"
echo "gpu_budget=${gpu_budget}"
echo "run_type=${RUN_TYPE}"

sbatch_cmd=(
  sbatch
  --partition="${PARTITION}"
  --time="${SBATCH_TIME}"
  --gres="gpu:${gpu_budget%% *}"
  --export=ALL,SCENARIO_DIR="${SCENARIO_DIR}",BASE_PATH="${BASE_PATH}",MAX_SCENES="${MAX_SCENES}",METHOD_FILTER="${METHOD_FILTER}",BASELINE_RUN_TYPE="${RUN_TYPE}"
  scripts/baselines/full/run_nuscenes_full_baselines.slurm
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'dry_run_command='
  printf '%q ' "${sbatch_cmd[@]}"
  printf '\n'
  exit 0
fi

"${sbatch_cmd[@]}"
