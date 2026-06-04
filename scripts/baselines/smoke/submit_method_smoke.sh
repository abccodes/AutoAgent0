#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
DATASET="${DATASET:-nuscenes}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/bigdata/aidan/outputs/slurm}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"
SMOKE_METHODS="${SMOKE_METHODS:-rap_vlm drivor_vlm rap_intervention_4cam drivor_intervention_4cam rule_based rap_impl_a drivor_impl_a rap_impl_b drivor_impl_b}"
SMOKE_METHODS_EXPORT="${SMOKE_METHODS// /:}"
DRY_RUN="${DRY_RUN:-0}"

default_scenario_for_dataset() {
    local dataset="$1"
    case "${dataset}" in
        nuscenes)
            echo "configs/benchmark/nuscenes_all_variants/scene-0010-easy-00.yaml"
            ;;
        waymo)
            sed -n '1p' configs/benchmark/scene_sets/waymo_3easy.txt
            ;;
        kitti360)
            sed -n '1p' configs/benchmark/scene_sets/kitti360_3easy.txt
            ;;
        *)
            echo "unsupported DATASET=${dataset}" >&2
            exit 2
            ;;
    esac
}

default_suite_for_dataset() {
    local dataset="$1"
    case "${dataset}" in
        nuscenes)
            echo "full"
            ;;
        waymo|kitti360)
            echo "3easy"
            ;;
        *)
            echo "unsupported DATASET=${dataset}" >&2
            exit 2
            ;;
    esac
}

cd "${REPO_ROOT}"

SUITE="${SUITE:-$(default_suite_for_dataset "${DATASET}")}"
SMOKE_SCENARIO_PATH="${SMOKE_SCENARIO_PATH:-$(default_scenario_for_dataset "${DATASET}")}"

if [[ ! -f "${SMOKE_SCENARIO_PATH}" ]]; then
    echo "missing smoke scenario: ${SMOKE_SCENARIO_PATH}" >&2
    exit 1
fi

method_count="$(wc -w <<< "${SMOKE_METHODS}" | tr -d '[:space:]')"

echo "dataset=${DATASET}"
echo "suite=${SUITE}"
echo "scenario=${SMOKE_SCENARIO_PATH}"
echo "smoke_run_id=${SMOKE_RUN_ID}"
echo "methods=${SMOKE_METHODS}"
echo "method_count=${method_count}"
echo "scene_method_runs=${method_count}"
echo "gpu_budget=6"
echo "partition=${PARTITION}"
echo "sbatch_time=${SBATCH_TIME}"

sbatch_cmd=(
    sbatch
    --partition="${PARTITION}"
    --time="${SBATCH_TIME}"
    --gres="gpu:6"
    --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET=${DATASET},SUITE=${SUITE},SMOKE_SCENARIO_PATH=${SMOKE_SCENARIO_PATH},SMOKE_RUN_ID=${SMOKE_RUN_ID},SMOKE_METHODS=${SMOKE_METHODS_EXPORT},HUGSIM_PYTHON_BIN=${HUGSIM_PYTHON_BIN}"
    scripts/baselines/smoke/run_method_smoke.slurm
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run_command='
    printf '%q ' "${sbatch_cmd[@]}"
    printf '\n'
    exit 0
fi

"${sbatch_cmd[@]}"
