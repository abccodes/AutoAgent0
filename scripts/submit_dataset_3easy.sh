#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 <dataset:{waymo|kitti360}> [planner_name] [planner_path]" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
DATASET="${1:?missing dataset}"
PLANNER_NAME="${2:-${PLANNER_NAME:-rap_vlm}}"
PLANNER_PATH="${3:-${PLANNER_PATH:-}}"
PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/bigdata/aidan/outputs/slurm}"
SCENES_FILE="${SCENES_FILE:-}"
BASE_PATH="${BASE_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

default_scenes_file() {
    local dataset="$1"
    case "${dataset}" in
        waymo)
            echo "configs/benchmark/scene_sets/waymo_3easy.txt"
            ;;
        kitti360)
            echo "configs/benchmark/scene_sets/kitti360_3easy.txt"
            ;;
        *)
            echo "unsupported dataset: ${dataset}" >&2
            exit 2
            ;;
    esac
}

default_base_path() {
    local dataset="$1"
    case "${dataset}" in
        waymo)
            echo "configs/sim/waymo_base_local.yaml"
            ;;
        kitti360)
            echo "configs/sim/kitti360_base_local.yaml"
            ;;
        *)
            echo "unsupported dataset: ${dataset}" >&2
            exit 2
            ;;
    esac
}

default_planner_path() {
    local planner_name="$1"
    case "${planner_name}" in
        rap_vlm)
            echo "configs/planners/rap_vlm_intervention_4cam_0428.yaml"
            ;;
        drivor_vlm)
            echo "configs/planners/drivor_vlm_intervention_4cam_0428.yaml"
            ;;
        rule_based)
            echo "configs/planners/rule_based_local_aidan.yaml"
            ;;
        rule_based_vlm)
            echo "configs/planners/rule_based_local_aidan.yaml"
            ;;
        *)
            echo "unsupported planner_name: ${planner_name}" >&2
            exit 2
            ;;
    esac
}

infer_gpu_count() {
    local planner_name="$1"
    case "${planner_name}" in
        rap_vlm|drivor_vlm|rule_based_vlm)
            echo 2
            ;;
        *)
            echo 1
            ;;
    esac
}

if [[ -z "${SCENES_FILE}" ]]; then
    SCENES_FILE="$(default_scenes_file "${DATASET}")"
fi

if [[ -z "${BASE_PATH}" ]]; then
    BASE_PATH="$(default_base_path "${DATASET}")"
fi

if [[ -z "${PLANNER_PATH}" ]]; then
    PLANNER_PATH="$(default_planner_path "${PLANNER_NAME}")"
fi

if [[ ! -f "${SCENES_FILE}" ]]; then
    echo "missing scenes file: ${SCENES_FILE}" >&2
    exit 1
fi

if [[ ! -f "${PLANNER_PATH}" ]]; then
    echo "missing planner config: ${PLANNER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${BASE_PATH}" ]]; then
    echo "missing base config: ${BASE_PATH}" >&2
    exit 1
fi

scene_count="$(wc -l < "${SCENES_FILE}" | tr -d '[:space:]')"
if [[ "${scene_count}" -eq 0 ]]; then
    echo "no scenes listed in ${SCENES_FILE}" >&2
    exit 1
fi

GPUS="$(infer_gpu_count "${PLANNER_NAME}")"
planner_tag="$(basename "${PLANNER_PATH}" .yaml)"
job_name="hugsim-${DATASET}-${planner_tag}-3easy"
array_spec="0-$((scene_count - 1))%${scene_count}"

mkdir -p "${SLURM_LOG_DIR}"
cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "dataset=${DATASET}"
echo "planner_name=${PLANNER_NAME}"
echo "planner_path=${PLANNER_PATH}"
echo "base_path=${BASE_PATH}"
echo "scenes_file=${SCENES_FILE}"
echo "scene_count=${scene_count}"
echo "gpus=${GPUS}"
echo "partition=${PARTITION}"
echo "time=${SBATCH_TIME}"
echo "array=${array_spec}"
echo "scenes:"
cat "${SCENES_FILE}"

sbatch_cmd=(
    sbatch
    --partition="${PARTITION}"
    --time="${SBATCH_TIME}"
    --job-name="${job_name}"
    --output="${SLURM_LOG_DIR}/${job_name}-%A_%a.out"
    --array="${array_spec}"
    --gres="gpu:${GPUS}"
    --export="ALL,REPO_ROOT=${REPO_ROOT},BASE_PATH=${BASE_PATH},PLANNER_PATH=${PLANNER_PATH},PLANNER_NAME=${PLANNER_NAME},SCENES_FILE=${SCENES_FILE}"
    scripts/run_scene_array_item.slurm
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run_command='
    printf '%q ' "${sbatch_cmd[@]}"
    printf '\n'
    exit 0
fi

"${sbatch_cmd[@]}"
