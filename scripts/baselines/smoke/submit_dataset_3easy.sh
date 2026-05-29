#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <dataset:{waymo|kitti360}> <baseline_id> [run_type:{canonical|debug}]" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
DATASET="${1:?missing dataset}"
BASELINE_ID="${2:?missing baseline_id}"
RUN_TYPE="${3:-${RUN_TYPE:-debug}}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
PARTITION="${PARTITION:-gpu02}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/bigdata/aidan/outputs/slurm}"
SUITE="${SUITE:-3easy}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${RUN_TYPE}" != "canonical" && "${RUN_TYPE}" != "debug" ]]; then
    echo "unsupported run_type: ${RUN_TYPE}" >&2
    exit 2
fi

read_registry_field() {
    local field="$1"
    "${HUGSIM_PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${BASELINE_ID}" "${DATASET}" "${field}"
import os
import sys
from omegaconf import OmegaConf

repo_root, baseline_id, dataset, field = sys.argv[1:]
cfg = OmegaConf.load(os.path.join(repo_root, "configs", "baselines", "registry.yaml"))
entry = cfg.baselines.get(baseline_id)
if entry is None:
    raise SystemExit(f"unknown baseline_id: {baseline_id}")
dataset_entry = entry.datasets.get(dataset)
if dataset_entry is None:
    raise SystemExit(f"baseline {baseline_id} does not support dataset {dataset}")
if field == "planner_name":
    print(entry.planner_name)
elif field == "planner_path":
    print(entry.planner_path)
elif field == "base_path":
    print(dataset_entry.base_path)
elif field == "run_variant":
    print(entry.get("run_variant", "default"))
else:
    raise SystemExit(f"unsupported field: {field}")
PY
}

resolve_scene_file() {
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

SCENES_FILE="${SCENES_FILE:-$(resolve_scene_file "${DATASET}")}"
PLANNER_NAME="${PLANNER_NAME:-$(read_registry_field planner_name)}"
PLANNER_PATH="${PLANNER_PATH:-$(read_registry_field planner_path)}"
BASE_PATH="${BASE_PATH:-$(read_registry_field base_path)}"
RUN_VARIANT="${RUN_VARIANT:-$(read_registry_field run_variant)}"

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
job_name="hugsim-${DATASET}-${BASELINE_ID}-${SUITE}"
array_spec="0-$((scene_count - 1))%${scene_count}"

mkdir -p "${SLURM_LOG_DIR}"
cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "dataset=${DATASET}"
echo "baseline_id=${BASELINE_ID}"
echo "planner_name=${PLANNER_NAME}"
echo "planner_path=${PLANNER_PATH}"
echo "base_path=${BASE_PATH}"
echo "suite=${SUITE}"
echo "run_type=${RUN_TYPE}"
echo "run_variant=${RUN_VARIANT}"
echo "scenes_file=${SCENES_FILE}"
echo "scene_count=${scene_count}"
echo "gpus=${GPUS}"

sbatch_cmd=(
    sbatch
    --partition="${PARTITION}"
    --time="${SBATCH_TIME}"
    --job-name="${job_name}"
    --output="${SLURM_LOG_DIR}/${job_name}-%A_%a.out"
    --array="${array_spec}"
    --gres="gpu:${GPUS}"
    --export="ALL,REPO_ROOT=${REPO_ROOT},BASELINE_ID=${BASELINE_ID},BASE_PATH=${BASE_PATH},PLANNER_PATH=${PLANNER_PATH},PLANNER_NAME=${PLANNER_NAME},SCENES_FILE=${SCENES_FILE},BASELINE_RUN_TYPE=${RUN_TYPE},BASELINE_SUITE=${SUITE},BASELINE_RUN_VARIANT=${RUN_VARIANT}"
    scripts/baselines/common/run_scene_array_item.slurm
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run_command='
    printf '%q ' "${sbatch_cmd[@]}"
    printf '\n'
    exit 0
fi

"${sbatch_cmd[@]}"
