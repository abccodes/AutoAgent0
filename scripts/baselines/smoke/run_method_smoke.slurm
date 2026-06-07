#!/bin/bash
#SBATCH --job-name=hugsim-method-smoke
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
DATASET="${DATASET:-nuscenes}"
SUITE="${SUITE:-full}"
SMOKE_SCENARIO_PATH="${SMOKE_SCENARIO_PATH:?missing SMOKE_SCENARIO_PATH}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:?missing SMOKE_RUN_ID}"
SMOKE_METHODS="${SMOKE_METHODS:-rap_vlm drivor_vlm rap_intervention_4cam drivor_intervention_4cam rule_based rap_impl_a drivor_impl_a rap_impl_b drivor_impl_b}"
SMOKE_METHODS="${SMOKE_METHODS//:/ }"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/bigdata/aidan/outputs/slurm/method_smoke/${SMOKE_RUN_ID}/${DATASET}}"
VLM_PRELOAD_LOCK_PATH="${VLM_PRELOAD_LOCK_PATH:-${SLURM_LOG_DIR}/vlm_preload.lock}"
SMOKE_PLAN_RESPONSE_TIMEOUT_SEC="${SMOKE_PLAN_RESPONSE_TIMEOUT_SEC:-600}"

mkdir -p /bigdata/aidan/outputs/slurm "${SLURM_LOG_DIR}"
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

method_enabled() {
    local method="$1"
    for selected in ${SMOKE_METHODS}; do
        if [[ "${selected}" == "${method}" ]]; then
            return 0
        fi
    done
    return 1
}

read_registry_field() {
    local baseline_id="$1"
    local field="$2"
    "${HUGSIM_PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${baseline_id}" "${DATASET}" "${field}"
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
else:
    raise SystemExit(f"unsupported field: {field}")
PY
}

run_method() {
    local baseline_id="$1"
    local gpu_set="$2"
    local ad_cuda="$3"
    local planner_vlm_device="$4"

    if ! method_enabled "${baseline_id}"; then
        return 0
    fi

    local planner_name planner_path base_path log_path
    planner_name="$(read_registry_field "${baseline_id}" planner_name)"
    planner_path="$(read_registry_field "${baseline_id}" planner_path)"
    base_path="$(read_registry_field "${baseline_id}" base_path)"
    log_path="${SLURM_LOG_DIR}/${baseline_id}.log"

    echo "[${baseline_id}] start gpu_set=${gpu_set} planner=${planner_name} scenario=${SMOKE_SCENARIO_PATH}"
    CUDA_VISIBLE_DEVICES="${gpu_set}" \
    REPO_ROOT="${REPO_ROOT}" \
    DATA_TYPE="${DATASET}" \
    BASELINE_ID="${baseline_id}" \
    BASELINE_SUITE="${SUITE}" \
    BASELINE_RUN_TYPE="debug" \
    BASELINE_RUN_VARIANT="${SMOKE_RUN_ID}" \
    PLANNER_NAME="${planner_name}" \
    PLANNER_PATH="${planner_path}" \
    SCENARIO_PATH="${SMOKE_SCENARIO_PATH}" \
    BASE_PATH="${base_path}" \
    HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN}" \
    SIM_CUDA="inherit" \
    AD_CUDA="${ad_cuda}" \
    SETUP_VLM_ENV=0 \
    PLANNER_VLM_PRELOAD_LOCK_PATH="${VLM_PRELOAD_LOCK_PATH}" \
    HUGSIM_PLAN_RESPONSE_TIMEOUT_SEC="${SMOKE_PLAN_RESPONSE_TIMEOUT_SEC}" \
    RAP_DEVICE_OVERRIDE="cuda:0" \
    DRIVOR_DEVICE_OVERRIDE="cuda:0" \
    PLANNER_VLM_DEVICE_OVERRIDE="${planner_vlm_device}" \
    bash scripts/baselines/common/run_single_nuscenes_scene.sh "${planner_name}" "${SMOKE_SCENARIO_PATH}" inherit "${ad_cuda}" \
        > "${log_path}" 2>&1
    echo "[${baseline_id}] done"
}

wait_wave() {
    local status=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    return "${status}"
}

echo "dataset=${DATASET}"
echo "suite=${SUITE}"
echo "scenario=${SMOKE_SCENARIO_PATH}"
echo "smoke_run_id=${SMOKE_RUN_ID}"
echo "methods=${SMOKE_METHODS}"
echo "gpu_budget=6"
echo "vlm_preload_lock_path=${VLM_PRELOAD_LOCK_PATH}"

status=0

pids=()
run_method "rap_vlm" "0,1" "inherit" "cuda:1" & pids+=("$!")
run_method "rap_intervention_4cam" "2,3" "inherit" "cuda:1" & pids+=("$!")
run_method "drivor_vlm" "4" "inherit" "cuda:0" & pids+=("$!")
run_method "rule_based" "5" "-1" "cuda:0" & pids+=("$!")
if ! wait_wave "${pids[@]}"; then status=1; fi

pids=()
run_method "drivor_intervention_4cam" "0" "inherit" "cuda:0" & pids+=("$!")
run_method "drivor_impl_a" "1" "inherit" "cuda:0" & pids+=("$!")
run_method "drivor_impl_b" "2" "inherit" "cuda:0" & pids+=("$!")
run_method "rap_impl_a" "3,4" "inherit" "cuda:1" & pids+=("$!")
if ! wait_wave "${pids[@]}"; then status=1; fi

pids=()
run_method "rap_impl_b" "0,1" "inherit" "cuda:1" & pids+=("$!")
if ! wait_wave "${pids[@]}"; then status=1; fi

"${HUGSIM_PYTHON_BIN}" scripts/baselines/smoke/check_method_smoke.py \
    --dataset "${DATASET}" \
    --suite "${SUITE}" \
    --run_id "${SMOKE_RUN_ID}" \
    --scenario_path "${SMOKE_SCENARIO_PATH}" \
    --methods "${SMOKE_METHODS}" || status=1

exit "${status}"
