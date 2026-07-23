#!/bin/bash
#SBATCH --job-name=uncertainty-ab
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM-uncertainty-port}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_ab30}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0522_full.yaml}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
GPU_WORKERS="${GPU_WORKERS:-4}"
BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE:-canonical}"

# AB_CONFIG selects which planner YAML to use and dictates the run variant tag.
# Set by the submit wrapper: baseline | critic_only | full
AB_CONFIG="${AB_CONFIG:?AB_CONFIG must be set (baseline|critic_only|full)}"

case "${AB_CONFIG}" in
    baseline)
        PLANNER_PATH_REL="configs/planners/autoagent0/ab_test/drivor_ab_baseline.yaml"
        RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-ab-$(date +%Y%m%d)-baseline}"
        ;;
    critic_only)
        PLANNER_PATH_REL="configs/planners/autoagent0/ab_test/drivor_ab_critic_only.yaml"
        RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-ab-$(date +%Y%m%d)-critic_only}"
        ;;
    full)
        PLANNER_PATH_REL="configs/planners/autoagent0/ab_test/drivor_ab_full.yaml"
        RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-ab-$(date +%Y%m%d)-full}"
        ;;
    *)
        echo "unsupported AB_CONFIG=${AB_CONFIG}" >&2
        exit 2
        ;;
esac

FAMILY="drivor"
BASELINE_ID="drivor_autoagent0"
PLANNER_NAME="drivor_vlm"
VLM_PRELOAD_LOCK_PATH="${VLM_PRELOAD_LOCK_PATH:-/bigdata/aidan/outputs/slurm/ab_test/vlm_preload_${AB_CONFIG}.lock}"
PLAN_RESPONSE_TIMEOUT_SEC="${PLAN_RESPONSE_TIMEOUT_SEC:-600}"

mkdir -p /bigdata/aidan/outputs/slurm
mkdir -p /bigdata/aidan/outputs/slurm/ab_test
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

mapfile -t SCENES < <(find -L "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
    echo "no scenarios found under ${SCENARIO_DIR}" >&2
    exit 1
fi

GPU_WORKERS="$(( GPU_WORKERS < 1 ? 1 : GPU_WORKERS ))"

echo "job_kind=uncertainty_ab"
echo "ab_config=${AB_CONFIG}"
echo "run_variant=${RUN_VARIANT_TAG}"
echo "planner_path=${PLANNER_PATH_REL}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scene_count=${#SCENES[@]}"
echo "gpu_workers=${GPU_WORKERS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "vlm_preload_lock_path=${VLM_PRELOAD_LOCK_PATH}"
mkdir -p "$(dirname "${VLM_PRELOAD_LOCK_PATH}")"

WORKER_LOG_DIR="/bigdata/aidan/outputs/slurm/ab_test/${AB_CONFIG}"
mkdir -p "${WORKER_LOG_DIR}"

run_worker() {
    local worker_id="$1"
    local gpu_id="$2"
    for idx in "${!SCENES[@]}"; do
        if (( idx % GPU_WORKERS != worker_id )); then
            continue
        fi
        local scene="${SCENES[$idx]}"
        local output_path
        output_path="$("${HUGSIM_PYTHON_BIN}" scripts/baselines/common/resolve_output_path.py \
            --baseline_id "${BASELINE_ID}" \
            --dataset nuscenes \
            --suite full \
            --run_type "${BASELINE_RUN_TYPE}" \
            --run_variant "${RUN_VARIANT_TAG}" \
            --scenario_path "${scene}")"

        if [[ -f "${output_path}/eval.json" ]]; then
            echo "[${AB_CONFIG} worker ${worker_id}] skip complete ${scene}"
            continue
        fi

        local scene_slug
        scene_slug="$(basename "${scene}" .yaml)"
        local scene_log="${WORKER_LOG_DIR}/${scene_slug}.worker${worker_id}.log"

        echo "[${AB_CONFIG} worker ${worker_id} gpu ${gpu_id}] run ${scene}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        REPO_ROOT="${REPO_ROOT}" \
        BASELINE_ID="${BASELINE_ID}" \
        BASELINE_SUITE="full" \
        BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE}" \
        BASELINE_RUN_VARIANT="${RUN_VARIANT_TAG}" \
        PLANNER_NAME="${PLANNER_NAME}" \
        PLANNER_PATH="${PLANNER_PATH_REL}" \
        SCENARIO_PATH="${scene}" \
        BASE_PATH="${BASE_PATH}" \
        HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN}" \
        SIM_CUDA="inherit" \
        AD_CUDA="inherit" \
        SETUP_VLM_ENV=0 \
        PLANNER_VLM_PRELOAD_LOCK_PATH="${VLM_PRELOAD_LOCK_PATH}" \
        HUGSIM_PLAN_RESPONSE_TIMEOUT_SEC="${PLAN_RESPONSE_TIMEOUT_SEC}" \
        DRIVOR_DEVICE_OVERRIDE="cuda:0" \
        RAP_DEVICE_OVERRIDE="cuda:0" \
        PLANNER_VLM_DEVICE_OVERRIDE="cuda:0" \
        if ! bash scripts/baselines/common/run_single_nuscenes_scene.sh "${PLANNER_NAME}" "${scene}" inherit inherit \
                > "${scene_log}" 2>&1; then
            echo "[${AB_CONFIG} worker ${worker_id}] FAILED ${scene} (see ${scene_log})"
        fi
    done
}

phase_status=0
pids=()
for worker_id in $(seq 0 $((GPU_WORKERS - 1))); do
    run_worker "${worker_id}" "${worker_id}" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        phase_status=1
    fi
done

echo "done ab_config=${AB_CONFIG} run_variant=${RUN_VARIANT_TAG} status=${phase_status}"
exit "${phase_status}"
