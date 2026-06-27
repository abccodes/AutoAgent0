#!/bin/bash
#SBATCH --job-name=uncertainty-calibration
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0522_full.yaml}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
MAX_SCENES="${MAX_SCENES:-0}"
GPU_WORKERS="${GPU_WORKERS:-6}"
BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE:-canonical}"
PLANNER_FAMILY="${PLANNER_FAMILY:-drivor}"
RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-calibration-$(date +%Y%m%d)}"
DRIVOR_PLANNER_PATH="${DRIVOR_PLANNER_PATH:-configs/planners/autoagent0/drivor_autoagent0_recovery_loop_0605.yaml}"
RAP_PLANNER_PATH="${RAP_PLANNER_PATH:-configs/planners/autoagent0/rap_autoagent0_recovery_loop_0605.yaml}"
CALIBRATION_HORIZON_STEPS="${CALIBRATION_HORIZON_STEPS:-20}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
VLM_PRELOAD_LOCK_PATH="${VLM_PRELOAD_LOCK_PATH:-/bigdata/aidan/outputs/slurm/calibration/vlm_preload.lock}"
PLAN_RESPONSE_TIMEOUT_SEC="${PLAN_RESPONSE_TIMEOUT_SEC:-600}"

case "${PLANNER_FAMILY}" in
    drivor) FAMILIES=("drivor") ;;
    rap)    FAMILIES=("rap") ;;
    both)   FAMILIES=("drivor" "rap") ;;
    *)
        echo "unsupported PLANNER_FAMILY=${PLANNER_FAMILY}" >&2
        exit 2
        ;;
esac

mkdir -p /bigdata/aidan/outputs/slurm
mkdir -p /bigdata/aidan/outputs/slurm/calibration
mkdir -p /bigdata/aidan/.cache /bigdata/aidan/.cache/matplotlib /bigdata/aidan/.home

export HOME="/bigdata/aidan/.home"
export XDG_CACHE_HOME="/bigdata/aidan/.cache"
export MPLCONFIGDIR="/bigdata/aidan/.cache/matplotlib"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

mapfile -t SCENES < <(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
    echo "no scenarios found under ${SCENARIO_DIR}" >&2
    exit 1
fi
if [[ "${MAX_SCENES}" -gt 0 ]]; then
    SCENES=("${SCENES[@]:0:${MAX_SCENES}}")
fi

GPU_WORKERS="$(( GPU_WORKERS < 1 ? 1 : GPU_WORKERS ))"

echo "job_kind=uncertainty_calibration"
echo "planner_family=${PLANNER_FAMILY}"
echo "run_variant=${RUN_VARIANT_TAG}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scene_count=${#SCENES[@]}"
echo "base_path=${BASE_PATH}"
echo "gpu_workers=${GPU_WORKERS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "calibration_horizon_steps=${CALIBRATION_HORIZON_STEPS}"
echo "vlm_preload_lock_path=${VLM_PRELOAD_LOCK_PATH}"
echo "plan_response_timeout_sec=${PLAN_RESPONSE_TIMEOUT_SEC}"
mkdir -p "$(dirname "${VLM_PRELOAD_LOCK_PATH}")"

run_worker() {
    local worker_id="$1"
    local gpu_id="$2"
    local family="$3"
    local baseline_id="${family}_autoagent0"
    local planner_name
    local planner_path

    case "${family}" in
        drivor)
            planner_name="drivor_vlm"
            planner_path="${DRIVOR_PLANNER_PATH}"
            ;;
        rap)
            planner_name="rap_vlm"
            planner_path="${RAP_PLANNER_PATH}"
            ;;
        *)
            echo "[worker ${worker_id}] unsupported family=${family}" >&2
            return 1
            ;;
    esac

    local worker_log_dir="/bigdata/aidan/outputs/slurm/calibration/${baseline_id}"
    mkdir -p "${worker_log_dir}"

    for idx in "${!SCENES[@]}"; do
        if (( idx % GPU_WORKERS != worker_id )); then
            continue
        fi

        local scene="${SCENES[$idx]}"
        local output_path
        output_path="$("${HUGSIM_PYTHON_BIN}" scripts/baselines/common/resolve_output_path.py \
            --baseline_id "${baseline_id}" \
            --dataset nuscenes \
            --suite full \
            --run_type "${BASELINE_RUN_TYPE}" \
            --run_variant "${RUN_VARIANT_TAG}" \
            --scenario_path "${scene}")"

        if [[ -f "${output_path}/eval.json" ]]; then
            echo "[${baseline_id} worker ${worker_id}] skip complete ${scene}"
            continue
        fi

        local scene_slug
        scene_slug="$(basename "${scene}" .yaml)"
        local scene_log="${worker_log_dir}/${scene_slug}.worker${worker_id}.log"

        echo "[${baseline_id} worker ${worker_id} gpu ${gpu_id}] run ${scene}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        REPO_ROOT="${REPO_ROOT}" \
        BASELINE_ID="${baseline_id}" \
        BASELINE_SUITE="full" \
        BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE}" \
        BASELINE_RUN_VARIANT="${RUN_VARIANT_TAG}" \
        PLANNER_NAME="${planner_name}" \
        PLANNER_PATH="${planner_path}" \
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
        bash scripts/baselines/common/run_single_nuscenes_scene.sh "${planner_name}" "${scene}" inherit inherit \
            > "${scene_log}" 2>&1
    done
}

phase_a_status=0
for family in "${FAMILIES[@]}"; do
    echo "=== Phase A: collecting corpus for family=${family} ==="
    pids=()
    for worker_id in $(seq 0 $((GPU_WORKERS - 1))); do
        run_worker "${worker_id}" "${worker_id}" "${family}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            phase_a_status=1
        fi
    done
done

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
    echo "SKIP_ANALYSIS=1 set; exiting after Phase A with status=${phase_a_status}"
    exit "${phase_a_status}"
fi

echo "=== Phase B: calibration analysis ==="
analysis_status=0
for family in "${FAMILIES[@]}"; do
    baseline_id="${family}_autoagent0"
    run_root="/bigdata/aidan/outputs/benchmark/out/baselines/${baseline_id}/nuscenes/full/${RUN_VARIANT_TAG}"
    out_dir="${run_root}/_calibration"
    mkdir -p "${out_dir}"
    echo "[analysis ${family}] runs=${run_root} out=${out_dir}"
    if ! "${HUGSIM_PYTHON_BIN}" scripts/calibrate_uncertainty.py \
            --runs "${run_root}" \
            --horizon-steps "${CALIBRATION_HORIZON_STEPS}" \
            --out-dir "${out_dir}" \
            2>&1 | tee "${out_dir}/analysis.log"; then
        echo "[analysis ${family}] analysis script returned non-zero; corpus preserved"
        analysis_status=1
    fi
done

if [[ "${phase_a_status}" -ne 0 ]]; then
    echo "Phase A had failing workers; exiting with status=${phase_a_status}"
    exit "${phase_a_status}"
fi
exit "${analysis_status}"
