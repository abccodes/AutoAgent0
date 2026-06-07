#!/bin/bash
#SBATCH --job-name=hugsim-full-baselines
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%j.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0522_full.yaml}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
MAX_SCENES="${MAX_SCENES:-0}"
METHOD_FILTER="${METHOD_FILTER:-all}"
BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE:-canonical}"

mkdir -p /bigdata/aidan/outputs/slurm
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

run_method() {
    local baseline_id="$1"
    local planner_name="$2"
    local planner_path="$3"
    local gpu_set="$4"
    local ad_cuda="$5"
    local planner_vlm_device="$6"

    local method_log_dir="/bigdata/aidan/outputs/slurm/full_baselines/${baseline_id}"
    mkdir -p "${method_log_dir}"

    for scene in "${SCENES[@]}"; do
        local output_path
        output_path="$("${HUGSIM_PYTHON_BIN}" scripts/baselines/common/resolve_output_path.py \
            --baseline_id "${baseline_id}" \
            --dataset nuscenes \
            --suite full \
            --run_type "${BASELINE_RUN_TYPE}" \
            --scenario_path "${scene}")"

        if [[ -f "${output_path}/eval.json" ]]; then
            echo "[${baseline_id}] skip complete ${scene}"
            continue
        fi

        local scene_slug
        scene_slug="$(basename "${scene}" .yaml)"
        local scene_log="${method_log_dir}/${scene_slug}.log"

        CUDA_VISIBLE_DEVICES="${gpu_set}" \
        REPO_ROOT="${REPO_ROOT}" \
        BASELINE_ID="${baseline_id}" \
        BASELINE_SUITE="full" \
        BASELINE_RUN_TYPE="${BASELINE_RUN_TYPE}" \
        PLANNER_NAME="${planner_name}" \
        PLANNER_PATH="${planner_path}" \
        SCENARIO_PATH="${scene}" \
        BASE_PATH="${BASE_PATH}" \
        HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN}" \
        SIM_CUDA="inherit" \
        AD_CUDA="${ad_cuda}" \
        SETUP_VLM_ENV=0 \
        RAP_DEVICE_OVERRIDE="cuda:0" \
        DRIVOR_DEVICE_OVERRIDE="cuda:0" \
        PLANNER_VLM_DEVICE_OVERRIDE="${planner_vlm_device}" \
        bash scripts/baselines/common/run_single_nuscenes_scene.sh "${planner_name}" "${scene}" inherit "${ad_cuda}" \
            > "${scene_log}" 2>&1
    done
}

should_run() {
    local method_name="$1"
    case "${METHOD_FILTER}" in
        all) return 0 ;;
        rap_only) [[ "${method_name}" == rap_* ]] ;;
        drivor_only) [[ "${method_name}" == drivor_* ]] ;;
        *) [[ "${method_name}" == "${METHOD_FILTER}" ]] ;;
    esac
}

drivor_a_gpus="0"
drivor_b_gpus="1"
rap_a_gpus="2,3"
rap_b_gpus="4,5"

case "${METHOD_FILTER}" in
    rap_only)
        rap_a_gpus="0,1"
        rap_b_gpus="2,3"
        ;;
    drivor_only)
        drivor_a_gpus="0"
        drivor_b_gpus="1"
        ;;
    rap_impl_a)
        rap_a_gpus="0,1"
        ;;
    rap_impl_b)
        rap_b_gpus="0,1"
        ;;
    drivor_impl_a)
        drivor_a_gpus="0"
        ;;
    drivor_impl_b)
        drivor_b_gpus="0"
        ;;
esac

pids=()
names=()

if should_run "drivor_impl_a"; then
    run_method "drivor_impl_a" "drivor_vlm" "configs/planners/choice_a_rule_merge/drivor_vlm_intervention_4cam_rule_merge_0522.yaml" "${drivor_a_gpus}" "inherit" "cuda:0" &
    pids+=("$!")
    names+=("drivor_impl_a")
fi
if should_run "drivor_impl_b"; then
    run_method "drivor_impl_b" "drivor_vlm" "configs/planners/choice_b_rule_gate/drivor_vlm_intervention_4cam_rule_gate_0522.yaml" "${drivor_b_gpus}" "inherit" "cuda:0" &
    pids+=("$!")
    names+=("drivor_impl_b")
fi
if should_run "rap_impl_a"; then
    run_method "rap_impl_a" "rap_vlm" "configs/planners/choice_a_rule_merge/rap_vlm_intervention_4cam_rule_merge_0522.yaml" "${rap_a_gpus}" "inherit" "cuda:1" &
    pids+=("$!")
    names+=("rap_impl_a")
fi
if should_run "rap_impl_b"; then
    run_method "rap_impl_b" "rap_vlm" "configs/planners/choice_b_rule_gate/rap_vlm_intervention_4cam_rule_gate_0522.yaml" "${rap_b_gpus}" "inherit" "cuda:1" &
    pids+=("$!")
    names+=("rap_impl_b")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "no methods selected for METHOD_FILTER=${METHOD_FILTER}" >&2
    exit 1
fi

status=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "[${names[$idx]}] FAILED"
        status=1
    fi
done

exit "${status}"
