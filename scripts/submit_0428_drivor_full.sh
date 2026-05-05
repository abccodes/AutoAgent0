#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0428_drivor.yaml}"
LAUNCHER_1GPU="${LAUNCHER_1GPU:-${REPO_ROOT}/scripts/run_0428_chunk_1gpu.slurm}"
LAUNCHER_2GPU="${LAUNCHER_2GPU:-${REPO_ROOT}/scripts/run_0428_chunk_2gpu.slurm}"
SBATCH_WAIT="${SBATCH_WAIT:-1}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-12}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-6}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
BASE_MAX_PARALLEL_SCENES="${BASE_MAX_PARALLEL_SCENES:-6}"
VLM_MAX_PARALLEL_SCENES="${VLM_MAX_PARALLEL_SCENES:-3}"

declare -a PLANNER_NAMES=(
    "drivor"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
    "drivor_vlm"
)

declare -a PLANNER_PATHS=(
    "configs/planners/drivor.yaml"
    "configs/planners/drivor_vlm_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_0428.yaml"
    "configs/planners/drivor_vlm_intervention_1cam_0428.yaml"
    "configs/planners/drivor_vlm_intervention_4cam_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention_1cam_0428.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention_4cam_0428.yaml"
)

declare -a JOB_LABELS=(
    "drivor"
    "drivor_vlm"
    "drivor_vlm_default_trajectory"
    "drivor_intervention_vlm_1_cam"
    "drivor_intervention_vlm_4_cam"
    "drivor_intervention_vlm_default_trajectories_1_cam"
    "drivor_intervention_vlm_default_trajectories_4_cam"
)

declare -a GPUS_PER_JOB=(
    "1"
    "2"
    "2"
    "2"
    "2"
    "2"
    "2"
)

mapfile -t SCENARIOS < <(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
    echo "no scenario YAMLs found under ${SCENARIO_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "partition=${PARTITION}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scene_count=${#SCENARIOS[@]}"
echo "sbatch_wait=${SBATCH_WAIT}"
echo "base_batch_size=${BASE_BATCH_SIZE}"
echo "vlm_batch_size=${VLM_BATCH_SIZE}"

resolve_pending_scenes() {
    local planner_name="$1"
    local planner_path="$2"
    local recompute_all="$3"
    local scenario
    for scenario in "${SCENARIOS[@]}"; do
        local output_path
        output_path="$("${HUGSIM_PYTHON_BIN}" scripts/resolve_output_path.py \
            --planner_name "${planner_name}" \
            --scenario_path "${scenario}" \
            --base_path "${BASE_PATH}" \
            --planner_path "${planner_path}")"
        if [[ "${recompute_all}" == "1" || ! -f "${output_path}/eval.json" ]]; then
            printf '%s\n' "${scenario}"
        fi
    done
}

for idx in "${!PLANNER_NAMES[@]}"; do
    planner_name="${PLANNER_NAMES[$idx]}"
    planner_path="${PLANNER_PATHS[$idx]}"
    job_label="${JOB_LABELS[$idx]}"
    gpus="${GPUS_PER_JOB[$idx]}"
    launcher="${LAUNCHER_1GPU}"
    batch_size="${BASE_BATCH_SIZE}"
    max_parallel_scenes="${BASE_MAX_PARALLEL_SCENES}"
    recompute_all="0"
    if [[ "${gpus}" == "2" ]]; then
        launcher="${LAUNCHER_2GPU}"
        batch_size="${VLM_BATCH_SIZE}"
        max_parallel_scenes="${VLM_MAX_PARALLEL_SCENES}"
    fi
    if [[ "${job_label}" == "drivor_vlm_default_trajectory" ]]; then
        recompute_all="1"
    fi

    mapfile -t pending_scenes < <(resolve_pending_scenes "${planner_name}" "${planner_path}" "${recompute_all}")
    pending_count="${#pending_scenes[@]}"
    echo "submitting planner=${job_label} gpus_per_job=${gpus} pending_scenes=${pending_count} batch_size=${batch_size}"
    if (( pending_count == 0 )); then
        continue
    fi

    batch_index=0
    for ((start=0; start<pending_count; start+=batch_size)); do
        batch_index=$((batch_index + 1))
        scene_file="$(mktemp "/tmp/${job_label}.XXXXXX.scenes")"
        printf '%s\n' "${pending_scenes[@]:start:batch_size}" > "${scene_file}"
        sbatch_args=(
            --partition="${PARTITION}"
            --job-name="hugsim-0428-${job_label}-b${batch_index}"
            --export=ALL,REPO_ROOT="${REPO_ROOT}",PLANNER_NAME="${planner_name}",PLANNER_PATH="${planner_path}",BASE_PATH="${BASE_PATH}",SCENES_FILE="${scene_file}",MAX_PARALLEL_SCENES="${max_parallel_scenes}"
        )
        if [[ "${SBATCH_WAIT}" == "1" ]]; then
            sbatch_args+=(--wait)
        fi
        sbatch "${sbatch_args[@]}" "${launcher}"
        if [[ "${SBATCH_WAIT}" == "1" ]]; then
            rm -f "${scene_file}"
        else
            echo "left scene list at ${scene_file} because sbatch_wait=0"
        fi
    done
done
