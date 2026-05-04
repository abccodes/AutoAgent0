#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
SCENARIO_DIR="${SCENARIO_DIR:-configs/benchmark/nuscenes_all_variants}"
SCENARIO_GLOB="${SCENARIO_GLOB:-*.yaml}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_0428_drivor.yaml}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
BASE_MAX_JOBS="${BASE_MAX_JOBS:-6}"
VLM_MAX_JOBS="${VLM_MAX_JOBS:-3}"

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

declare -a FORCE_RERUN_ALL=(
    "0"
    "0"
    "1"
    "0"
    "0"
    "0"
    "0"
)

mapfile -t SCENARIOS < <(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name "${SCENARIO_GLOB}" | sort)
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
    echo "no scenario YAMLs found under ${SCENARIO_DIR} matching ${SCENARIO_GLOB}" >&2
    exit 1
fi

mkdir -p /bigdata/aidan/outputs/slurm
cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "partition=${PARTITION}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "scenario_glob=${SCENARIO_GLOB}"
echo "scene_count=${#SCENARIOS[@]}"
echo "base_path=${BASE_PATH}"
echo "sbatch_time=${SBATCH_TIME}"
echo "base_max_jobs=${BASE_MAX_JOBS}"
echo "vlm_max_jobs=${VLM_MAX_JOBS}"

submitted=0
skipped=0

for idx in "${!PLANNER_NAMES[@]}"; do
    planner_name="${PLANNER_NAMES[$idx]}"
    planner_path="${PLANNER_PATHS[$idx]}"
    job_label="${JOB_LABELS[$idx]}"
    force_rerun_all="${FORCE_RERUN_ALL[$idx]}"

    if [[ "${job_label}" == "drivor" ]]; then
        gpus="1"
        max_jobs="${BASE_MAX_JOBS}"
    else
        gpus="2"
        max_jobs="${VLM_MAX_JOBS}"
    fi

    echo "preparing planner=${job_label} planner_name=${planner_name} gpus_per_job=${gpus} max_jobs=${max_jobs} force_rerun_all=${force_rerun_all}"

    pending_file="$(mktemp "/tmp/${job_label}.XXXXXX.scenes")"
    pending_count=0

    for scenario in "${SCENARIOS[@]}"; do
        scene_tag="$(basename "${scenario}" .yaml)"
        output_path="$("${HUGSIM_PYTHON_BIN}" scripts/resolve_output_path.py \
            --planner_name "${planner_name}" \
            --scenario_path "${scenario}" \
            --base_path "${BASE_PATH}" \
            --planner_path "${planner_path}")"

        if [[ "${force_rerun_all}" != "1" && -f "${output_path}/eval.json" ]]; then
            echo "skipping completed planner=${job_label} scene=${scene_tag} output=${output_path}"
            skipped=$((skipped + 1))
            continue
        fi

        echo "${scenario}" >> "${pending_file}"
        pending_count=$((pending_count + 1))
    done

    if [[ "${pending_count}" -eq 0 ]]; then
        rm -f "${pending_file}"
        echo "no pending scenes for planner=${job_label}"
        continue
    fi

    array_spec="0-$((pending_count - 1))%${max_jobs}"
    echo "submitting planner=${job_label} pending_scenes=${pending_count} array=${array_spec}"
    sbatch --wait \
        --partition="${PARTITION}" \
        --time="${SBATCH_TIME}" \
        --job-name="hugsim-${job_label}" \
        --output="/bigdata/aidan/outputs/slurm/hugsim-${job_label}-%A_%a.out" \
        --array="${array_spec}" \
        --gres="gpu:${gpus}" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",BASE_PATH="${BASE_PATH}",PLANNER_PATH="${planner_path}",PLANNER_NAME="${planner_name}",SCENES_FILE="${pending_file}" \
        scripts/run_scene_array_item.slurm
    rm -f "${pending_file}"
    submitted=$((submitted + pending_count))
done

echo "submitted=${submitted}"
echo "skipped=${skipped}"
