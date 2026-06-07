#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
PARTITION="${PARTITION:-gpu02}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local_drivor_debug.yaml}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"

PLANNERS=(
    "configs/planners/drivor_vlm_intervention.yaml"
    "configs/planners/drivor_vlm_default_trajectory_intervention.yaml"
    "configs/planners/drivor_vlm.yaml"
    "configs/planners/drivor_vlm_default_trajectory_config.yaml"
)

SCENES=(
    "configs/benchmark/nuscenes_all_variants/scene-0010-easy-00.yaml"
    "configs/benchmark/nuscenes_all_variants/scene-0013-easy-00.yaml"
    "configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml"
)

mkdir -p /bigdata/aidan/outputs/slurm
cd "${REPO_ROOT}"

for planner_path in "${PLANNERS[@]}"; do
    planner_tag="$(basename "${planner_path}" .yaml)"
    for scene in "${SCENES[@]}"; do
        scene_tag="$(basename "${scene}" .yaml)"
        job_name="drv-${planner_tag}-${scene_tag}"
        log_path="/bigdata/aidan/outputs/slurm/${job_name}-%j.out"
        echo "submitting ${job_name}"
        sbatch \
            --partition="${PARTITION}" \
            --time="${SBATCH_TIME}" \
            --job-name="${job_name}" \
            --output="${log_path}" \
            --export=ALL,REPO_ROOT="${REPO_ROOT}",BASE_PATH="${BASE_PATH}",PLANNER_PATH="${planner_path}",SCENARIO_PATH="${scene}" \
            scripts/run_single_drivor_vlm_benchmark_debug.slurm
    done
done
