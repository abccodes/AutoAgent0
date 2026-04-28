#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/scripts/run_qwen3_vl_8b_gpu02_10sample_compare_debug.slurm}"

CONFIGS=(
    "${REPO_ROOT}/configs/planners/rap_vlm_intervention_compare_gate_multiview_score_front_only.yaml"
    "${REPO_ROOT}/configs/planners/rap_vlm_intervention_compare_gate_front_only_score_front_only.yaml"
)

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "launcher=${LAUNCHER}"
echo "submitting ${#CONFIGS[@]} gate-vs-score camera test jobs"

for planner_path in "${CONFIGS[@]}"; do
    echo "submitting planner_path=${planner_path}"
    sbatch --export=ALL,PLANNER_PATH="${planner_path}" "${LAUNCHER}"
done

echo "expected output root=/bigdata/aidan/outputs/benchmark/out/debug/qwen3-vl-8b-instruct_nusc_camera_test"
