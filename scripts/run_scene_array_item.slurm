#!/bin/bash
#SBATCH --job-name=hugsim-array-item
#SBATCH --output=/bigdata/aidan/outputs/slurm/%x-%A_%a.out
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM}"
SCENES_FILE="${SCENES_FILE:?missing SCENES_FILE}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

if [[ ! -f "${SCENES_FILE}" ]]; then
    echo "missing scenes file: ${SCENES_FILE}" >&2
    exit 1
fi

SCENARIO_PATH="$(sed -n "$((TASK_ID + 1))p" "${SCENES_FILE}")"
if [[ -z "${SCENARIO_PATH}" ]]; then
    echo "no scenario for task_id=${TASK_ID} in ${SCENES_FILE}" >&2
    exit 1
fi

export SCENARIO_PATH

cd "${REPO_ROOT}"
bash scripts/run_single_scene.slurm
