#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 <planner:{rap|rap_vlm}> <scenario_dir> [partition] [max_jobs]" >&2
    exit 2
fi

PLANNER_NAME="${1:?missing planner name}"
SCENARIO_DIR="${2:?missing scenario directory}"
PARTITION="${3:-gpu02}"
MAX_JOBS="${4:-0}"
PLANNER_PATH_ENV="${PLANNER_PATH:-}"
BASE_PATH_ENV="${BASE_PATH:-}"
JOB_LABEL="${SUBMIT_JOB_LABEL:-}"

case "${PLANNER_NAME}" in
    rap|drivor)
        GPUS=1
        ;;
    rap_vlm|drivor_vlm)
        GPUS=2
        ;;
    *)
        echo "unsupported planner: ${PLANNER_NAME}" >&2
        exit 2
        ;;
esac

if [[ -z "${JOB_LABEL}" ]]; then
    if [[ -n "${PLANNER_PATH_ENV}" ]]; then
        JOB_LABEL="$(basename "${PLANNER_PATH_ENV}")"
        JOB_LABEL="${JOB_LABEL%.yaml}"
    else
        JOB_LABEL="${PLANNER_NAME}"
    fi
fi
JOB_LABEL="$(printf '%s' "${JOB_LABEL}" | tr '/:[:space:]' '____')"
JOB_NAME="hugsim-${JOB_LABEL}"

if [[ -z "${BASE_PATH_ENV}" && "${PLANNER_PATH_ENV}" == *"rap_vlm_default_trajectory_config.yaml" ]]; then
    BASE_PATH_ENV="configs/sim/nuscenes_base_local_rap_vlm_default_trajectory_config.yaml"
fi

if [[ -z "${BASE_PATH_ENV}" && "${PLANNER_PATH_ENV}" == *"drivor_vlm"* ]]; then
    BASE_PATH_ENV="configs/sim/nuscenes_base_local_drivor_vlm.yaml"
fi

mapfile -t SCENARIOS < <(find "${SCENARIO_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
    echo "no scenario YAMLs found under ${SCENARIO_DIR}" >&2
    exit 1
fi

echo "planner=${PLANNER_NAME}"
echo "partition=${PARTITION}"
echo "gpus_per_job=${GPUS}"
echo "scenario_count=${#SCENARIOS[@]}"
echo "scenario_dir=${SCENARIO_DIR}"
echo "job_name=${JOB_NAME}"
if [[ -n "${PLANNER_PATH_ENV}" ]]; then
    echo "planner_path=${PLANNER_PATH_ENV}"
fi
if [[ -n "${BASE_PATH_ENV}" ]]; then
    echo "base_path=${BASE_PATH_ENV}"
fi

submitted=0
for scenario in "${SCENARIOS[@]}"; do
    if [[ "${MAX_JOBS}" -gt 0 ]]; then
        while true; do
            running="$(squeue -u "${USER}" -h -o '%j' | grep -c "^${JOB_NAME}\$" || true)"
            if [[ "${running}" -lt "${MAX_JOBS}" ]]; then
                break
            fi
            sleep 10
        done
    fi

    echo "submitting ${scenario}"
    export_vars="ALL,PLANNER_NAME=${PLANNER_NAME},SETUP_VLM_ENV=0,SCENARIO_PATH=${scenario}"
    if [[ -n "${PLANNER_PATH_ENV}" ]]; then
        export_vars="${export_vars},PLANNER_PATH=${PLANNER_PATH_ENV}"
    fi
    if [[ -n "${BASE_PATH_ENV}" ]]; then
        export_vars="${export_vars},BASE_PATH=${BASE_PATH_ENV}"
    fi
    sbatch \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --gres="gpu:${GPUS}" \
        --export="${export_vars}" \
        scripts/run_single_scene.slurm
    submitted=$((submitted + 1))
done

echo "submitted=${submitted}"
