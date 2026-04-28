#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 <planner:{rap|rap_vlm|drivor|drivor_vlm}> <scenario_dir> [partition] [max_jobs]" >&2
    exit 2
fi

PLANNER_NAME="${1:?missing planner name}"
SCENARIO_DIR="${2:?missing scenario directory}"
PARTITION="${3:-gpu02}"
MAX_JOBS="${4:-0}"
PLANNER_PATH_ENV="${PLANNER_PATH:-}"
BASE_PATH_ENV="${BASE_PATH:-}"
JOB_LABEL="${SUBMIT_JOB_LABEL:-}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
RUN_PREFIX="${SUBMIT_RUN_PREFIX:-hugsim}"
GPU_BUDGET="${GPU_BUDGET:-0}"

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
GPUS="${GPUS_PER_JOB:-${GPUS}}"

if [[ -z "${JOB_LABEL}" ]]; then
    if [[ -n "${PLANNER_PATH_ENV}" ]]; then
        JOB_LABEL="$(basename "${PLANNER_PATH_ENV}")"
        JOB_LABEL="${JOB_LABEL%.yaml}"
    else
        JOB_LABEL="${PLANNER_NAME}"
    fi
fi
JOB_LABEL="$(printf '%s' "${JOB_LABEL}" | tr '/:[:space:]' '____')"
JOB_NAME="${RUN_PREFIX}-${JOB_LABEL}"

if [[ -z "${BASE_PATH_ENV}" && "${PLANNER_PATH_ENV}" == *"rap_vlm_default_trajectory_config.yaml" ]]; then
    BASE_PATH_ENV="configs/sim/nuscenes_base_local_rap_vlm_default_trajectory_config.yaml"
fi

if [[ -z "${BASE_PATH_ENV}" && "${PLANNER_PATH_ENV}" == *"drivor_vlm"* ]]; then
    BASE_PATH_ENV="configs/sim/nuscenes_base_local_drivor.yaml"
fi

if [[ -n "${PLANNER_PATH_ENV}" && ! -f "${PLANNER_PATH_ENV}" ]]; then
    echo "missing planner config: ${PLANNER_PATH_ENV}" >&2
    exit 1
fi

if [[ -n "${BASE_PATH_ENV}" && ! -f "${BASE_PATH_ENV}" ]]; then
    echo "missing base config: ${BASE_PATH_ENV}" >&2
    exit 1
fi

if [[ ! -x "${HUGSIM_PYTHON_BIN}" ]]; then
    echo "missing HUGSIM python: ${HUGSIM_PYTHON_BIN}" >&2
    exit 1
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
if [[ "${GPU_BUDGET}" -gt 0 ]]; then
    echo "gpu_budget=${GPU_BUDGET}"
fi
if [[ -n "${PLANNER_PATH_ENV}" ]]; then
    echo "planner_path=${PLANNER_PATH_ENV}"
fi
if [[ -n "${BASE_PATH_ENV}" ]]; then
    echo "base_path=${BASE_PATH_ENV}"
fi

submitted=0
skipped=0

current_gpu_demand() {
    squeue -u "${USER}" -h -o '%j|%b' | awk -F'|' -v prefix="${RUN_PREFIX}-" '
        index($1, prefix) != 1 { next }
        {
            gpu = 0
            n = split($2, parts, ",")
            for (i = 1; i <= n; i++) {
                if (parts[i] ~ /gpu/) {
                    m = split(parts[i], segs, ":")
                    val = segs[m]
                    if (val ~ /^[0-9]+$/) {
                        gpu += val + 0
                    } else if (parts[i] ~ /gpu$/) {
                        gpu += 1
                    }
                }
            }
            total += gpu
        }
        END { print total + 0 }
    '
}

for scenario in "${SCENARIOS[@]}"; do
    if [[ "${MAX_JOBS}" -gt 0 || "${GPU_BUDGET}" -gt 0 ]]; then
        while true; do
            max_jobs_ok=1
            gpu_budget_ok=1

            if [[ "${MAX_JOBS}" -gt 0 ]]; then
                running="$(squeue -u "${USER}" -h -o '%j' | grep -c "^${JOB_NAME}\$" || true)"
                if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
                    max_jobs_ok=0
                fi
            fi

            if [[ "${GPU_BUDGET}" -gt 0 ]]; then
                current_gpu="$(current_gpu_demand)"
                if (( current_gpu + GPUS > GPU_BUDGET )); then
                    gpu_budget_ok=0
                fi
            fi

            if [[ "${max_jobs_ok}" -eq 1 && "${gpu_budget_ok}" -eq 1 ]]; then
                break
            fi
            sleep 10
        done
    fi

    resolve_args=(
        scripts/resolve_output_path.py
        --planner_name "${PLANNER_NAME}"
        --scenario_path "${scenario}"
        --base_path "${BASE_PATH_ENV}"
    )
    if [[ -n "${PLANNER_PATH_ENV}" ]]; then
        resolve_args+=(--planner_path "${PLANNER_PATH_ENV}")
    fi
    output_path="$("${HUGSIM_PYTHON_BIN}" "${resolve_args[@]}")"
    if [[ -d "${output_path}" ]] && find "${output_path}" -mindepth 1 -print -quit | grep -q .; then
        echo "skipping existing ${scenario} output=${output_path}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "submitting ${scenario} output=${output_path}"
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
echo "skipped=${skipped}"
