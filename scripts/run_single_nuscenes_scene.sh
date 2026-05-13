#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "usage: $0 <planner:{rap|rap_vlm|drivor|drivor_vlm|rule_based|rule_based_vlm}> [scenario_yaml] [sim_cuda|inherit] [ad_cuda|inherit]" >&2
    exit 2
fi

PLANNER_NAME="${1:?missing planner name}"
SCENARIO_PATH="${2:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}"
DEFAULT_CUDA_ID="0"
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]; then
    DEFAULT_CUDA_ID="inherit"
fi
SIM_CUDA="${3:-${SIM_CUDA:-${DEFAULT_CUDA_ID}}}"
AD_CUDA="${4:-${AD_CUDA:-${DEFAULT_CUDA_ID}}}"
BASE_PATH="${BASE_PATH:-configs/sim/nuscenes_base_local.yaml}"
CAMERA_PATH="${CAMERA_PATH:-configs/sim/nuscenes_camera.yaml}"
KINEMATIC_PATH="${KINEMATIC_PATH:-configs/sim/kinematic.yaml}"
PLANNER_PATH="${PLANNER_PATH:-configs/planners/${PLANNER_NAME}.yaml}"
HUGSIM_PYTHON_BIN="${HUGSIM_PYTHON_BIN:-/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python}"
INCLUDE_PRIVILEGED_PIPE="${INCLUDE_PRIVILEGED_PIPE:-}"

case "${PLANNER_NAME}" in
    drivor|drivor_vlm)
        AD_NAME="drivor"
        ;;
    rap|rap_vlm)
        AD_NAME="rap"
        ;;
    rule_based|rule_based_vlm)
        AD_NAME="rule_based"
        ;;
    *)
        echo "unsupported planner: ${PLANNER_NAME}" >&2
        exit 2
        ;;
esac

if [[ -z "${INCLUDE_PRIVILEGED_PIPE}" ]]; then
    case "${AD_NAME}" in
        rule_based)
            INCLUDE_PRIVILEGED_PIPE="true"
            ;;
        *)
            INCLUDE_PRIVILEGED_PIPE="false"
            ;;
    esac
fi

if [[ ! -f "${SCENARIO_PATH}" ]]; then
    echo "missing scenario config: ${SCENARIO_PATH}" >&2
    exit 1
fi

if [[ ! -f "${BASE_PATH}" ]]; then
    echo "missing base config: ${BASE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PLANNER_PATH}" ]]; then
    echo "missing planner config: ${PLANNER_PATH}" >&2
    exit 1
fi

if [[ ! -x "${HUGSIM_PYTHON_BIN}" ]]; then
    echo "missing HUGSIM python: ${HUGSIM_PYTHON_BIN}" >&2
    exit 1
fi

#comment out for mlcav comaptibility
# mkdir -p /bigdata/aidan/outputs/benchmark/out

export PYTHONPATH="${PWD}:${PWD}/sim${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

HUGSIM_ENV_ROOT="$(cd "$(dirname "${HUGSIM_PYTHON_BIN}")/.." && pwd)"
TORCH_LIB_DIR="${HUGSIM_ENV_ROOT}/lib/python3.11/site-packages/torch/lib"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${HUGSIM_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Debug: surface HUGSIM env torch lib and LD_LIBRARY_PATH for troubleshooting
echo "RUN DEBUG: HUGSIM_PYTHON_BIN=${HUGSIM_PYTHON_BIN}"
echo "RUN DEBUG: HUGSIM_ENV_ROOT=${HUGSIM_ENV_ROOT}"
echo "RUN DEBUG: TORCH_LIB_DIR=${TORCH_LIB_DIR} (exists=$( [[ -d \"${TORCH_LIB_DIR}\" ]] && echo true || echo false ))"
echo "RUN DEBUG: LD_LIBRARY_PATH after HUGSIM torch lib add=${LD_LIBRARY_PATH:-unset}"
echo "RUN DEBUG: PATH contains python? -> $(which python 2>/dev/null || echo none)"
echo "RUN DEBUG: HUGSIM_PYTHON_BIN executable? -> $( [[ -x \"${HUGSIM_PYTHON_BIN}\" ]] && echo yes || echo no )"

# Some CUDA wheel dependencies ship shared libraries outside torch/lib.
# Include those packaged NVIDIA library directories so tinycudann can resolve
# libnvrtc and related CUDA runtime libs on nodes without system CUDA paths.
#comment out for mlcav compatibility
# for extra_lib_dir in \
#     "${HUGSIM_ENV_ROOT}"/lib/python*/site-packages/nvidia/*/lib \
#     /bigdata/aidan/.home/envs/vlm/lib/python*/site-packages/nvidia/*/lib
# do
#     if [[ -d "${extra_lib_dir}" ]]; then
#         export LD_LIBRARY_PATH="${extra_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
#     fi
# done

echo "planner=${PLANNER_NAME}"
echo "ad=${AD_NAME}"
echo "scenario=${SCENARIO_PATH}"
echo "base=${BASE_PATH}"
echo "planner_config=${PLANNER_PATH}"
echo "hugsim_python=${HUGSIM_PYTHON_BIN}"
echo "sim_cuda=${SIM_CUDA} ad_cuda=${AD_CUDA}"
echo "include_privileged_pipe=${INCLUDE_PRIVILEGED_PIPE}"
echo "ld_library_path=${LD_LIBRARY_PATH:-unset}"

CLOSED_LOOP_EXTRA_ARGS=()
case "$(printf '%s' "${INCLUDE_PRIVILEGED_PIPE}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
        CLOSED_LOOP_EXTRA_ARGS+=(--include_privileged_pipe)
        ;;
esac

echo "closed loop extra args=${CLOSED_LOOP_EXTRA_ARGS}"

if ! "${HUGSIM_PYTHON_BIN}" -c "from simple_knn._C import distCUDA2" >/dev/null 2>&1; then
    echo "simple_knn missing in ${HUGSIM_PYTHON_BIN}; bootstrapping local CUDA extension"
    SIMPLE_KNN_LOCK_DIR="${TMPDIR:-/tmp}/hugsim-simple-knn-install.lock"
    until mkdir "${SIMPLE_KNN_LOCK_DIR}" 2>/dev/null; do
        echo "waiting for simple_knn install lock: ${SIMPLE_KNN_LOCK_DIR}"
        sleep 5
    done

    if ! "${HUGSIM_PYTHON_BIN}" -c "from simple_knn._C import distCUDA2" >/dev/null 2>&1; then
        if ! "${HUGSIM_PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
            echo "pip missing in ${HUGSIM_PYTHON_BIN}; bootstrapping pip with ensurepip"
            "${HUGSIM_PYTHON_BIN}" -m ensurepip --upgrade
        fi
        "${HUGSIM_PYTHON_BIN}" -m pip install --no-build-isolation ./submodules/simple-knn
    fi
    rmdir "${SIMPLE_KNN_LOCK_DIR}" >/dev/null 2>&1 || true
fi

if [[ "${SIM_CUDA}" == "inherit" ]]; then
    "${HUGSIM_PYTHON_BIN}" closed_loop.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad "${AD_NAME}" \
        --ad_cuda "${AD_CUDA}" \
        --include_privileged_pipe "${CLOSED_LOOP_EXTRA_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${SIM_CUDA}" \
    "${HUGSIM_PYTHON_BIN}" closed_loop.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad "${AD_NAME}" \
        --ad_cuda "${AD_CUDA}" \
        --include_privileged_pipe "${CLOSED_LOOP_EXTRA_ARGS[@]}"
fi
