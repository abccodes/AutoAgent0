#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "usage: $0 <planner:{rap|rap_vlm|drivor|drivor_vlm|rule_based|rule_based_vlm}> [scenario_yaml] [sim_cuda|inherit] [ad_cuda|inherit]" >&2
    exit 2
fi

default_planner_path() {
    local planner_name="$1"
    case "${planner_name}" in
        rule_based|rule_based_vlm)
            echo "configs/planners/rule_based_local_aidan.yaml"
            ;;
        *)
            echo "configs/planners/${planner_name}.yaml"
            ;;
    esac
}

infer_dataset_type() {
    local scenario_path="$1"
    local python_bin="$2"
    "${python_bin}" - <<'PY' "${scenario_path}"
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
print(str(cfg.get("data_type", "nuscenes")).strip().lower())
PY
}

preflight_hugsim_assets() {
    local base_path="$1"
    local scenario_path="$2"
    local python_bin="$3"
    "${python_bin}" - <<'PY' "${base_path}" "${scenario_path}"
import os
import sys
from omegaconf import OmegaConf

base_path, scenario_path = sys.argv[1:]
base = OmegaConf.load(base_path)
scenario = OmegaConf.load(scenario_path)
model_base = os.path.expanduser(str(base.get("model_base", "")))
realcar_path = os.path.expanduser(str(base.get("realcar_path", "")))
scene_name = str(scenario.scene_name)

missing = []
if not os.path.isdir(model_base):
    missing.append(f"processed scene root: {model_base}")
if not os.path.isdir(realcar_path):
    missing.append(f"3DRealCar root: {realcar_path}")
if missing:
    raise SystemExit("missing HUGSIM assets:\n- " + "\n- ".join(missing))

direct = os.path.join(model_base, scene_name)
candidates = []
if os.path.isfile(os.path.join(direct, "cfg.yaml")):
    candidates.append(direct)
else:
    for root, dirs, _files in os.walk(model_base):
        if scene_name not in dirs:
            continue
        candidate = os.path.join(root, scene_name)
        if os.path.isfile(os.path.join(candidate, "cfg.yaml")):
            candidates.append(candidate)
if len(candidates) != 1:
    detail = "not found" if not candidates else f"ambiguous: {candidates}"
    raise SystemExit(
        f"processed scene {scene_name!r} {detail} under model_base={model_base!r}"
    )

for index, control in enumerate(scenario.get("plan_list", []) or []):
    if len(control) < 6:
        missing.append(f"plan_list[{index}] has no dynamic model path")
        continue
    model_ref = os.path.expanduser(str(control[5]))
    model_dir = model_ref if os.path.isabs(model_ref) else os.path.join(realcar_path, model_ref)
    if not os.path.isdir(model_dir):
        missing.append(f"plan_list[{index}] dynamic model directory: {model_dir}")
        continue
    for filename in ("gs.pth", "wlh.json"):
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            missing.append(f"plan_list[{index}] dynamic model file: {path}")
if missing:
    raise SystemExit("missing HUGSIM assets:\n- " + "\n- ".join(missing))
print(f"scene_model_path={candidates[0]}")
PY
}

default_base_path_for_dataset() {
    local data_type="$1"
    case "${data_type}" in
        nuscenes)
            echo "configs/sim/nuscenes_base_local.yaml"
            ;;
        waymo)
            echo "configs/sim/waymo_base_local.yaml"
            ;;
        kitti360)
            echo "configs/sim/kitti360_base_local.yaml"
            ;;
        *)
            echo "unsupported data_type: ${data_type}" >&2
            exit 2
            ;;
    esac
}

default_camera_path_for_dataset() {
    local data_type="$1"
    case "${data_type}" in
        nuscenes)
            echo "configs/sim/nuscenes_camera.yaml"
            ;;
        waymo)
            echo "configs/sim/waymo_camera.yaml"
            ;;
        kitti360)
            echo "configs/sim/kitti360_camera.yaml"
            ;;
        *)
            echo "unsupported data_type: ${data_type}" >&2
            exit 2
            ;;
    esac
}

PLANNER_NAME="${1:?missing planner name}"
SCENARIO_PATH="${2:-${SCENARIO_PATH:-configs/benchmark/nuscenes/scene-0383-easy-00.yaml}}"
DEFAULT_CUDA_ID="0"
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]; then
    DEFAULT_CUDA_ID="inherit"
fi
SIM_CUDA="${3:-${SIM_CUDA:-${DEFAULT_CUDA_ID}}}"
AD_CUDA="${4:-${AD_CUDA:-${DEFAULT_CUDA_ID}}}"
KINEMATIC_PATH="${KINEMATIC_PATH:-configs/sim/kinematic.yaml}"
PLANNER_PATH="${PLANNER_PATH:-$(default_planner_path "${PLANNER_NAME}")}"
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

if [[ ! -x "${HUGSIM_PYTHON_BIN}" ]]; then
    echo "missing HUGSIM python: ${HUGSIM_PYTHON_BIN}" >&2
    exit 1
fi

DATA_TYPE="${DATA_TYPE:-$(infer_dataset_type "${SCENARIO_PATH}" "${HUGSIM_PYTHON_BIN}")}"
BASE_PATH="${BASE_PATH:-$(default_base_path_for_dataset "${DATA_TYPE}")}"
CAMERA_PATH="${CAMERA_PATH:-$(default_camera_path_for_dataset "${DATA_TYPE}")}"

if [[ ! -f "${BASE_PATH}" ]]; then
    echo "missing base config: ${BASE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PLANNER_PATH}" ]]; then
    echo "missing planner config: ${PLANNER_PATH}" >&2
    exit 1
fi

preflight_hugsim_assets "${BASE_PATH}" "${SCENARIO_PATH}" "${HUGSIM_PYTHON_BIN}"

mkdir -p /bigdata/aidan/outputs/benchmark/out

export PYTHONPATH="${PWD}:${PWD}/sim${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

HUGSIM_ENV_ROOT="$(cd "$(dirname "${HUGSIM_PYTHON_BIN}")/.." && pwd)"
TORCH_LIB_DIR="${HUGSIM_ENV_ROOT}/lib/python3.11/site-packages/torch/lib"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${HUGSIM_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Some CUDA wheel dependencies ship shared libraries outside torch/lib.
# Include those packaged NVIDIA library directories so tinycudann can resolve
# libnvrtc and related CUDA runtime libs on nodes without system CUDA paths.
for extra_lib_dir in \
    "${HUGSIM_ENV_ROOT}"/lib/python*/site-packages/nvidia/*/lib \
    "${VLM_ENV_DIR:-}"/lib/python*/site-packages/nvidia/*/lib \
    /bigdata/aidan/.home/envs/vlm/lib/python*/site-packages/nvidia/*/lib
do
    if [[ -d "${extra_lib_dir}" ]]; then
        export LD_LIBRARY_PATH="${extra_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done

echo "planner=${PLANNER_NAME}"
echo "ad=${AD_NAME}"
echo "data_type=${DATA_TYPE}"
echo "scenario=${SCENARIO_PATH}"
echo "base=${BASE_PATH}"
echo "camera=${CAMERA_PATH}"
echo "planner_config=${PLANNER_PATH}"
echo "hugsim_python=${HUGSIM_PYTHON_BIN}"
echo "sim_cuda=${SIM_CUDA} ad_cuda=${AD_CUDA}"
echo "include_privileged_pipe=${INCLUDE_PRIVILEGED_PIPE}"
echo "ld_library_path=${LD_LIBRARY_PATH:-unset}"

if [[ -n "${BASELINE_ID:-}" && -n "${BASELINE_SUITE:-}" ]]; then
    resolve_cmd=(
        "${HUGSIM_PYTHON_BIN}" scripts/baselines/common/resolve_output_path.py
        --baseline_id "${BASELINE_ID}"
        --dataset "${DATA_TYPE}"
        --suite "${BASELINE_SUITE}"
        --run_type "${BASELINE_RUN_TYPE:-canonical}"
        --scenario_path "${SCENARIO_PATH}"
    )
    if [[ -n "${BASELINE_RUN_VARIANT:-}" ]]; then
        resolve_cmd+=(--run_variant "${BASELINE_RUN_VARIANT}")
    fi
    BENCHMARK_OUTPUT_ROOT_OVERRIDE="$("${resolve_cmd[@]}")"
    BENCHMARK_OUTPUT_ROOT_OVERRIDE="$(dirname "${BENCHMARK_OUTPUT_ROOT_OVERRIDE}")"
    export BENCHMARK_OUTPUT_ROOT_OVERRIDE
    echo "benchmark_output_root_override=${BENCHMARK_OUTPUT_ROOT_OVERRIDE}"
fi

if ! "${HUGSIM_PYTHON_BIN}" -c "from simple_knn._C import distCUDA2" >/dev/null 2>&1; then
    echo "warning: simple_knn import check failed in ${HUGSIM_PYTHON_BIN}; skipping auto-bootstrap"
fi

if [[ "${SIM_CUDA}" == "inherit" ]]; then
    "${HUGSIM_PYTHON_BIN}" main.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad "${AD_NAME}" \
        --ad_cuda "${AD_CUDA}" \
        --include_privileged_pipe "${INCLUDE_PRIVILEGED_PIPE}"
else
    CUDA_VISIBLE_DEVICES="${SIM_CUDA}" \
    "${HUGSIM_PYTHON_BIN}" main.py \
        --scenario_path "${SCENARIO_PATH}" \
        --base_path "${BASE_PATH}" \
        --camera_path "${CAMERA_PATH}" \
        --kinematic_path "${KINEMATIC_PATH}" \
        --planner_path "${PLANNER_PATH}" \
        --ad "${AD_NAME}" \
        --ad_cuda "${AD_CUDA}" \
        --include_privileged_pipe "${INCLUDE_PRIVILEGED_PIPE}"
fi
