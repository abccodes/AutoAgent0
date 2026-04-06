#!/usr/bin/env bash

export HUGSIM_ROOT="/data/guest_rui/ztrs_workspace/HUGSIM"
export HUGSIM_CACHE_ROOT="${HUGSIM_CACHE_ROOT:-/data/guest_rui/.cache/hugsim}"

export PIXI_HOME="${HUGSIM_CACHE_ROOT}/pixi"
export PIP_CACHE_DIR="${HUGSIM_CACHE_ROOT}/pip"
export UV_CACHE_DIR="${HUGSIM_CACHE_ROOT}/uv"
export XDG_CACHE_HOME="${HUGSIM_CACHE_ROOT}/xdg"
export TORCH_HOME="${HUGSIM_CACHE_ROOT}/torch"
mkdir -p "${PIXI_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"


export LD_LIBRARY_PATH="${HUGSIM_ROOT}/.pixi/envs/default/lib/python3.11/site-packages/torch/lib:${HUGSIM_ROOT}/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${HUGSIM_ROOT}" || return 1
