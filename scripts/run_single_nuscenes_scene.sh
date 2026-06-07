#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/baselines/common/run_single_nuscenes_scene.sh "$@"
