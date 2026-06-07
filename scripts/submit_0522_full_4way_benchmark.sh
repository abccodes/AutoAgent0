#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/baselines/full/submit_nuscenes_full_baselines.sh "$@"
