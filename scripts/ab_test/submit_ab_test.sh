#!/usr/bin/env bash
# Submit the three A/B configs (baseline / critic_only / full) as separate
# SLURM jobs, each producing its own run-variant directory.
#
# Usage: bash scripts/ab_test/submit_ab_test.sh
#
# Optional overrides: DATE_TAG, PARTITION, TIME, GPU_WORKERS
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/bigdata/aidan/HUGSIM-uncertainty-port}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
PARTITION="${PARTITION:-gpu02}"
TIME="${TIME:-24:00:00}"
GPU_WORKERS="${GPU_WORKERS:-4}"

cd "${REPO_ROOT}"

for CFG in baseline critic_only full; do
    JOB="ab-${DATE_TAG}-${CFG}"
    RUN_VARIANT_TAG="ab-${DATE_TAG}-${CFG}"
    echo "submitting AB_CONFIG=${CFG} run_variant=${RUN_VARIANT_TAG}"
    sbatch \
        --job-name="${JOB}" \
        --partition="${PARTITION}" \
        --time="${TIME}" \
        --gres="gpu:${GPU_WORKERS}" \
        --export="ALL,AB_CONFIG=${CFG},RUN_VARIANT_TAG=${RUN_VARIANT_TAG},GPU_WORKERS=${GPU_WORKERS}" \
        scripts/ab_test/run_ab_test.slurm
done

echo ""
echo "3 jobs submitted. Track with:"
echo "  squeue -u ${USER:-\$USER}"
echo "  ls -lt /bigdata/aidan/outputs/slurm/ab-${DATE_TAG}-* 2>/dev/null"
echo ""
echo "Output roots per config:"
for CFG in baseline critic_only full; do
    echo "  ${CFG}: /bigdata/aidan/outputs/benchmark/out/baselines/drivor_autoagent0/nuscenes/full/ab-${DATE_TAG}-${CFG}"
done
echo ""
echo "After all 3 finish, aggregate PDMS with:"
echo "  python scripts/ab_test/aggregate_ab_results.py --date-tag ${DATE_TAG}"
