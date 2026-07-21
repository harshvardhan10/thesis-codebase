#!/bin/bash
set -euo pipefail

PRECOMPUTE_JOB="$(sbatch --parsable precompute_chexmask_mask_cache.slurm)"
PRECOMPUTE_JOB="${PRECOMPUTE_JOB%%;*}"

FINALIZE_JOB="$(
    sbatch \
        --parsable \
        --dependency="afterok:${PRECOMPUTE_JOB}" \
        finalize_chexmask_mask_cache.slurm
)"
FINALIZE_JOB="${FINALIZE_JOB%%;*}"

TRAIN_JOB="$(
    sbatch \
        --parsable \
        --dependency="afterok:${FINALIZE_JOB}" \
        train_a0_maskcache_common_views.slurm
)"
TRAIN_JOB="${TRAIN_JOB%%;*}"

echo "Precompute mask cache: ${PRECOMPUTE_JOB}"
echo "Finalize manifests:    ${FINALIZE_JOB}"
echo "Train lung + heart:    ${TRAIN_JOB}"
