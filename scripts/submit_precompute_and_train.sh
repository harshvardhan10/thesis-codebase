#!/usr/bin/env bash
set -euo pipefail

# Run from the ALBEF repository root after copying the bundle files there.
PRECOMPUTE_JOB="$(sbatch --parsable precompute_chexmask_common_views.slurm)"
FINALIZE_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${PRECOMPUTE_JOB}" \
    finalize_chexmask_precomputed.slurm
)"
TRAIN_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${FINALIZE_JOB}" \
    train_precomputed_common_views.slurm
)"

echo "Precompute array job: ${PRECOMPUTE_JOB}"
echo "Finalize job:         ${FINALIZE_JOB}"
echo "Training array job:   ${TRAIN_JOB}"
echo
echo "The training array starts only if all precompute shards and validation pass."
