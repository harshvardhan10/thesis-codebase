#!/bin/bash -l
#SBATCH --job-name=chexmask-maskcache
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_maskcache_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_maskcache_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
SOURCE_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite"
LOCAL_DB="${TMPDIR:-/tmp}/mimic_chexmask_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sqlite"

MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_valid.json"
OUTPUT_ROOT="/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common"
STAGING_DIR="${OUTPUT_ROOT}/_staging"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

mkdir -p \
    "${OUTPUT_ROOT}" \
    "${STAGING_DIR}" \
    /home/vault/iwi5/iwi5362h/logs

cleanup() {
    rm -f "${LOCAL_DB}" "${LOCAL_DB}-shm" "${LOCAL_DB}-wal"
}
trap cleanup EXIT

echo "======================================================================"
echo "Compact CheXmask mask-cache generation"
echo "Host:         $(hostname)"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}/4"
echo "Manifest:     ${MANIFEST}"
echo "Source DB:    ${SOURCE_DB}"
echo "Local DB:     ${LOCAL_DB}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "======================================================================"

cp "${SOURCE_DB}" "${LOCAL_DB}"

python -u scripts/precompute_chexmask_mask_cache.py \
    --manifest "${MANIFEST}" \
    --chexmask_db "${LOCAL_DB}" \
    --output_root "${OUTPUT_ROOT}" \
    --staging_dir "${STAGING_DIR}" \
    --project_root "${PROJECT_DIR}" \
    --image_key image \
    --shard_index "${SLURM_ARRAY_TASK_ID}" \
    --num_shards 4 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --compression_level 6 \
    --progress_every 100

echo
du -sh "${OUTPUT_ROOT}/lung" "${OUTPUT_ROOT}/heart" || true
df -h "${OUTPUT_ROOT}"
