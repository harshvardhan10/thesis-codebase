#!/bin/bash -l
#SBATCH --job-name=chexmask-precompute
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_precompute_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_precompute_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
SOURCE_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite"
LOCAL_DB="${TMPDIR:-/tmp}/mimic_chexmask_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sqlite"
COMMON_MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_valid.json"
OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/data/chexmask/precomputed/common"
LUNG_ROOT="${OUTPUT_ROOT}/lung"
HEART_ROOT="${OUTPUT_ROOT}/heart"
STAGING_DIR="${OUTPUT_ROOT}/_staging"
NUM_SHARDS=4

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p \
  "${LUNG_ROOT}" \
  "${HEART_ROOT}" \
  "${STAGING_DIR}" \
  /home/vault/iwi5/iwi5362h/logs

for path in \
  "${SOURCE_DB}" \
  "${COMMON_MANIFEST}" \
  scripts/precompute_chexmask_common_views.py
do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 2
  fi
done

cleanup() {
  rm -f "${LOCAL_DB}" "${LOCAL_DB}-shm" "${LOCAL_DB}-wal"
}
trap cleanup EXIT

echo "======================================================================"
echo "CheXmask offline precomputation"
echo "Host:         $(hostname)"
echo "Shard:        ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Source DB:    ${SOURCE_DB}"
echo "Local DB:     ${LOCAL_DB}"
echo "Manifest:     ${COMMON_MANIFEST}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Started:      $(date --iso-8601=seconds)"
echo "======================================================================"

df -h "${OUTPUT_ROOT}" "${TMPDIR:-/tmp}" || true

echo "[1/2] Copying SQLite index to node-local storage..."
cp "${SOURCE_DB}" "${LOCAL_DB}"
ls -lh "${LOCAL_DB}"

echo "[2/2] Generating lung and heart images..."
python -u scripts/precompute_chexmask_common_views.py \
  --manifest "${COMMON_MANIFEST}" \
  --chexmask_db "${LOCAL_DB}" \
  --lung_output_root "${LUNG_ROOT}" \
  --heart_output_root "${HEART_ROOT}" \
  --staging_dir "${STAGING_DIR}" \
  --image_key image \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${SLURM_ARRAY_TASK_ID}" \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --chunksize 4 \
  --png_compress_level 1

echo "Finished: $(date --iso-8601=seconds)"
