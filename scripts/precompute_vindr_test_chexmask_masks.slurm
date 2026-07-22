#!/bin/bash -l
#SBATCH --job-name=vindr-test-masks
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_test_masks_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_test_masks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"

# ---------------------------------------------------------------------------
# VinDr paths
# Change VINDR_TEST_IMAGES only if your test PNG folder has another location.
# ---------------------------------------------------------------------------
VINDR_LABELS="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"
VINDR_TEST_IMAGES="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"

# Official CheXmask VinDr file already downloaded with the MIMIC CheXmask files.
CHEXMASK_CSV="/home/vault/iwi5/iwi5362h/data/chexmask/OriginalResolution/VinDr-CXR.csv"

# Persistent filtered SQLite index. Your earlier build_chexmask_indexes.slurm
# may already have created this file.
CHEXMASK_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/vindr_chexmask.sqlite"
LOCAL_DB="${TMPDIR:-/tmp}/vindr_chexmask_${SLURM_JOB_ID}.sqlite"

# This manifest is only an ID-to-image-path list used for mask generation.
MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/vindr_test_chexmask.json"

# Keep the ~3,000 one-bit masks on Woody, not Vault.
OUTPUT_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test"
STAGING_DIR="${OUTPUT_ROOT}/_staging"
FINAL_MANIFEST_DIR="${PROJECT_DIR}/data/chexmask_manifests"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

mkdir -p \
    "$(dirname "${CHEXMASK_DB}")" \
    "$(dirname "${MANIFEST}")" \
    "${OUTPUT_ROOT}" \
    "${STAGING_DIR}" \
    "${FINAL_MANIFEST_DIR}" \
    /home/vault/iwi5/iwi5362h/logs

cleanup() {
    rm -f "${LOCAL_DB}" "${LOCAL_DB}-shm" "${LOCAL_DB}-wal"
}
trap cleanup EXIT

echo "======================================================================"
echo "VinDr test CheXmask mask generation"
echo "Host:              $(hostname)"
echo "Labels:            ${VINDR_LABELS}"
echo "Test images:       ${VINDR_TEST_IMAGES}"
echo "CheXmask CSV:      ${CHEXMASK_CSV}"
echo "Persistent DB:     ${CHEXMASK_DB}"
echo "Output masks:      ${OUTPUT_ROOT}"
echo "======================================================================"

for required in \
    "${VINDR_LABELS}" \
    "${VINDR_TEST_IMAGES}" \
    "${CHEXMASK_CSV}" \
    scripts/build_vindr_test_mask_manifest.py \
    scripts/build_chexmask_sqlite_index.py \
    scripts/validate_chexmask_db_for_manifest.py \
    scripts/precompute_chexmask_mask_cache.py \
    scripts/finalize_chexmask_mask_manifests.py
do
    if [[ ! -e "${required}" ]]; then
        echo "[ERROR] Required path is missing: ${required}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# 1. Build a strict manifest for exactly the VinDr test IDs.
# ---------------------------------------------------------------------------
python -u scripts/build_vindr_test_mask_manifest.py \
    --labels_csv "${VINDR_LABELS}" \
    --images_root "${VINDR_TEST_IMAGES}" \
    --output_manifest "${MANIFEST}" \
    --id_col image_id \
    --extension .png

# ---------------------------------------------------------------------------
# 2. Reuse the existing filtered VinDr SQLite index when it is valid.
#    Otherwise rebuild it from VinDr-CXR.csv.
# ---------------------------------------------------------------------------
DB_VALID=0
if [[ -f "${CHEXMASK_DB}" ]]; then
    set +e
    python -u scripts/validate_chexmask_db_for_manifest.py \
        --manifest "${MANIFEST}" \
        --database "${CHEXMASK_DB}"
    VALIDATE_STATUS=$?
    set -e

    if [[ "${VALIDATE_STATUS}" -eq 0 ]]; then
        DB_VALID=1
        echo "[Index] Existing VinDr CheXmask DB is valid; reusing it."
    else
        echo "[Index] Existing DB is incomplete or invalid; rebuilding it."
    fi
fi

if [[ "${DB_VALID}" -ne 1 ]]; then
    python -u scripts/build_chexmask_sqlite_index.py \
        --chexmask_csv "${CHEXMASK_CSV}" \
        --output_db "${CHEXMASK_DB}" \
        --ids_csv "${VINDR_LABELS}" \
        --ids_csv_col image_id \
        --compression_level 1 \
        --strict_coverage

    python -u scripts/validate_chexmask_db_for_manifest.py \
        --manifest "${MANIFEST}" \
        --database "${CHEXMASK_DB}"
fi

# ---------------------------------------------------------------------------
# 3. Copy the compact filtered DB to node-local storage.
# ---------------------------------------------------------------------------
cp "${CHEXMASK_DB}" "${LOCAL_DB}"

# ---------------------------------------------------------------------------
# 4. Generate lung and heart 1-bit masks.
#    VinDr test has only about 3,000 images, so one job with eight workers is
#    sufficient; a four-task array would add scheduling and synchronization
#    overhead without meaningful benefit.
# ---------------------------------------------------------------------------
python -u scripts/precompute_chexmask_mask_cache.py \
    --manifest "${MANIFEST}" \
    --chexmask_db "${LOCAL_DB}" \
    --output_root "${OUTPUT_ROOT}" \
    --staging_dir "${STAGING_DIR}" \
    --project_root "${PROJECT_DIR}" \
    --image_key image \
    --shard_index 0 \
    --num_shards 1 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --compression_level 6 \
    --progress_every 100

# ---------------------------------------------------------------------------
# 5. Verify every lung and heart mask and write view-specific JSON manifests.
# ---------------------------------------------------------------------------
python -u scripts/finalize_chexmask_mask_manifests.py \
    --manifest "${MANIFEST}" \
    --mask_root "${OUTPUT_ROOT}" \
    --staging_dir "${STAGING_DIR}" \
    --output_dir "${FINAL_MANIFEST_DIR}" \
    --num_shards 1 \
    --image_key image \
    --verify_all

echo
echo "======================================================================"
echo "FINAL OUTPUTS"
echo "======================================================================"

echo "[Mask counts]"
echo -n "lung:  "
find "${OUTPUT_ROOT}/lung" -type f -name '*.png' | wc -l
echo -n "heart: "
find "${OUTPUT_ROOT}/heart" -type f -name '*.png' | wc -l

echo
echo "[Mask sizes]"
du -sh "${OUTPUT_ROOT}/lung" "${OUTPUT_ROOT}/heart"

echo
echo "[Generated manifests]"
ls -lh \
    "${FINAL_MANIFEST_DIR}/vindr_test_chexmask_lung_maskcache.json" \
    "${FINAL_MANIFEST_DIR}/vindr_test_chexmask_heart_maskcache.json" \
    "${FINAL_MANIFEST_DIR}/vindr_test_chexmask_maskcache_stats.json"

echo
shownicerquota.pl || true
