#!/bin/bash -l
#SBATCH --job-name=vindr-val-masks
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_val_masks_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_val_masks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"

# The fixed 500-image validation split created earlier.
VINDR_LABELS="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/checkpoint_validation/cardiomegaly_labels.csv"

# Your resized 256x256 VinDr training PNGs.
VINDR_IMAGES="/home/woody/iwi5/iwi5362h/data/vindr_cxr/train"

# Same original-resolution CheXmask CSV used for the test masks.
CHEXMASK_CSV="/home/vault/iwi5/iwi5362h/data/chexmask/OriginalResolution/VinDr-CXR.csv"

# Separate filtered database for only these 500 validation images.
CHEXMASK_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/vindr_checkpoint_val_chexmask.sqlite"
LOCAL_DB="${TMPDIR:-/tmp}/vindr_checkpoint_val_chexmask_${SLURM_JOB_ID}.sqlite"

# Manifest containing only the selected validation image IDs and PNG paths.
MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/vindr_checkpoint_val_chexmask.json"

# Masks will be saved here.
OUTPUT_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/train"

# Use a separate staging directory for this validation run.
STAGING_DIR="${OUTPUT_ROOT}/_staging_checkpoint_val"

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
echo "VinDr checkpoint-validation mask generation"
echo "Host:              $(hostname)"
echo "Validation labels: ${VINDR_LABELS}"
echo "Train images:      ${VINDR_IMAGES}"
echo "CheXmask CSV:      ${CHEXMASK_CSV}"
echo "Filtered DB:       ${CHEXMASK_DB}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "======================================================================"

for required in \
    "${VINDR_LABELS}" \
    "${VINDR_IMAGES}" \
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
# 1. Build a manifest for exactly the selected validation images.
# ---------------------------------------------------------------------------
python -u scripts/build_vindr_test_mask_manifest.py \
    --labels_csv "${VINDR_LABELS}" \
    --images_root "${VINDR_IMAGES}" \
    --output_manifest "${MANIFEST}" \
    --id_col image_id \
    --extension .png

# ---------------------------------------------------------------------------
# 2. Reuse the validation-specific SQLite database when valid.
#    Otherwise build it from the full VinDr CheXmask CSV.
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
        echo "[Index] Existing validation database is valid."
    else
        echo "[Index] Existing validation database is invalid; rebuilding."
        rm -f "${CHEXMASK_DB}"
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
# 3. Copy the small filtered database to node-local storage.
# ---------------------------------------------------------------------------
cp "${CHEXMASK_DB}" "${LOCAL_DB}"

# ---------------------------------------------------------------------------
# 4. Generate lung and heart masks using your existing working code.
#    This code decodes the original-resolution RLE and creates a mask matching
#    the actual 256x256 PNG dimensions.
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
    --progress_every 50

# ---------------------------------------------------------------------------
# 5. Verify all selected masks and produce view-specific manifests.
# ---------------------------------------------------------------------------
python -u scripts/finalize_chexmask_mask_manifests.py \
    --manifest "${MANIFEST}" \
    --mask_root "${OUTPUT_ROOT}" \
    --staging_dir "${STAGING_DIR}" \
    --output_dir "${FINAL_MANIFEST_DIR}" \
    --num_shards 1 \
    --image_key image \
    --verify_all

# ---------------------------------------------------------------------------
# 6. Verify that every selected image has both masks.
# ---------------------------------------------------------------------------
python - "${VINDR_LABELS}" "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

labels_csv = Path(sys.argv[1])
output_root = Path(sys.argv[2])

df = pd.read_csv(labels_csv)
image_ids = df["image_id"].astype(str).tolist()

missing_lung = []
missing_heart = []
bad_sizes = []

for image_id in image_ids:
    lung_path = output_root / "lung" / image_id[:2] / f"{image_id}.png"
    heart_path = output_root / "heart" / image_id[:2] / f"{image_id}.png"

    if not lung_path.exists():
        missing_lung.append(str(lung_path))

    if not heart_path.exists():
        missing_heart.append(str(heart_path))

    if lung_path.exists() and heart_path.exists():
        with Image.open(lung_path) as lung:
            lung_size = lung.size
        with Image.open(heart_path) as heart:
            heart_size = heart.size

        if lung_size != (256, 256) or heart_size != (256, 256):
            bad_sizes.append(
                {
                    "image_id": image_id,
                    "lung_size": lung_size,
                    "heart_size": heart_size,
                }
            )

print(f"Selected images: {len(image_ids)}")
print(f"Missing lung masks: {len(missing_lung)}")
print(f"Missing heart masks: {len(missing_heart)}")
print(f"Incorrect mask dimensions: {len(bad_sizes)}")

if missing_lung:
    print("First missing lung masks:")
    print("\n".join(missing_lung[:10]))

if missing_heart:
    print("First missing heart masks:")
    print("\n".join(missing_heart[:10]))

if bad_sizes:
    print("First incorrect sizes:")
    for row in bad_sizes[:10]:
        print(row)

if missing_lung or missing_heart or bad_sizes:
    raise SystemExit(1)

print("Validation lung and heart mask generation completed successfully.")
PY

echo
echo "======================================================================"
echo "FINAL MASK COUNTS"
echo "======================================================================"

python - "${VINDR_LABELS}" "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

labels_csv = Path(sys.argv[1])
root = Path(sys.argv[2])

ids = set(pd.read_csv(labels_csv)["image_id"].astype(str))

for view in ["lung", "heart"]:
    count = sum(
        1
        for image_id in ids
        if (root / view / image_id[:2] / f"{image_id}.png").exists()
    )
    print(f"{view}: {count}/{len(ids)}")
PY

du -sh "${OUTPUT_ROOT}/lung" "${OUTPUT_ROOT}/heart"
shownicerquota.pl || true