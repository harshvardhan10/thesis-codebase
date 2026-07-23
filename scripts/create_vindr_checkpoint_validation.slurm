#!/bin/bash -l
#SBATCH --job-name=vindr-val-setup
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_val_setup_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_val_setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
DATA_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr"

CHEXMASK_CSV="/home/vault/iwi5/iwi5362h/data/chexmask/OriginalResolution/VinDr-CXR.csv"
ANNOTATIONS_CSV="${DATA_ROOT}/annotations/annotations_train.csv"

LABELS_CSV="${DATA_ROOT}/annotations/image_labels_train.csv"
IMAGES_ROOT="${DATA_ROOT}/train"

VAL_DIR="${DATA_ROOT}/annotations/checkpoint_validation"
VAL_LABELS_CSV="${VAL_DIR}/cardiomegaly_labels.csv"
VAL_BOXES_CSV="${VAL_DIR}/cardiomegaly_boxes.csv"
MASK_OUTPUT_ROOT="${DATA_ROOT}/chexmask_mask_cache/train"

NUM_POSITIVE=200
NUM_NEGATIVE=300
SEED=42
MIN_DICE_RCA_MEAN=0.7

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for required in \
    "scripts/build_vindr_checkpoint_val_split.py" \
    "scripts/decode_vindr_validation_chexmask.py" \
    "${LABELS_CSV}" \
    "${ANNOTATIONS_CSV}" \
    "${IMAGES_ROOT}" \
    "${CHEXMASK_CSV}"
do
    if [[ ! -e "${required}" ]]; then
        echo "[ERROR] Missing required path: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${VAL_DIR}" "${MASK_OUTPUT_ROOT}"

echo "======================================================================"
echo "VinDr checkpoint-validation setup"
echo "Host:                 $(hostname)"
echo "Labels CSV:           ${LABELS_CSV}"
echo "Annotations CSV:      ${ANNOTATIONS_CSV}"
echo "CheXmask CSV:         ${CHEXMASK_CSV}"
echo "Images root:          ${IMAGES_ROOT}"
echo "Validation labels:    ${VAL_LABELS_CSV}"
echo "Validation boxes:     ${VAL_BOXES_CSV}"
echo "Mask output root:     ${MASK_OUTPUT_ROOT}"
echo "Positive/negative:    ${NUM_POSITIVE}/${NUM_NEGATIVE}"
echo "Seed:                 ${SEED}"
echo "======================================================================"

# =============================================================================
# Step 1: create the fixed 500-image validation split
# =============================================================================

echo "Creating fixed VinDr validation split..."

python -u scripts/build_vindr_checkpoint_val_split.py \
    --labels_csv "${LABELS_CSV}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --images_root "${IMAGES_ROOT}" \
    --chexmask_csv "${CHEXMASK_CSV}" \
    --output_labels_csv "${VAL_LABELS_CSV}" \
    --output_annotations_csv "${VAL_BOXES_CSV}" \
    --label Cardiomegaly \
    --num_positive "${NUM_POSITIVE}" \
    --num_negative "${NUM_NEGATIVE}" \
    --min_dice_rca_mean "${MIN_DICE_RCA_MEAN}" \
    --seed "${SEED}"

# =============================================================================
# Step 2: decode lung and heart masks only for the selected 500 images
# =============================================================================

echo "Decoding validation lung and heart masks..."

python -u scripts/decode_vindr_validation_chexmask.py \
    --selected_labels_csv "${VAL_LABELS_CSV}" \
    --chexmask_csv "${CHEXMASK_CSV}" \
    --images_root "${IMAGES_ROOT}" \
    --output_root "${MASK_OUTPUT_ROOT}"

# =============================================================================
# Verification
# =============================================================================

echo "[Verify] Checking split and mask counts..."

python - "${VAL_LABELS_CSV}" "${VAL_BOXES_CSV}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

labels_path = Path(sys.argv[1])
boxes_path = Path(sys.argv[2])

labels = pd.read_csv(labels_path)
boxes = pd.read_csv(boxes_path)
id_col = labels.columns[0]

n = len(labels)
pos = int(labels["Cardiomegaly"].sum())
neg = int((labels["Cardiomegaly"] == 0).sum())
cardio_boxes = int((boxes["class_name"].astype(str) == "Cardiomegaly").sum())

print(f"Validation rows:          {n}")
print(f"Cardiomegaly positives:  {pos}")
print(f"Cardiomegaly negatives:  {neg}")
print(f"Cardiomegaly GT boxes:   {cardio_boxes}")

if n != 500 or pos != 200 or neg != 300:
    raise SystemExit(
        f"Unexpected split composition: rows={n}, positives={pos}, negatives={neg}"
    )
PY

python - "${VAL_LABELS_CSV}" "${MASK_OUTPUT_ROOT}" <<'PYMASK'
import sys
from pathlib import Path

import pandas as pd

labels = pd.read_csv(sys.argv[1])
mask_root = Path(sys.argv[2])
id_col = labels.columns[0]
image_ids = labels[id_col].astype(str).tolist()

for view in ("lung", "heart"):
    missing = [
        image_id
        for image_id in image_ids
        if not (mask_root / view / image_id[:2] / f"{image_id}.png").exists()
    ]
    print(f"{view.capitalize()} masks verified: {len(image_ids) - len(missing)}/{len(image_ids)}")
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} selected {view} masks. First IDs: {missing[:10]}"
        )
PYMASK

echo "======================================================================"
echo "[Done] Fixed validation split and masks are ready."
echo "Labels: ${VAL_LABELS_CSV}"
echo "Boxes:  ${VAL_BOXES_CSV}"
echo "Lung:   ${MASK_OUTPUT_ROOT}/lung"
echo "Heart:  ${MASK_OUTPUT_ROOT}/heart"
echo "======================================================================"
