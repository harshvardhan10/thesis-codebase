#!/bin/bash -l
#SBATCH --job-name=cxr-orientation-qc
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/cxr_orientation_qc_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/cxr_orientation_qc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

echo "============================================================"
echo "Starting CXR orientation QC"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
unset PYTHONPATH

source ~/.bashrc
conda activate albef-notebook

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
SCRIPT_PATH="${PROJECT_DIR}/scripts/check_cxr_orientation.py"

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

# MIMIC-CXR / MIMIC-CXR-JPG
# Use the image root relative to paths inside data/mimic_cxr.json.
# Example: if JSON has image paths like files/p10/...jpg,
# set MIMIC_IMAGES_ROOT to the MIMIC-CXR-JPG root.
MIMIC_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/mimic_cxr/extracted/2.1.0"
MIMIC_JSON="${PROJECT_DIR}/data/mimic_cxr.json"

# VinDr train
VINDR_TRAIN_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr-cxr/png/train"
VINDR_TRAIN_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_train.csv"

# VinDr test
VINDR_TEST_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr-cxr/png/test"
VINDR_TEST_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"

# Output directory
OUT_DIR="${PROJECT_DIR}/orientation_qc_results"
mkdir -p "${OUT_DIR}"
mkdir -p /home/woody/iwi5/iwi5362h/logs

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
echo "[Check] Python:"
which python
python --version

echo "[Check] Required files/directories"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Missing script: ${SCRIPT_PATH}"
    exit 1
fi

if [ ! -f "${MIMIC_JSON}" ]; then
    echo "ERROR: Missing MIMIC JSON: ${MIMIC_JSON}"
    exit 1
fi

if [ ! -d "${MIMIC_IMAGES_ROOT}" ]; then
    echo "ERROR: Missing MIMIC image root: ${MIMIC_IMAGES_ROOT}"
    exit 1
fi

if [ ! -f "${VINDR_TRAIN_CSV}" ]; then
    echo "ERROR: Missing VinDr train CSV: ${VINDR_TRAIN_CSV}"
    exit 1
fi

if [ ! -d "${VINDR_TRAIN_IMAGES_ROOT}" ]; then
    echo "ERROR: Missing VinDr train image root: ${VINDR_TRAIN_IMAGES_ROOT}"
    exit 1
fi

if [ ! -f "${VINDR_TEST_CSV}" ]; then
    echo "ERROR: Missing VinDr test CSV: ${VINDR_TEST_CSV}"
    exit 1
fi

if [ ! -d "${VINDR_TEST_IMAGES_ROOT}" ]; then
    echo "ERROR: Missing VinDr test image root: ${VINDR_TEST_IMAGES_ROOT}"
    exit 1
fi

echo "All sanity checks passed."

# ------------------------------------------------------------
# Run 1: Full MIMIC-CXR
# ------------------------------------------------------------
echo "============================================================"
echo "[1/3] Running orientation QC on full MIMIC-CXR"
echo "Start: $(date)"
echo "============================================================"

python -u "${SCRIPT_PATH}" \
    --dataset_name "MIMIC-CXR-JPG" \
    --images_root "${MIMIC_IMAGES_ROOT}" \
    --manifest_json "${MIMIC_JSON}" \
    --output_csv "${OUT_DIR}/orientation_mimic_full.csv" \
    2>&1 | tee "${OUT_DIR}/orientation_mimic_full.log"

echo "[Done] MIMIC-CXR orientation QC finished at $(date)"

# ------------------------------------------------------------
# Run 2: VinDr train
# ------------------------------------------------------------
echo "============================================================"
echo "[2/3] Running orientation QC on VinDr train"
echo "Start: $(date)"
echo "============================================================"

python -u "${SCRIPT_PATH}" \
    --dataset_name "VinDr-CXR-train" \
    --images_root "${VINDR_TRAIN_IMAGES_ROOT}" \
    --manifest_csv "${VINDR_TRAIN_CSV}" \
    --id_col image_id \
    --filename_template "{image_id}.png" \
    --output_csv "${OUT_DIR}/orientation_vindr_train.csv" \
    2>&1 | tee "${OUT_DIR}/orientation_vindr_train.log"

echo "[Done] VinDr train orientation QC finished at $(date)"

# ------------------------------------------------------------
# Run 3: VinDr test
# ------------------------------------------------------------
echo "============================================================"
echo "[3/3] Running orientation QC on VinDr test"
echo "Start: $(date)"
echo "============================================================"

python -u "${SCRIPT_PATH}" \
    --dataset_name "VinDr-CXR-test" \
    --images_root "${VINDR_TEST_IMAGES_ROOT}" \
    --manifest_csv "${VINDR_TEST_CSV}" \
    --id_col image_id \
    --filename_template "{image_id}.png" \
    --output_csv "${OUT_DIR}/orientation_vindr_test.csv" \
    2>&1 | tee "${OUT_DIR}/orientation_vindr_test.log"

echo "[Done] VinDr test orientation QC finished at $(date)"

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
echo "============================================================"
echo "Final quick summary"
echo "============================================================"

echo ""
echo "MIMIC-CXR:"
python - <<EOF
import pandas as pd
p = "${OUT_DIR}/orientation_mimic_full.csv"
df = pd.read_csv(p)
print("rows:", len(df))
print(df["canvas_orientation"].value_counts(dropna=False))
print(df["anatomy_orientation"].value_counts(dropna=False))
print("vertical candidates:", (df["anatomy_orientation"] == "vertical_lungs_candidate").sum())
EOF

echo ""
echo "VinDr train:"
python - <<EOF
import pandas as pd
p = "${OUT_DIR}/orientation_vindr_train.csv"
df = pd.read_csv(p)
print("rows:", len(df))
print(df["canvas_orientation"].value_counts(dropna=False))
print(df["anatomy_orientation"].value_counts(dropna=False))
print("vertical candidates:", (df["anatomy_orientation"] == "vertical_lungs_candidate").sum())
EOF

echo ""
echo "VinDr test:"
python - <<EOF
import pandas as pd
p = "${OUT_DIR}/orientation_vindr_test.csv"
df = pd.read_csv(p)
print("rows:", len(df))
print(df["canvas_orientation"].value_counts(dropna=False))
print(df["anatomy_orientation"].value_counts(dropna=False))
print("vertical candidates:", (df["anatomy_orientation"] == "vertical_lungs_candidate").sum())
EOF

echo "============================================================"
echo "All orientation QC runs completed."
echo "End time: $(date)"
echo "Outputs saved in: ${OUT_DIR}"
echo "============================================================"