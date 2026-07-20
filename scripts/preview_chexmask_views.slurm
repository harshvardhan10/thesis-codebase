#!/bin/bash -l
#SBATCH --job-name=chexmask-preview
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/chexmask_preview_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/chexmask_preview_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python
conda activate albef-notebook

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"

PREVIEW_SCRIPT="${PROJECT_DIR}/scripts/preview_chexmask_views.py"
MIMIC_MANIFEST="${PROJECT_DIR}/data/mimic_cxr.json"
CHEXMASK_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite"

# Store preview images in the project directory for convenient inspection.
OUTPUT_DIR="${PROJECT_DIR}/previews/chexmask"

MAX_IMAGES=30
MIN_RCA_MEAN=0.0

mkdir -p \
    "${OUTPUT_DIR}" \
    /home/woody/iwi5/iwi5362h/logs

cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "CheXmask preview generation"
echo "Host:            $(hostname)"
echo "Started:         $(date --iso-8601=seconds)"
echo "Project:         ${PROJECT_DIR}"
echo "Preview script:  ${PREVIEW_SCRIPT}"
echo "Manifest:        ${MIMIC_MANIFEST}"
echo "CheXmask DB:     ${CHEXMASK_DB}"
echo "Output:          ${OUTPUT_DIR}"
echo "Max images:      ${MAX_IMAGES}"
echo "Min RCA mean:    ${MIN_RCA_MEAN}"
echo "Python:          $(which python)"
echo "======================================================================"

for required_file in \
    "${PREVIEW_SCRIPT}" \
    "${MIMIC_MANIFEST}" \
    "${CHEXMASK_DB}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Required file not found: ${required_file}" >&2
        exit 2
    fi
    ls -lh "${required_file}"
done

echo
echo "[COMMAND]"
printf ' %q' \
    python -u "${PREVIEW_SCRIPT}" \
    --manifest "${MIMIC_MANIFEST}" \
    --chexmask_db "${CHEXMASK_DB}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_images "${MAX_IMAGES}" \
    --min_rca_mean "${MIN_RCA_MEAN}"
echo
echo

# ---------------------------------------------------------------------------
# Generate original | lung-only | heart-only panels
# ---------------------------------------------------------------------------
python -u "${PREVIEW_SCRIPT}" \
    --manifest "${MIMIC_MANIFEST}" \
    --chexmask_db "${CHEXMASK_DB}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_images "${MAX_IMAGES}" \
    --min_rca_mean "${MIN_RCA_MEAN}"

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
echo
echo "======================================================================"
echo "Generated previews"
echo "======================================================================"

PREVIEW_COUNT="$(
    find "${OUTPUT_DIR}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | wc -l
)"

echo "Preview count: ${PREVIEW_COUNT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo

find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -printf '%f\n' \
    | sort \
    | head -n 40

if [[ "${PREVIEW_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No preview images were generated." >&2
    exit 3
fi

echo
du -sh "${OUTPUT_DIR}"
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"
