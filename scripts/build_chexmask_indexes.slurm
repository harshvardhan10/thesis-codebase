#!/bin/bash -l
#SBATCH --job-name=chexmask-index
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_index_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_index_%A_%a.err
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

CHEXMASK_ROOT="/home/vault/iwi5/iwi5362h/data/chexmask"
CHEXMASK_CSV_DIR="${CHEXMASK_ROOT}/OriginalResolution"
INDEX_DIR="${CHEXMASK_ROOT}/index"

MIMIC_CSV="${CHEXMASK_CSV_DIR}/MIMIC-CXR-JPG.csv"
VINDR_CSV="${CHEXMASK_CSV_DIR}/VinDr-CXR.csv"

MIMIC_MANIFEST="${PROJECT_DIR}/data/mimic_cxr.json"
VINDR_LABELS="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"

INDEX_SCRIPT="${PROJECT_DIR}/scripts/build_chexmask_sqlite_index.py"

mkdir -p \
    "${INDEX_DIR}" \
    /home/woody/iwi5/iwi5362h/logs

cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Select array task
# ---------------------------------------------------------------------------
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        DATASET_NAME="MIMIC-CXR-JPG"
        INPUT_CSV="${MIMIC_CSV}"
        OUTPUT_DB="${INDEX_DIR}/mimic_chexmask.sqlite"

        COMMAND=(
            python -u "${INDEX_SCRIPT}"
            --chexmask_csv "${INPUT_CSV}"
            --output_db "${OUTPUT_DB}"
            --manifest "${MIMIC_MANIFEST}"
            --image_key image
            --compression_level 1
            --strict_coverage
        )
        ;;

    1)
        DATASET_NAME="VinDr-CXR"
        INPUT_CSV="${VINDR_CSV}"
        OUTPUT_DB="${INDEX_DIR}/vindr_chexmask.sqlite"

        COMMAND=(
            python -u "${INDEX_SCRIPT}"
            --chexmask_csv "${INPUT_CSV}"
            --output_db "${OUTPUT_DB}"
            --ids_csv "${VINDR_LABELS}"
            --ids_csv_col image_id
            --compression_level 1
            --strict_coverage
        )
        ;;

    *)
        echo "[ERROR] Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

MISSING_FILE="${OUTPUT_DB}.missing_ids.txt"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "CheXmask compressed-index build"
echo "Dataset:       ${DATASET_NAME}"
echo "Host:          $(hostname)"
echo "Started:       $(date --iso-8601=seconds)"
echo "Project:       ${PROJECT_DIR}"
echo "Input CSV:     ${INPUT_CSV}"
echo "Output DB:     ${OUTPUT_DB}"
echo "Missing IDs:   ${MISSING_FILE}"
echo "Python:        $(which python)"
echo "======================================================================"

required_paths=(
    "${INDEX_SCRIPT}"
    "${INPUT_CSV}"
)

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    required_paths+=("${MIMIC_MANIFEST}")
else
    required_paths+=("${VINDR_LABELS}")
fi

for required_path in "${required_paths[@]}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "[ERROR] Required file not found: ${required_path}" >&2
        exit 3
    fi
    ls -lh "${required_path}"
done

echo
echo "[COMMAND]"
printf ' %q' "${COMMAND[@]}"
echo
echo

# ---------------------------------------------------------------------------
# Build index
#
# --strict_coverage may deliberately return nonzero after creating the DB if
# requested IDs are missing. Capture the status so diagnostics still print.
# ---------------------------------------------------------------------------
set +e
"${COMMAND[@]}"
BUILD_STATUS=$?
set -e

echo
echo "======================================================================"
echo "POST-BUILD DIAGNOSTICS"
echo "======================================================================"

if [[ -f "${OUTPUT_DB}" ]]; then
    echo "[Database]"
    ls -lh "${OUTPUT_DB}"
    du -h "${OUTPUT_DB}"

    echo
    echo "[SQLite row count]"
    python - "${OUTPUT_DB}" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
try:
    count = connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
    metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
finally:
    connection.close()

print(f"rows_in_masks={count:,}")
for key in sorted(metadata):
    print(f"{key}={metadata[key]}")
PY
else
    echo "[ERROR] Database was not created: ${OUTPUT_DB}" >&2
fi

echo
if [[ -f "${MISSING_FILE}" ]]; then
    MISSING_COUNT="$(grep -cve '^[[:space:]]*$' "${MISSING_FILE}" || true)"
    echo "[Missing IDs]"
    echo "count=${MISSING_COUNT}"
    echo "file=${MISSING_FILE}"

    if [[ "${MISSING_COUNT}" -gt 0 ]]; then
        echo
        echo "First 30 missing IDs:"
        head -n 30 "${MISSING_FILE}"
    else
        echo "No missing IDs."
    fi
else
    echo "[WARNING] Missing-ID report was not created: ${MISSING_FILE}"
fi

echo
echo "Build exit status: ${BUILD_STATUS}"
echo "Finished:          $(date --iso-8601=seconds)"
echo "======================================================================"

# Preserve strict-coverage failure status after printing diagnostics.
exit "${BUILD_STATUS}"
