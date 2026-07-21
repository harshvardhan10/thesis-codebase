#!/bin/bash -l
#SBATCH --job-name=chexmask-validate
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_validate_fast_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_validate_fast_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
SOURCE_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite"
LOCAL_DB="${TMPDIR:-/tmp}/mimic_chexmask_${SLURM_JOB_ID}.sqlite"
OUTPUT_DIR="${PROJECT_DIR}/data/chexmask_manifests"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p \
    "${OUTPUT_DIR}" \
    /home/vault/iwi5/iwi5362h/logs

echo "======================================================================"
echo "CheXmask manifest validation"
echo "Host:       $(hostname)"
echo "Started:    $(date --iso-8601=seconds)"
echo "Source DB:  ${SOURCE_DB}"
echo "Local DB:   ${LOCAL_DB}"
echo "TMPDIR:     ${TMPDIR:-/tmp}"
echo "======================================================================"

if [[ ! -f "${SOURCE_DB}" ]]; then
    echo "[ERROR] Missing source database: ${SOURCE_DB}" >&2
    exit 2
fi

if [[ ! -f "scripts/filter_chexmask_valid_manifest_fast.py" ]]; then
    echo "[ERROR] Missing validator script." >&2
    exit 3
fi

cleanup() {
    rm -f "${LOCAL_DB}" "${LOCAL_DB}-shm" "${LOCAL_DB}-wal"
}
trap cleanup EXIT

echo "[1/3] Copying SQLite database to node-local storage..."
START_COPY="$(date +%s)"
cp "${SOURCE_DB}" "${LOCAL_DB}"
END_COPY="$(date +%s)"

echo "[1/3] Copy completed in $((END_COPY - START_COPY)) seconds."
ls -lh "${SOURCE_DB}" "${LOCAL_DB}"

echo
echo "[2/3] Checking local SQLite database..."
python - "${LOCAL_DB}" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
try:
    result = connection.execute("PRAGMA quick_check").fetchone()[0]
    rows = connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
finally:
    connection.close()

print(f"quick_check={result}", flush=True)
print(f"rows={rows:,}", flush=True)

if result != "ok":
    raise SystemExit("SQLite quick_check failed")
PY

echo
echo "[3/3] Validating manifest against node-local database..."
START_VALIDATE="$(date +%s)"

python -u scripts/filter_chexmask_valid_manifest_fast.py \
    --manifest data/mimic_cxr.json \
    --chexmask_db "${LOCAL_DB}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_key image \
    --min_rca_mean 0.0

END_VALIDATE="$(date +%s)"

echo
echo "======================================================================"
echo "Validation completed"
echo "Validation time: $((END_VALIDATE - START_VALIDATE)) seconds"
echo "Finished:        $(date --iso-8601=seconds)"
echo "Outputs:"
ls -lh "${OUTPUT_DIR}"
echo "======================================================================"
