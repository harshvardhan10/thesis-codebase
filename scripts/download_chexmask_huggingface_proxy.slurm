#!/bin/bash -l
#SBATCH --job-name=chexmask-hf
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_hf_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_hf_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

PROXY="http://proxy.nhr.fau.de:80"
export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"
unset all_proxy ALL_PROXY no_proxy NO_PROXY

ROOT="/home/vault/iwi5/iwi5362h/data/chexmask"
OUTPUT_DIR="${ROOT}/OriginalResolution"
CHECKSUM_FILE="${ROOT}/SHA256SUMS.physionet.txt"

mkdir -p "${OUTPUT_DIR}" /home/woody/iwi5/iwi5362h/logs

FILES=(
    "MIMIC-CXR-JPG.csv"
    "VinDr-CXR.csv"
)

FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
REPO="ming0100/chexmask"
URL="https://huggingface.co/datasets/${REPO}/resolve/main/OriginalResolution/${FILE}"

FINAL_PATH="${OUTPUT_DIR}/${FILE}"
PARTIAL_PATH="${FINAL_PATH}.part"
LOCK_PATH="${FINAL_PATH}.lock"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "[ERROR] Another job is already downloading ${FILE}."
    exit 2
fi

echo "============================================================"
echo "Host:        $(hostname)"
echo "Started:     $(date --iso-8601=seconds)"
echo "File:        ${FILE}"
echo "URL:         ${URL}"
echo "Destination: ${FINAL_PATH}"
echo "Proxy:       ${HTTPS_PROXY}"
echo "============================================================"

if [[ ! -f "${CHECKSUM_FILE}" ]]; then
    echo "[ERROR] Missing official PhysioNet checksum file:"
    echo "        ${CHECKSUM_FILE}"
    exit 3
fi

EXPECTED_SHA256="$(
    grep -F "OriginalResolution/${FILE}" "${CHECKSUM_FILE}" \
    | head -n 1 \
    | awk '{print $1}'
)"

if [[ -z "${EXPECTED_SHA256}" ]]; then
    echo "[ERROR] No checksum entry found for ${FILE}."
    grep -F "${FILE}" "${CHECKSUM_FILE}" || true
    exit 4
fi

verify_file() {
    local path="$1"
    echo "${EXPECTED_SHA256}  ${path}" | sha256sum --check -
}

if [[ -f "${FINAL_PATH}" ]]; then
    echo "[INFO] Final file already exists; verifying."
    if verify_file "${FINAL_PATH}"; then
        echo "[DONE] Existing file is valid."
        ls -lh "${FINAL_PATH}"
        exit 0
    fi

    INVALID_PATH="${FINAL_PATH}.invalid.$(date +%Y%m%d_%H%M%S)"
    echo "[WARNING] Existing file failed verification."
    echo "[WARNING] Moving it to ${INVALID_PATH}"
    mv "${FINAL_PATH}" "${INVALID_PATH}"
fi

if [[ -f "${PARTIAL_PATH}" ]]; then
    echo "[INFO] Resuming partial download:"
    ls -lh "${PARTIAL_PATH}"
fi

wget \
    --inet4-only \
    --continue \
    --timeout=120 \
    --read-timeout=900 \
    --tries=50 \
    --waitretry=15 \
    --retry-connrefused \
    --output-document="${PARTIAL_PATH}" \
    "${URL}"

if [[ ! -s "${PARTIAL_PATH}" ]]; then
    echo "[ERROR] Downloaded file is empty: ${PARTIAL_PATH}"
    exit 5
fi

echo "[VERIFY] Checking against the official PhysioNet SHA256."
verify_file "${PARTIAL_PATH}"

mv "${PARTIAL_PATH}" "${FINAL_PATH}"
sync

echo "============================================================"
echo "[DONE] Download and verification completed."
echo "Finished: $(date --iso-8601=seconds)"
ls -lh "${FINAL_PATH}"
du -h "${FINAL_PATH}"
echo "============================================================"
