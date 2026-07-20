#!/bin/bash -l
#SBATCH --job-name=chexmask-mimic
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_mimic_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_mimic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de


set -euo pipefail

VERSION="1.0.0"
BASE_URL="https://physionet.org/files/chexmask-cxr-segmentation-data/${VERSION}"
CHEXMASK_ROOT="${CHEXMASK_ROOT:-/home/vault/iwi5/iwi5362h/data/chexmask}"
OUTPUT_DIR="${CHEXMASK_ROOT}/OriginalResolution"

mkdir -p "${OUTPUT_DIR}"

download_resumable() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl \
            --location \
            --fail \
            --show-error \
            --retry 30 \
            --retry-delay 10 \
            --connect-timeout 30 \
            --continue-at - \
            --output "${output}" \
            "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --continue \
            --tries=50 \
            --timeout=60 \
            --waitretry=10 \
            --retry-connrefused \
            --output-document="${output}" \
            "${url}"
    else
        echo "[ERROR] Neither curl nor wget is available." >&2
        exit 1
    fi
}

download_chexmask_file() {
    local filename="$1"
    local url="${BASE_URL}/OriginalResolution/${filename}"
    local destination="${OUTPUT_DIR}/${filename}"
    local partial="${destination}.part"
    local lock_file="${destination}.lock"

    exec 9>"${lock_file}"
    if ! flock -n 9; then
        echo "[ERROR] Another process is already downloading ${filename}."
        exit 2
    fi

    echo "============================================================"
    echo "CheXmask version : ${VERSION}"
    echo "File             : ${filename}"
    echo "URL              : ${url}"
    echo "Destination      : ${destination}"
    echo "Host             : $(hostname)"
    echo "Started          : $(date --iso-8601=seconds)"
    echo "============================================================"

    if [[ -f "${destination}" ]]; then
        echo "[INFO] File already exists:"
        ls -lh "${destination}"
        echo "[INFO] Delete it manually if you want to download it again."
        exit 0
    fi

    if [[ -f "${partial}" ]]; then
        echo "[INFO] Resuming partial download:"
        ls -lh "${partial}"
    fi

    download_resumable "${url}" "${partial}"

    if [[ ! -s "${partial}" ]]; then
        echo "[ERROR] Downloaded file is empty: ${partial}" >&2
        exit 3
    fi

    mv "${partial}" "${destination}"
    sync

    echo "============================================================"
    echo "[DONE] Download completed."
    echo "Finished         : $(date --iso-8601=seconds)"
    ls -lh "${destination}"
    du -h "${destination}"
    echo "============================================================"
}


download_chexmask_file "MIMIC-CXR-JPG.csv"
