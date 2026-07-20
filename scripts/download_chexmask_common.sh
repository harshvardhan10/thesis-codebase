#!/bin/bash
set -euo pipefail

VERSION="1.0.0"
BASE_URL="https://physionet.org/files/chexmask-cxr-segmentation-data/${VERSION}"
CHECKSUMS_URL="${BASE_URL}/SHA256SUMS.txt"
CHEXMASK_ROOT="${CHEXMASK_ROOT:-/home/vault/iwi5/iwi5362h/data/chexmask/${VERSION}}"
OUTPUT_DIR="${CHEXMASK_ROOT}/OriginalResolution"
mkdir -p "${OUTPUT_DIR}"

download_resumable() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --location --fail --show-error \
            --retry 30 --retry-delay 10 --connect-timeout 30 \
            --continue-at - --output "${output}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=50 --timeout=60 --waitretry=10 \
            --retry-connrefused --output-document="${output}" "${url}"
    else
        echo "[ERROR] Neither curl nor wget is available." >&2
        exit 1
    fi
}

download_small_file() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --location --fail --show-error --silent \
            --retry 10 --retry-delay 5 --output "${output}" "${url}"
    else
        wget --quiet --tries=10 --output-document="${output}" "${url}"
    fi
}

verify_sha256() {
    local file_path="$1"
    local relative_path="$2"
    local checksum_file="$3"
    local expected
    expected="$(
        grep -F "${relative_path}" "${checksum_file}" \
        | head -n 1 \
        | awk '{print $1}'
    )"
    if [[ -z "${expected}" ]]; then
        echo "[ERROR] No SHA256 entry found for ${relative_path}" >&2
        grep -F "$(basename "${relative_path}")" "${checksum_file}" >&2 || true
        return 1
    fi
    echo "${expected}  ${file_path}" | sha256sum --check -
}

download_chexmask_file() {
    local filename="$1"
    local relative_path="OriginalResolution/${filename}"
    local url="${BASE_URL}/${relative_path}"
    local destination="${OUTPUT_DIR}/${filename}"
    local partial="${destination}.part"
    local lock_file="${destination}.lock"
    local checksum_file

    exec 9>"${lock_file}"
    if ! flock -n 9; then
        echo "[ERROR] Another process is already downloading ${filename}."
        exit 2
    fi

    checksum_file="$(mktemp)"
    trap 'rm -f "${checksum_file}"' EXIT
    download_small_file "${CHECKSUMS_URL}" "${checksum_file}"

    echo "============================================================"
    echo "CheXmask version : ${VERSION}"
    echo "File             : ${filename}"
    echo "URL              : ${url}"
    echo "Destination      : ${destination}"
    echo "Host             : $(hostname)"
    echo "Started          : $(date --iso-8601=seconds)"
    echo "============================================================"

    if [[ -f "${destination}" ]]; then
        echo "[INFO] Existing destination found; verifying."
        if verify_sha256 "${destination}" "${relative_path}" "${checksum_file}"; then
            echo "[DONE] Existing file is valid."
            ls -lh "${destination}"
            exit 0
        fi
        invalid="${destination}.invalid.$(date +%Y%m%d_%H%M%S)"
        echo "[WARNING] Existing file failed checksum; moving to ${invalid}"
        mv "${destination}" "${invalid}"
    fi

    if [[ -f "${partial}" ]]; then
        echo "[INFO] Resuming partial file:"
        ls -lh "${partial}"
    fi

    download_resumable "${url}" "${partial}"

    echo "[INFO] Verifying downloaded file."
    verify_sha256 "${partial}" "${relative_path}" "${checksum_file}"

    mv "${partial}" "${destination}"
    sync

    echo "============================================================"
    echo "[DONE] Download and checksum verification completed."
    echo "Finished         : $(date --iso-8601=seconds)"
    ls -lh "${destination}"
    du -h "${destination}"
    echo "============================================================"
}
