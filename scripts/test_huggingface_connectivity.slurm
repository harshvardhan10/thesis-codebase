#!/bin/bash -l
#SBATCH --job-name=hf-nettest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/hf_nettest_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/hf_nettest_%j.err

set -u

PROXY="http://proxy.nhr.fau.de:80"

# Small regular Hugging Face file.
SMALL_URL="https://huggingface.co/bert-base-uncased/resolve/main/config.json"

# Large public model file. Only the first 1 MiB is requested.
# Following redirects tests the storage/CDN/Xet route used for large files.
LARGE_URL="https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"

TMP_ROOT="/tmp/hf_nettest_${SLURM_JOB_ID}"
mkdir -p "${TMP_ROOT}"

declare -a TEST_NAMES=()
declare -a TEST_CODES=()

record_result() {
    local name="$1"
    local code="$2"
    TEST_NAMES+=("${name}")
    TEST_CODES+=("${code}")
    echo "[RESULT] ${name}: exit_code=${code}"
    echo
}

clear_proxy() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    unset all_proxy ALL_PROXY no_proxy NO_PROXY
}

set_proxy() {
    export http_proxy="${PROXY}"
    export https_proxy="${PROXY}"
    export HTTP_PROXY="${PROXY}"
    export HTTPS_PROXY="${PROXY}"
    unset all_proxy ALL_PROXY no_proxy NO_PROXY
}

echo "============================================================"
echo "Hugging Face connectivity test"
echo "Host:       $(hostname)"
echo "Started:    $(date --iso-8601=seconds)"
echo "Proxy:      ${PROXY}"
echo "Small URL:  ${SMALL_URL}"
echo "Large URL:  ${LARGE_URL}"
echo "Temp dir:   ${TMP_ROOT}"
echo "============================================================"
echo

echo "[DNS]"
getent ahostsv4 huggingface.co || true
getent ahostsv4 proxy.nhr.fau.de || true
echo

# ---------------------------------------------------------------------------
# Test 1: direct small-file download with wget
# ---------------------------------------------------------------------------
clear_proxy
echo "[TEST 1] Direct wget: small Hugging Face file"
env | grep -i proxy || true

wget \
    --inet4-only \
    --server-response \
    --timeout=45 \
    --tries=1 \
    --output-document="${TMP_ROOT}/direct_config.json" \
    "${SMALL_URL}"
code=$?
record_result "direct_wget_small" "${code}"

if [[ -s "${TMP_ROOT}/direct_config.json" ]]; then
    ls -lh "${TMP_ROOT}/direct_config.json"
    head -c 200 "${TMP_ROOT}/direct_config.json"
    echo
fi

# ---------------------------------------------------------------------------
# Test 2: proxy small-file download with wget
# ---------------------------------------------------------------------------
set_proxy
echo "[TEST 2] Proxy wget: small Hugging Face file"
env | grep -i proxy || true

wget \
    --inet4-only \
    --server-response \
    --timeout=45 \
    --tries=1 \
    --output-document="${TMP_ROOT}/proxy_config.json" \
    "${SMALL_URL}"
code=$?
record_result "proxy_wget_small" "${code}"

if [[ -s "${TMP_ROOT}/proxy_config.json" ]]; then
    ls -lh "${TMP_ROOT}/proxy_config.json"
    head -c 200 "${TMP_ROOT}/proxy_config.json"
    echo
fi

# ---------------------------------------------------------------------------
# Test 3: direct ranged large-file request with curl
# ---------------------------------------------------------------------------
clear_proxy
echo "[TEST 3] Direct curl: first 1 MiB of a large Hugging Face file"

curl -4 \
    --verbose \
    --location \
    --fail \
    --show-error \
    --range 0-1048575 \
    --connect-timeout 45 \
    --max-time 180 \
    --output "${TMP_ROOT}/direct_large.part" \
    "${LARGE_URL}"
code=$?
record_result "direct_curl_large_range" "${code}"

if [[ -s "${TMP_ROOT}/direct_large.part" ]]; then
    ls -lh "${TMP_ROOT}/direct_large.part"
fi

# ---------------------------------------------------------------------------
# Test 4: proxy ranged large-file request with curl
# ---------------------------------------------------------------------------
set_proxy
echo "[TEST 4] Proxy curl: first 1 MiB of a large Hugging Face file"

curl -4 \
    --verbose \
    --proxy "${PROXY}" \
    --location \
    --fail \
    --show-error \
    --range 0-1048575 \
    --connect-timeout 45 \
    --max-time 180 \
    --output "${TMP_ROOT}/proxy_large.part" \
    "${LARGE_URL}"
code=$?
record_result "proxy_curl_large_range" "${code}"

if [[ -s "${TMP_ROOT}/proxy_large.part" ]]; then
    ls -lh "${TMP_ROOT}/proxy_large.part"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "SUMMARY"
for i in "${!TEST_NAMES[@]}"; do
    printf "%-32s %s\n" "${TEST_NAMES[$i]}" "${TEST_CODES[$i]}"
done
echo "============================================================"
echo
echo "Interpretation:"
echo "  exit 0  = request succeeded"
echo "  exit 4  = wget network failure"
echo "  exit 5  = wget SSL verification failure"
echo "  exit 22 = curl HTTP error such as 403/404"
echo "  exit 28 = curl timeout"
echo
echo "A small-file success but large-range failure usually means that"
echo "huggingface.co is reachable but the redirected CDN/Xet host is blocked."

# Succeed if any Hugging Face route worked; otherwise fail the Slurm job.
for code in "${TEST_CODES[@]}"; do
    if [[ "${code}" -eq 0 ]]; then
        exit 0
    fi
done

exit 1
