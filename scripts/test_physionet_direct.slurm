#!/bin/bash -l
#SBATCH --job-name=physionet-proxy-test
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/physionet_proxy_test_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/physionet_proxy_test_%j.err

set -u

PROXY="http://proxy.nhr.fau.de:80"

export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

# Prevent unrelated proxy variables from interfering.
unset all_proxy ALL_PROXY no_proxy NO_PROXY

URL="https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/SHA256SUMS.txt"
WGET_OUT="/tmp/chexmask_sha256_wget_${SLURM_JOB_ID}.txt"
CURL_OUT="/tmp/chexmask_sha256_curl_${SLURM_JOB_ID}.txt"

echo "============================================================"
echo "Host:     $(hostname)"
echo "Date:     $(date --iso-8601=seconds)"
echo "Proxy:    ${PROXY}"
echo "URL:      ${URL}"
echo "============================================================"

echo "[Environment]"
env | grep -i proxy || true

echo
echo "[DNS: proxy IPv4]"
getent ahostsv4 proxy.nhr.fau.de || true

echo
echo "[TEST 1] GitHub through proxy"
wget \
  --inet4-only \
  --timeout=30 \
  --tries=1 \
  --output-document=/dev/null \
  "https://github.com/"
GITHUB_STATUS=$?
echo "[TEST 1] exit_code=${GITHUB_STATUS}"

echo
echo "[TEST 2] PhysioNet checksum through proxy with wget"
wget \
  --inet4-only \
  --server-response \
  --debug \
  --timeout=60 \
  --tries=1 \
  --output-document="${WGET_OUT}" \
  "${URL}"
WGET_STATUS=$?
echo "[TEST 2] exit_code=${WGET_STATUS}"

if [[ -s "${WGET_OUT}" ]]; then
    echo "[TEST 2] downloaded file:"
    ls -lh "${WGET_OUT}"
    cat "${WGET_OUT}"
else
    echo "[TEST 2] no data downloaded"
fi

echo
echo "[TEST 3] PhysioNet checksum through proxy with curl"
curl -4 \
  --verbose \
  --proxy "${PROXY}" \
  --location \
  --fail \
  --show-error \
  --connect-timeout 60 \
  --max-time 120 \
  --output "${CURL_OUT}" \
  "${URL}"
CURL_STATUS=$?
echo "[TEST 3] exit_code=${CURL_STATUS}"

if [[ -s "${CURL_OUT}" ]]; then
    echo "[TEST 3] downloaded file:"
    ls -lh "${CURL_OUT}"
    cat "${CURL_OUT}"
else
    echo "[TEST 3] no data downloaded"
fi

echo
echo "============================================================"
echo "Summary"
echo "GitHub wget exit:    ${GITHUB_STATUS}"
echo "PhysioNet wget exit: ${WGET_STATUS}"
echo "PhysioNet curl exit: ${CURL_STATUS}"
echo "============================================================"

# Return success only if at least one PhysioNet request succeeded.
if [[ ${WGET_STATUS} -eq 0 || ${CURL_STATUS} -eq 0 ]]; then
    exit 0
fi

exit 1
