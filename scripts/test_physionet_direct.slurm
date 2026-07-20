#!/bin/bash -l
#SBATCH --job-name=physionet-test
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/physionet_test_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/physionet_test_%j.err

set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

echo "Running on: $(hostname)"
env | grep -i proxy || true

wget \
  --inet4-only \
  --timeout=60 \
  --tries=1 \
  --server-response \
  --output-document=/tmp/SHA256SUMS.txt \
  "https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/SHA256SUMS.txt"

cat /tmp/SHA256SUMS.txt