#!/bin/bash -l
#SBATCH --job-name=chexmask-maskfinal
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_maskfinal_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_maskfinal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_valid.json"
MASK_ROOT="/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common"
OUTPUT_DIR="${PROJECT_DIR}/data/chexmask_manifests"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u scripts/finalize_chexmask_mask_manifests.py \
    --manifest "${MANIFEST}" \
    --mask_root "${MASK_ROOT}" \
    --staging_dir "${MASK_ROOT}/_staging" \
    --output_dir "${OUTPUT_DIR}" \
    --num_shards 4 \
    --image_key image \
    --verify_all

du -sh "${MASK_ROOT}/lung" "${MASK_ROOT}/heart"
