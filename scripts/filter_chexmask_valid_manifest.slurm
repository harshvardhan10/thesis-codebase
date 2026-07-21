#!/bin/bash -l
#SBATCH --job-name=chexmask-validate
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_validate_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_validate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u scripts/filter_chexmask_valid_manifest.py \
  --manifest data/mimic_cxr.json \
  --chexmask_db /home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite \
  --output_dir data/chexmask_manifests \
  --image_key image \
  --min_rca_mean 0.0
