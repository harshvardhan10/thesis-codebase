#!/bin/bash -l
#SBATCH --job-name=inspect-chexmask
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/inspect_chexmask_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/inspect_chexmask_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u scripts/inspect_chexmask_excluded.py \
  --manifest data/mimic_cxr.json \
  --excluded_tsv data/chexmask_manifests/mimic_cxr_chexmask_excluded.tsv \
  --output_dir previews/chexmask_excluded \
  --image_key image \
  --caption_key caption \
  --panel_size 768 \
  --sheet_columns 2 \
  --sheet_rows 3 \
  --sheet_thumb_width 900
