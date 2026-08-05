#!/bin/bash -l
#SBATCH --job-name=biovil_t_grounding
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/biovil_t_grounding_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/biovil_t_grounding_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PYTHONUNBUFFERED=1
export HF_HOME=/home/woody/iwi5/iwi5362h/.cache/huggingface

module load python
conda activate biovilt-clean

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

python -u scripts_new/extract_biovil_t_phrase_grounding.py \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir /home/vault/iwi5/iwi5362h/results/biovil_t_phrase_grounding/diagnostic_50 \
  --target_labels Cardiomegaly "Pleural effusion" \
  --cases_per_label 50 \
  --seed 2026 \
  --device cuda
