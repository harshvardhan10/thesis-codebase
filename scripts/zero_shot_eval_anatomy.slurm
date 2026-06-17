#!/bin/bash -l
#
#SBATCH --job-name=zero_shot_A3_full
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1          # 1 x A40
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/zero_shot_A3_full_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/zero_shot_A3_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

export PYTHONPATH=/home/woody/iwi5/iwi5362h/ALBEF:$PYTHONPATH

python -u scripts/zero_shot_eval_vindr.py \
  --config configs/Pretrain_anatomy_prior_A3.yaml \
  --checkpoints \
    output_anatomy_A3_lamda_0.001/checkpoint_best.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir results/anatomy_prior_screening_full/A3_0.001/zero_shot_A3_with_f1 \
  --save_scores \
  --save_scores_csv

echo "Job finished at $(date)"
