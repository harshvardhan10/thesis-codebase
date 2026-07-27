#!/bin/bash -l
#
#SBATCH --job-name=vindr-3view-ensemble
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_ensemble_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_ensemble_%A_%a.err
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

python scripts/ensemble_zero_shot_scores.py \
    --score_files \
        /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/val_auc_ckpt/original/vindr_zero_shot_scores_checkpoint_best_macro_auc_stable.npz \
        /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/val_auc_ckpt/lung/vindr_zero_shot_scores_lung_checkpoint_best_macro_auc_stable.npz \
        /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/val_auc_ckpt/heart/vindr_zero_shot_scores_heart_checkpoint_best_macro_auc_stable.npz \
    --model_names original lung heart \
    --output_dir /home/vault/iwi5/iwi5362h/outputs/ensemble/stable \
    --ensemble_name stable \
    --threshold 0.3948 \
    --save_scores_csv

echo "Job finished at $(date)"
