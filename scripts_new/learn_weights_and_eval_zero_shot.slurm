#!/bin/bash -l
#
#SBATCH --job-name=learn_wt_eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/learn_wt_eval_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/learn_wt_eval_%j.err
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

python scripts_new/learn_dual_checkpoint_ensemble_weights.py \
  --model_names original lung heart \
  --best_macro_val_files \
    /home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/best_macro_auc_stable_scores.npz \
    /home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/best_macro_auc_stable_scores.npz \
    /home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/best_macro_auc_stable_scores.npz \
  --best_cardio_val_files \
    /home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/best_cardiomegaly_auc_scores.npz \
    /home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/best_cardiomegaly_auc_scores.npz \
    /home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/best_cardiomegaly_auc_scores.npz \
  --config configs_new/Pretrain_A0_original_dual_auc.yaml \
  --method validation_zscore \
  --weight_step 0.01 \
  --output_dir /home/vault/iwi5/iwi5362h/results/multiview/learned_weight_ensemble_results \
  --best_macro_test_files \
      /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_auc_ckpt_threshold_tuned_val_set/original/vindr_zero_shot_scores_checkpoint_best_macro_auc_stable.npz \
      /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_auc_ckpt_threshold_tuned_val_set/lung/vindr_zero_shot_scores_lung_checkpoint_best_macro_auc_stable.npz \
      /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_auc_ckpt_threshold_tuned_val_set/heart/vindr_zero_shot_scores_heart_checkpoint_best_macro_auc_stable.npz \
  --best_cardio_test_files \
    /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_cardiomegaly_auc_ckpt_threshold_tuned_val_set/original/vindr_zero_shot_scores_checkpoint_best_cardiomegaly_auc.npz \
      /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_cardiomegaly_auc_ckpt_threshold_tuned_val_set/lung/vindr_zero_shot_scores_lung_checkpoint_best_cardiomegaly_auc.npz \
      /home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_cardiomegaly_auc_ckpt_threshold_tuned_val_set/heart/vindr_zero_shot_scores_heart_checkpoint_best_cardiomegaly_auc.npz \

