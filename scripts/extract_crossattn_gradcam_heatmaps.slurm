#!/bin/bash -l
#
#SBATCH --job-name=extract_crossattn_heatmaps_validation_set
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/extract_crossattn_validation_set_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/extract_crossattn_validation_set_%j.err
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

python -u scripts/extract_crossattn_gradcam_heatmaps.py \
  --config configs/Pretrain_A0_original_dual_auc.yaml \
  --checkpoint /home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/checkpoint_validation_2000/multilabel_labels.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/train \
  --output_dir /home/vault/iwi5/iwi5362h/results/multiview/validation/original/heatmaps_crossattn_gradcam_layer8 \
  --layers_to_use 8 \
  --only_labels Cardiomegaly \
  --device cuda

echo "Job finished at $(date)"
