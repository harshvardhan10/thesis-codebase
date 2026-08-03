#!/bin/bash -l
#
#SBATCH --job-name=extract_albef_itc_gradcam
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/extract_albef_itc_gradcam%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/extract_albef_itc_gradcam%j.err
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

python -u scripts_new/extract_albef_itc_gradcam_heatmaps.py \
  --config configs_new/Pretrain_A0_original_dual_auc.yaml \
  --checkpoint /home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir /home/vault/iwi5/iwi5362h/results/heatmaps_itc/original \
  --device cuda \
  --view_type original \
  --target_labels Cardiomegaly "Pleural effusion" \

echo "Job finished at $(date)"
