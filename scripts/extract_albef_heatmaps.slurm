#!/bin/bash -l
#SBATCH --job-name=extract-albef-heatmaps
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1                   # one launcher task
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/extract_albef_heatmaps_%j.out   # stdout log
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/extract_albef_heatmaps_%j.err    # stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de


unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python
conda activate albef-env

# Unbuffered Python so logs flush quickly
export PYTHONUNBUFFERED=1

# Optional: fewer OpenMP threads per process
export OMP_NUM_THREADS=1

# HuggingFace cache (shared for all jobs)
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

python - u extract_albef_cross_attn_gradcam.py \
  --config configs/Pretrain.yaml \
  --checkpoint output_mimic_a40_transformations/checkpoint_29.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir /home/woody/iwi5/iwi5362h/ALBEF/results/zero_shot_vindr_results/heatmaps_vindr

echo "Job finished at $(date)"
