#!/bin/bash -l
#SBATCH --job-name=evaluate-froc-vindr-gated
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/evaluate_froc_vindr_gated_%j.out   # stdout log
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/evaluate_froc_vindr_gated_%j.err    # stderr log
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

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

export PYTHONPATH=/home/woody/iwi5/iwi5362h/ALBEF:$PYTHONPATH

python -u scripts/evaluate_froc_vindr_gated.py \
  --heatmaps_dir results/zero_shot_vindr_results/heatmaps_vindr_raw_vis \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --ann_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/annotations_test.csv \
  --meta_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/test_meta.csv \
  --output_dir /home/woody/iwi5/iwi5362h/ALBEF/results/zero_shot_vindr_results/froc/froc_top_10_labels_score_analysis \
  --scores_npz /home/woody/iwi5/iwi5362h/ALBEF/results/zero_shot_vindr_results/classification/vindr_zero_shot_scores_checkpoint_29.npz \
  --thr_min 0.35 \
  --thr_max 0.95 \
  --n_thr 14 \
  --top_k_labels 10 \
  --top_k_boxes 1 \
  --score_fusion mul

echo "Job finished at $(date)"