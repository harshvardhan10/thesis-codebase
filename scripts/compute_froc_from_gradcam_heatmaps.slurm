#!/bin/bash -l
#
#SBATCH --job-name=froc_from_gradcam_A3_subset
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/froc_from_gradcam_A3_subset_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/froc_from_gradcam_A3_subset_%j.err
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

python -u scripts/compute_froc_from_gradcam_heatmaps.py \
  --heatmaps_dir results/anatomy_prior_screening_15k/A3/heatmaps_crossattn_gradcam_layer8 \
  --annotations_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/annotations_test.csv \
  --meta_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/test_meta.csv \
  --output_dir results/anatomy_prior_screening_15k/A3/froc_gradcam_layer8 \
  --label Cardiomegaly \
  --cam_key cam_vis_up \
  --threshold 0.50 \
  --min_box_area_frac 0.002 \
  --score_mode max \
  --match_mode quadrant \
  --prefix A3

echo "Job finished at $(date)"
