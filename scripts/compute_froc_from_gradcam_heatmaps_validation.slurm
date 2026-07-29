#!/bin/bash -l
#
#SBATCH --job-name=froc_from_heatmap_validation
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/froc_from_heatmap_validation_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/froc_from_heatmap_validation_%j.err
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

python -u scripts/compute_froc_from_gradcam_heatmaps_validation.py \
  --mode validation \
  --heatmaps_dir /home/vault/iwi5/iwi5362h/results/multiview/validation/original/heatmaps_crossattn_gradcam_layer8 \
  --evaluation_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/checkpoint_validation_2000/multilabel_labels.csv \
  --annotations_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/annotations_train.csv \
  --output_dir /home/vault/iwi5/iwi5362h/results/A0_validation_froc_search \
  --label "Cardiomegaly" \
  --expected_view original \
  --expected_layer 8 \
  --expected_num_images 2000 \
  --thresholds "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90" \
  --min_area_patches_grid "0,1,2,4" \
  --score_mode sum_sqrt_area \
  --match_mode quadrant \
  --froc_targets "0.10,0.25,0.50" \
  --prefix A0_validation

echo "Job finished at $(date)"
