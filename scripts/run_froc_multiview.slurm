#!/bin/bash -l
#
# Evaluate original, lung-only, heart-only, and ensemble heatmaps independently.
#
#SBATCH --job-name=froc_crossattn_multiview
#SBATCH --array=0-3
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/froc_crossattn_multiview_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/froc_crossattn_multiview_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail
unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1

PROJECT_ROOT=/home/woody/iwi5/iwi5362h/ALBEF
RESULTS_ROOT="/home/vault/iwi5/iwi5362h/results/multiview/best_val_auc"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_NAMES=(
    "original"
    "lung_only"
    "heart_only"
    "ensemble_equal"
)

HEATMAP_DIRS=(
    "${RESULTS_ROOT}/original/heatmaps_crossattn_gradcam_layer8"
    "${RESULTS_ROOT}/lung_only/heatmaps_crossattn_gradcam_layer8"
    "${RESULTS_ROOT}/heart_only/heatmaps_crossattn_gradcam_layer8"
    "${RESULTS_ROOT}/ensemble_equal/heatmaps_crossattn_gradcam_layer8"
)

MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
HEATMAP_DIR="${HEATMAP_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${RESULTS_ROOT}/froc_comparison/${MODEL_NAME}"

ANNOTATIONS_CSV=/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/annotations_test.csv
META_CSV=/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv

if [[ ! -f "${HEATMAP_DIR}/crossattn_gradcam_index.csv" ]]; then
    echo "ERROR: Missing heatmap index: ${HEATMAP_DIR}/crossattn_gradcam_index.csv" >&2
    exit 1
fi

echo "Model: ${MODEL_NAME}"
echo "Heatmaps: ${HEATMAP_DIR}"
echo "FROC output: ${OUTPUT_DIR}"

python -u scripts/compute_froc_from_gradcam_heatmaps.py \
    --heatmaps_dir "${HEATMAP_DIR}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --meta_csv "${META_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --label Cardiomegaly \
    --cam_key cam_vis_up \
    --threshold 0.50 \
    --min_box_area_frac 0.002 \
    --score_mode max \
    --match_mode quadrant \
    --froc_targets 0.10,0.25,0.50 \
    --prefix "${MODEL_NAME}"

echo "FROC evaluation completed for ${MODEL_NAME}"
echo "Job finished at $(date)"
