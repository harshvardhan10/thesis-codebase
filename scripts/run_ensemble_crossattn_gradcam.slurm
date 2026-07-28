#!/bin/bash -l
#
# Create an equal-weight ensemble of the already-extracted Grad-CAM heatmaps.
#
#SBATCH --job-name=ensemble_crossattn_multiview
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/ensemble_crossattn_multiview_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/ensemble_crossattn_multiview_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail
unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1

PROJECT_ROOT=/home/woody/iwi5/iwi5362h/ALBEF
RESULTS_ROOT="/home/vault/iwi5/iwi5362h/results/multiview/pretrain_loss_last_ckpt"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# These must be the three completed extraction output directories.
ORIGINAL_HEATMAPS="${RESULTS_ROOT}/original/heatmaps_crossattn_gradcam_layer8"
LUNG_HEATMAPS="${RESULTS_ROOT}/lung_only/heatmaps_crossattn_gradcam_layer8"
HEART_HEATMAPS="${RESULTS_ROOT}/heart_only/heatmaps_crossattn_gradcam_layer8"
ENSEMBLE_HEATMAPS="${RESULTS_ROOT}/ensemble_equal/heatmaps_crossattn_gradcam_layer8"

for heatmap_dir in \
    "${ORIGINAL_HEATMAPS}" \
    "${LUNG_HEATMAPS}" \
    "${HEART_HEATMAPS}"; do
    if [[ ! -f "${heatmap_dir}/crossattn_gradcam_index.csv" ]]; then
        echo "ERROR: Missing heatmap index: ${heatmap_dir}/crossattn_gradcam_index.csv" >&2
        exit 1
    fi
done

python -u scripts/ensemble_crossattn_gradcam_heatmaps.py \
    --heatmap_dirs \
        "${ORIGINAL_HEATMAPS}" \
        "${LUNG_HEATMAPS}" \
        "${HEART_HEATMAPS}" \
    --model_names \
        original \
        lung_only \
        heart_only \
    --weights 1 1 1 \
    --output_dir "${ENSEMBLE_HEATMAPS}" \
    --label Cardiomegaly \
    --cam_keys cam_raw cam_vis cam_vis_up \
    --normalization per_model_minmax

test -f "${ENSEMBLE_HEATMAPS}/crossattn_gradcam_index.csv"
test -f "${ENSEMBLE_HEATMAPS}/crossattn_gradcam_ensemble_manifest.json"

echo "Ensemble heatmaps: ${ENSEMBLE_HEATMAPS}"
echo "Job finished at $(date)"
