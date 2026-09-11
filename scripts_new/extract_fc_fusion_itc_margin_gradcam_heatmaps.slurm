#!/bin/bash -l
#SBATCH --job-name=fc-fusion-gradcam
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/fc_fusion_gradcam_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/fc_fusion_gradcam_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

PROJECT_ROOT="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_ROOT}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

SCRIPT="${PROJECT_ROOT}/scripts_new/extract_fc_fusion_itc_margin_gradcam_heatmaps.py"

CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml"
CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_fusion_dual_auc/checkpoint_best_cardiomegaly_auc.pth"

LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv"
IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"
LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/fc_fusion_itc_margin_gradcam_test_best_cardio"

mkdir -p "${OUTPUT_DIR}"

for path in \
    "${SCRIPT}" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${LABELS_CSV}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required file: ${path}" >&2
        exit 1
    fi
done

for path in "${IMAGES_ROOT}" "${LUNG_MASK_ROOT}" "${HEART_MASK_ROOT}"
do
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Missing required directory: ${path}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "FC-FUSION ITC-MARGIN GRAD-CAM EXTRACTION"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Node:                $(hostname)"
echo "Config:              ${CONFIG}"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Labels CSV:          ${LABELS_CSV}"
echo "Images root:         ${IMAGES_ROOT}"
echo "Lung mask root:      ${LUNG_MASK_ROOT}"
echo "Heart mask root:     ${HEART_MASK_ROOT}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

nvidia-smi || true

python -u "${SCRIPT}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --labels_csv "${LABELS_CSV}" \
    --images_root "${IMAGES_ROOT}" \
    --lung_mask_root "${LUNG_MASK_ROOT}" \
    --heart_mask_root "${HEART_MASK_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_labels Cardiomegaly "Pleural Effusion" \
    --vit_layer -2 \
    --device cuda

echo "[Done] FC-fusion Grad-CAM extraction finished."