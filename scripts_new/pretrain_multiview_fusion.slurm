#!/bin/bash -l
#SBATCH --job-name=albef-mv-pretrain
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/albef_mv_pretrain_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/albef_mv_pretrain_%j.err
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

TRAIN_SCRIPT="${PROJECT_ROOT}/Pretrain_A0_multiview_fusion_dual_auc.py"
CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml"
MODEL_MODULE="${PROJECT_ROOT}/models/model_pretrain_multiview_fusion.py"
VALIDATION_MODULE="${PROJECT_ROOT}/scripts_new/vindr_multiview_classification_validation.py"

ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_fusion_dual_auc"
mkdir -p "${OUTPUT_DIR}"

for path in \
    "${TRAIN_SCRIPT}" \
    "${CONFIG}" \
    "${MODEL_MODULE}" \
    "${VALIDATION_MODULE}" \
    "${ORIGINAL_CHECKPOINT}" \
    "${LUNG_CHECKPOINT}" \
    "${HEART_CHECKPOINT}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 1
    fi
done

if ! grep -q "from scripts_new.vindr_multiview_classification_validation import" "${TRAIN_SCRIPT}"; then
    echo "[ERROR] ${TRAIN_SCRIPT} must import the multi-view validator from scripts_new/." >&2
    exit 1
fi

echo "============================================================"
echo "Multi-view ALBEF FULL PRETRAINING"
echo "Slurm job ID:        ${SLURM_JOB_ID}"
echo "Config:              ${CONFIG}"
echo "Original checkpoint: ${ORIGINAL_CHECKPOINT}"
echo "Lung checkpoint:     ${LUNG_CHECKPOINT}"
echo "Heart checkpoint:    ${HEART_CHECKPOINT}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

nvidia-smi || true

python -u "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --lung_checkpoint "${LUNG_CHECKPOINT}" \
    --heart_checkpoint "${HEART_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --text_encoder bert-base-uncased \
    --device cuda \
    --seed 42 \
    --distributed false \
    --auto_resume true

echo "============================================================"
echo "[Done] Multi-view ALBEF pretraining job finished cleanly."
echo "Best checkpoints should be under: ${OUTPUT_DIR}"
echo "============================================================"
