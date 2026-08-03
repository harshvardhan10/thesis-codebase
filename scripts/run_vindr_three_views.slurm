#!/bin/bash -l
#SBATCH --job-name=vindr-3view-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_eval_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ============================================================
# Environment
# ============================================================

PROJECT_ROOT="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_ROOT}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Shared paths and settings
# ============================================================

CONFIG="${PROJECT_ROOT}/configs/Pretrain_A0_original_dual_auc.yaml"

LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"
IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"

LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"

OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_cardiomegaly_auc_ckpt"

BATCH_SIZE=64
NUM_WORKERS=8
THRESHOLD=0.3948

# ============================================================
# Replace these checkpoint paths
# ============================================================

ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_cardiomegaly_auc.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_cardiomegaly_auc.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_cardiomegaly_auc.pth"

# ============================================================
# Select evaluation from array index
# ============================================================

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        VIEW_NAME="original"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr.py"
        CHECKPOINT="${ORIGINAL_CHECKPOINT}"
        OUTPUT_DIR="${OUTPUT_ROOT}/original"
        ;;

    1)
        VIEW_NAME="lung"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr_maskcache.py"
        CHECKPOINT="${LUNG_CHECKPOINT}"
        MASK_ROOT="${LUNG_MASK_ROOT}"
        OUTPUT_DIR="${OUTPUT_ROOT}/lung"
        ;;

    2)
        VIEW_NAME="heart"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr_maskcache.py"
        CHECKPOINT="${HEART_CHECKPOINT}"
        MASK_ROOT="${HEART_MASK_ROOT}"
        OUTPUT_DIR="${OUTPUT_ROOT}/heart"
        ;;

    *)
        echo "[ERROR] Unknown array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

# ============================================================
# Validate common inputs
# ============================================================

for path in \
    "${CONFIG}" \
    "${LABELS_CSV}" \
    "${SCRIPT}" \
    "${CHECKPOINT}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ ! -d "${IMAGES_ROOT}" ]]; then
    echo "[ERROR] Images directory does not exist: ${IMAGES_ROOT}" >&2
    exit 1
fi

echo "============================================================"
echo "Slurm job ID:       ${SLURM_JOB_ID}"
echo "Array job ID:       ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:      ${SLURM_ARRAY_TASK_ID}"
echo "Evaluation view:    ${VIEW_NAME}"
echo "Checkpoint:         ${CHECKPOINT}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

# ============================================================
# Run original or masked evaluation
# ============================================================

if [[ "${VIEW_NAME}" == "original" ]]; then

    python -u "${SCRIPT}" \
        --config "${CONFIG}" \
        --checkpoints "${CHECKPOINT}" \
        --labels_csv "${LABELS_CSV}" \
        --images_root "${IMAGES_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --device cuda \
        --threshold "${THRESHOLD}" \
        --save_scores

else

    if [[ ! -d "${MASK_ROOT}" ]]; then
        echo "[ERROR] Mask directory does not exist: ${MASK_ROOT}" >&2
        exit 1
    fi

    python -u "${SCRIPT}" \
        --config "${CONFIG}" \
        --checkpoints "${CHECKPOINT}" \
        --labels_csv "${LABELS_CSV}" \
        --images_root "${IMAGES_ROOT}" \
        --view_type "${VIEW_NAME}" \
        --mask_root "${MASK_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --device cuda \
        --threshold "${THRESHOLD}" \
        --save_scores

fi

echo "[Done] ${VIEW_NAME} evaluation completed successfully."