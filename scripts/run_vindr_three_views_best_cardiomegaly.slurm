#!/bin/bash -l
#SBATCH --job-name=vindr-3view-cardio-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_cardio_eval_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_cardio_eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

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

LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"
IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"

LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"

OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_cardiomegaly_auc_ckpt_threshold_tuned_val_set"

BATCH_SIZE=64
NUM_WORKERS=8

# Used only by the existing inference scripts. Final F1 below uses thresholds
# selected from the corresponding 2,000-image validation scores.
LEGACY_THRESHOLD=0.3948

F1_EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_zero_shot_f1_from_validation.py"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        VIEW_NAME="original"
        CONFIG="${PROJECT_ROOT}/configs/Pretrain_A0_original_dual_auc.yaml"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr.py"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_cardiomegaly_auc.pth"
        VALIDATION_SCORES="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/best_cardiomegaly_auc_scores.npz"
        OUTPUT_DIR="${OUTPUT_ROOT}/original"
        ;;

    1)
        VIEW_NAME="lung"
        CONFIG="${PROJECT_ROOT}/configs/Pretrain_A0_lung_dual_auc.yaml"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr_maskcache.py"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_cardiomegaly_auc.pth"
        VALIDATION_SCORES="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/best_cardiomegaly_auc_scores.npz"
        MASK_ROOT="${LUNG_MASK_ROOT}"
        OUTPUT_DIR="${OUTPUT_ROOT}/lung"
        ;;

    2)
        VIEW_NAME="heart"
        CONFIG="${PROJECT_ROOT}/configs/Pretrain_A0_heart_dual_auc.yaml"
        SCRIPT="${PROJECT_ROOT}/scripts/zero_shot_eval_vindr_maskcache.py"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_cardiomegaly_auc.pth"
        VALIDATION_SCORES="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/best_cardiomegaly_auc_scores.npz"
        MASK_ROOT="${HEART_MASK_ROOT}"
        OUTPUT_DIR="${OUTPUT_ROOT}/heart"
        ;;

    *)
        echo "[ERROR] Unknown array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

for path in \
    "${CONFIG}" \
    "${LABELS_CSV}" \
    "${SCRIPT}" \
    "${CHECKPOINT}" \
    "${VALIDATION_SCORES}" \
    "${F1_EVALUATOR}"
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

if [[ "${VIEW_NAME}" != "original" && ! -d "${MASK_ROOT}" ]]; then
    echo "[ERROR] Mask directory does not exist: ${MASK_ROOT}" >&2
    exit 1
fi

echo "============================================================"
echo "Slurm job ID:        ${SLURM_JOB_ID}"
echo "Array job ID:        ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Evaluation view:     ${VIEW_NAME}"
echo "Config:              ${CONFIG}"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Validation scores:   ${VALIDATION_SCORES}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

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
        --threshold "${LEGACY_THRESHOLD}" \
        --save_scores
else
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
        --threshold "${LEGACY_THRESHOLD}" \
        --save_scores
fi

# Locate the NPZ without assuming whether the masked evaluator adds
# "lung_" or "heart_" to its filename.
CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pth)"

mapfile -t TEST_SCORE_CANDIDATES < <(
    find "${OUTPUT_DIR}" -maxdepth 1 -type f \
        -name "*${CHECKPOINT_STEM}.npz" -print | sort
)

if [[ "${#TEST_SCORE_CANDIDATES[@]}" -ne 1 ]]; then
    echo "[ERROR] Expected exactly one test-score NPZ for ${CHECKPOINT_STEM}" >&2
    printf '[ERROR] Candidate: %s\n' \
        "${TEST_SCORE_CANDIDATES[@]:-none}" >&2
    exit 1
fi

TEST_SCORES="${TEST_SCORE_CANDIDATES[0]}"

# Produce reportable F1 using thresholds selected on validation, not test.
python -u "${F1_EVALUATOR}" \
    --validation_scores "${VALIDATION_SCORES}" \
    --test_scores "${TEST_SCORES}" \
    --config "${CONFIG}" \
    --model_name "${VIEW_NAME}_best_cardiomegaly_auc" \
    --output_dir "${OUTPUT_DIR}"

echo "[Done] ${VIEW_NAME} zero-shot evaluation and final metrics completed."