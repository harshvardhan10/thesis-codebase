#!/bin/bash -l
#SBATCH --job-name=vindr-bare-3view
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_bare_3view_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_bare_3view_%A_%a.err
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

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
VAL_LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/checkpoint_validation_2000/multilabel_labels.csv"
VAL_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/train"
VAL_LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/train/lung"
VAL_HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/train/heart"

TEST_LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv"
TEST_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"
TEST_LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
TEST_HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"

OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot_bare_prompt_itc_val_calibrated"

BARE_PROMPT_EVALUATOR="${PROJECT_ROOT}/scripts_new/zero_shot_eval_vindr_pos_neg_bare_prompts.py"
F1_EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_zero_shot_f1_from_validation.py"

BATCH_SIZE=64
NUM_WORKERS=8

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        VIEW_NAME="original"
        CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_original_dual_auc.yaml"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
        ;;
    1)
        VIEW_NAME="lung"
        CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_lung_dual_auc.yaml"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
        ;;
    2)
        VIEW_NAME="heart"
        CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_heart_dual_auc.yaml"
        CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"
        ;;
    *)
        echo "[ERROR] Unknown array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

VIEW_OUTPUT_DIR="${OUTPUT_ROOT}/${VIEW_NAME}"
VAL_OUTPUT_DIR="${VIEW_OUTPUT_DIR}/validation"
TEST_OUTPUT_DIR="${VIEW_OUTPUT_DIR}/test"
FINAL_OUTPUT_DIR="${VIEW_OUTPUT_DIR}/final_metrics"
mkdir -p "${VAL_OUTPUT_DIR}" "${TEST_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}"

for path in \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${BARE_PROMPT_EVALUATOR}" \
    "${F1_EVALUATOR}" \
    "${VAL_LABELS_CSV}" \
    "${TEST_LABELS_CSV}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 1
    fi
done

for path in "${VAL_IMAGES_ROOT}" "${TEST_IMAGES_ROOT}"
do
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Required image directory does not exist: ${path}" >&2
        exit 1
    fi
done

if [[ "${VIEW_NAME}" == "lung" ]]; then
    VAL_MASK_ROOT="${VAL_LUNG_MASK_ROOT}"
    TEST_MASK_ROOT="${TEST_LUNG_MASK_ROOT}"
elif [[ "${VIEW_NAME}" == "heart" ]]; then
    VAL_MASK_ROOT="${VAL_HEART_MASK_ROOT}"
    TEST_MASK_ROOT="${TEST_HEART_MASK_ROOT}"
fi

if [[ "${VIEW_NAME}" != "original" ]]; then
    for path in "${VAL_MASK_ROOT}" "${TEST_MASK_ROOT}"
    do
        if [[ ! -d "${path}" ]]; then
            echo "[ERROR] Required mask directory does not exist: ${path}" >&2
            exit 1
        fi
    done
fi

echo "============================================================"
echo "Slurm job ID:        ${SLURM_JOB_ID}"
echo "Array job ID:        ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Evaluation view:     ${VIEW_NAME}"
echo "Config:              ${CONFIG}"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Validation CSV:      ${VAL_LABELS_CSV}"
echo "Test CSV:            ${TEST_LABELS_CSV}"
echo "Output directory:    ${VIEW_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

run_split() {
    local split_name="$1"
    local labels_csv="$2"
    local images_root="$3"
    local output_dir="$4"
    local mask_root="${5:-}"

    echo "[Run] Generating ${VIEW_NAME} ${split_name} bare-prompt scores"

    local command=(
        python -u "${BARE_PROMPT_EVALUATOR}"
        --config "${CONFIG}"
        --checkpoints "${CHECKPOINT}"
        --labels_csv "${labels_csv}"
        --images_root "${images_root}"
        --view_type "${VIEW_NAME}"
        --output_dir "${output_dir}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --device cuda
    )

    if [[ "${VIEW_NAME}" != "original" ]]; then
        command+=(--mask_root "${mask_root}")
    fi

    "${command[@]}"
}

if [[ "${VIEW_NAME}" == "original" ]]; then
    run_split "validation" "${VAL_LABELS_CSV}" "${VAL_IMAGES_ROOT}" "${VAL_OUTPUT_DIR}"
    run_split "test" "${TEST_LABELS_CSV}" "${TEST_IMAGES_ROOT}" "${TEST_OUTPUT_DIR}"
else
    run_split "validation" "${VAL_LABELS_CSV}" "${VAL_IMAGES_ROOT}" "${VAL_OUTPUT_DIR}" "${VAL_MASK_ROOT}"
    run_split "test" "${TEST_LABELS_CSV}" "${TEST_IMAGES_ROOT}" "${TEST_OUTPUT_DIR}" "${TEST_MASK_ROOT}"
fi

CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pth)"
SCORE_FILENAME="vindr_bare_pair_itc_${VIEW_NAME}_${CHECKPOINT_STEM}_scores.npz"
VALIDATION_SCORES="${VAL_OUTPUT_DIR}/${SCORE_FILENAME}"
TEST_SCORES="${TEST_OUTPUT_DIR}/${SCORE_FILENAME}"

for path in "${VALIDATION_SCORES}" "${TEST_SCORES}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Expected score file was not generated: ${path}" >&2
        exit 1
    fi
done

echo "[Run] Selecting per-label F1 thresholds on validation and applying them to test"
python -u "${F1_EVALUATOR}" \
    --validation_scores "${VALIDATION_SCORES}" \
    --test_scores "${TEST_SCORES}" \
    --config "${CONFIG}" \
    --model_name "${VIEW_NAME}_bare_prompt_itc_best_macro_auc_stable" \
    --output_dir "${FINAL_OUTPUT_DIR}"

echo "[Done] ${VIEW_NAME}: validation scores, test scores, and final test metrics completed."
