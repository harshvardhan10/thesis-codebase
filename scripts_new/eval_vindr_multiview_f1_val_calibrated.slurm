#!/bin/bash -l
#SBATCH --job-name=vindr-bare-mvfusion
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_bare_mvfusion_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_bare_mvfusion_%j.err
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

# ---------------------------------------------------------------------------
# VinDr validation/test data
# ---------------------------------------------------------------------------
VAL_LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/checkpoint_validation_2000/multilabel_labels.csv"
VAL_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/train"
VAL_LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/train/lung"
VAL_HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/train/heart"

TEST_LABELS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv"
TEST_IMAGES_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test"
TEST_LUNG_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
TEST_HEART_MASK_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"

# ---------------------------------------------------------------------------
# Learned multi-view fusion model
# ---------------------------------------------------------------------------
# Change these two paths only if you store the new config/output elsewhere.
CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml"
CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_fusion_dual_auc/checkpoint_best_cardiomegaly_auc.pth"

# The evaluator below must import:
#   models.model_pretrain_multiview_fusion.ALBEF
MULTIVIEW_EVALUATOR="${PROJECT_ROOT}/scripts_new/zero_shot_eval_vindr_multiview_pos_neg_bare_prompts.py"
F1_EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_zero_shot_f1_from_validation.py"

OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot_bare_prompt_itc_val_calibrated/multiview_fusion/best_cardiomegaly_auc"
VAL_OUTPUT_DIR="${OUTPUT_ROOT}/validation"
TEST_OUTPUT_DIR="${OUTPUT_ROOT}/test"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/final_metrics"
mkdir -p "${VAL_OUTPUT_DIR}" "${TEST_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}"

# Fused inference runs three online ViT branches, so start below the old
# single-view batch size of 64. Increase only after checking A40 memory usage.
BATCH_SIZE=32
NUM_WORKERS=8

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
for path in \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${MULTIVIEW_EVALUATOR}" \
    "${F1_EVALUATOR}" \
    "${VAL_LABELS_CSV}" \
    "${TEST_LABELS_CSV}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 1
    fi
done

for path in \
    "${VAL_IMAGES_ROOT}" \
    "${VAL_LUNG_MASK_ROOT}" \
    "${VAL_HEART_MASK_ROOT}" \
    "${TEST_IMAGES_ROOT}" \
    "${TEST_LUNG_MASK_ROOT}" \
    "${TEST_HEART_MASK_ROOT}"
do
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Required directory does not exist: ${path}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Slurm job ID:          ${SLURM_JOB_ID}"
echo "Model:                 learned multiview fusion"
echo "Config:                ${CONFIG}"
echo "Checkpoint:            ${CHECKPOINT}"
echo "Validation CSV:        ${VAL_LABELS_CSV}"
echo "Test CSV:              ${TEST_LABELS_CSV}"
echo "Validation lung masks: ${VAL_LUNG_MASK_ROOT}"
echo "Validation heart masks:${VAL_HEART_MASK_ROOT}"
echo "Test lung masks:       ${TEST_LUNG_MASK_ROOT}"
echo "Test heart masks:      ${TEST_HEART_MASK_ROOT}"
echo "Output directory:      ${OUTPUT_ROOT}"
echo "Batch size:            ${BATCH_SIZE}"
echo "CUDA_VISIBLE_DEVICES=  ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

run_split() {
    local split_name="$1"
    local labels_csv="$2"
    local images_root="$3"
    local lung_mask_root="$4"
    local heart_mask_root="$5"
    local output_dir="$6"

    echo "[Run] Generating multiview-fusion ${split_name} bare-prompt ITC scores"

    python -u "${MULTIVIEW_EVALUATOR}" \
        --config "${CONFIG}" \
        --checkpoints "${CHECKPOINT}" \
        --labels_csv "${labels_csv}" \
        --images_root "${images_root}" \
        --lung_mask_root "${lung_mask_root}" \
        --heart_mask_root "${heart_mask_root}" \
        --output_dir "${output_dir}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --device cuda
}

# ---------------------------------------------------------------------------
# 1. Generate scores on the fixed 2,000-image VinDr validation set.
# 2. Generate scores independently on VinDr test.
# ---------------------------------------------------------------------------
run_split \
    "validation" \
    "${VAL_LABELS_CSV}" \
    "${VAL_IMAGES_ROOT}" \
    "${VAL_LUNG_MASK_ROOT}" \
    "${VAL_HEART_MASK_ROOT}" \
    "${VAL_OUTPUT_DIR}"

run_split \
    "test" \
    "${TEST_LABELS_CSV}" \
    "${TEST_IMAGES_ROOT}" \
    "${TEST_LUNG_MASK_ROOT}" \
    "${TEST_HEART_MASK_ROOT}" \
    "${TEST_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Locate the exact score files produced by the evaluator.
# ---------------------------------------------------------------------------
CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pth)"
SCORE_FILENAME="vindr_bare_pair_itc_multiview_fusion_${CHECKPOINT_STEM}_scores.npz"
VALIDATION_SCORES="${VAL_OUTPUT_DIR}/${SCORE_FILENAME}"
TEST_SCORES="${TEST_OUTPUT_DIR}/${SCORE_FILENAME}"

for path in "${VALIDATION_SCORES}" "${TEST_SCORES}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Expected score file was not generated: ${path}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# IMPORTANT:
# Threshold fitting happens ONLY on validation_scores.
# The resulting per-label thresholds are then applied unchanged to test_scores.
# ---------------------------------------------------------------------------
echo "[Run] Selecting per-label F1 thresholds on validation and applying them unchanged to test"
python -u "${F1_EVALUATOR}" \
    --validation_scores "${VALIDATION_SCORES}" \
    --test_scores "${TEST_SCORES}" \
    --config "${CONFIG}" \
    --model_name "multiview_fusion_bare_prompt_itc_best_cardiomegaly_auc" \
    --output_dir "${FINAL_OUTPUT_DIR}"

echo "[Done] Fused validation scores, test scores, and validation-calibrated test F1 metrics completed."
