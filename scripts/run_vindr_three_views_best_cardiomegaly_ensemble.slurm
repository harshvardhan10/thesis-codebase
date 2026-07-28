#!/bin/bash -l
#SBATCH --job-name=vindr-3view-cardio-ensemble
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_cardio_ensemble_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_cardio_ensemble_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

PROJECT_ROOT="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_ROOT}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ENSEMBLE_SCRIPT="${PROJECT_ROOT}/scripts/ensemble_zero_shot_scores.py"
CONFIG="${PROJECT_ROOT}/configs/Pretrain_A0_original_dual_auc.yaml"

INDIVIDUAL_ROOT="/home/vault/iwi5/iwi5362h/outputs/vindr_zero_shot/best_val_auc_ckpt_threshold_tuned_val_set"
OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/outputs/ensemble/best_val_auc_ckpt_threshold_tuned_val_set"

ORIGINAL_VALIDATION="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/best_macro_auc_stable_scores.npz"
LUNG_VALIDATION="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/best_macro_auc_stable_scores.npz"
HEART_VALIDATION="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/best_macro_auc_stable_scores.npz"

CHECKPOINT_STEM="checkpoint_best_macro_auc_stable"

# Resolve score filenames without assuming whether the masked evaluator inserts
# "lung_" or "heart_" into its filename.
resolve_test_scores() {
    local view_dir="$1"
    local -a candidates=()

    mapfile -t candidates < <(
        find "${view_dir}" -maxdepth 1 -type f \
            -name "*${CHECKPOINT_STEM}.npz" -print | sort
    )

    if [[ "${#candidates[@]}" -ne 1 ]]; then
        echo "[ERROR] Expected exactly one test-score NPZ in ${view_dir}" >&2
        printf '[ERROR] Candidate: %s\n' "${candidates[@]:-none}" >&2
        return 1
    fi

    printf '%s\n' "${candidates[0]}"
}

ORIGINAL_TEST="$(resolve_test_scores "${INDIVIDUAL_ROOT}/original")"
LUNG_TEST="$(resolve_test_scores "${INDIVIDUAL_ROOT}/lung")"
HEART_TEST="$(resolve_test_scores "${INDIVIDUAL_ROOT}/heart")"

for path in \
    "${ENSEMBLE_SCRIPT}" \
    "${CONFIG}" \
    "${ORIGINAL_VALIDATION}" \
    "${LUNG_VALIDATION}" \
    "${HEART_VALIDATION}" \
    "${ORIGINAL_TEST}" \
    "${LUNG_TEST}" \
    "${HEART_TEST}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Slurm job ID:       ${SLURM_JOB_ID}"
echo "Original test:      ${ORIGINAL_TEST}"
echo "Lung test:          ${LUNG_TEST}"
echo "Heart test:         ${HEART_TEST}"
echo "Output root:        ${OUTPUT_ROOT}"
echo "============================================================"

# Primary ensemble: standardize every model and label using validation-set
# statistics, then average the aligned scores.
python -u "${ENSEMBLE_SCRIPT}" \
    --validation_score_files \
        "${ORIGINAL_VALIDATION}" \
        "${LUNG_VALIDATION}" \
        "${HEART_VALIDATION}" \
    --test_score_files \
        "${ORIGINAL_TEST}" \
        "${LUNG_TEST}" \
        "${HEART_TEST}" \
    --model_names original lung heart \
    --method validation_zscore \
    --config "${CONFIG}" \
    --ensemble_name best_cardiomegaly_auc_validation_zscore \
    --output_dir "${OUTPUT_ROOT}/validation_zscore" \
    --save_scores_csv

# Raw-score mean retained as a directly comparable baseline. Its F1 thresholds
# are still selected from the validation ensemble and transferred to test.
python -u "${ENSEMBLE_SCRIPT}" \
    --validation_score_files \
        "${ORIGINAL_VALIDATION}" \
        "${LUNG_VALIDATION}" \
        "${HEART_VALIDATION}" \
    --test_score_files \
        "${ORIGINAL_TEST}" \
        "${LUNG_TEST}" \
        "${HEART_TEST}" \
    --model_names original lung heart \
    --method raw_mean \
    --config "${CONFIG}" \
    --ensemble_name best_cardiomegaly_auc_raw_mean \
    --output_dir "${OUTPUT_ROOT}/raw_mean" \
    --save_scores_csv

echo "[Done] Validation-standardized and raw-mean ensembles completed."