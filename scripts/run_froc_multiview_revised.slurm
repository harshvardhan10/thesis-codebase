#!/bin/bash -l
#
# Evaluate original, lung-only, heart-only, and ensemble heatmaps independently.
#
#SBATCH --job-name=froc_crossattn_multiview
#SBATCH --array=0-3
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
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
OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/results/multiview/best_val_auc"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCRIPT="scripts/compute_froc_from_gradcam_heatmaps_revised.py"

EVALUATION_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv"
ANNOTATIONS_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/annotations_test.csv"
META_CSV="/home/woody/iwi5/iwi5362h/data/vindr_cxr/test_meta.csv"

LABEL="Cardiomegaly"
EXPECTED_NUM_IMAGES=3000
EXPECTED_LAYER=8

# Frozen using the 2,000-image A0 validation set.
CAM_THRESHOLD=0.30
MIN_AREA_PATCHES=0

SCORE_MODE="sum_sqrt_area"
MATCH_MODE="quadrant"
FROC_TARGETS="0.10,0.25,0.50"

ORIGINAL_HEATMAPS="${OUTPUT_ROOT}/original/heatmaps_crossattn_gradcam_layer8"
LUNG_HEATMAPS="${OUTPUT_ROOT}/lung_only/heatmaps_crossattn_gradcam_layer8"
HEART_HEATMAPS="${OUTPUT_ROOT}/heart_only/heatmaps_crossattn_gradcam_layer8"
ENSEMBLE_HEATMAPS="${OUTPUT_ROOT}/ensemble_equal/heatmaps_crossattn_gradcam_layer8"

run_froc () {
    local view_name="$1"
    local expected_view="$2"
    local heatmaps_dir="$3"
    local output_dir="${OUTPUT_ROOT}/${view_name}"

    echo "======================================================================"
    echo "[Start] View: ${view_name}"
    echo "[Input] ${heatmaps_dir}"
    echo "[Output] ${output_dir}"
    echo "======================================================================"

    if [[ ! -d "${heatmaps_dir}" ]]; then
        echo "[Error] Missing heatmap directory: ${heatmaps_dir}" >&2
        exit 1
    fi

    if [[ ! -f "${heatmaps_dir}/crossattn_gradcam_index.csv" ]]; then
        echo "[Error] Missing heatmap index:"
        echo "        ${heatmaps_dir}/crossattn_gradcam_index.csv" >&2
        exit 1
    fi

    mkdir -p "${output_dir}"

    python -u "${SCRIPT}" \
        --mode test \
        --heatmaps_dir "${heatmaps_dir}" \
        --evaluation_csv "${EVALUATION_CSV}" \
        --annotations_csv "${ANNOTATIONS_CSV}" \
        --meta_csv "${META_CSV}" \
        --output_dir "${output_dir}" \
        --label "${LABEL}" \
        --prefix "${view_name}_test" \
        --expected_view "${expected_view}" \
        --expected_layer "${EXPECTED_LAYER}" \
        --expected_num_images "${EXPECTED_NUM_IMAGES}" \
        --threshold "${CAM_THRESHOLD}" \
        --min_area_patches "${MIN_AREA_PATCHES}" \
        --score_mode "${SCORE_MODE}" \
        --match_mode "${MATCH_MODE}" \
        --froc_targets "${FROC_TARGETS}"

    echo "[Finished] ${view_name}"
}

# ---------------------------------------------------------------------------
# Individual views
# ---------------------------------------------------------------------------

run_froc \
    "original" \
    "original" \
    "${ORIGINAL_HEATMAPS}"

run_froc \
    "lung" \
    "lung_only" \
    "${LUNG_HEATMAPS}"

run_froc \
    "heart" \
    "heart_only" \
    "${HEART_HEATMAPS}"

# ---------------------------------------------------------------------------
# Ensemble
#
# The current FROC CLI accepts only original/lung_only/heart_only for
# --expected_view. Therefore, omit that argument for ensemble CAM files while
# retaining layer, label, shape, coverage, and image-ID validation.
# ---------------------------------------------------------------------------

echo "======================================================================"
echo "[Start] View: ensemble"
echo "======================================================================"

if [[ ! -d "${ENSEMBLE_HEATMAPS}" ]]; then
    echo "[Error] Missing ensemble heatmap directory: ${ENSEMBLE_HEATMAPS}" >&2
    exit 1
fi

if [[ ! -f "${ENSEMBLE_HEATMAPS}/crossattn_gradcam_index.csv" ]]; then
    echo "[Error] Missing ensemble heatmap index:"
    echo "        ${ENSEMBLE_HEATMAPS}/crossattn_gradcam_index.csv" >&2
    exit 1
fi

ENSEMBLE_OUTPUT="${OUTPUT_ROOT}/ensemble"
mkdir -p "${ENSEMBLE_OUTPUT}"

python -u "${SCRIPT}" \
    --mode test \
    --heatmaps_dir "${ENSEMBLE_HEATMAPS}" \
    --evaluation_csv "${EVALUATION_CSV}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --meta_csv "${META_CSV}" \
    --output_dir "${ENSEMBLE_OUTPUT}" \
    --label "${LABEL}" \
    --prefix "ensemble_test" \
    --expected_layer "${EXPECTED_LAYER}" \
    --expected_num_images "${EXPECTED_NUM_IMAGES}" \
    --threshold "${CAM_THRESHOLD}" \
    --min_area_patches "${MIN_AREA_PATCHES}" \
    --score_mode "${SCORE_MODE}" \
    --match_mode "${MATCH_MODE}" \
    --froc_targets "${FROC_TARGETS}"

echo "======================================================================"
echo "[Done] FROC evaluation completed for all four views."
echo "[Results] ${OUTPUT_ROOT}"
echo "======================================================================"
