#!/bin/bash -l
#
#SBATCH --job-name=extract_crossattn_multiview
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/extract_crossattn_multiview_pretrain_last_ckpt_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/extract_crossattn_multiview_pretrain_last_ckpt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface

PROJECT_ROOT=/home/woody/iwi5/iwi5362h/ALBEF
cd "${PROJECT_ROOT}" || exit 1

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

VIEWS=(
  "original"
  "lung_only"
  "heart_only"
)

CHECKPOINTS=(
  "/home/vault/iwi5/iwi5362h/outputs/A0_original_chexmask_common/checkpoint_last.pth"
  "/home/vault/iwi5/iwi5362h/outputs/A0_lung_chexmask_common/checkpoint_last.pth"
  "/home/vault/iwi5/iwi5362h/outputs/A0_heart_chexmask_common/checkpoint_last.pth"
)

# original does not use a mask, so its entry is empty.
MASK_ROOTS=(
  ""
  "/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/lung"
  "/home/woody/iwi5/iwi5362h/data/vindr_cxr/chexmask_mask_cache/test/heart"
)

OUTPUT_DIRS=(
  "/home/vault/iwi5/iwi5362h/results/multiview/pretrain_loss_last_ckpt/original/heatmaps_crossattn_gradcam_layer8"
  "/home/vault/iwi5/iwi5362h/results/multiview/pretrain_loss_last_ckpt/lung_only/heatmaps_crossattn_gradcam_layer8"
  "/home/vault/iwi5/iwi5362h/results/multiview/pretrain_loss_last_ckpt/heart_only/heatmaps_crossattn_gradcam_layer8"
)

VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
MASK_ROOT="${MASK_ROOTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "View: ${VIEW}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Mask root: ${MASK_ROOT:-not required}"
echo "Output directory: ${OUTPUT_DIR}"

# Common arguments shared by all three models.
ARGS=(
  --config configs/Pretrain.yaml
  --checkpoint "${CHECKPOINT}"
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test
  --output_dir "${OUTPUT_DIR}"
  --view "${VIEW}"
  --layers_to_use 8
  --only_labels Cardiomegaly
  --device cuda
)

# Only masked views require --mask_root.
if [[ "${VIEW}" != "original" ]]; then
  ARGS+=(--mask_root "${MASK_ROOT}")
fi

python -u scripts/extract_crossattn_gradcam_heatmaps.py "${ARGS[@]}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "Extraction failed for ${VIEW} with exit code ${EXIT_CODE}"
  exit "${EXIT_CODE}"
fi

echo "Extraction completed successfully for ${VIEW}"
echo "Job finished at $(date)"