#!/bin/bash -l
#SBATCH --job-name=anatomy_itc
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/anatomy_itc_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/anatomy_itc_%A_%a.err

set -euo pipefail
unset SLURM_EXPORT_ENV

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1

PROJECT=/home/woody/iwi5/iwi5362h/ALBEF
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# ==========================================================
# EXPERIMENT NAMES
# ==========================================================

NAMES=(
    "A1"
    "A2"
    "A3_lambda_0.01"
    "A3_lambda_0.005"
    "A3_lambda_0.001"
    "A4"
    "A5"
)

# ==========================================================
# CONFIG FILES
# ==========================================================

CONFIGS=(
    "configs/Pretrain_anatomy_prior_A1.yaml"
    "configs/Pretrain_anatomy_prior_A2.yaml"
    "configs/Pretrain_anatomy_prior_A3.yaml"
    "configs/Pretrain_anatomy_prior_A3.yaml"
    "configs/Pretrain_anatomy_prior_A3.yaml"
    "configs/Pretrain_anatomy_prior_A4.yaml"
    "configs/Pretrain_A5_patch_head.yaml"
)

# ==========================================================
# CHECKPOINT FILES
# ==========================================================

CHECKPOINTS=(
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A1_support_full/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A2_support_full/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A3_support_full/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A3_lamda_0.005/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A3_lamda_0.001/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A4_cls_preserve/checkpoint_best.pth"
    "/home/woody/iwi5/iwi5362h/ALBEF/output_anatomy_A5_patch_head_cm_lambda_0.01/checkpoint_best.pth"
)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

OUTPUT_ROOT=/home/vault/iwi5/iwi5362h/results/anatomy_itc_margin_layer10

# ==========================================================
# SELECT EXPERIMENT BASED ON SLURM ARRAY INDEX
# ==========================================================

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as a job array}

NAME=${NAMES[$TASK_ID]}
CONFIG=${CONFIGS[$TASK_ID]}
CKPT=${CHECKPOINTS[$TASK_ID]}

OUT="$OUTPUT_ROOT/$NAME"

if [[ "$CONFIG" == REPLACE_WITH_* || "$CKPT" == REPLACE_WITH_* ]]; then
    echo "ERROR: Fill config/checkpoint for $NAME"
    exit 2
fi

if [[ ! -f "$CONFIG" || ! -f "$CKPT" ]]; then
    echo "ERROR: Config or checkpoint missing for $NAME"
    exit 2
fi

echo "Experiment: $NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Output: $OUT"

# Verify that A5 includes the learned patch-head checkpoint.
A5_ARGS=()
if [[ "$NAME" == "A5" ]]; then
    A5_ARGS+=(--require_a5_patch_head)
fi

mkdir -p "$OUT"

# ==========================================================
# EXTRACT ITC MARGIN GRAD-CAM
# ==========================================================

python -u scripts_new/extract_anatomy_itc_margin_gradcam_a1_a5.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test_canonical.csv \
    --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
    --output_dir "$OUT" \
    --device cuda \
    --view_type original \
    --vit_layer_index 10 \
    --target_labels Cardiomegaly "Pleural effusion" \
    "${A5_ARGS[@]}"

echo "Finished $NAME"
