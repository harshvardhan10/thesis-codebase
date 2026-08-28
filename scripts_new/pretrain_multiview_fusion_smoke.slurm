#!/bin/bash -l
#SBATCH --job-name=albef-mv-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/albef_mv_smoke_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/albef_mv_smoke_%j.err
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
# Better CUDA stack traces for a tiny diagnostic run.
export CUDA_LAUNCH_BLOCKING=1

# ---------------------------------------------------------------------------
# Code/config paths
# ---------------------------------------------------------------------------
TRAIN_SCRIPT="${PROJECT_ROOT}/Pretrain_A0_multiview_fusion_dual_auc.py"
CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml"
MODEL_MODULE="${PROJECT_ROOT}/models/model_pretrain_multiview_fusion.py"
VALIDATION_MODULE="${PROJECT_ROOT}/scripts_new/vindr_multiview_classification_validation.py"

# ---------------------------------------------------------------------------
# Single-view A0 checkpoints used to initialize the three visual branches.
# These are CLI overrides, so placeholder paths inside the YAML are ignored.
# ---------------------------------------------------------------------------
ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"

# Keep smoke-test artifacts separate from the real experiment.
SMOKE_ROOT="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_fusion_smoke"
SMOKE_OUTPUT_DIR="${SMOKE_ROOT}/job_${SLURM_JOB_ID}"
SMOKE_CONFIG="${SMOKE_OUTPUT_DIR}/smoke_config.yaml"
mkdir -p "${SMOKE_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
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

# The trainer must import the validator from scripts_new in this repo layout.
if ! grep -q "from scripts_new.vindr_multiview_classification_validation import" "${TRAIN_SCRIPT}"; then
    echo "[ERROR] ${TRAIN_SCRIPT} still imports vindr_multiview_classification_validation from scripts/." >&2
    echo "[ERROR] Change it to: from scripts_new.vindr_multiview_classification_validation import (...)" >&2
    exit 1
fi

# Check that Python can import the two new modules before allocating model memory.
python - <<'PY'
from models.model_pretrain_multiview_fusion import ALBEF  # noqa: F401
from scripts_new.vindr_multiview_classification_validation import (  # noqa: F401
    VinDrMultiViewClassificationValidationRunner,
)
print("[Smoke preflight] Multi-view model and validation modules import successfully.")
PY

# ---------------------------------------------------------------------------
# Build a temporary smoke-test config from the REAL training config.
# We only override settings needed to make the diagnostic run tiny.
# ---------------------------------------------------------------------------
python - "${CONFIG}" "${SMOKE_CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with source.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Tiny training workload: enough to exercise data loading, synchronized
# original/lung/heart augmentation, all three ViTs, FC fusion, momentum path,
# ITC + ITM + MLM, backward(), optimizer.step(), and checkpoint serialization.
cfg["batch_size"] = 2
cfg["num_workers"] = 2
cfg["train_subset_size"] = 16
cfg["train_subset_seed"] = 42
cfg["debug_max_batches"] = 2
cfg["print_freq"] = 1
cfg["verify_training_files_at_start"] = False
cfg["retain_last_checkpoint_after_success"] = True

cfg["schedular"]["epochs"] = 1
cfg["schedular"]["warmup_epochs"] = 0

# Do NOT mix VinDr validation into the first diagnostic. The full job below
# restores validation-every-epoch and dual-AUC checkpoint selection.
cfg["vindr_validation"]["enabled"] = False

with destination.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(f"[Smoke config] Wrote {destination}")
PY

echo "============================================================"
echo "Multi-view ALBEF PRETRAINING SMOKE TEST"
echo "Slurm job ID:        ${SLURM_JOB_ID}"
echo "Project root:        ${PROJECT_ROOT}"
echo "Base config:         ${CONFIG}"
echo "Smoke config:        ${SMOKE_CONFIG}"
echo "Original checkpoint: ${ORIGINAL_CHECKPOINT}"
echo "Lung checkpoint:     ${LUNG_CHECKPOINT}"
echo "Heart checkpoint:    ${HEART_CHECKPOINT}"
echo "Output directory:    ${SMOKE_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

nvidia-smi || true

python -u "${TRAIN_SCRIPT}" \
    --config "${SMOKE_CONFIG}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --lung_checkpoint "${LUNG_CHECKPOINT}" \
    --heart_checkpoint "${HEART_CHECKPOINT}" \
    --output_dir "${SMOKE_OUTPUT_DIR}" \
    --text_encoder bert-base-uncased \
    --device cuda \
    --seed 42 \
    --distributed false \
    --auto_resume false

# ---------------------------------------------------------------------------
# Post-run smoke assertions
# ---------------------------------------------------------------------------
SMOKE_CHECKPOINT="${SMOKE_OUTPUT_DIR}/checkpoint_last.pth"
if [[ ! -f "${SMOKE_CHECKPOINT}" ]]; then
    echo "[ERROR] Smoke training finished without checkpoint_last.pth" >&2
    exit 1
fi

python - "${SMOKE_CHECKPOINT}" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")

for key in ("model", "optimizer", "lr_scheduler", "epoch"):
    if key not in checkpoint:
        raise RuntimeError(f"Smoke checkpoint is missing required key: {key}")

state = checkpoint["model"]
required_prefixes = (
    "visual_encoder_original.",
    "visual_encoder_lung.",
    "visual_encoder_heart.",
    "view_fusion.",
    "view_fusion_norm.",
    "visual_encoder_original_m.",
    "visual_encoder_lung_m.",
    "visual_encoder_heart_m.",
    "view_fusion_m.",
)

for prefix in required_prefixes:
    if not any(name.startswith(prefix) for name in state):
        raise RuntimeError(f"Smoke checkpoint has no parameters with prefix: {prefix}")

print(f"[Smoke verify] checkpoint={path}")
print(f"[Smoke verify] epoch={checkpoint['epoch']}")
print(f"[Smoke verify] model_tensors={len(state)}")
print("[Smoke verify] All three online/momentum ViTs and fusion modules are present.")
PY

echo "============================================================"
echo "[PASS] Multi-view pretraining smoke test completed successfully."
echo "[PASS] Two training batches executed and checkpoint serialization verified."
echo "[NEXT] Submit pretrain_multiview_fusion.slurm for the full run."
echo "============================================================"
