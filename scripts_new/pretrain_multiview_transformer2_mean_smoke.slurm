#!/bin/bash -l
#SBATCH --job-name=albef-mv-tf2-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/albef_mv_tf2_smoke_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/albef_mv_tf2_smoke_%j.err
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
export OMP_NUM_THREADS=1

TRAIN_SCRIPT="${PROJECT_ROOT}/Pretrain_A0_multiview_transformer_fusion_dual_auc.py"
BASE_CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml"
MODEL_MODULE="${PROJECT_ROOT}/models/model_pretrain_multiview_transformer_fusion.py"
VALIDATION_MODULE="${PROJECT_ROOT}/scripts_new/vindr_multiview_classification_validation.py"

ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_transformer2_mean_smoke_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
SMOKE_CONFIG="${OUTPUT_DIR}/smoke_config.yaml"

for path in \
    "${TRAIN_SCRIPT}" \
    "${BASE_CONFIG}" \
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

# Make a temporary smoke config without editing the real experiment config.
BASE_CONFIG="${BASE_CONFIG}" SMOKE_CONFIG="${SMOKE_CONFIG}" python - <<'PY'
import os
import yaml

with open(os.environ["BASE_CONFIG"], "r") as handle:
    config = yaml.safe_load(handle)

# 2 DDP ranks x 32 samples/rank x 2 steps = 128 samples.
config["batch_size"] = 32
config["train_subset_size"] = 128
config["debug_max_batches"] = 2
config["numeric_debug_every"] = 1
config["enable_finite_checks"] = True
config["retain_last_checkpoint_after_success"] = True

config["schedular"]["epochs"] = 1
config["schedular"]["warmup_epochs"] = 0
config["vindr_validation"]["enabled"] = False

with open(os.environ["SMOKE_CONFIG"], "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Confirm legacy launcher availability before consuming GPU time.
python -m torch.distributed.launch --help >/dev/null

echo "============================================================"
echo "2-LAYER TRANSFORMER MEAN-FUSION SMOKE TEST"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Node:                 $(hostname)"
echo "Config:               ${SMOKE_CONFIG}"
echo "Batch/GPU:            32"
echo "Global batch:         64"
echo "Steps/rank:           2"
echo "DDP workers:          2"
echo "Output:               ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

nvidia-smi || true

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --use_env \
    "${TRAIN_SCRIPT}" \
    --config "${SMOKE_CONFIG}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --lung_checkpoint "${LUNG_CHECKPOINT}" \
    --heart_checkpoint "${HEART_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --text_encoder bert-base-uncased \
    --device cuda \
    --seed 42 \
    --distributed true \
    --dist_url env:// \
    --auto_resume false

if [[ ! -f "${OUTPUT_DIR}/checkpoint_last.pth" ]]; then
    echo "[ERROR] Smoke training finished but checkpoint_last.pth is missing." >&2
    exit 1
fi

echo "============================================================"
echo "[PASS] Transformer mean-fusion smoke test completed."
echo "[PASS] Inspect GPU peak memory before launching the full run."
echo "============================================================"
