#!/bin/bash -l
#SBATCH --job-name=albef-mv-residual
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/albef_mv_residual_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/albef_mv_residual_%j.err
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

TRAIN_SCRIPT="${PROJECT_ROOT}/Pretrain_A0_multiview_residual_fusion_dual_auc.py"
CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml"
MODEL_MODULE="${PROJECT_ROOT}/models/model_pretrain_multiview_residual_fusion.py"
VALIDATION_MODULE="${PROJECT_ROOT}/scripts_new/vindr_multiview_classification_validation.py"

ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_residual_fusion_dual_auc"
mkdir -p "${OUTPUT_DIR}"

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

# Refuse to launch a full 215k-image run with a leftover debug configuration.
CONFIG="${CONFIG}" python - <<'PY_CHECK'
import os, yaml
with open(os.environ['CONFIG'], 'r') as handle:
    cfg = yaml.safe_load(handle)
if not cfg.get('vindr_validation', {}).get('enabled', False):
    raise RuntimeError('FULL TRAINING REFUSED: vindr_validation.enabled is false')
if cfg.get('debug_max_batches') is not None:
    raise RuntimeError('FULL TRAINING REFUSED: debug_max_batches is still set')
if cfg.get('train_subset_size') is not None:
    raise RuntimeError('FULL TRAINING REFUSED: train_subset_size is still set')
if int(cfg.get('batch_size', -1)) != 32:
    raise RuntimeError('Expected batch_size=32 per GPU for global batch 64')
print('[Preflight] Full-run config verified: validation ON, no subset/debug limit, 32/GPU.')
PY_CHECK

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

unset RANK || true
unset WORLD_SIZE || true
unset LOCAL_RANK || true
unset LOCAL_WORLD_SIZE || true

python -m torch.distributed.launch --help >/dev/null
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "MULTI-VIEW ALBEF: ORIGINAL-ANCHORED RESIDUAL FUSION"
echo "Slurm job ID:         ${SLURM_JOB_ID}"
echo "Node:                 $(hostname)"
echo "Config:               ${CONFIG}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "DDP workers:          2"
echo "Batch/GPU:            32"
echo "Global batch:         64"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"

nvidia-smi || true

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --use_env \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --lung_checkpoint "${LUNG_CHECKPOINT}" \
    --heart_checkpoint "${HEART_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --text_encoder bert-base-uncased \
    --device cuda \
    --seed 42 \
    --distributed true \
    --dist_url env:// \
    --auto_resume true

echo "============================================================"
echo "[Done] ORIGINAL-ANCHORED RESIDUAL FUSION pretraining job finished cleanly."
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
