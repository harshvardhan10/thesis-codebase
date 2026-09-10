#!/bin/bash -l
#SBATCH --job-name=albef-mv-residual-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --exclude=a0429
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/albef_mv_residual_smoke_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/albef_mv_residual_smoke_%j.err
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
BASE_CONFIG="${PROJECT_ROOT}/configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml"
MODEL_MODULE="${PROJECT_ROOT}/models/model_pretrain_multiview_residual_fusion.py"
VALIDATION_MODULE="${PROJECT_ROOT}/scripts_new/vindr_multiview_classification_validation.py"
ORIGINAL_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc/checkpoint_best_macro_auc_stable.pth"
LUNG_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc/checkpoint_best_macro_auc_stable.pth"
HEART_CHECKPOINT="/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc/checkpoint_best_macro_auc_stable.pth"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_multiview_residual_fusion_dual_auc_smoke_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
SMOKE_CONFIG="${OUTPUT_DIR}/smoke_config.yaml"

for path in "${TRAIN_SCRIPT}" "${BASE_CONFIG}" "${MODEL_MODULE}" \
            "${VALIDATION_MODULE}" "${ORIGINAL_CHECKPOINT}" \
            "${LUNG_CHECKPOINT}" "${HEART_CHECKPOINT}"
do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing: ${path}" >&2; exit 1; }
done

BASE_CONFIG="${BASE_CONFIG}" SMOKE_CONFIG="${SMOKE_CONFIG}" python - <<'PY_SMOKE'
import os, yaml
with open(os.environ['BASE_CONFIG'], 'r') as handle:
    cfg = yaml.safe_load(handle)
cfg['batch_size'] = 32
cfg['train_subset_size'] = 128
cfg['debug_max_batches'] = 2
cfg['numeric_debug_every'] = 1
cfg['enable_finite_checks'] = True
cfg['retain_last_checkpoint_after_success'] = True
cfg['schedular']['epochs'] = 1
cfg['schedular']['warmup_epochs'] = 0
cfg['vindr_validation']['enabled'] = False
with open(os.environ['SMOKE_CONFIG'], 'w') as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY_SMOKE

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

unset RANK || true
unset WORLD_SIZE || true
unset LOCAL_RANK || true
unset LOCAL_WORLD_SIZE || true

echo "========================================"
echo "Node: ${HOSTNAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "========================================"

python -m torch.distributed.launch --help >/dev/null
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "SMOKE TEST: ORIGINAL-ANCHORED RESIDUAL FUSION"
echo "2 A40s | 32/GPU | global batch 64 | 2 steps/rank"
echo "Output: ${OUTPUT_DIR}"
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

[[ -f "${OUTPUT_DIR}/checkpoint_last.pth" ]] || {
    echo "[ERROR] Smoke run ended without checkpoint_last.pth" >&2
    exit 1
}
echo "[PASS] ORIGINAL-ANCHORED RESIDUAL FUSION smoke test completed; inspect peak GPU memory."
