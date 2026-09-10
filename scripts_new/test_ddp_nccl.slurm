#!/bin/bash -l
#SBATCH --job-name=ddp-test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=00:10:00
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/ddp_test_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/ddp_test_%j.err

set -euo pipefail

PROJECT_ROOT="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_ROOT}"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

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

nvidia-smi

python - <<'PY'
import torch
print("Visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
assert torch.cuda.device_count() == 2
PY

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# NCCL diagnostics / workaround for a0429 P2P hang
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1

# The current A40 pair on a0429 hangs on NCCL P2P/IPC collectives.
export NCCL_P2P_DISABLE=1

# Single-node training: InfiniBand is unnecessary.
export NCCL_IB_DISABLE=1

echo "===== GPU TOPOLOGY ====="
nvidia-smi topo -m || true

echo "===== NCCL SETTINGS ====="
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --use_env \
    "${PROJECT_ROOT}/test_ddp_nccl.py"