#!/bin/bash -l
#SBATCH --job-name=a0-chexmask
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_chexmask_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_chexmask_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

VIEWS=(lung heart)
VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"

EXPECTED_GPUS=2

echo "======================================================================"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Array task:           ${SLURM_ARRAY_TASK_ID}"
echo "Host:                 $(hostname)"
echo "View:                 ${VIEW}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE:   ${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_NTASKS:         ${SLURM_NTASKS:-unset}"
echo "======================================================================"

nvidia-smi -L

VISIBLE_GPUS="$(
python - <<'PY'
import os
import torch

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
print(f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
print(f"torch.cuda.device_count={torch.cuda.device_count()}", flush=True)

for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}", flush=True)

print(torch.cuda.device_count())
PY
)"

echo "${VISIBLE_GPUS}"

GPU_COUNT="$(echo "${VISIBLE_GPUS}" | tail -n 1)"

if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS}" ]]; then
    echo "[ERROR] Requested ${EXPECTED_GPUS} GPUs, but PyTorch sees ${GPU_COUNT}." >&2
    echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    echo "[ERROR] Do not launch distributed training until both GPUs are visible." >&2
    exit 10
fi

MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"

echo "MASTER_PORT=${MASTER_PORT}"
echo "Launching ${EXPECTED_GPUS} ranks for ${VIEW}"

python -m torch.distributed.launch \
    --nproc_per_node="${EXPECTED_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    --use_env \
    Pretrain_A0_chexmask.py \
    --config "configs/Pretrain_A0_${VIEW}_chexmask.yaml" \
    --output_dir "/home/vault/iwi5/iwi5362h/outputs/A0_${VIEW}_chexmask" \
    --checkpoint "" \
    --resume false \
    --auto_resume true \
    --distributed true
