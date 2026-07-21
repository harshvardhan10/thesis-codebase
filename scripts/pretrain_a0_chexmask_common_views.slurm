#!/bin/bash -l
#SBATCH --job-name=a0-common
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_common_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_common_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

VIEWS=(original lung heart)
VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
EXPECTED_GPUS=2

COMMON_MANIFEST="data/chexmask_manifests/mimic_cxr_chexmask_common_valid.json"
CONFIG="configs/Pretrain_A0_${VIEW}_chexmask_common.yaml"
OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_${VIEW}_chexmask_common"

for required_file in \
    "${COMMON_MANIFEST}" \
    "${CONFIG}" \
    "Pretrain_A0_chexmask.py"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Missing required file: ${required_file}" >&2
        exit 2
    fi
done

echo "======================================================================"
echo "View:                 ${VIEW}"
echo "Config:               ${CONFIG}"
echo "Common manifest:      ${COMMON_MANIFEST}"
echo "Output:               ${OUTPUT_DIR}"
echo "Host:                 $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-unset}"
echo "======================================================================"

nvidia-smi -L

GPU_COUNT="$(
python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS}" ]]; then
    echo "[ERROR] Requested ${EXPECTED_GPUS} GPUs, but PyTorch sees ${GPU_COUNT}." >&2
    exit 10
fi

RECORD_COUNT="$(
python - "${COMMON_MANIFEST}" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))))
PY
)"
echo "Common training records: ${RECORD_COUNT}"

MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"

python -m torch.distributed.launch \
    --nproc_per_node="${EXPECTED_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    --use_env \
    Pretrain_A0_chexmask.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint "" \
    --resume false \
    --auto_resume true \
    --distributed true
