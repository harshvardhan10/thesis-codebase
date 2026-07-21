#!/bin/bash -l
#SBATCH --job-name=a0-maskcache
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_maskcache_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_maskcache_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

VIEWS=(lung heart)
VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"

BASE_CONFIG="configs/Pretrain_A0_${VIEW}_maskcache_common.yaml"
SOURCE_MASK_ROOT="/home/vault/iwi5/iwi5362h/data/chexmask/mask_cache/common/${VIEW}"
LOCAL_MASK_ROOT="${TMPDIR:-/tmp}/chexmask_mask_cache_${VIEW}_${SLURM_JOB_ID}"
RUNTIME_CONFIG="${TMPDIR:-/tmp}/Pretrain_A0_${VIEW}_maskcache_runtime_${SLURM_JOB_ID}.yaml"

OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_${VIEW}_chexmask_common"
EXPECTED_GPUS=2

for required in \
    "${BASE_CONFIG}" \
    "${SOURCE_MASK_ROOT}" \
    "Pretrain_A0_maskcache.py"
do
    if [[ ! -e "${required}" ]]; then
        echo "[ERROR] Missing required path: ${required}" >&2
        exit 2
    fi
done

echo "======================================================================"
echo "View:                 ${VIEW}"
echo "Host:                 $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Persistent masks:     ${SOURCE_MASK_ROOT}"
echo "Local masks:          ${LOCAL_MASK_ROOT}"
echo "Output:               ${OUTPUT_DIR}"
echo "======================================================================"

nvidia-smi -L

GPU_COUNT="$(
python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS}" ]]; then
    echo "[ERROR] Expected ${EXPECTED_GPUS} visible GPUs, got ${GPU_COUNT}" >&2
    exit 10
fi

MASK_BYTES="$(du -sb "${SOURCE_MASK_ROOT}" | awk '{print $1}')"
AVAILABLE_BYTES="$(df -PB1 "${TMPDIR:-/tmp}" | awk 'NR==2 {print $4}')"
REQUIRED_BYTES="$((MASK_BYTES + 2 * 1024 * 1024 * 1024))"

ACTIVE_MASK_ROOT="${SOURCE_MASK_ROOT}"

if [[ "${AVAILABLE_BYTES}" -ge "${REQUIRED_BYTES}" ]]; then
    echo "[Cache] Copying compact mask cache to node-local storage."
    mkdir -p "${LOCAL_MASK_ROOT}"
    cp -a "${SOURCE_MASK_ROOT}/." "${LOCAL_MASK_ROOT}/"
    ACTIVE_MASK_ROOT="${LOCAL_MASK_ROOT}"
else
    echo "[Cache] Insufficient local space; using persistent compact masks."
    echo "[Cache] required=${REQUIRED_BYTES}, available=${AVAILABLE_BYTES}"
fi

python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${ACTIVE_MASK_ROOT}" <<'PY'
import sys
import yaml

source, destination, mask_root = sys.argv[1:]
with open(source) as handle:
    config = yaml.safe_load(handle)

config["mask_root"] = mask_root

with open(destination, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"Runtime mask_root={mask_root}", flush=True)
PY

MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"

python -m torch.distributed.launch \
    --nproc_per_node="${EXPECTED_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    --use_env \
    Pretrain_A0_maskcache.py \
    --config "${RUNTIME_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint "" \
    --resume false \
    --auto_resume true \
    --distributed true
