#!/bin/bash -l
#SBATCH --job-name=a0-precomputed
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_precomputed_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_precomputed_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
VIEWS=(lung heart)
VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
EXPECTED_GPUS=2

SOURCE_ROOT="/home/vault/iwi5/iwi5362h/data/chexmask/precomputed/common/${VIEW}"
SOURCE_MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_${VIEW}_precomputed.json"
CONFIG="${PROJECT_DIR}/configs/Pretrain_A0_${VIEW}_precomputed_common.yaml"
OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/A0_${VIEW}_chexmask_common"

LOCAL_PARENT="${TMPDIR:-/tmp}/chexmask_precomputed_${SLURM_JOB_ID}_${VIEW}"
LOCAL_ROOT="${LOCAL_PARENT}/${VIEW}"
LOCAL_MANIFEST="${LOCAL_PARENT}/mimic_cxr_chexmask_common_${VIEW}_local.json"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

mkdir -p \
  "${OUTPUT_DIR}" \
  "${LOCAL_PARENT}" \
  /home/vault/iwi5/iwi5362h/logs

for required_path in \
  "${SOURCE_MANIFEST}" \
  "${CONFIG}" \
  "${SOURCE_ROOT}" \
  Pretrain_A0_precomputed.py
do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Missing required path: ${required_path}" >&2
    exit 2
  fi
done

echo "======================================================================"
echo "Precomputed ${VIEW} A0 training"
echo "Host:                 $(hostname)"
echo "View:                 ${VIEW}"
echo "Source root:          ${SOURCE_ROOT}"
echo "Source manifest:      ${SOURCE_MANIFEST}"
echo "Local root:           ${LOCAL_ROOT}"
echo "Output:               ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-unset}"
echo "Started:              $(date --iso-8601=seconds)"
echo "======================================================================"

nvidia-smi -L

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS}" ]]; then
  echo "[ERROR] Requested ${EXPECTED_GPUS} GPUs, but PyTorch sees ${GPU_COUNT}." >&2
  exit 10
fi

SOURCE_BYTES="$(du -sb "${SOURCE_ROOT}" | awk '{print $1}')"
AVAILABLE_BYTES="$(df -PB1 "${LOCAL_PARENT}" | tail -n 1 | awk '{print $4}')"
# Require source size plus 10% and 2 GiB safety margin.
REQUIRED_BYTES="$((SOURCE_BYTES + SOURCE_BYTES / 10 + 2147483648))"

TRAIN_MANIFEST="${SOURCE_MANIFEST}"

if [[ "${AVAILABLE_BYTES}" -ge "${REQUIRED_BYTES}" ]]; then
  echo "[Stage] Copying ${VIEW} view to node-local storage."
  echo "[Stage] Source bytes:    ${SOURCE_BYTES}"
  echo "[Stage] Available bytes: ${AVAILABLE_BYTES}"

  mkdir -p "${LOCAL_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --whole-file --info=stats2 \
      "${SOURCE_ROOT}/" "${LOCAL_ROOT}/"
  else
    cp -a "${SOURCE_ROOT}/." "${LOCAL_ROOT}/"
  fi

  python - "${SOURCE_MANIFEST}" "${SOURCE_ROOT}" "${LOCAL_ROOT}" "${LOCAL_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

source_manifest = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
local_root = Path(sys.argv[3]).resolve()
local_manifest = Path(sys.argv[4]).resolve()

records = json.loads(source_manifest.read_text())
for record in records:
    source_image = Path(record["image"]).resolve()
    try:
        relative = source_image.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Manifest image is outside source root: {source_image}"
        ) from exc
    local_image = local_root / relative
    if not local_image.is_file():
        raise FileNotFoundError(local_image)
    record["image"] = str(local_image)

local_manifest.parent.mkdir(parents=True, exist_ok=True)
local_manifest.write_text(json.dumps(records, indent=2) + "\n")
print(f"Local manifest: {local_manifest}")
print(f"Records: {len(records):,}")
PY

  TRAIN_MANIFEST="${LOCAL_MANIFEST}"
else
  echo "[WARNING] Not enough node-local space; using shared precomputed images." >&2
  echo "[WARNING] Required bytes:  ${REQUIRED_BYTES}" >&2
  echo "[WARNING] Available bytes: ${AVAILABLE_BYTES}" >&2
fi

MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"

echo "Training manifest: ${TRAIN_MANIFEST}"
echo "Master port:       ${MASTER_PORT}"

python -m torch.distributed.launch \
  --nproc_per_node="${EXPECTED_GPUS}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  --use_env \
  Pretrain_A0_precomputed.py \
  --config "${CONFIG}" \
  --train_file_override "${TRAIN_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "" \
  --resume false \
  --auto_resume true \
  --distributed true

 echo "Finished: $(date --iso-8601=seconds)"
