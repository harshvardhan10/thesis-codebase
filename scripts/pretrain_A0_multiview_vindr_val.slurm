#!/bin/bash -l
#SBATCH --job-name=a0-multiview-val
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_multiview_val_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_multiview_val_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

VIEWS=(original lung heart)
CONFIGS=(
    "configs/Pretrain_A0_original_common_vindr_val.yaml"
    "configs/Pretrain_A0_lung_maskcache_common_vindr_val.yaml"
    "configs/Pretrain_A0_heart_maskcache_common_vindr_val.yaml"
)
OUTPUT_DIRS=(
    "/home/vault/iwi5/iwi5362h/outputs/A0_original_common_vindr_val"
    "/home/vault/iwi5/iwi5362h/outputs/A0_lung_common_vindr_val"
    "/home/vault/iwi5/iwi5362h/outputs/A0_heart_common_vindr_val"
)
TRAIN_MASK_ROOTS=(
    ""
    "/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common/lung"
    "/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common/heart"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
VIEW="${VIEWS[$TASK_ID]}"
BASE_CONFIG="${CONFIGS[$TASK_ID]}"
OUTPUT_DIR="${OUTPUT_DIRS[$TASK_ID]}"
SOURCE_TRAIN_MASK_ROOT="${TRAIN_MASK_ROOTS[$TASK_ID]}"

EXPECTED_GPUS=2
TRAIN_SCRIPT="Pretrain_A0_multiview_vindr_checkpoint_selection.py"
VALIDATION_MODULE="scripts/vindr_online_validation.py"
RUNTIME_CONFIG="${TMPDIR:-/tmp}/Pretrain_A0_${VIEW}_vindr_val_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"

for required in "${BASE_CONFIG}" "${TRAIN_SCRIPT}" "${VALIDATION_MODULE}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[ERROR] Missing required path: ${required}" >&2
        exit 2
    fi
done

if [[ "${VIEW}" != "original" && ! -d "${SOURCE_TRAIN_MASK_ROOT}" ]]; then
    echo "[ERROR] Missing training mask root: ${SOURCE_TRAIN_MASK_ROOT}" >&2
    exit 3
fi

mkdir -p "${OUTPUT_DIR}"

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS}" ]]; then
    echo "[ERROR] Expected ${EXPECTED_GPUS} visible GPUs, got ${GPU_COUNT}" >&2
    exit 10
fi

ACTIVE_TRAIN_MASK_ROOT="${SOURCE_TRAIN_MASK_ROOT}"

# Only lung/heart need the large MIMIC training mask cache.
# VinDr validation masks remain on persistent storage because the validation
# subset is small and its root is configured separately in the YAML.
if [[ "${VIEW}" != "original" ]]; then
    LOCAL_TRAIN_MASK_ROOT="${TMPDIR:-/tmp}/chexmask_common_${VIEW}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
    MASK_BYTES="$(du -sb "${SOURCE_TRAIN_MASK_ROOT}" | awk '{print $1}')"
    AVAILABLE_BYTES="$(df -PB1 "${TMPDIR:-/tmp}" | awk 'NR==2 {print $4}')"
    REQUIRED_BYTES="$((MASK_BYTES + 2 * 1024 * 1024 * 1024))"

    if [[ "${AVAILABLE_BYTES}" -ge "${REQUIRED_BYTES}" ]]; then
        echo "[Cache] Copying ${VIEW} MIMIC mask cache to node-local storage."
        mkdir -p "${LOCAL_TRAIN_MASK_ROOT}"
        cp -a "${SOURCE_TRAIN_MASK_ROOT}/." "${LOCAL_TRAIN_MASK_ROOT}/"
        ACTIVE_TRAIN_MASK_ROOT="${LOCAL_TRAIN_MASK_ROOT}"
    else
        echo "[Cache] Insufficient local space; using persistent training masks."
        echo "[Cache] required=${REQUIRED_BYTES}, available=${AVAILABLE_BYTES}"
    fi
fi

# Create a runtime YAML. For lung/heart, only the MIMIC training mask root is
# replaced with the node-local copy. The VinDr validation mask root remains the
# persistent path from the base YAML.
python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${VIEW}" "${ACTIVE_TRAIN_MASK_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination, view, active_train_mask_root = sys.argv[1:]

with open(source) as handle:
    config = yaml.safe_load(handle)

configured_view = str(config.get("view_name", "")).strip().lower()
if configured_view != view:
    raise ValueError(
        f"Config/view mismatch: config view_name={configured_view!r}, array view={view!r}"
    )

val_cfg = config.get("vindr_validation", {})
if str(val_cfg.get("view_type", "")).strip().lower() != view:
    raise ValueError(
        "vindr_validation.view_type must match view_name: "
        f"{val_cfg.get('view_type')!r} vs {view!r}"
    )

if view == "original":
    config.pop("mask_root", None)
    val_cfg["mask_root"] = None
else:
    config["mask_root"] = active_train_mask_root

# Validate all fixed VinDr validation inputs before launching two processes.
for key in (
    "classification_labels_csv",
    "localization_labels_csv",
    "localization_annotations_csv",
    "images_root",
):
    path = val_cfg.get(key)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Missing vindr_validation.{key}: {path}")

if view != "original":
    validation_mask_root = val_cfg.get("mask_root")
    if not validation_mask_root or not Path(validation_mask_root).is_dir():
        raise FileNotFoundError(
            f"Missing VinDr validation mask root for {view}: {validation_mask_root}"
        )

config["vindr_validation"] = val_cfg

with open(destination, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"[Runtime config] {destination}")
print(f"[Runtime config] view_name={view}")
print(f"[Runtime config] training mask_root={config.get('mask_root')}")
print(f"[Runtime config] validation mask_root={val_cfg.get('mask_root')}")
PY

MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + TASK_ID))"

echo "======================================================================"
echo "View:                 ${VIEW}"
echo "Host:                 $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Base config:          ${BASE_CONFIG}"
echo "Runtime config:       ${RUNTIME_CONFIG}"
echo "Training masks:       ${ACTIVE_TRAIN_MASK_ROOT:-not used}"
echo "Output:               ${OUTPUT_DIR}"
echo "Master port:          ${MASTER_PORT}"
echo "======================================================================"

nvidia-smi -L

python -m torch.distributed.launch \
    --nproc_per_node="${EXPECTED_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    --use_env \
    "${TRAIN_SCRIPT}" \
    --config "${RUNTIME_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint "" \
    --resume false \
    --auto_resume true \
    --distributed true
