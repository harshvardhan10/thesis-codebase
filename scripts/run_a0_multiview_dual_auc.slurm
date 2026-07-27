#!/bin/bash -l
#SBATCH --job-name=a0-multiview-dual-auc
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_multiview_dual_auc_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_multiview_dual_auc_%A_%a.err
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
    "configs/Pretrain_A0_original_dual_auc.yaml"
    "configs/Pretrain_A0_lung_dual_auc.yaml"
    "configs/Pretrain_A0_heart_dual_auc.yaml"
)
OUTPUT_DIRS=(
    "/home/vault/iwi5/iwi5362h/outputs/A0_original_dual_auc"
    "/home/vault/iwi5/iwi5362h/outputs/A0_lung_dual_auc"
    "/home/vault/iwi5/iwi5362h/outputs/A0_heart_dual_auc"
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
EXPECTED_VALIDATION_IMAGES=2000
TRAIN_SCRIPT="Pretrain_A0_multiview_dual_auc.py"
VALIDATION_MODULE="scripts/vindr_classification_validation.py"
RUNTIME_CONFIG="${TMPDIR:-/tmp}/Pretrain_A0_${VIEW}_dual_auc_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"

for required in "${BASE_CONFIG}" "${TRAIN_SCRIPT}" "${VALIDATION_MODULE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing required file: ${required}" >&2
        exit 2
    fi
done

# Do not launch an expensive run unless the trainer explicitly consumes the two
# checkpoint-selection objectives.
for metric_key in \
    "macro_auc_stable" \
    "cardiomegaly_auc"; do
    if ! grep -q "${metric_key}" "${TRAIN_SCRIPT}"; then
        echo "[ERROR] ${TRAIN_SCRIPT} does not reference ${metric_key}." >&2
        echo "[ERROR] Update the trainer to save the two requested checkpoints first." >&2
        exit 4
    fi
done

for checkpoint_name in \
    "checkpoint_best_macro_auc_stable.pth" \
    "checkpoint_best_cardiomegaly_auc.pth"; do
    if ! grep -q "${checkpoint_name}" "${TRAIN_SCRIPT}"; then
        echo "[ERROR] ${TRAIN_SCRIPT} does not save ${checkpoint_name}." >&2
        exit 4
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

# Only lung/heart need the large MIMIC training mask cache. The smaller VinDr
# validation mask cache remains at the path specified in each base YAML.
if [[ "${VIEW}" != "original" ]]; then
    LOCAL_TRAIN_MASK_ROOT="${TMPDIR:-/tmp}/chexmask_common_${VIEW}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
    MASK_BYTES="$(du -sb "${SOURCE_TRAIN_MASK_ROOT}" | awk '{print $1}')"
    AVAILABLE_BYTES="$(df -PB1 "${TMPDIR:-/tmp}" | awk 'NR==2 {print $4}')"
    REQUIRED_BYTES="$((MASK_BYTES + 2 * 1024 * 1024 * 1024))"

    if [[ "${AVAILABLE_BYTES}" -ge "${REQUIRED_BYTES}" ]]; then
        echo "[Cache] Copying ${VIEW} MIMIC masks to node-local storage."
        mkdir -p "${LOCAL_TRAIN_MASK_ROOT}"
        cp -a "${SOURCE_TRAIN_MASK_ROOT}/." "${LOCAL_TRAIN_MASK_ROOT}/"
        ACTIVE_TRAIN_MASK_ROOT="${LOCAL_TRAIN_MASK_ROOT}"
    else
        echo "[Cache] Insufficient local space; using persistent training masks."
        echo "[Cache] required=${REQUIRED_BYTES}, available=${AVAILABLE_BYTES}"
    fi
fi

# Build and validate a job-specific runtime YAML. For lung/heart, only the
# MIMIC training mask root is replaced by the node-local copy. VinDr validation
# continues to use the mask_root configured under vindr_validation.
python - \
    "${BASE_CONFIG}" \
    "${RUNTIME_CONFIG}" \
    "${VIEW}" \
    "${ACTIVE_TRAIN_MASK_ROOT}" \
    "${EXPECTED_VALIDATION_IMAGES}" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

(
    source,
    destination,
    view,
    active_train_mask_root,
    expected_validation_images,
) = sys.argv[1:]
expected_validation_images = int(expected_validation_images)

with open(source, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

if not isinstance(config, dict):
    raise TypeError(f"Top-level YAML object must be a mapping: {source}")

if bool(config.get("retain_last_checkpoint_after_success", False)):
    raise ValueError(
        "retain_last_checkpoint_after_success must be false so only the two "
        "best checkpoints remain after successful completion"
    )

configured_view = str(config.get("view_name", "")).strip().lower()
if configured_view != view:
    raise ValueError(
        f"Config/view mismatch: view_name={configured_view!r}, array view={view!r}"
    )

val_cfg = config.get("vindr_validation")
if not isinstance(val_cfg, dict):
    raise ValueError("vindr_validation must be configured as a mapping")

validation_view = str(val_cfg.get("view_type", "")).strip().lower()
if validation_view != view:
    raise ValueError(
        "vindr_validation.view_type must match view_name: "
        f"{validation_view!r} vs {view!r}"
    )

if view == "original":
    config.pop("mask_root", None)
    val_cfg["mask_root"] = None
else:
    config["mask_root"] = active_train_mask_root

labels_csv = Path(str(val_cfg.get("labels_csv", "")))
images_root = Path(str(val_cfg.get("images_root", "")))
if not labels_csv.is_file():
    raise FileNotFoundError(
        f"Missing vindr_validation.labels_csv: {labels_csv}"
    )
if not images_root.is_dir():
    raise FileNotFoundError(
        f"Missing vindr_validation.images_root: {images_root}"
    )

macro_auc_labels = val_cfg.get("macro_auc_labels")
if (
    not isinstance(macro_auc_labels, list)
    or not macro_auc_labels
    or not all(isinstance(label, str) and label.strip() for label in macro_auc_labels)
):
    raise ValueError(
        "vindr_validation.macro_auc_labels must be a non-empty list of labels"
    )
if len(macro_auc_labels) != len(set(macro_auc_labels)):
    raise ValueError("vindr_validation.macro_auc_labels contains duplicates")

with open(labels_csv, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError(f"Validation CSV has no header: {labels_csv}")
    id_column = "image_id" if "image_id" in reader.fieldnames else reader.fieldnames[0]
    csv_labels = [name for name in reader.fieldnames if name != id_column]
    unknown = [label for label in macro_auc_labels if label not in csv_labels]
    if unknown:
        raise ValueError(
            "macro_auc_labels absent from validation CSV: "
            f"{unknown}; available={csv_labels}"
        )

    rows = list(reader)

if len(rows) != expected_validation_images:
    raise ValueError(
        f"Expected {expected_validation_images} validation rows, found {len(rows)} "
        f"in {labels_csv}"
    )

image_ids = [str(row[id_column]).strip() for row in rows]
if any(not image_id for image_id in image_ids):
    raise ValueError(f"Validation CSV contains an empty {id_column}")
if len(image_ids) != len(set(image_ids)):
    raise ValueError("Validation CSV contains duplicate image IDs")

missing_images = [
    images_root / f"{image_id}.png"
    for image_id in image_ids
    if not (images_root / f"{image_id}.png").is_file()
]
if missing_images:
    preview = "\n".join(f"  {path}" for path in missing_images[:10])
    raise FileNotFoundError(
        f"Missing {len(missing_images)} VinDr validation images:\n{preview}"
    )

if view != "original":
    validation_mask_root = Path(str(val_cfg.get("mask_root", "")))
    if not validation_mask_root.is_dir():
        raise FileNotFoundError(
            f"Missing VinDr validation mask root for {view}: "
            f"{validation_mask_root}"
        )
    missing_masks = [
        validation_mask_root / image_id[:2] / f"{image_id}.png"
        for image_id in image_ids
        if not (
            validation_mask_root / image_id[:2] / f"{image_id}.png"
        ).is_file()
    ]
    if missing_masks:
        preview = "\n".join(f"  {path}" for path in missing_masks[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} VinDr validation masks:\n{preview}"
        )

min_positive = int(val_cfg.get("min_positive_per_label", 20))
min_negative = int(val_cfg.get("min_negative_per_label", 20))
insufficient = []
for label in macro_auc_labels:
    positives = sum(int(float(row[label])) for row in rows)
    negatives = len(rows) - positives
    if positives < min_positive or negatives < min_negative:
        insufficient.append((label, positives, negatives))
if insufficient:
    raise ValueError(
        "Configured macro_auc_labels fail the support thresholds "
        f"(min_positive={min_positive}, min_negative={min_negative}): "
        f"{insufficient}"
    )

config["vindr_validation"] = val_cfg

with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"[Runtime config] {destination}")
print(f"[Runtime config] view_name={view}")
print(f"[Runtime config] training mask_root={config.get('mask_root')}")
print(f"[Runtime config] validation labels={labels_csv}")
print(f"[Runtime config] validation images={len(rows)}")
print(f"[Runtime config] macro_auc_labels={len(macro_auc_labels)}")
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
