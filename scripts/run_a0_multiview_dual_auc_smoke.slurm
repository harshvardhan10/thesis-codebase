#!/bin/bash -l
#SBATCH --job-name=a0-dual-auc-smoke
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_dual_auc_smoke_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_dual_auc_smoke_%A_%a.err
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
TRAIN_MASK_ROOTS=(
    ""
    "/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common/lung"
    "/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common/heart"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#VIEWS[@]} )); then
    echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
fi

VIEW="${VIEWS[$TASK_ID]}"
BASE_CONFIG="${CONFIGS[$TASK_ID]}"
TRAIN_MASK_ROOT="${TRAIN_MASK_ROOTS[$TASK_ID]}"

TRAIN_SCRIPT="Pretrain_A0_multiview_dual_auc.py"
VALIDATION_MODULE="scripts/vindr_classification_validation.py"
EXPECTED_VALIDATION_IMAGES=2000
EXPECTED_STABLE_LABELS=15

# Each submission gets isolated output, so a previous smoke run cannot resume
# accidentally or overwrite its results.
OUTPUT_DIR="/home/vault/iwi5/iwi5362h/outputs/smoke/A0_${VIEW}_dual_auc_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
RUNTIME_CONFIG="${TMPDIR:-/tmp}/Pretrain_A0_${VIEW}_dual_auc_smoke_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"

for required in "${BASE_CONFIG}" "${TRAIN_SCRIPT}" "${VALIDATION_MODULE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing required file: ${required}" >&2
        exit 3
    fi
done

if [[ "${VIEW}" != "original" && ! -d "${TRAIN_MASK_ROOT}" ]]; then
    echo "[ERROR] Missing training mask root: ${TRAIN_MASK_ROOT}" >&2
    exit 4
fi

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -ne 1 ]]; then
    echo "[ERROR] Expected 1 visible GPU, got ${GPU_COUNT}" >&2
    exit 5
fi

python -m py_compile "${TRAIN_SCRIPT}" "${VALIDATION_MODULE}"
mkdir -p "${OUTPUT_DIR}"

# Build a smoke-only runtime configuration. The base YAML files are not edited.
# Training is reduced to two batches, while validation deliberately keeps all
# 2,000 images so every configured stable label still has valid support.
python - \
    "${BASE_CONFIG}" \
    "${RUNTIME_CONFIG}" \
    "${VIEW}" \
    "${TRAIN_MASK_ROOT}" \
    "${EXPECTED_VALIDATION_IMAGES}" \
    "${EXPECTED_STABLE_LABELS}" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

(
    source,
    destination,
    expected_view,
    train_mask_root,
    expected_validation_images,
    expected_stable_labels,
) = sys.argv[1:]
expected_validation_images = int(expected_validation_images)
expected_stable_labels = int(expected_stable_labels)

with open(source, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

if not isinstance(config, dict):
    raise TypeError(f"Top-level YAML object must be a mapping: {source}")

view = str(config.get("view_name", "")).strip().lower()
if view != expected_view:
    raise ValueError(
        f"Config/view mismatch: view_name={view!r}, expected={expected_view!r}"
    )

train_files = config.get("train_file")
if isinstance(train_files, str):
    train_files = [train_files]
if not isinstance(train_files, list) or not train_files:
    raise ValueError("train_file must contain at least one annotation file")
for train_file in train_files:
    if not Path(train_file).is_file():
        raise FileNotFoundError(f"Missing training annotation: {train_file}")

if view == "original":
    config.pop("mask_root", None)
else:
    if not train_mask_root:
        raise ValueError(f"Training mask root is empty for view={view}")
    config["mask_root"] = train_mask_root

# Two real optimizer steps exercise loading, augmentation, tokenization,
# forward/backward, gradient update, and scheduler behavior.
config["batch_size"] = 8
config["num_workers"] = 2
config["train_subset_size"] = 16
config["train_subset_seed"] = 42
config["debug_max_batches"] = 2
config["print_freq"] = 1
config["retain_last_checkpoint_after_success"] = False

scheduler = config.get("schedular")
if not isinstance(scheduler, dict):
    raise ValueError("schedular must be configured as a mapping")
scheduler["epochs"] = 1
scheduler["warmup_epochs"] = 0

val_cfg = config.get("vindr_validation")
if not isinstance(val_cfg, dict):
    raise ValueError("vindr_validation must be configured as a mapping")
if not bool(val_cfg.get("enabled", False)):
    raise ValueError("vindr_validation.enabled must be true")

validation_view = str(val_cfg.get("view_type", "")).strip().lower()
if validation_view != view:
    raise ValueError(
        "vindr_validation.view_type must match view_name: "
        f"{validation_view!r} vs {view!r}"
    )

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

macro_labels = val_cfg.get("macro_auc_labels")
if not isinstance(macro_labels, list):
    raise ValueError("vindr_validation.macro_auc_labels must be a list")
if len(macro_labels) != expected_stable_labels:
    raise ValueError(
        f"Expected {expected_stable_labels} stable labels, "
        f"found {len(macro_labels)}"
    )
if len(macro_labels) != len(set(macro_labels)):
    raise ValueError("vindr_validation.macro_auc_labels contains duplicates")

with open(labels_csv, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError(f"Validation CSV has no header: {labels_csv}")
    id_column = (
        "image_id" if "image_id" in reader.fieldnames else reader.fieldnames[0]
    )
    csv_labels = [name for name in reader.fieldnames if name != id_column]
    rows = list(reader)

if len(rows) != expected_validation_images:
    raise ValueError(
        f"Expected {expected_validation_images} validation rows, "
        f"found {len(rows)} in {labels_csv}"
    )

unknown_labels = [label for label in macro_labels if label not in csv_labels]
if unknown_labels:
    raise ValueError(
        f"macro_auc_labels absent from validation CSV: {unknown_labels}"
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
        f"Missing {len(missing_images)} validation images:\n{preview}"
    )

if view == "original":
    val_cfg["mask_root"] = None
else:
    validation_mask_root = Path(str(val_cfg.get("mask_root", "")))
    if not validation_mask_root.is_dir():
        raise FileNotFoundError(
            f"Missing validation mask root for {view}: {validation_mask_root}"
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
            f"Missing {len(missing_masks)} validation masks:\n{preview}"
        )

min_positive = int(val_cfg.get("min_positive_per_label", 20))
min_negative = int(val_cfg.get("min_negative_per_label", 20))
insufficient = []
for label in macro_labels:
    positives = sum(int(float(row[label])) for row in rows)
    negatives = len(rows) - positives
    if positives < min_positive or negatives < min_negative:
        insufficient.append((label, positives, negatives))
if insufficient:
    raise ValueError(
        "Configured stable labels fail the support thresholds: "
        f"{insufficient}"
    )

val_cfg["validate_every"] = 1
val_cfg["batch_size"] = 64
val_cfg["num_workers"] = 4
val_cfg["max_images"] = None
val_cfg["threshold_steps"] = 50
val_cfg["early_stopping"] = {
    "enabled": False,
    "min_epochs": 1,
    "patience": 1,
}
config["vindr_validation"] = val_cfg

with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"[Smoke config] path={destination}")
print(f"[Smoke config] view={view}")
print("[Smoke config] train_batches=2 batch_size=8 epochs=1")
print(f"[Smoke config] validation_images={len(rows)}")
print(f"[Smoke config] stable_labels={len(macro_labels)}")
PY

MASTER_PORT="$((30000 + (SLURM_ARRAY_JOB_ID % 20000) + TASK_ID))"

echo "======================================================================"
echo "Smoke view:           ${VIEW}"
echo "Host:                 $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Base config:          ${BASE_CONFIG}"
echo "Runtime config:       ${RUNTIME_CONFIG}"
echo "Training masks:       ${TRAIN_MASK_ROOT:-not used}"
echo "Output:               ${OUTPUT_DIR}"
echo "Master port:          ${MASTER_PORT}"
echo "======================================================================"
nvidia-smi -L

# A one-process distributed launch follows the same DDP code path as the
# two-GPU production job without requiring two GPUs for the smoke test.
python -m torch.distributed.launch \
    --nproc_per_node=1 \
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
    --auto_resume false \
    --distributed true

# Verify the observable contract of a successful run.
python - "${OUTPUT_DIR}" "${VIEW}" "${EXPECTED_STABLE_LABELS}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_view = sys.argv[2]
expected_stable_labels = int(sys.argv[3])

expected_checkpoints = {
    "checkpoint_best_cardiomegaly_auc.pth",
    "checkpoint_best_macro_auc_stable.pth",
}
actual_checkpoints = {path.name for path in output_dir.glob("*.pth")}
if actual_checkpoints != expected_checkpoints:
    raise RuntimeError(
        f"Checkpoint contract failed: expected={sorted(expected_checkpoints)}, "
        f"actual={sorted(actual_checkpoints)}"
    )

required_nonempty = [
    "config.yaml",
    "log.txt",
    "best_cardiomegaly_auc_scores.npz",
    "best_macro_auc_stable_scores.npz",
]
missing = [
    name
    for name in required_nonempty
    if not (output_dir / name).is_file()
    or (output_dir / name).stat().st_size == 0
]
if missing:
    raise RuntimeError(f"Missing or empty smoke outputs: {missing}")

with open(output_dir / "log.txt", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if len(rows) != 1:
    raise RuntimeError(f"Expected exactly one epoch log row, found {len(rows)}")

row = rows[0]
if row.get("view_name") != expected_view:
    raise RuntimeError(
        f"Logged view mismatch: {row.get('view_name')!r} vs {expected_view!r}"
    )
if row.get("validation_ran") is not True:
    raise RuntimeError("Validation did not run")
if row.get("improved_cardiomegaly_auc") is not True:
    raise RuntimeError("Cardiomegaly checkpoint was not selected")
if row.get("improved_macro_auc_stable") is not True:
    raise RuntimeError("Stable macro-AUC checkpoint was not selected")
if int(row.get("val_num_valid_auc_labels", -1)) != expected_stable_labels:
    raise RuntimeError(
        "Stable-label count mismatch: "
        f"{row.get('val_num_valid_auc_labels')} vs {expected_stable_labels}"
    )

for key in (
    "train_loss_total",
    "val_cardiomegaly_auc",
    "val_macro_auc_stable",
    "val_macro_auc_all_evaluable",
):
    value = row.get(key)
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError(f"Missing or non-finite metric {key}: {value}")

if (output_dir / "checkpoint_last.pth").exists():
    raise RuntimeError("checkpoint_last.pth was not removed after clean completion")

print("======================================================================")
print(f"[SMOKE PASS] view={expected_view}")
print(f"[SMOKE PASS] output={output_dir}")
print(
    "[SMOKE PASS] cardio_auc="
    f"{float(row['val_cardiomegaly_auc']):.6f} "
    "macro_stable="
    f"{float(row['val_macro_auc_stable']):.6f}"
)
print("[SMOKE PASS] exactly two .pth checkpoints remain")
print("======================================================================")
PY
