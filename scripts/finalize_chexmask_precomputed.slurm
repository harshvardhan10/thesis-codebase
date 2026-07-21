#!/bin/bash -l
#SBATCH --job-name=chexmask-finalize
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_finalize_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_finalize_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

module load python
conda activate albef-notebook

PROJECT_DIR="/home/woody/iwi5/iwi5362h/ALBEF"
OUTPUT_ROOT="/home/vault/iwi5/iwi5362h/data/chexmask/precomputed/common"
SOURCE_DB="/home/vault/iwi5/iwi5362h/data/chexmask/index/mimic_chexmask.sqlite"
COMMON_MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_valid.json"
LUNG_MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_lung_precomputed.json"
HEART_MANIFEST="${PROJECT_DIR}/data/chexmask_manifests/mimic_cxr_chexmask_common_heart_precomputed.json"
NUM_SHARDS=4

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u scripts/finalize_chexmask_precomputed_manifests.py \
  --source_manifest "${COMMON_MANIFEST}" \
  --staging_dir "${OUTPUT_ROOT}/_staging" \
  --num_shards "${NUM_SHARDS}" \
  --lung_manifest "${LUNG_MANIFEST}" \
  --heart_manifest "${HEART_MANIFEST}" \
  --chexmask_db "${SOURCE_DB}" \
  --image_key image \
  --verify_samples 30 \
  --seed 42

echo
python - "${COMMON_MANIFEST}" "${LUNG_MANIFEST}" "${HEART_MANIFEST}" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    with open(path) as handle:
        print(f"{path}: {len(json.load(handle)):,} records")
PY

du -sh "${OUTPUT_ROOT}/lung" "${OUTPUT_ROOT}/heart"
