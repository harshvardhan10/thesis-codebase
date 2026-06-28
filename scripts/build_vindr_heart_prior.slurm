#!/bin/bash -l
#SBATCH --job-name=vindr-heart-prior
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/vindr_heart_prior_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/vindr_heart_prior_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# -----------------------------
# Environment
# -----------------------------
module load python
conda activate albef-notebook

# -----------------------------
# Paths
# -----------------------------
cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

SCRIPT_PATH="build_vindr_heart_prior_xrv.py"

# Change these paths if your VinDr train paths are different
VINDR_TRAIN_IMAGES="/home/woody/iwi5/iwi5362h/data/vindr_cxr/train"
VINDR_TRAIN_LABELS="/home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_train.csv"

OUTPUT_DIR="/home/woody/iwi5/iwi5362h/ALBEF/anatomy_priors"
OUTPUT_NPY="${OUTPUT_DIR}/vindr_train_heart_prior_256.npy"

mkdir -p "${OUTPUT_DIR}"

# TorchXRayVision model cache
XRV_CACHE_DIR="/home/hpc/iwi5/iwi5362h/.torchxrayvision"
mkdir -p "${XRV_CACHE_DIR}"

echo "Python: $(which python)"
echo "Images root: ${VINDR_TRAIN_IMAGES}"
echo "Index CSV: ${VINDR_TRAIN_LABELS}"
echo "Output prior: ${OUTPUT_NPY}"

# -----------------------------
# Run heart prior extraction
# -----------------------------
python "${SCRIPT_PATH}" \
  --images_root "${VINDR_TRAIN_IMAGES}" \
  --index_csv "${VINDR_TRAIN_LABELS}" \
  --output_npy "${OUTPUT_NPY}" \
  --image_ext png \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --heart_threshold 0.5 \
  --output_size 256 \
  --cache_dir "${XRV_CACHE_DIR}"

echo "Job finished on: $(date)"