#!/bin/bash -l
#SBATCH --job-name=albef-A6-sanity
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_A6_sanity_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_A6_sanity_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface
export HF_HOME=/home/woody/iwi5/iwi5362h/.cache/huggingface

# Important: unique port for each distributed job
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
echo "MASTER_PORT: $MASTER_PORT"

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

python -u -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  --use_env \
  extract_a6_0_identity_patch_head_heatmaps.py \
  --config configs/Pretrain_A5_patch_head.yaml \
  --checkpoint output_anatomy_A0_full/checkpoint_best.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir results/anatomy_prior_screening_full/A6.0/patch_head_heatmaps \
  --only_labels Cardiomegaly \
  --layers_to_use 8 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --no_save_attn_debug