#!/bin/bash -l
#SBATCH --job-name=albef-A5-patch-head-cm-lambda-0
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_A5_patch_head_lambda_0_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_A5_patch_head_lambda_0_%j.err
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

mkdir -p ./output_anatomy_A5_patch_head_cm_lambda_0

python -u -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  --use_env \
  Pretrain_anatomy_prior_A5_patch_head.py \
  --config ./configs/Pretrain_A5_patch_head.yaml \
  --output_dir ./output_anatomy_A5_patch_head_cm_lambda_0 \
  --text_encoder bert-base-uncased \
  --device cuda \
  --distributed True \
  --auto_resume True

