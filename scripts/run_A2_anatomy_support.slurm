#!/bin/bash -l
#SBATCH --job-name=albef-A2-debug
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_A2_debug_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_A2_debug_%j.err
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

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

mkdir -p ./output_anatomy_A2_support_debug

python -u -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    Pretrain_anatomy_prior.py \
    --config ./configs/Pretrain_anatomy_prior_A2.yaml \
    --output_dir ./output_anatomy_A2_support_debug

echo "Job finished at $(date)"