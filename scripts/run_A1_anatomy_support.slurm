#!/bin/bash -l
#SBATCH --job-name=albef-A1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_A1_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_A1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface
export HF_HOME=/home/woody/iwi5/iwi5362h/.cache/huggingface

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

python -u -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    Pretrain_anatomy_prior.py \
    --config ./configs/Pretrain_anatomy_prior_A1.yaml \
    --output_dir ./output_anatomy_A1_support_debug