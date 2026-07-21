#!/bin/bash -l
#SBATCH --job-name=a0-chexmask
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/a0_chexmask_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/a0_chexmask_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail
unset SLURM_EXPORT_ENV

module load python
conda activate albef-notebook
cd /home/woody/iwi5/iwi5362h/ALBEF

VIEWS=(lung heart)
VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
  Pretrain_A0_chexmask.py \
  --config "configs/Pretrain_A0_${VIEW}_chexmask.yaml" \
  --output_dir "/home/vault/iwi5/iwi5362h/outputs/A0_${VIEW}_chexmask" \
  --checkpoint "" \
  --resume false \
  --auto_resume true \
  --distributed true
