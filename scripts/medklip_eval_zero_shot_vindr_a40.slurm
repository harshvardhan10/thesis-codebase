#!/bin/bash -l
#
#SBATCH --job-name=chexzero-vindr-zeroshot-a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1          # 1 x A40
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/chexzero_vindr_zeroshot_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/chexzero_vindr_zeroshot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python
conda activate chexzero-env

cd /home/woody/iwi5/iwi5362h/CheXzero || exit 1

export PYTHONPATH=/home/woody/iwi5/iwi5362h/CheXzero:$PYTHONPATH

python -u zero_shot_vindr/medklip_zero_shot_vindr.py \
  --config /home/woody/iwi5/iwi5362h/MedKLIP/zero_shot_vindr/MedKLIP_config.yaml \
  --model_paths /home/woody/iwi5/iwi5362h/MedKLIP/checkpoints \
  --vindr_root /home/woody/iwi5/iwi5362h/data/vindr_cxr \
  --out_dir "vindr_medklip_full_$(date)"

echo "Job finished at $(date)"
