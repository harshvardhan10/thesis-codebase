#!/bin/bash -l
#
#SBATCH --job-name=albef-rtx2
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:2                 # 2 x RTX3080
#SBATCH --time=24:00:00
#SBATCH --ntasks=1                   # one launcher task; torch.distributed will spawn 2 procs
#SBATCH --cpus-per-task=8            # CPU cores for dataloading etc.
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_%j.out   # stdout log
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_%j.err    # stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de   # <-- change this

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# --------- Modules / Conda env ---------

module load python
conda activate albef-env

# Unbuffered Python so logs flush quickly
export PYTHONUNBUFFERED=1

# Optional: fewer OpenMP threads per process
export OMP_NUM_THREADS=1

# --------- Go to repo ---------

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1
mkdir -p output_mimic_rtx3080

echo "Using config: ./configs/Pretrain_rtx3080.yaml"
echo "Output dir:  ./output_mimic_rtx3080"

# --------- Launch distributed training on 2 GPUs ---------

python -u -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    Pretrain.py \
    --config ./configs/Pretrain_rtx3080.yaml \
    --output_dir ./output_mimic_rtx3080

echo "Job finished at $(date)"
