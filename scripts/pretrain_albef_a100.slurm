#!/bin/bash -l
#SBATCH --job-name=albef-pretrain
#SBATCH --gres=gpu:v100:4  # trying with v100 as a100 is not available
#SBATCH --partition=v100
#SBATCH --time=24:00:00            # adjust as needed
#SBATCH --ntasks=1                 # one launcher task; torch.distributed will spawn 4 procs
#SBATCH --cpus-per-task=8          # CPU cores for data loading etc.
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_%j.out # stdout log (create logs/ before submitting!)
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_%j.err  # stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de  # <-- change this

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# --------- Modules / Conda env ---------

module load python              # cluster's Python module
conda activate albef-env

# --------- Logging & NCCL debug settings ---------

# Python: unbuffered output
export PYTHONUNBUFFERED=1

# Make stdio line-buffered to get "live" logs into the file
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdbuf.so
export _STDBUF_O=L
export _STDBUF_E=L

# Optional but useful if NCCL gives trouble
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=1

# --------- Run training (single node, 4 GPUs) ---------

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

echo "Launching distributed training..."
echo "Using config: ./configs/Pretrain.yaml"
echo "Output dir:  ./output_mimic"

# Using the legacy launch module (what ALBEF README uses)
stdbuf -oL -eL python -u -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    Pretrain.py \
    --config ./configs/Pretrain.yaml \
    --output_dir ./output_mimic

echo "Job finished at $(date)"
