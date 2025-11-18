#!/bin/bash -l
#
#SBATCH --job-name=albef-a100-1
#SBATCH --partition=a40
#SBATCH --gres=gpu:1                 # debugging with 1 GPU
#SBATCH --time=4:00:00
#SBATCH --ntasks=1                   # one launcher task
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/albef_a100_1_%j.out   # stdout log
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/albef_a100_1_%j.err    # stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# --------- Modules / Conda env ---------

module load python
conda activate albef-env

# Unbuffered Python so logs flush quickly
export PYTHONUNBUFFERED=1

# Optional: fewer OpenMP threads per process
export OMP_NUM_THREADS=1

# HuggingFace cache (shared for all jobs)
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface


# --------- Go to repo ---------

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1
mkdir -p output_mimic_rtx3080

echo "Using config: ./configs/Pretrain_debug.yaml"
echo "Output dir:  ./output_mimic_a100_debug"
echo "HF cache dir: $TRANSFORMERS_CACHE"

# -----------------------------------------------------------------------------------
# 4) Ensure BERT tokenizer/model are cached
# -----------------------------------------------------------------------------------
python - << 'EOF'
from transformers import BertTokenizer, BertModel
try:
    print("Checking BERT cache...")
    BertTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    print("HF models found in cache.")
except Exception as e:
    print("Downloading BERT tokenizer/model due to cache miss...")
    BertTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    print("Completed downloading BERT resources.")
EOF

# --------- Launch distributed training on 8 GPUs ---------

python -u -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    Pretrain.py \
    --config ./configs/Pretrain_debug.yaml \
    --output_dir ./output_mimic_a100_debug

echo "Job finished at $(date)"
