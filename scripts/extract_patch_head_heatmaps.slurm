#!/bin/bash -l
#
#SBATCH --job-name=extract_patch_head_heatmaps_A5_0.01_ckpt_best
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/extract_patch_head_heatmaps_A5_0.01_ckpt_best_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/extract_patch_head_heatmaps_A5_0.01_ckpt_best_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

unset SLURM_EXPORT_ENV

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python
conda activate albef-notebook

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5362h/.cache/huggingface
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
echo "MASTER_PORT: $MASTER_PORT"

cd /home/woody/iwi5/iwi5362h/ALBEF || exit 1

export PYTHONPATH=/home/woody/iwi5/iwi5362h/ALBEF:$PYTHONPATH

#python -u scripts/extract_patch_head_heatmaps_batched_distributed.py \
#  --config configs/Pretrain_A5_patch_head.yaml \
#  --checkpoint \
#    output_anatomy_A5_patch_head_cm_lambda_0.01/checkpoint_best.pth \
#  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
#  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
#  --output_dir results/anatomy_prior_screening_full/A5_lambda_0.01/patch_head_heatmaps_ckpt_best \
#  --only_labels Cardiomegaly \
#  --layers_to_use 8 \
#  --device cuda \
#  --overwrite

python -u -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  --use_env \
  scripts/extract_patch_head_heatmaps_batched_distributed.py \
  --config ./configs/Pretrain_A5_patch_head.yaml \
  --checkpoint ./output_anatomy_A5_patch_head_cm_lambda_0.01/checkpoint_best.pth \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir results/anatomy_prior_screening_full/A5_lambda_0.01/patch_head_heatmaps_ckpt_best \
  --only_labels Cardiomegaly \
  --layers_to_use 8 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --no_save_attn_debug

echo "Job finished at $(date)"
