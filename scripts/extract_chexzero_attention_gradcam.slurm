#!/bin/bash -l
#
#SBATCH --job-name=extract-chexzero-gradcam
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/extract-chexzero-gradcam_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/extract-chexzero-gradcam_%j.err
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

python -u zero_shot_vindr.py \
  --vindr_root /home/woody/iwi5/iwi5362h/data/vindr_cxr \
  --model_paths /home/woody/iwi5/iwi5362h/CheXzero/checkpoints \
  --pretrained \
  --device cuda

python extract_chexzero_multi_checkpoint_attention_gradcam.py \
  --model_paths /home/woody/iwi5/iwi5362h/CheXzero/checkpoints \
  --labels_csv /home/woody/iwi5/iwi5362h/data/vindr_cxr/annotations/image_labels_test.csv \
  --images_root /home/woody/iwi5/iwi5362h/data/vindr_cxr/test \
  --output_dir /home/vault/chexzero_heatmaps \
  --target_labels Cardiomegaly "Pleural effusion" \
  --positive_template "{}" \
  --negative_template "no {}" \
  --attention_layer -1 \
  --output_map_size 256 \
  --device cuda

echo "Job finished at $(date)"
