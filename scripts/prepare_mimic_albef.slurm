#!/bin/bash -l
#SBATCH --job-name=prep-mimic-albef
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/prepare_mimic_albef-%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/prepare_mimic_albef-%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# activate your environment
module add python
conda activate albef-env   # change if your env has a different name

cd $HOME/thesis-codebase/scripts || exit 1

python prepare_mimic_albef.py \
  --base-dir $WORK/data/mimic_cxr/extracted/2.1.0/files \
  --negbio-csv $WORK/data/mimic_cxr/extracted/2.1.0/mimic-cxr-2.0.0-negbio.csv \
  --metadata-csv $WORK/data/mimic_cxr/extracted/2.1.0/mimic-cxr-2.0.0-metadata.csv \
  --output-json $WORK/ALBEF/data/mimic_cxr.json \
  --first-image-only

echo "Job finished at $(date)"
