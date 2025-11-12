#!/bin/bash
#SBATCH --job-name=unzip-mimic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de
#SBATCH --output=$WORK/logs/unzip-mimic-%j.out
#SBATCH --error=$WORK/logs/unzip-mimic-%j.err

echo "Starting unzip at $(date)"

cd $WORK/data/mimic_cxr || exit 1

mkdir -p extracted
unzip -o Mimic_CXR_Org.zip -d extracted

echo "Finished unzip at $(date)"
