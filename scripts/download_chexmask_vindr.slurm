#!/bin/bash -l
#SBATCH --job-name=chexmask-vindr
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/chexmask_vindr_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/chexmask_vindr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/download_chexmask_common.sh"
download_chexmask_file "VinDr-CXR.csv"
