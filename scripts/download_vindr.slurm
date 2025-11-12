#!/bin/bash
#SBATCH --job-name=download-vindr-zip
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

# ==== Setup dynamic log filename ====
timestamp=$(date +%Y%m%d-%H%M%S)
LOGDIR=$WORK/logs
mkdir -p "$LOGDIR"
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/download-vindr-${timestamp}-%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/download-vindr-${timestamp}-%j.err

echo "Download started at $(date)"
cd $WORK/data/vindr_cxr

# === Download ZIP ===
ZIP_URL="https://physionet.org/content/vindr-cxr/get-zip/1.0.0/"
ZIP_FILE="vindr-cxr.zip"

wget -c --progress=dot:mega -O "$ZIP_FILE" "$ZIP_URL"

echo "Download completed at $(date)"
ls -lh "$ZIP_FILE"

# === Confirm success in log ===
if [ -f "$ZIP_FILE" ]; then
    echo "Download successful and file exists."
else
    echo "Download failed or file missing."
fi
