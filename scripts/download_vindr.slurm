#!/bin/bash
#SBATCH --job-name=download-vindr-zip
#SBATCH --output=download-vindr-zip-%j.out
#SBATCH --error=download-vindr-zip-%j.err
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

echo "Download started at $(date)"
cd $WORK/data/vindr_cxr

# Replace the URL below with the actual direct ZIP file link
ZIP_URL="https://physionet.org/static/published-projects/vindr-cxr/vindr-cxr-1.0.0.zip"

# Use wget to download with visual progress dots per MB
wget --progress=dot:mega -O vindr-cxr.zip "$ZIP_URL"

echo "Download completed at $(date)"
ls -lh vindr-cxr.zip
