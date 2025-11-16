#!/bin/bash
#SBATCH --job-name=download-mimic-reports
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=$WORK/logs/download_txt_all-%j.out
#SBATCH --error=$WORK/logs/download_txt_all-%j.err

BASE_DIR="$WORK/data/mimic_cxr/extracted/2.1.0/files"
LOG_DIR="$WORK/logs"
mkdir -p "$LOG_DIR"

echo "Download started at $(date)"

for pdir in "$BASE_DIR"/p1[0-9]/*; do
  if [ -d "$pdir" ]; then
    for study_path in "$pdir"/s*/; do
      study_id=$(basename "$study_path")
      txt_path="$pdir/$study_id.txt"

      if [ ! -f "$txt_path" ]; then
        echo "Downloading $study_id.txt to $txt_path..."
        wget -q -c \
          "https://physionet.org/files/mimic-cxr/2.0.0/files/$(basename "$(dirname "$pdir")")/$(basename "$pdir")/$study_id.txt" \
          -O "$txt_path"
      fi
    done
  fi
done

echo "Download finished at $(date)"
