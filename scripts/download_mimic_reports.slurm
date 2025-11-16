#!/bin/bash
#SBATCH --job-name=download-txt-reports
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=$WORK/logs/download_txt_reports-%j.out
#SBATCH --error=$WORK/logs/download_txt_reports-%j.err

echo "SLURM job started at $(date)"
echo "Running on node: $HOSTNAME"

BASE_DIR="$WORK/data/mimic_cxr/extracted/2.1.0/files"
LOG_DIR="$WORK/logs"
mkdir -p "$LOG_DIR"

echo "Scanning directories under: $BASE_DIR"

for pdir in "$BASE_DIR"/p1[0-9]/*; do
  if [ -d "$pdir" ]; then
    for study_path in "$pdir"/s*/; do
      study_id=$(basename "$study_path")
      txt_path="$pdir/$study_id.txt"

      if [ ! -f "$txt_path" ]; then
        echo "Downloading $study_id.txt..."
        wget -q -c \
          "https://physionet.org/files/mimic-cxr/2.1.0/files/$(basename "$(dirname "$pdir")")/$(basename "$pdir")/$study_id.txt" \
          -O "$txt_path"
      fi
    done
  fi
done

echo "Download completed at $(date)"
