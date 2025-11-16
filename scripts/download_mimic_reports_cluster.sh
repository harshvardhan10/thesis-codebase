#!/bin/bash

BASE_DIR="$WORK/data/mimic_cxr/extracted/2.1.0/files"
LOG_FILE="$WORK/logs/download_mimic_reports_$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Download started at $(date)" | tee -a "$LOG_FILE"
echo "Scanning directories under $BASE_DIR" | tee -a "$LOG_FILE"

for pdir in "$BASE_DIR"/p1[0-9]/*; do
  if [ -d "$pdir" ]; then
    echo "Processing $pdir" | tee -a "$LOG_FILE"
    for study_path in "$pdir"/s*/; do
      study_id=$(basename "$study_path")
      txt_path="$pdir/$study_id.txt"

      if [ ! -f "$txt_path" ]; then
        echo "Downloading $study_id.txt..." | tee -a "$LOG_FILE"
        wget -q -c \
          "https://physionet.org/files/mimic-cxr/2.1.0/files/$(basename "$(dirname "$pdir")")/$(basename "$pdir")/$study_id.txt" \
          -O "$txt_path"

        if [ $? -eq 0 ]; then
          echo "Downloaded $study_id.txt" | tee -a "$LOG_FILE"
        else
          echo "Failed to download $study_id.txt" | tee -a "$LOG_FILE"
        fi
      fi
    done
  fi
done

echo "Download finished at $(date)" | tee -a "$LOG_FILE"
