#!/bin/bash
# Expert review — May 12 2018 orca detections
# 190 detections clustered UTC 07-11 (PDT 00-04)
# Peak hour UTC 08 (105 detections)
#
# Prerequisites:
#   cd ~/perch-hoplite
#   source ~/perch-hoplite/venv/bin/activate
#   Use Chrome (incognito) — Safari has audio playback issues
#
# John Ryan (and other experts) open: http://134.89.11.107:7862
#
# Instructions for reviewers:
#   - Listen to each 5-second clip
#   - Click orca_call if you confirm orca
#   - Click humpback_song / dolphin_call if misclassified
#   - Click unlabeled if unsure
#   - Labels auto-save on each click

cd ~/perch-hoplite
source ~/perch-hoplite/venv/bin/activate

# Step 1: Create filtered CSV — May 12 orca detections only, sorted by logit score
echo "Creating filtered CSV for May 12 2018 orca detections..."
awk -F',' 'NR==1 || ($6=="orca_call" && $3 ~ /20180512/)' \
    /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180501_20180531_v7_clean_detections.csv \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180512_v7_orca_detections.csv

echo "May 12 orca detections:"
wc -l /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180512_v7_orca_detections.csv

# Step 2: Launch Gradio review on port 7862
pkill -f "port 7862" 2>/dev/null || true

nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180501_20180531_32kHz \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v7_clean.pt \
    --target-label orca_call \
    --detections-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180512_v7_orca_detections.csv \
    --num-results 25 \
    --detections-offset 0 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/05 \
    --serve --port 7862 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_may2018_orca_7862.log 2>&1 &

echo ""
echo "Gradio running at http://134.89.11.107:7862"
echo "Open in Chrome (incognito) — Safari has audio issues"
echo ""
echo "190 total detections — run in batches:"
echo "  Batch 1 (clips 1-25):   --detections-offset 0   (current)"
echo "  Batch 2 (clips 26-50):  --detections-offset 25"
echo "  Batch 3 (clips 51-75):  --detections-offset 50"
echo "  Batch 4 (clips 76-100): --detections-offset 75"
echo "  ... up to offset 175 for all 190 detections"
echo ""
echo "NOTE: October 2020 expert review runs simultaneously on port 7861"
echo "  http://134.89.11.107:7861"
