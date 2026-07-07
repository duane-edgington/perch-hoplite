#!/bin/bash
# Expert review — October 2020 high-confidence orca detections
# 81 detections, logit >= 3.0, October 5-12 2020 event window
#
# Prerequisites:
#   source ~/perch-hoplite/venv/bin/activate
#   Use Chrome (incognito recommended) — Safari has audio playback issues
#
# John Ryan (and other experts) open: http://134.89.11.107:7861
#
# Instructions for reviewers:
#   - Listen to each 5-second clip
#   - Click orca_call if you confirm orca
#   - Click humpback_song if it is humpback (likely common)
#   - Click dolphin_call if it is dolphin
#   - Click unlabeled if unsure (removes from training)
#   - Labels auto-save on each click

cd ~/perch-hoplite
source ~/perch-hoplite/venv/bin/activate

# Batch 1 (clips 1-25)
pkill -f "phase2_classify" 2>/dev/null || true

nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20201001_20201031_32kHz \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v7_clean.pt \
    --target-label orca_call \
    --detections-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20201001_20201031_v7_orca_logit3_detections.csv \
    --num-results 25 \
    --detections-offset 0 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2020/10 \
    --serve --port 7861 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_oct2020_logit3_7861.log 2>&1 &

echo "Gradio running at http://134.89.11.107:7861"
echo "Open in Chrome (incognito) — Safari has audio issues"
echo ""
echo "After batch 1, run batch 2 with --detections-offset 25"
echo "After batch 2, run batch 3 with --detections-offset 50"
echo "(81 total detections = 4 batches of 25)"
