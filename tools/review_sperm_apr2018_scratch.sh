#!/bin/bash
# review_sperm_apr2018_scratch.sh
#
# Gradio review of the 53 April 2018 sperm_whale candidates produced by orca_v11.
# Writes labels ONLY to the scratch copy of the April 2018 database. The frozen originals
# (db/MARS_20180401_20180430_32kHz_norm and db/MARS_20180501_20180531_32kHz_norm) are never written.
#
# Usage (on spark-ae0e, from ~/perch-hoplite):
#   bash review_sperm_apr2018_scratch.sh <annotator-name>
#   e.g. bash review_sperm_apr2018_scratch.sh expert_jane
#
# Use a different annotator name for each reviewer. Labels are saved under that name, so the
# experts' decisions can be compared with each other and with duane's.
#
# Re-annotating a window overwrites the earlier label. The first review (duane, 53 labels saved
# Nov 20 2026) is in the scratch database already. Reviewers who want an independent record should
# work from a second scratch copy, so their labels do not overwrite duane's.

set -u

ANNOTATOR="${1:-}"
if [ -z "$ANNOTATOR" ]; then
    echo "usage: bash $0 <annotator-name>"
    exit 1
fi

DB="/mnt/PAM_Analysis/perch-hoplite/scratch_review_frozen_2018/SCRATCH_APR2018_32kHz_norm"
CSV="/mnt/PAM_Analysis/perch-hoplite/results/sperm_candidates_apr2018_review.csv"
CLF="/mnt/PAM_Analysis/perch-hoplite/models/orca_v11.pt"
AUDIO="/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04"
LOG="/mnt/PAM_Analysis/perch-hoplite/logs/sperm_review_apr2018_scratch_${ANNOTATOR}.log"

# Refuse to run against the frozen originals.
case "$DB" in
    *scratch_review_frozen_2018*) ;;
    *) echo "ERROR: DB is not a scratch copy: $DB"; exit 1 ;;
esac

for f in "$CSV" "$CLF"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -d "$DB" ] || { echo "ERROR: missing scratch database $DB"; exit 1; }
[ -d "$AUDIO" ] || { echo "ERROR: missing audio folder $AUDIO"; exit 1; }

pkill -f "phase2_classify.py review"
sleep 3
if ss -ltn | grep -q ":7878"; then
    echo "ERROR: port 7878 is still in use; stop the other session first"
    exit 1
fi

cd "$HOME/perch-hoplite" || exit 1
nohup python3 phase2_classify.py review \
    --db-dir "$DB" \
    --classifier "$CLF" \
    --target-label sperm_whale \
    --detections-csv "$CSV" \
    --detections-offset 0 \
    --num-results 53 \
    --classes sperm_whale,other,unlabeled \
    --annotator-id "$ANNOTATOR" \
    --audio-dir "$AUDIO" \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7878 \
    > "$LOG" 2>&1 &

sleep 10
echo "launched for annotator '$ANNOTATOR' against scratch copy:"
echo "  $DB"
echo "log: $LOG"
tail -3 "$LOG"
echo "open http://134.89.11.107:7878"
