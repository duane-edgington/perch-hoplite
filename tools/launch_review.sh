#!/bin/bash
# launch_review.sh  -  start one Gradio review session (spark-ae0e, port 7878)
#
# Usage:
#   bash ~/perch-hoplite/tools/launch_review.sh YYYY-MM REVIEW_CSV NUM_RESULTS AUDIO_DIR|auto [--dry-run]
#
# Example:
#   bash ~/perch-hoplite/tools/launch_review.sh 2020-04 review_apr2020_pass2_chunk5.csv 25 auto
#
# What it does, in order:
#   1. shows what is on the GPU (nvidia-smi)
#   2. stops any running "phase2_classify.py review" session and waits for port 7878 to free
#   3. finds the month's *_norm database and (with "auto") the month's audio directory
#   4. launches the review session in the background, logging to logs/<REVIEW_CSV name>.log
#   5. prints the log tail
# Nothing is deleted. --dry-run prints the launch command and stops before step 2.

set -u

DB_ROOT="${DB_ROOT:-/mnt/PAM_Analysis/perch-hoplite/db}"
RESULTS="${RESULTS:-/mnt/PAM_Analysis/perch-hoplite/results}"
LOGS="${LOGS:-/mnt/PAM_Analysis/perch-hoplite/logs}"
MODEL="${MODEL:-/mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt}"
PORT="${PORT:-7878}"

if [ $# -lt 4 ]; then
  echo "usage: $0 YYYY-MM REVIEW_CSV NUM_RESULTS AUDIO_DIR|auto [--dry-run]"
  exit 1
fi
YM="$1"; CSV_NAME="$2"; NUM="$3"; AUDIO="$4"; DRY="${5:-}"
Y="${YM%-*}"; M="${YM#*-}"

CSV_PATH="$RESULTS/$CSV_NAME"
[ -f "$CSV_PATH" ] || { echo "ERROR: review CSV not found: $CSV_PATH"; exit 1; }

DB=$(ls -d "$DB_ROOT"/MARS_${Y}${M}01_*_32kHz_norm 2>/dev/null | head -1)
[ -n "$DB" ] || { echo "ERROR: no *_norm database for $YM under $DB_ROOT"; exit 1; }

if [ "$AUDIO" = "auto" ]; then
  AUDIO=""
  for root in /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz /mnt/PAM_20TiB_32k \
              /mnt/PAM_32k_Archive/resampled_32kHz /mnt/SL500/resampled_32kHz /mnt/SL500; do
    if [ -d "$root/$Y/$M" ] && [ -n "$(ls "$root/$Y/$M" 2>/dev/null | head -1)" ]; then
      AUDIO="$root/$Y/$M"; break
    fi
  done
  [ -n "$AUDIO" ] || { echo "ERROR: audio for $YM not found in the usual locations; pass the directory explicitly"; exit 1; }
fi
[ -d "$AUDIO" ] || { echo "ERROR: audio directory not found: $AUDIO"; exit 1; }

LOG="$LOGS/${CSV_NAME%.csv}.log"
echo "month:  $YM"
echo "db:     $DB"
echo "audio:  $AUDIO ($(ls "$AUDIO" | wc -l) files)"
echo "review: $CSV_PATH ($NUM results)"
echo "log:    $LOG"

CMD=(python3 phase2_classify.py review
     --db-dir "$DB" --classifier "$MODEL" --target-label orca_call
     --detections-csv "$CSV_PATH" --detections-offset 0 --num-results "$NUM"
     --classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled
     --annotator-id duane --audio-dir "$AUDIO"
     --spectrogram-type mel --colormap viridis --serve --port "$PORT")

if [ "$DRY" = "--dry-run" ]; then
  echo; echo "DRY RUN - would run (from ~/perch-hoplite):"; echo "  ${CMD[*]}"
  exit 0
fi

echo; echo "--- GPU:"; nvidia-smi | grep -E "python|No running"
pkill -f "phase2_classify.py review"
sleep 3
if ss -ltn | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is still in use after stopping the old session; not launching"
  exit 1
fi
echo "port $PORT free"

cd ~/perch-hoplite || exit 1
nohup "${CMD[@]}" > "$LOG" 2>&1 &
sleep 5
echo; echo "--- log tail:"; tail -3 "$LOG"
echo; echo "Open http://134.89.11.107:$PORT   (first clip's score should fall inside this chunk's score range)"
