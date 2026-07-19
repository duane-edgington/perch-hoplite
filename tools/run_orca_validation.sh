#!/usr/bin/env bash
# run_orca_validation.sh
# ---------------------------------------------------------------------------
# Multi-month orca validation inference (v4 cross-season classifier).
#
# Runs inference over the four resampled months for which we have ground-truth
# regions, at a LOW logit floor (0.0) so that a single run captures every orca
# logit >= 0. The companion scorer (tools/score_orca_regions.py) then sweeps the
# decision threshold (+1.0 / +1.16 / +1.5 / +2.0) in post-processing WITHOUT
# re-running inference.
#
# Why v4: cross-season generalist. Applying ONE model uniformly across all four
# months keeps the month-to-month counts comparable (mixing v1/v2/v4 would mix
# decision boundaries). v4's F1-optimal orca threshold is +1.16 — the scorer
# brackets it.
#
# Known ground-truth regions (see CLAUDE.md "Annotation State"):
#   April 2018   — Apr 13 confirmed Bigg's event (POSITIVE, expect retention)
#   May 2018     — May 12 confirmed event (POSITIVE); May 14 secondary (probable)
#   October 2020 — confirmed ZERO orca vocalizations (NEGATIVE, pure specificity)
#   April 2026   — Apr 17–24 CA51A/CA50B window, humpback FPs (NEGATIVE for orca)
#
# Usage (walk-away; ~30 min/month per known /tmp write speed, item #3):
#   nohup bash tools/run_orca_validation.sh > \
#     /mnt/PAM_Analysis/perch-hoplite/logs/orca_validation.log 2>&1 &
#   tail -f /mnt/PAM_Analysis/perch-hoplite/logs/orca_validation.log
# ---------------------------------------------------------------------------
set -euo pipefail

DB_BASE=/mnt/PAM_Analysis/perch-hoplite/db
RESULTS=/mnt/PAM_Analysis/perch-hoplite/results
LOGS=/mnt/PAM_Analysis/perch-hoplite/logs
MODEL=/mnt/PAM_Analysis/perch-hoplite/models/orca_v4.pt

# Capture floor. 0.0 keeps the current-default baseline column in the sweep.
# If /tmp write times are painful, raise to 1.0 — you lose the 0.0 baseline but
# still cover the +1.0..+2.0 ladder. (The real fix is item #8: --output-format full.)
FLOOR=0.0

# month_key : db_dir_name : output_csv_name
MONTHS=(
  "apr2018:MARS_20180401_20180430_32kHz_norm:MARS_20180401_20180430_v4_orcaval.csv"
  "may2018:MARS_20180501_20180531_32kHz_norm:MARS_20180501_20180531_v4_orcaval.csv"
  "oct2020:MARS_20201001_20201031_32kHz_norm:MARS_20201001_20201031_v4_orcaval.csv"
  "apr2026:MARS_20260401_20260430_32kHz_norm:MARS_20260401_20260430_v4_orcaval.csv"
)

mkdir -p "$RESULTS" "$LOGS"
echo "=== Orca validation inference — v4 @ floor ${FLOOR} — $(date) ==="
echo "Model: $MODEL"
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; exit 1; }

for entry in "${MONTHS[@]}"; do
  IFS=":" read -r key db_name csv_name <<< "$entry"
  db="$DB_BASE/$db_name"
  out="$RESULTS/$csv_name"
  echo
  echo "--- [$key] $(date) ---"
  if [ ! -d "$db" ]; then
    echo "WARNING: DB missing, skipping: $db"
    continue
  fi
  echo "  db:  $db"
  echo "  out: $out"
  time python3 phase2_classify.py infer \
      --db-dir "$db" \
      --classifier "$MODEL" \
      --output-csv "$out" \
      --logit-threshold "$FLOOR"
  rows=$(( $(wc -l < "$out") - 1 ))
  echo "  wrote $rows detection rows"
done

echo
echo "=== done — $(date) ==="
echo "Score with:"
echo "  python3 tools/score_orca_regions.py --results-dir $RESULTS"
