#!/usr/bin/env bash
# archive_tsne_by_day.sh
# Generate the by-day orca t-SNE at 3 perplexities x 2 styles (12 figures total)
# straight into figures/, then register each with provenance. One run replaces
# ~12 hand-typed register_figure.py calls.
#
# Matrix: {april2018, 4days} x {px10, px30, px50} x {analysis(light), presentation(dark)}
#
# Run from repo root:  bash tools/archive_tsne_by_day.sh
set -u

APRIL=/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm
MAY=/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm
FIG=figures

echo "=== 1/2 generating 12 plots into $FIG/ ==="
for px in 10 30 50; do
  for style in analysis presentation; do
    echo "  -- perplexity=$px style=$style"
    python3 tools/plot_tsne_orca_by_day.py \
        --april-db "$APRIL" --may-db "$MAY" \
        --out-dir "$FIG" --perplexity "$px" --style "$style" \
        >/dev/null || { echo "GENERATION FAILED at px=$px style=$style"; exit 1; }
  done
done

echo "=== 2/2 registering figures ==="
shopt -s nullglob
count=0
for f in "$FIG"/tsne_orca_by_day_*_px*.png; do
  name=$(basename "$f")
  case "$name" in
    *_4days_*) dayset="4 confirmed days (Apr 13/18/25 + May 12 2018)";;
    *)         dayset="April 2018 by confirmed day (Apr 13/18/25)";;
  esac
  case "$name" in
    *_pres.png) styletag="presentation (dark theme)"; styarg="presentation";;
    *)          styletag="analysis (light theme)";   styarg="analysis";;
  esac
  px=$(echo "$name" | sed -E 's/.*_px([0-9]+).*/\1/')

  python3 tools/register_figure.py \
    --saved-name "$name" --original-name "$name" \
    --computer spark-ae0e --type tsne_plot \
    --caption "t-SNE of confirmed orca_call embeddings (Perch V2, cosine) — ${dayset}, perplexity=${px}, ${styletag}. Apr 25 (evening encounter) forms a distinct cluster separated from the Apr 13 morning Bigg's event WITHIN the same month; separation is ROBUST across perplexity 10/30/50 and spans 10 recordings over ~3.5h (not a single-recording/boat artifact). Apr 18 is partially distinct. Exploratory — t-SNE distances/sizes are not meaningful. Finding #14 (extended April 2018 orca presence)." \
    --command "python3 tools/plot_tsne_orca_by_day.py --april-db MARS_20180401_20180430_32kHz_norm --may-db MARS_20180501_20180531_32kHz_norm --perplexity ${px} --style ${styarg} --out-dir figures" \
    --notes "Confirmed-orca windows only (filename-date filter). Apr 25 within-month separation survives perplexity + multi-recording confounds; remaining question (pod vs individual vs call-type vs evening acoustic context) pending direct listening. Expert reviewer: D. Edgington." \
    && count=$((count+1))
done

echo
echo "Registered $count / 12 figures."
sidecars=$(ls "$FIG"/tsne_orca_by_day_*_px*.json 2>/dev/null | wc -l)
echo "Sidecars present: $sidecars (expect 12)."
if [ "$count" -eq 12 ] && [ "$sidecars" -eq 12 ]; then
  echo "OK — all 12 generated and registered."
else
  echo "WARNING — count mismatch; check output above before committing."
fi
