#!/usr/bin/env bash
# register_aug2015_figures.sh — rename + register the 12 keeper framegrabs from the
# August 2015 orca review session (D. Edgington, 2026-08-28, DuaneEM1).
#
# 15 grabs were taken for 14 reviewed clips; 3 were duplicate pairs where the first
# shot cut off the audio control (labels #6, #7, #10). The keepers are the scrolled
# versions. Labels #4 and #5 were never screenshotted.
#
# Run from the repo root on DuaneEM1:  bash tools/register_aug2015_figures.sh
set -u

cd "$(dirname "$0")/.." || exit 1
[ -f tools/register_figure.py ] || { echo "run from repo root"; exit 1; }

REVCMD='nohup python3 phase2_classify.py review --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150801_20150831_32kHz_norm --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt --target-label orca_call --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_aug2015_union16.csv --detections-offset 0 --num-results 16 --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled --annotator-id duane --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2015/08 --spectrogram-type mel --colormap viridis --serve --port 7878'

SESSION_NOTE='August 2015 orca review session, 14 clips = union of orca_v4 >=1.16 and orca_v10 >=1.00, presented score-descending. Session outcome: 13 dolphin_call, 1 orca_call, 0 unlabeled. NOTE the displayed score may be either model: the review set mixes v4 scores (for the 6 windows v4 flagged >=1.16) and v10 scores (for the 8 v10 contributed), so scores in this set are NOT on a single scale.'

# ---------- 1. drop the three duplicate parked grabs ----------
echo "== removing 3 duplicate parked grabs =="
git rm -q --cached figures/gradio_aug2015_tmp_104147.png 2>/dev/null
git rm -q --cached figures/gradio_aug2015_tmp_104218.png 2>/dev/null
git rm -q --cached figures/gradio_aug2015_tmp_104352.png 2>/dev/null
rm -f figures/gradio_aug2015_tmp_104147.png \
      figures/gradio_aug2015_tmp_104218.png \
      figures/gradio_aug2015_tmp_104352.png

# ---------- 2. rename the 12 keepers to convention ----------
echo "== renaming 12 keepers =="
cd figures || exit 1
mv gradio_aug2015_tmp_103222.png gradio_aug08_2015_dolphin_550s_wid226600.png
mv gradio_aug2015_tmp_103314.png gradio_aug04_2015_dolphin_385s_wid262377.png
mv gradio_aug2015_tmp_103421.png gradio_aug05_2015_dolphin_235s_wid43572.png
mv gradio_aug2015_tmp_104158.png gradio_aug10_2015_dolphin_595s_wid57310.png
mv gradio_aug2015_tmp_104230.png gradio_aug05_2015_dolphin_155s_wid43556.png
mv gradio_aug2015_tmp_104247.png gradio_aug08_2015_dolphin_145s_wid11672.png
mv gradio_aug2015_tmp_104333.png gradio_aug28_2015_ORCA_325s_wid255405.png
mv gradio_aug2015_tmp_104403.png gradio_aug14_2015_dolphin_235s_wid97893.png
mv gradio_aug2015_tmp_104438.png gradio_aug01_2015_dolphin_205s_wid398682.png
mv gradio_aug2015_tmp_104525.png gradio_aug08_2015_dolphin_55s_wid156658.png
mv gradio_aug2015_tmp_104608.png gradio_aug20_2015_dolphin_535s_wid405468.png
mv gradio_aug2015_tmp_104623.png gradio_aug06_2015_dolphin_510s_wid138989.png
cd ..
ls figures/gradio_aug*_2015_*.png | wc -l

# ---------- 3. register ----------
reg () {   # reg <saved> <origtime> <wav> <offset> <score> <label> <caption> <notes>
  python3 tools/register_figure.py \
    --saved-name "$1" \
    --original-name "Screenshot 2026-08-28 at $2.png" \
    --computer DuaneEM1 \
    --type gradio_screenshot \
    --wav "$3" --offset "$4" \
    --spectrogram mel --colormap viridis \
    --classifier orca_v10.pt --db MARS_20150801_20150831_32kHz_norm \
    --score "$5" --label "$6" \
    --caption "$7" \
    --notes "$8 $SESSION_NOTE" \
    --command "$REVCMD"
}

echo "== registering 12 =="

reg gradio_aug28_2015_ORCA_325s_wid255405.png "10.43.33 AM" \
  MARS_20150828_212219_resampled_32kHz.wav 325 1.406 orca_call \
  "ORCA CALL candidate, confirmed by ear (D. Edgington), 2015-08-28 21:22 UTC. The ONLY orca_call label in August 2015 and the first non-spring orca candidate of the full-archive campaign. Narrow harmonic stack ~2-4 kHz with a gentle upward inflection, ~1.8-3.0 s into the window; NO energy above ~5 kHz. Visually distinct from all 11 dolphin_call clips in the same session, which sit at 4-16 kHz with steep sweeps. Mel/viridis, 5 s window with 30 s context." \
  "Label #9, wid=255405. UNCONFIRMED -- pending review by J. Ryan. Duane's call: looks and sounds like an orca but lacks the higher frequencies, and is isolated from other calls. Both readings remain open: a DISTANT orca (range strips highs first, and Bigg's killer whales are acoustically cryptic -- long silences then a few isolated calls), or a distant dolphin high-passed by the same propagation. SCORES: v10 1.406 (5th month-wide), v4 0.512 (~20th) -- v4 did score it, below the 1.16 review cutoff, so it entered the set via v10. LOCAL CONTEXT (from a -10 logit-threshold diagnostic run): the most anomalous window in 3 hours. Within its own recording v10's next-highest is -1.155 (median ~-5); across 20:00-23:00 (2160 windows) v10 p99 = -1.99 and next-highest = -0.452. Neighbours at 320 s (-5.767) and 330 s (-3.378) are ordinary background, so the isolation is real and not an artifact of the 0.0 cutoff. CAVEAT: local-outlier magnitude partly measures how quiet the water was; Aug 28 21:00 was acoustically quiet. Month-wide this is mid-pack among windows already labelled dolphin."

reg gradio_aug08_2015_dolphin_55s_wid156658.png "10.45.25 AM" \
  MARS_20150808_015545_resampled_32kHz.wav 55 1.525 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-08 01:56 UTC. Dense harmonic banding across 4-16 kHz. orca_v4's TOP-scoring window of August 2015. Mel/viridis, 5 s window with 30 s context." \
  "Label #12, wid=156658. Score 1.525 is the orca_v4 score (v10 scored this window 1.367, 7th month-wide). Aug 8 is the month's acoustic hotspot -- 3 recordings (01:55, 12:45, 15:25) contribute 6 of v4's top 15 windows, all reviewed as dolphin."

reg gradio_aug14_2015_dolphin_235s_wid97893.png "10.44.03 AM" \
  MARS_20150814_110503_resampled_32kHz.wav 235 1.506 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-14 11:09 UTC. Energy concentrated at 8-10 kHz. orca_v10's TOP-scoring window of August 2015 (1.993) and orca_v4's second (1.506). Mel/viridis, 5 s window with 30 s context." \
  "Label #10, wid=97893. Displayed score 1.506 is the orca_v4 score; orca_v10 scored 1.993 -- the highest v10 score anywhere in the month, and still BELOW v10's 2.31 operating threshold. Duplicate grab (10.44.03 AM) kept over the earlier 10.43.52 AM shot, which cut off the audio control."

reg gradio_aug06_2015_dolphin_510s_wid138989.png "10.46.23 AM" \
  MARS_20150806_222909_resampled_32kHz.wav 510 1.394 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-06 22:37 UTC. Descending sweep from ~16 kHz early in the window plus a harmonic stack near 5 kHz. Mel/viridis, 5 s window with 30 s context." \
  "Label #14, wid=138989. Score 1.394 is the orca_v10 score (6th month-wide); v4 scored 0.889, below the 1.16 cutoff, so this window entered the review set via v10."

reg gradio_aug20_2015_dolphin_535s_wid405468.png "10.46.08 AM" \
  MARS_20150820_111926_resampled_32kHz.wav 535 1.337 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-20 11:28 UTC. Faint -- only a trace of energy near 2 kHz around 3.7 s. Mel/viridis, 5 s window with 30 s context." \
  "Label #13, wid=405468. Score 1.337 is the orca_v10 score (8th month-wide); v4 did not place this window in its top 15. A v10-contributed clip that Duane's ear resolved as dolphin -- relevant to the v4-vs-v10 comparison."

reg gradio_aug01_2015_dolphin_205s_wid398682.png "10.44.38 AM" \
  MARS_20150801_072345_resampled_32kHz.wav 205 1.283 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-01 07:27 UTC, the first full day after the July 2015 deployment month. Mel/viridis, 5 s window with 30 s context." \
  "Label #11, wid=398682. Displayed score 1.283 is the orca_v4 score; v10 scored 1.574 (3rd month-wide). Aug 1 is a COMPLETE day (144 files), which is what makes the July 2015 month-boundary question (finding #29) answerable."

reg gradio_aug04_2015_dolphin_385s_wid262377.png "10.33.14 AM" \
  MARS_20150804_091503_resampled_32kHz.wav 385 1.246 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-04 09:21 UTC. Clear harmonic stack around 2 s. Mel/viridis, 5 s window with 30 s context." \
  "Label #2, wid=262377. Displayed score 1.246 is the orca_v4 score; v10 scored 1.672 (2nd month-wide)."

reg gradio_aug10_2015_dolphin_595s_wid57310.png "10.41.58 AM" \
  MARS_20150810_123546_resampled_32kHz.wav 595 1.191 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-10 12:45 UTC. Mel/viridis, 5 s window with 30 s context." \
  "Label #6, wid=57310. Score 1.191 is the orca_v10 score. NOTE this is the FINAL window of the recording (595-600 s), so the 30 s context pane is truncated at the file end. Duplicate grab (10.41.58 AM) kept over the earlier 10.41.47 AM shot, which cut off the audio control."

reg gradio_aug05_2015_dolphin_155s_wid43556.png "10.42.30 AM" \
  MARS_20150805_125503_resampled_32kHz.wav 155 1.173 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-05 12:57 UTC. Multiple whistles and a harmonic stack; the 30 s context is dense with dolphin activity. Mel/viridis, 5 s window with 30 s context." \
  "Label #7, wid=43556. Score 1.173 is the orca_v10 score. Same recording as wid=43572 (235 s), 80 s later -- a sustained dolphin group. Duplicate grab (10.42.30 AM) kept over the earlier 10.42.18 AM shot, which cut off the audio control. NOTE 2015-08-05 is the recorder-thrashing day: 152 files, including 8 files in under 4 minutes at 13:15-13:18 (see finding #31)."

reg gradio_aug05_2015_dolphin_235s_wid43572.png "10.34.21 AM" \
  MARS_20150805_125503_resampled_32kHz.wav 235 1.166 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-05 12:58 UTC. Textbook sweeping whistles 8-16 kHz; the 30 s context shows many more. Mel/viridis, 5 s window with 30 s context." \
  "Label #3, wid=43572. Score 1.166 is the orca_v10 score. Same recording as wid=43556 (155 s), 80 s earlier."

reg gradio_aug08_2015_dolphin_145s_wid11672.png "10.42.47 AM" \
  MARS_20150808_152545_resampled_32kHz.wav 145 1.535 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-08 15:28 UTC. Descending sweep near 8-10 kHz. Mel/viridis, 5 s window with 30 s context." \
  "Label #8, wid=11672. Score 1.535 is the orca_v10 score (4th month-wide); v4 scored 1.017, below the 1.16 cutoff, so this entered via v10. Third of the three Aug 8 recordings that make that date the month's hotspot."

reg gradio_aug08_2015_dolphin_550s_wid226600.png "10.32.22 AM" \
  MARS_20150808_124545_resampled_32kHz.wav 550 1.068 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-08-08 12:54 UTC. Mel/viridis, 5 s window with 30 s context." \
  "Label #1, wid=226600. Score 1.068 is the orca_v10 score. This recording contributed TWO windows to v10's top 14 (530 s and 550 s), the only sub-bout structure found in the month."

echo
echo "== done. verify, then commit: =="
echo "  ls figures/gradio_aug*_2015_*.json | wc -l    # expect 12"
echo "  git status --short"
