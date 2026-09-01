#!/usr/bin/env bash
# register_oct2015_figures.sh — rename + register 24 keeper Gradio framegrabs from
# October 2015 orca review (D. Edgington, 2026-08-31, DuaneEM1).
#
# Session summary:
#   Pass 1  (16 clips, 6 min, 8:53-9:05 AM): 15 orca + 1 dolphin (94% precision)
#   Pass 2  (28 clips, 20 min, 1:51-2:08 PM): 16 orca + 4 dolphin + 1 humpback + 8 unlabeled
#   Total DB: 31 orca, 5 dolphin, 1 humpback_song
#
# DROPPED (5):
#   185732  pre-click of wid=70269  (superseded by 185811)
#   190051  pre-click of wid=70263  (superseded by 190105)
#   190136  pre-click of wid=124571 (superseded by 190502)
#   052109  CKWP Facebook post -> docs/ (handled separately)
#   191339  @MBayWhaleWatch Apr 2024 X post -> docs/ (handled separately)
#   191648  @MBayWhaleWatch Oct 2015 X post -> docs/ (already in finding #44)
#
# KEPT: 24 Gradio framegrabs (14 pass-1 orca/dolphin + 9 pass-2 orca + 1 pass-1 orca)
#
# Run from the repo root on DuaneEM1:  bash tools/register_oct2015_figures.sh
set -u
cd "$(dirname "$0")/.." || exit 1
[ -f tools/register_figure.py ] || { echo "run from repo root"; exit 1; }

DB=MARS_20151001_20151031_32kHz_norm
AUD=/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2015/10

REV1='nohup python3 phase2_classify.py review --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20151001_20151031_32kHz_norm --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt --target-label orca_call --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_oct2015_pass1.csv --detections-offset 0 --num-results 16 --classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled --annotator-id duane --audio-dir '"$AUD"' --spectrogram-type mel --colormap viridis --serve --port 7878'
REV2='nohup python3 phase2_classify.py review --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20151001_20151031_32kHz_norm --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt --target-label orca_call --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_oct2015_pass2.csv --detections-offset 0 --num-results 28 --classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled --annotator-id duane --audio-dir '"$AUD"' --spectrogram-type mel --colormap viridis --serve --port 7878'

P1='PASS 1 (16 clips, 6 min, 94% precision): union of v4>=1.16 and v10>=2.31, score-descending.'
P2='PASS 2 (28 clips, 20 min): v10>=0.20 on Oct 26 and Oct 27, excluding pass-1 windows.'
SCALE='Scores are v10 throughout (review set built from v10 CSV).'
MULTI='NOTE: 8 windows left UNLABELED by Duane because multiple species appeared to vocalize simultaneously in the same 5 s window (orca + dolphin or orca + humpback). The radio button cannot represent multi-label -- these are deliberate skips, not missed clicks.'

echo "== removing 3 pre-click grabs =="
for n in 185732 190051 190136; do
  git rm -q --cached "figures/gradio_oct2015_tmp_${n}.png" 2>/dev/null
  rm -f "figures/gradio_oct2015_tmp_${n}.png"
done
echo "== removing 3 non-Gradio grabs from figures/ =="
for n in 052109 191339 191648; do
  git rm -q --cached "figures/gradio_oct2015_tmp_${n}.png" 2>/dev/null
  rm -f "figures/gradio_oct2015_tmp_${n}.png"
done

echo "== renaming 24 keepers =="
cd figures || exit 1

# ---- Pass 1 orca (14) ----
mv gradio_oct2015_tmp_190502.png gradio_oct26_2015_ORCA_320s_wid124571.png   # 3.732 -- CAMPAIGN HIGH
mv gradio_oct2015_tmp_185941.png gradio_oct27_2015_ORCA_365s_wid456711.png   # 3.149
mv gradio_oct2015_tmp_185857.png gradio_oct26_2015_ORCA_330s_wid70271.png    # 3.027
mv gradio_oct2015_tmp_185835.png gradio_oct26_2015_ORCA_235s_wid70252.png    # 2.865
mv gradio_oct2015_tmp_185919.png gradio_oct26_2015_ORCA_270s_wid112801.png   # 2.534
mv gradio_oct2015_tmp_185649.png gradio_oct26_2015_ORCA_80s_wid479338.png    # 2.387
mv gradio_oct2015_tmp_190003.png gradio_oct26_2015_ORCA_175s_wid479357.png   # 2.658
mv gradio_oct2015_tmp_190105.png gradio_oct26_2015_ORCA_290s_wid70263.png    # 2.791 harmonic arches
mv gradio_oct2015_tmp_185709.png gradio_oct26_2015_ORCA_390s_wid70283.png    # 1.340
mv gradio_oct2015_tmp_185532.png gradio_oct27_2015_ORCA_310s_wid159016.png   # 2.454
mv gradio_oct2015_tmp_185557.png gradio_oct07_2015_ORCA_5s_wid59774.png      # 2.475 isolated single
mv gradio_oct2015_tmp_185350.png gradio_oct26_2015_ORCA_190s_wid479360.png   # 1.168
mv gradio_oct2015_tmp_185811.png gradio_oct26_2015_ORCA_320s_wid70269.png    # 2.610
mv gradio_oct2015_tmp_185443.png gradio_oct05_2015_dolphin_395s_wid343047.png  # dolphin pass 1

# ---- Pass 2 orca (9) ----
mv gradio_oct2015_tmp_120054.png gradio_oct26_2015_ORCA_305s_wid70266.png    # 1.849
mv gradio_oct2015_tmp_120214.png gradio_oct26_2015_ORCA_340s_wid70273.png    # 1.653
mv gradio_oct2015_tmp_120324.png gradio_oct26_2015_ORCA_145s_wid124536.png   # 1.744
mv gradio_oct2015_tmp_120412.png gradio_oct26_2015_ORCA_315s_wid124570.png   # 1.392
mv gradio_oct2015_tmp_120547.png gradio_oct26_2015_ORCA_150s_wid112777.png   # 0.804
mv gradio_oct2015_tmp_120605.png gradio_oct26_2015_ORCA_280s_wid479378.png   # 1.814
mv gradio_oct2015_tmp_120840.png gradio_oct26_2015_ORCA_555s_wid112858.png   # 1.624
mv gradio_oct2015_tmp_115141.png gradio_oct26_2015_ORCA_155s_wid112778.png   # 0.488
mv gradio_oct2015_tmp_115648.png gradio_oct27_2015_ORCA_370s_wid159028.png   # 0.856

cd ..
echo -n "renamed: "; ls figures/gradio_oct*_2015_*.png | wc -l

reg () {
  python3 tools/register_figure.py \
    --saved-name "$1" --original-name "Screenshot 2026-08-31 at $2.png" \
    --computer DuaneEM1 --type gradio_screenshot \
    --wav "$3" --offset "$4" --spectrogram mel --colormap viridis \
    --classifier orca_v10.pt --db "$DB" \
    --score "$5" --label "$6" --caption "$7" --notes "$8" --command "$9"
}

echo "== registering 24 =="

# ---- Campaign's highest score: the 3.732 ----
reg gradio_oct26_2015_ORCA_320s_wid124571.png "9.05.02 AM" \
  MARS_20151026_084928_resampled_32kHz.wav 320 3.732 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:54 PDT (08:54 UTC). THE HIGHEST-SCORING WINDOW OF THE ENTIRE CAMPAIGN TO DATE (v10=3.732). Extraordinary multi-harmonic structure spanning 2-16 kHz, with stacked harmonic arches visible at ~2, 4, 6, 8, 10, 12 kHz sweeping and bending over 2 seconds. Mel/viridis, 5 s window with 30 s context." \
  "Label #16, wid=124571. THREE GRABS TAKEN of this window: 9.01.36 AM (pre-click, unlabeled, dropped), 9.03.24 AM (unlabeled, dropped), 9.05.02 AM (orca_call, this one -- the keeper). The multi-harmonic structure with broad frequency coverage from ~500 Hz to 12 kHz is the richest call shape in the campaign. $P1 $SCALE $MULTI" "$REV1"

# ---- 3.149: Oct 27 ----
reg gradio_oct27_2015_ORCA_365s_wid456711.png "8.59.41 AM" \
  MARS_20151027_070928_resampled_32kHz.wav 365 3.149 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-27 00:15 PDT. Second-highest score of the campaign (v10=3.149). From the Oct 27 second bout (00:03-00:15 PDT), one day after the main encounter. Mel/viridis, 5 s window with 30 s context." \
  "Label #12, wid=456711. $P1 $SCALE" "$REV1"

# ---- 3.027 ----
reg gradio_oct26_2015_ORCA_330s_wid70271.png "8.58.57 AM" \
  MARS_20151026_085928_resampled_32kHz.wav 330 3.027 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:05 PDT. v10=3.027, third-highest campaign score. One of TEN calls in this single 10-minute recording -- the densest vocalization the campaign has found. Mel/viridis, 5 s window with 30 s context." \
  "Label #10, wid=70271. MARS_20151026_085928 holds 10 confirmed orca calls across 02:03-02:05 PDT, the most in any recording to date. $P1 $SCALE" "$REV1"

# ---- 2.865 ----
reg gradio_oct26_2015_ORCA_235s_wid70252.png "8.58.35 AM" \
  MARS_20151026_085928_resampled_32kHz.wav 235 2.865 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:04 PDT. v10=2.865. One of ten calls in MARS_20151026_085928. Mel/viridis, 5 s window with 30 s context." \
  "Label #9, wid=70252. $P1 $SCALE" "$REV1"

# ---- 2.791: harmonic arches ----
reg gradio_oct26_2015_ORCA_290s_wid70263.png "9.01.05 AM" \
  MARS_20151026_085928_resampled_32kHz.wav 290 2.791 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:04 PDT. v10=2.791. Visible stacked harmonic arches at 2-8 kHz with curved sweeping structure around 2.5-3.5 s -- one of the structurally clearest calls in the campaign. One of ten in MARS_20151026_085928. Mel/viridis, 5 s window with 30 s context." \
  "Label #14, wid=70263. A GRAB WAS TAKEN BEFORE CLICKING (9.00.51 AM, shows unlabeled, dropped). This is the scored version. $P1 $SCALE $MULTI" "$REV1"

# ---- 2.658 ----
reg gradio_oct26_2015_ORCA_175s_wid479357.png "9.00.03 AM" \
  MARS_20151026_082928_resampled_32kHz.wav 175 2.658 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:32 PDT. v10=2.658. From the build-up phase of the encounter, 30 min before the ten-call peak. Mel/viridis, 5 s window with 30 s context." \
  "Label #13, wid=479357. $P1 $SCALE" "$REV1"

# ---- 2.610 ----
reg gradio_oct26_2015_ORCA_320s_wid70269.png "8.58.11 AM" \
  MARS_20151026_085928_resampled_32kHz.wav 320 2.610 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:05 PDT. v10=2.610. One of ten calls in MARS_20151026_085928 at the encounter peak. Mel/viridis, 5 s window with 30 s context." \
  "Label #8, wid=70269. A GRAB WAS TAKEN BEFORE CLICKING (8.57.32 AM, shows unlabeled, dropped). This is the scored version. $P1 $SCALE" "$REV1"

# ---- 2.534 ----
reg gradio_oct26_2015_ORCA_270s_wid112801.png "8.59.19 AM" \
  MARS_20151026_080928_resampled_32kHz.wav 270 2.534 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:14 PDT. v10=2.534. From the early phase of the encounter, first hour of activity. Mel/viridis, 5 s window with 30 s context." \
  "Label #11, wid=112801. $P1 $SCALE" "$REV1"

# ---- 2.475: Oct 7 isolated single ----
reg gradio_oct07_2015_ORCA_5s_wid59774.png "8.57.09 AM" \
  MARS_20151007_101524_resampled_32kHz.wav 5 2.475 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-07 03:15 PDT. v10=2.475. An ISOLATED SINGLE, the day after MBWW logged 15 killer whales (10/6). NOTE this is at OFFSET 5s -- the very start of the file; the 30 s context begins at 0 s. Mel/viridis, 5 s window with 30 s context." \
  "Label #5, wid=59774. Offset 5s is the minimum non-zero offset and sits near the file boundary; worth checking for boundary artefact, though the score is well above threshold. Sighting-day context: 10/6 MBWW = 15 killer whales, no acoustic detection; our detection is the following night at 03:15 PDT -- consistent with animals present but silent during daylight, vocalizing after dark. $P1 $SCALE" "$REV1"

# ---- 2.454: Oct 27 ----
reg gradio_oct27_2015_ORCA_310s_wid159016.png "8.55.32 AM" \
  MARS_20151027_055928_resampled_32kHz.wav 310 2.454 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 23:03 PDT. v10=2.454. From the Oct 27 UTC file but Oct 26 local -- the late-night second bout (23:03-23:05 PDT Oct 26). Mel/viridis, 5 s window with 30 s context." \
  "Label #4, wid=159016. $P1 $SCALE" "$REV1"

# ---- 2.387 ----
reg gradio_oct26_2015_ORCA_80s_wid479338.png "8.56.49 AM" \
  MARS_20151026_082928_resampled_32kHz.wav 80 2.387 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:31 PDT. v10=2.387. Build-up phase. Mel/viridis, 5 s window with 30 s context." \
  "Label #6, wid=479338. $P1 $SCALE" "$REV1"

# ---- 2.610: 085928 320s — second one ----
reg gradio_oct26_2015_ORCA_190s_wid479360.png "8.53.50 AM" \
  MARS_20151026_082928_resampled_32kHz.wav 190 1.168 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:33 PDT. v10=1.168. Build-up phase 30 min before peak. Mel/viridis, 5 s window with 30 s context." \
  "Label #1, wid=479360. $P1 $SCALE" "$REV1"

# ---- 1.340 ----
reg gradio_oct26_2015_ORCA_390s_wid70283.png "8.57.32 AM" \
  MARS_20151026_085928_resampled_32kHz.wav 390 1.340 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:06 PDT. v10=1.340. One of ten calls in MARS_20151026_085928; final call of the peak recording, 30 s after the cluster. Mel/viridis, 5 s window with 30 s context." \
  "Label #7, wid=70283. $P1 $SCALE" "$REV1"

# ---- dolphin pass 1 ----
reg gradio_oct05_2015_dolphin_395s_wid343047.png "8.54.43 AM" \
  MARS_20151005_085148_resampled_32kHz.wav 395 1.192 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-10-05 01:58 PDT. The only non-orca label in pass 1. Oct 5 was the day of the recorder thrashing event (232 files, 97 short files in a 90-min block). Mel/viridis, 5 s window with 30 s context." \
  "Label #2, wid=343047. $P1 $SCALE" "$REV1"

# ---- Pass 2 orca (9) ----
reg gradio_oct26_2015_ORCA_305s_wid70266.png "1.00.54 PM" \
  MARS_20151026_085928_resampled_32kHz.wav 305 1.849 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:05 PDT. v10=1.849. Found only in pass-2 zoom-in; below pass-1 cutoff. One of ten calls in MARS_20151026_085928. Mel/viridis, 5 s window with 30 s context." \
  "Label #16, wid=70266. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_340s_wid70273.png "2.02.14 PM" \
  MARS_20151026_085928_resampled_32kHz.wav 340 1.653 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 02:05 PDT. v10=1.653. Pass-2 find in MARS_20151026_085928. Mel/viridis, 5 s window with 30 s context." \
  "Label #18, wid=70273. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_280s_wid479378.png "2.06.05 PM" \
  MARS_20151026_082928_resampled_32kHz.wav 280 1.814 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:34 PDT. v10=1.814. Pass-2 find; below pass-1 cutoff. Mel/viridis, 5 s window with 30 s context." \
  "Label #24, wid=479378. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_145s_wid124536.png "2.03.24 PM" \
  MARS_20151026_084928_resampled_32kHz.wav 145 1.744 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:52 PDT. v10=1.744. Pass-2 find in MARS_20151026_084928. Mel/viridis, 5 s window with 30 s context." \
  "Label #20, wid=124536. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_315s_wid124570.png "2.04.12 PM" \
  MARS_20151026_084928_resampled_32kHz.wav 315 1.392 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:54 PDT. v10=1.392. Pass-2 find. Mel/viridis, 5 s window with 30 s context." \
  "Label #21, wid=124570. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_555s_wid112858.png "2.08.40 PM" \
  MARS_20151026_080928_resampled_32kHz.wav 555 1.624 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:19 PDT. v10=1.624. Pass-2 find. Mel/vividis, 5 s window with 30 s context." \
  "Label #28, wid=112858. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_155s_wid112778.png "1.51.41 PM" \
  MARS_20151026_080928_resampled_32kHz.wav 155 0.488 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:12 PDT. v10=0.488 -- well below any threshold, found only by the pass-2 zoom-in. Mel/viridis, 5 s window with 30 s context." \
  "Label #3, wid=112778. Lowest-scoring confirmed orca of October 2015. $P2 $SCALE" "$REV2"

reg gradio_oct26_2015_ORCA_150s_wid112777.png "2.05.47 PM" \
  MARS_20151026_080928_resampled_32kHz.wav 150 0.804 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 01:12 PDT. v10=0.804. Pass-2 find, 5 s before wid=112778. Mel/viridis, 5 s window with 30 s context." \
  "Label #23, wid=112777. $P2 $SCALE" "$REV2"

reg gradio_oct27_2015_ORCA_370s_wid159028.png "1.56.48 PM" \
  MARS_20151027_055928_resampled_32kHz.wav 370 0.856 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-10-26 23:05 PDT. v10=0.856. Pass-2 find in the late-night second bout. Mel/viridis, 5 s window with 30 s context." \
  "Label #10, wid=159028. $P2 $SCALE" "$REV2"

echo
echo "== done. verify then commit: =="
echo "  ls figures/gradio_oct*_2015_*.json | wc -l   # expect 24"
echo "  ls figures/gradio_oct2015_tmp_*.png 2>/dev/null | wc -l  # expect 0"
echo "  git status --short | head -30"
