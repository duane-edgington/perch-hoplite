#!/usr/bin/env bash
# register_sep2015_figures.sh — rename + register the 22 keeper framegrabs from the
# September 2015 orca review (D. Edgington, 2026-08-30, DuaneEM1).
#
# 27 grabs were taken across three sessions for 47 labels + 1 deliberate skip. Dropped:
#   gradio_sep2015_tmp_021944  pre-click (button reads unlabeled; DB has ROV_noise)
#   gradio_sep2015_tmp_035946  pre-click (superseded by 035956, same wid=235704)
#   gradio_sep2015_tmp_023014  duplicate of 023101 (wid=110714), audio control cut off
#   gradio_sep2015_tmp_023533  duplicate of 023547 (wid=413969), audio control cut off
#   gradio_sep2015_tmp_022904  superseded by 042023 — a browser tooltip covered the header
#
# Run from the repo root on DuaneEM1:  bash tools/register_sep2015_figures.sh
set -u
cd "$(dirname "$0")/.." || exit 1
[ -f tools/register_figure.py ] || { echo "run from repo root"; exit 1; }

DB=MARS_20150901_20150930_32kHz_norm
AUD=/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2015/09

REV1='nohup python3 phase2_classify.py review --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150901_20150930_32kHz_norm --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt --target-label orca_call --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_sep2015_clusters.csv --detections-offset 0 --num-results 27 --classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled --annotator-id duane --audio-dir '"$AUD"' --spectrogram-type mel --colormap viridis --serve --port 7878'
REV2='nohup python3 phase2_classify.py review --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150901_20150930_32kHz_norm --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt --target-label orca_call --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_sep2015_pass2.csv --detections-offset 0 --num-results 21 --classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled --annotator-id duane --audio-dir '"$AUD"' --spectrogram-type mel --colormap viridis --serve --port 7878'

S1='PASS 1 (27 clips, 14 min): union of orca_v4 >=1.16, orca_v10 >=2.31, the full MARS_20150916_181020 bout at v10>=0.50, and the Sept 28 cluster at v10>=1.00. Score-descending.'
S2='PASS 2 (21 clips, 15 min): the zoom-in. Every previously-unheard window at v10 >= 0.20 inside the two episodes (Sept 17 06:20-06:50, Sept 28 04:00-09:00). All below the pass-1 review cutoff. 11 of 21 proved to be orca.'
SCALE='SCORE SCALE: pass-1 clips carry a v4 score where v4 selected them and a v10 score otherwise; pass-2 clips are all v10. Not a single scale across the set.'

echo "== dropping 5 superseded/pre-click grabs =="
for n in 021944 035946 023014 023533 022904; do
  git rm -q --cached "figures/gradio_sep2015_tmp_${n}.png" 2>/dev/null
  rm -f "figures/gradio_sep2015_tmp_${n}.png"
done

echo "== renaming 22 keepers =="
cd figures || exit 1
mv gradio_sep2015_tmp_042023.png gradio_sep17_2015_ORCA_80s_wid235701.png
mv gradio_sep2015_tmp_023216.png gradio_sep17_2015_ORCA_65s_wid235698.png
mv gradio_sep2015_tmp_035530.png gradio_sep17_2015_ORCA_295s_wid235744.png
mv gradio_sep2015_tmp_035716.png gradio_sep17_2015_ORCA_320s_wid413793.png
mv gradio_sep2015_tmp_040008.png gradio_sep17_2015_ORCA_370s_wid413803.png
mv gradio_sep2015_tmp_035956.png gradio_sep17_2015_ORCA_95s_wid235704.png
mv gradio_sep2015_tmp_034840.png gradio_sep17_2015_ORCA_100s_wid235705.png
mv gradio_sep2015_tmp_035022.png gradio_sep17_2015_ORCA_10s_wid235687.png
mv gradio_sep2015_tmp_022941.png gradio_sep28_2015_ORCA_170s_wid120157.png
mv gradio_sep2015_tmp_035844.png gradio_sep28_2015_ORCA_95s_wid120142.png
mv gradio_sep2015_tmp_035627.png gradio_sep28_2015_ORCA_380s_wid14959.png
mv gradio_sep2015_tmp_023439.png gradio_sep28_2015_ORCA_560s_wid153357.png
mv gradio_sep2015_tmp_022721.png gradio_sep28_2015_ORCA_435s_wid153332.png
mv gradio_sep2015_tmp_035656.png gradio_sep28_2015_ORCA_400s_wid153325.png
mv gradio_sep2015_tmp_035412.png gradio_sep28_2015_ORCA_145s_wid21512.png
mv gradio_sep2015_tmp_035759.png gradio_sep28_2015_ORCA_25s_wid348471.png
mv gradio_sep2015_tmp_023101.png gradio_sep28_2015_dolphin_355s_wid110714.png
mv gradio_sep2015_tmp_035823.png gradio_sep28_2015_dolphin_465s_wid162458.png
mv gradio_sep2015_tmp_034951.png gradio_sep28_2015_dolphin_100s_wid203785.png
mv gradio_sep2015_tmp_023547.png gradio_sep02_2015_dolphin_0s_wid413969.png
mv gradio_sep2015_tmp_022804.png gradio_sep16_2015_dolphin_470s_wid8975.png
mv gradio_sep2015_tmp_035500.png gradio_sep28_2015_unlabeled_315s_wid409592.png
cd ..
echo -n "renamed: "; ls figures/gradio_sep*_2015_*.png | wc -l

reg () {  # reg <saved> <origtime> <wav> <offset> <score> <label> <caption> <notes> <revcmd>
  python3 tools/register_figure.py \
    --saved-name "$1" --original-name "Screenshot 2026-08-30 at $2.png" \
    --computer DuaneEM1 --type gradio_screenshot \
    --wav "$3" --offset "$4" --spectrogram mel --colormap viridis \
    --classifier orca_v10.pt --db "$DB" \
    --score "$5" --label "$6" --caption "$7" --notes "$8" --command "$9"
}

echo "== registering 22 =="

reg gradio_sep17_2015_ORCA_80s_wid235701.png "4.20.23 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 80 3.128 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:41 UTC. THE HIGHEST-SCORING WINDOW OF THE ENTIRE CAMPAIGN TO DATE (orca_v10 = 3.128) and a confirmed orca. Harmonic pair at ~1.3 kHz and ~2 kHz running 0.2-1.5 s with a faint upper component near 4 kHz at ~1.8 s; no energy above ~5 kHz. Fourth of SIX orca calls in this single 10-minute recording. Mel/viridis, 5 s window with 30 s context." \
  "Label #16, wid=235701. v10 3.128 / v4 1.884. Above v10's 2.31 operating threshold. This grab REPLACES an earlier one (10.29.04 AM) in which a browser tooltip covered the header. The low-frequency harmonic structure with no high content matches the August 2015 candidate (finding #33) and is visibly distinct from the dolphin_call clips in the same session, which sit at 8-16 kHz with steep sweeps. $S1 $SCALE" "$REV1"

reg gradio_sep17_2015_ORCA_65s_wid235698.png "2.32.16 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 65 2.446 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:41 UTC. Harmonic stack 2-4 kHz with a distinct pulse at ~1.9 s. Sits just 15 s before the campaign's highest-scoring window (wid=235701, 3.128) in the same recording -- the pair that first flagged this as bout structure rather than scattered false positives." \
  "Label #20, wid=235698. v10 2.446 / v4 1.536. Above v10's 2.31 operating threshold. $S1 $SCALE" "$REV1"

reg gradio_sep17_2015_ORCA_295s_wid235744.png "3.55.30 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 295 1.023 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:45 UTC. Visible harmonic stack at 1-2 kHz around 1.3-2.2 s. Sub-threshold for both models, found only by the pass-2 zoom-in." \
  "Label #12, wid=235744. v10 1.023 -- BELOW v10's 2.31 operating threshold. $S2 $SCALE" "$REV2"

reg gradio_sep17_2015_ORCA_320s_wid413793.png "3.57.16 PM" \
  MARS_20150917_062020_resampled_32kHz.wav 320 1.067 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:25 UTC. Faint harmonic trace near 2 kHz at ~1.5-2.5 s. In the recording PRECEDING the six-call file -- evidence the encounter began ~20 minutes earlier than pass 1 showed." \
  "Label #16, wid=413793. v10 1.067 -- below threshold; found by the pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep17_2015_ORCA_370s_wid413803.png "4.00.08 PM" \
  MARS_20150917_062020_resampled_32kHz.wav 370 0.985 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:26 UTC. Second call in the 06:20 recording, 50 s after wid=413793." \
  "Label #20, wid=413803. v10 0.985 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep17_2015_ORCA_95s_wid235704.png "3.59.56 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 95 0.432 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:41 UTC. Faint; 15 s after the 3.128 window in the same recording." \
  "Label #21, wid=235704. v10 0.432 -- well below threshold; pass-2 zoom-in. NOTE an earlier grab of this same window (3.59.46 PM) was taken BEFORE the click landed and shows 'unlabeled'; it was discarded. This one is the scored version and matches the DB. $S2 $SCALE" "$REV2"

reg gradio_sep17_2015_ORCA_100s_wid235705.png "3.48.40 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 100 0.205 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:41 UTC. Faint call at ~4.2 s near 2 kHz. THE LOWEST-SCORING CONFIRMED ORCA of the month -- a direct illustration of why the zoom-in pass matters." \
  "Label #1, wid=235705. v10 0.205 -- barely above the 0.20 pass-2 floor and far below the 2.31 operating threshold, yet a real call. $S2 $SCALE" "$REV2"

reg gradio_sep17_2015_ORCA_10s_wid235687.png "3.50.22 PM" \
  MARS_20150917_064020_resampled_32kHz.wav 10 0.257 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-17 06:40 UTC. Faint low-frequency trace near 2 kHz in the first second. Earliest of the six calls in this recording." \
  "Label #4, wid=235687. v10 0.257 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_ORCA_170s_wid120157.png "2.29.41 PM" \
  MARS_20150928_050349_resampled_32kHz.wav 170 1.719 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 05:06 UTC. Low-frequency energy around 1-2 kHz early in the window. FIRST call of the Sept 28 episode, which runs ~2h50m to 07:53." \
  "Label #17, wid=120157. v10 1.719 / v4 1.033 -- below v10's 2.31 threshold. $S1 $SCALE" "$REV1"

reg gradio_sep28_2015_ORCA_95s_wid120142.png "3.58.44 PM" \
  MARS_20150928_050349_resampled_32kHz.wav 95 0.288 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 05:05 UTC. Second call in the opening recording of the Sept 28 episode, 75 s before wid=120157." \
  "Label #19, wid=120142. v10 0.288 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_ORCA_380s_wid14959.png "3.56.27 PM" \
  MARS_20150928_060349_resampled_32kHz.wav 380 0.664 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 06:10 UTC. The only call in the 06:03 recording; bridges the ~1 h gap between the 05:03 and 07:13 groups of the Sept 28 episode." \
  "Label #13, wid=14959. v10 0.664 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_ORCA_560s_wid153357.png "2.34.39 PM" \
  MARS_20150928_071349_resampled_32kHz.wav 560 1.557 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 07:23 UTC. One of FOUR orca calls in this single 10-minute recording -- the densest concentration of the Sept 28 episode." \
  "Label #25, wid=153357. v10 1.557 / v4 1.051 -- below v10's 2.31 threshold. $S1 $SCALE" "$REV1"

reg gradio_sep28_2015_ORCA_435s_wid153332.png "2.27.21 PM" \
  MARS_20150928_071349_resampled_32kHz.wav 435 1.187 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 07:20 UTC. Faint low-frequency energy near 2 kHz at ~3.5 s. Second of four calls in this recording." \
  "Label #4, wid=153332. v10 1.187 / v4 0.920 -- below threshold. $S1 $SCALE" "$REV1"

reg gradio_sep28_2015_ORCA_400s_wid153325.png "3.56.56 PM" \
  MARS_20150928_071349_resampled_32kHz.wav 400 0.411 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 07:20 UTC. Third of four calls in this recording, 35 s before wid=153332." \
  "Label #15, wid=153325. v10 0.411 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_ORCA_145s_wid21512.png "3.54.12 PM" \
  MARS_20150928_072349_resampled_32kHz.wav 145 0.236 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 07:26 UTC. Faint. Continues the episode into the recording after the four-call file." \
  "Label #10, wid=21512. v10 0.236 -- just above the 0.20 pass-2 floor. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_ORCA_25s_wid348471.png "3.57.59 PM" \
  MARS_20150928_075349_resampled_32kHz.wav 25 0.626 orca_call \
  "ORCA CALL confirmed by ear (D. Edgington), 2015-09-28 07:54 UTC. LAST confirmed call of the Sept 28 episode, ~2h50m after the first." \
  "Label #17, wid=348471. v10 0.626 -- below threshold; pass-2 zoom-in. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_dolphin_355s_wid110714.png "2.31.01 PM" \
  MARS_20150928_063349_resampled_32kHz.wav 355 2.349 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-09-28 06:39 UTC. A CAUTIONARY FIGURE: this is the highest-scoring window of Sept 28 in BOTH models and one of only four in the month above v10's 2.31 operating threshold -- and it is NOT an orca. The 30 s context shows the steep high-frequency sweeps that identify it." \
  "Label #18, wid=110714. v10 2.349 / v4 1.809. Of the four windows in September that cleared v10's 2.31 threshold, THIS ONE IS A FALSE POSITIVE for orca -- so the operating point yielded 3 true orca calls out of 4 above-threshold detections, against 18 confirmed orca calls in the month (recall ~17%). Duplicate grab (2.31.01 PM) kept over 2.30.14 PM, which cut off the audio control. $S1 $SCALE" "$REV1"

reg gradio_sep28_2015_dolphin_465s_wid162458.png "3.58.23 PM" \
  MARS_20150928_062349_resampled_32kHz.wav 465 0.673 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-09-28 06:31 UTC. Found in the pass-2 zoom-in of the Sept 28 orca episode -- dolphins and orcas were both present that morning." \
  "Label #18, wid=162458. v10 0.673. $S2 $SCALE" "$REV2"

reg gradio_sep28_2015_dolphin_100s_wid203785.png "3.49.51 PM" \
  MARS_20150928_085349_resampled_32kHz.wav 100 0.303 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-09-28 08:55 UTC. Steep high-frequency sweeps visible in the 30 s context at 18-30 s. An hour after the last confirmed orca call of the episode." \
  "Label #3, wid=203785. v10 0.303. $S2 $SCALE" "$REV2"

reg gradio_sep02_2015_dolphin_0s_wid413969.png "2.35.47 PM" \
  MARS_20150902_161010_resampled_32kHz.wav 0 1.720 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-09-02 16:10 UTC. Descending sweep from ~14 kHz in the first second. The only September detection outside the Sept 16-17 and Sept 28 dates. NOTE this is the FIRST window of the recording, so the 30 s context extends forward only." \
  "Label #26, wid=413969. v4 1.720 / v10 2.007. Duplicate grab (2.35.47 PM) kept over 2.35.33 PM, which cut off the audio control. $S1 $SCALE" "$REV1"

reg gradio_sep16_2015_dolphin_470s_wid8975.png "2.28.04 PM" \
  MARS_20150916_092020_resampled_32kHz.wav 470 1.728 dolphin_call \
  "Dolphin call confirmed by ear (D. Edgington), 2015-09-16 09:28 UTC. Morning of the day that later brought the ROV service (18:10) and the first orca call (23:40)." \
  "Label #14, wid=8975. v4 1.728 / v10 1.942. $S1 $SCALE" "$REV1"

reg gradio_sep28_2015_unlabeled_315s_wid409592.png "3.55.00 PM" \
  MARS_20150928_065349_resampled_32kHz.wav 315 0.480 unlabeled \
  "DELIBERATELY LEFT UNLABELED by D. Edgington -- a documented 'listened, could not tell' outcome, 2015-09-28 06:59 UTC, inside the Sept 28 orca episode. The only such decision in September 2015." \
  "Label #11, wid=409592. v10 0.480. THE BUTTON READS 'unlabeled' BECAUSE THE CLIP WAS SKIPPED, NOT BECAUSE THE GRAB WAS MISTIMED -- verified against the DB, which has NO annotation row for this window (session reported '20 labels saved, 1 unlabeled skipped'). Retained because a recorded 'can't tell' is real information and better provenance than forcing a marginal call. $S2 $SCALE" "$REV2"

echo
echo "== done. verify, then commit: =="
echo "  ls figures/gradio_sep*_2015_*.json | wc -l   # expect 22"
echo "  ls figures/gradio_sep2015_tmp_*.png 2>/dev/null | wc -l   # expect 0"
echo "  git status --short"
