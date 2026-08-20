# CLAUDE.md — Perch-Hoplite Project Context

This file provides context for Claude (AI assistant) when working on this
repository. It captures the project state, conventions, and key decisions
so that Claude can pick up where we left off without re-explaining everything.

---

## Project Summary

Applying Google Perch V2 bioacoustics embeddings + perch-hoplite agile
modeling to detect and classify marine mammal vocalizations in MARS
hydrophone data at MBARI. The pipeline runs entirely in PyTorch on a local
NVIDIA GB10 DGX — no TensorFlow, no Colab.

**Author:** Duane R. Edgington — MBARI (Monterey Bay Aquarium Research Institute)
**GitHub:** https://github.com/duane-edgington/perch-hoplite

---

## System Architecture

| Machine | Role | Notes |
|---|---|---|
| **ICEFISH** (Mac M1 Max) | Dev workstation, scp gateway | 134.89.114.25 / VPN 134.89.74.134 |
| **spark-ae0e** (134.89.11.107) | Primary compute — NVIDIA GB10 DGX | Working dir `~/perch-hoplite/` |
| **spark-0626** (134.89.11.174) | Spare DGX | |
| **thalassa** | NFS server | thalassa.shore.mbari.org |

**scp screenshots from Mac to spark** — macOS screenshot filenames contain spaces.
Use backslash-escaped spaces with a wildcard, e.g.:
```bash
scp ~/Desktop/Screenshot\ 2026-07-16\ at\ 5*.png duane@134.89.11.107:~/perch-hoplite/figures/
```
Quoted versions and underscore wildcards do NOT work for macOS screenshot filenames.

---

## Key Paths

```
# Permanent pipeline data (NFS)
/mnt/PAM_Analysis/perch-hoplite/
    db/          — embedding databases (normalized, suffix _norm)
    models/      — trained classifiers (.pt + .metrics.json)
    results/     — inference CSVs and plots
    logs/        — embedding and inference logs
    provenance/  — label and training JSON records
    example_clips/ — 10 peak-normalized 5s example clips + manifest.json

/mnt/PAM_Analysis/perch_weights/    — Perch V2 weights (ONNX-extracted)
    weights.npz
    graph_manifest.json

# Audio (32kHz resampled WAV)
/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/
    2018/04/    — April 2018 (4,320 files, 518,400 windows)
    2018/05/    — May 2018   (4,464 files, 535,680 windows)
    2020/10/    — October 2020 (4,504 files, 535,278 windows)

# Old location (read-only fallback — DO NOT write here)
/mnt/PAM_Analysis/duane_scratch/perch_hoplite/

# PyTorch Perch V2 port
~/perch-pytorch/
    perch_hoplite_torch_adapter.py  ← MUST be up to date (normalization fix)
    perch_embedder_torch.py
    perch_frontend_torch.py
    perch_weights/
    const__pad1_output_0.npy
```

---

## Critical: Low-Amplitude Normalization (July 9 2026)

**All embeddings must use per-window peak normalization to 0.25.**

MARS hydrophone audio has typical peak amplitude 0.0015–0.003. Without
normalization, the PyTorch port diverges from TF Perch V2 at cosine 0.43–0.94
on real MARS data (verified against live TF model on Colab A100).

The fix is in `perch_hoplite_torch_adapter.py` — `peak_normalize_windows()`
is called inside `embed()` before the model. This is transparent to callers.

**All DBs with `_norm` suffix were embedded with this fix (July 9 2026+).**
Old DBs without `_norm` suffix are pre-fix and should not be used for new work.

See: `docs/FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md`

---

## Venv

```bash
# Always activate before running anything
source ~/perch-hoplite/venv/bin/activate

# Key packages: torch 2.12.1+cu130, gradio==6.15.1 (pinned), librosa,
# soundfile, perch-hoplite (core, no TF extras)
```

**Gradio version must be 6.15.1** — other versions have audio playback issues.
**Browser for Gradio: Chrome (incognito)** — Safari has audio playback issues
with data: URIs.

---

## Classifier Versioning (new era — July 9 2026)

Old classifiers (v1_clean through v8_clean) were trained on un-normalized
embeddings and are retired. New versioning starts at v0:

| Version | ROC-AUC | top1_acc | cmap | F1† | Training DB | Notes |
|---|---|---|---|---|---|---|
| v0 | 0.9773 | 0.9405 | 0.8810 | — | April 2018 norm | Baseline normalized |
| v1 | 0.9533 | 0.9559 | 0.7999 | 0.799 | April + October 2020 norm | Cross-season |
| v2 | 0.9654 | 0.9438 | 0.8930 | 0.897 | April 2018 norm (expanded) | More dolphin/other labels |
| v3 | 0.9467 | 0.9481 | 0.7370 | — | April 2018 + Oct 2020 + April 2026 norm | 3-season, 17 Apr2026 humpback |
| v4 | 0.9590 | 0.9650 | 0.8297 | 0.830 | April 2018 + Oct 2020 + April 2026 norm | Best cross-season, 25 Apr2026 humpback |
| v5 | 0.9303 | 0.9301 | 0.5945 | — | 3-season context DB (30s Gaussian avg) | Context embedding experiment — WORSE than v4 |
| v6 | 0.9499 | 0.9409 | 0.7763 | — | 4-season combined | ship_noise inflated 1278→4496 ❌ |
| v7 | 0.9499 | 0.9409 | 0.7763 | — | 4-season, negative→orca_call fix | Identical to v6 ❌ |
| v8 | 0.9463 | 0.9347 | 0.6489 | — | 4-season, background→other fix | Still inflated ❌ |

**† F1 = macro F1 at F1-optimal per-class thresholds**, computed on the same held-out
eval split as cmap/ROC-AUC (`src/f1_metrics.py`, folded into `.metrics.json` every
training run — July 17 2026). v1/v2/v4 measured (all reproduced their table cmap exactly);
retrain v0/v3/v5–v8 to populate the rest. ⚠ **Macro F1 is inflated by low-support classes
and eval sets differ across versions — do not compare it across rows or read it as clean
skill. See the per-class note below.**

**Best for October 2020 analysis:** v1
**Best for April/May 2018 analysis:** v2
**Best for April 2026 / cross-season:** v4

### Per-class F1 — v1 / v2 / v4 (item #10, implemented July 17 2026)

Per-class precision/recall/F1 on the same held-out eval split as cmap/ROC-AUC
(`src/f1_metrics.py` → `.metrics.json`, every run). Reported at fixed logit ≥ 0.0
(inference default) and the F1-optimal per-class threshold. All three reproduced their
table cmap exactly (seed 42), so these attach to the exact rows.

**v2** — April 2018 only, n_eval=190:

| class | n | F1 @0.0 | F1 opt | opt thr |
|---|---|---|---|---|
| orca_call | 40 | 0.848 | 0.951 | +1.57 |
| dolphin_call | 35 | 0.522 | 0.703 | +2.30 |
| other | 7 | 0.364 | 0.833 | +2.47 |
| humpback_song | 5 | 1.000 | 1.000 | +0.75 ⚠ |
| ship_noise | 3 | 1.000 | 1.000 | +0.36 ⚠ |

**v1** — April 2018 + Oct 2020, n_eval=283:

| class | n | F1 @0.0 | F1 opt | opt thr |
|---|---|---|---|---|
| orca_call | 41 | 0.781 | 0.950 | +1.89 |
| dolphin_call | 41 | 0.554 | 0.706 | +2.05 |
| humpback_song | 40 | 0.444 | 0.559 | +1.08 |
| other | 11 | 0.545 | 0.778 | +2.19 |
| ship_noise | 3 | 1.000 | 1.000 | +1.10 ⚠ |

**v4** — 3-season, n_eval=296:

| class | n | F1 @0.0 | F1 opt | opt thr |
|---|---|---|---|---|
| orca_call | 45 | 0.841 | 0.947 | +1.16 |
| dolphin_call | 38 | 0.531 | 0.765 | +2.05 |
| humpback_song | 47 | 0.450 | 0.548 | +0.98 |
| other | 10 | 0.645 | 0.889 | +1.99 |
| ship_noise | 3 | 1.000 | 1.000 | +0.16 ⚠ |

**Credible vs insufficient support.** ⚠ = too few held-out examples to trust.
- **ship_noise is n=3 in *every* model** (only ~24 ship labels total, mostly April 2018).
  Its 1.0 is an artifact across the board — needs more ship labels before F1 means anything.
- **v2 humpback (n=5) = 1.0 was also an artifact** — confirmed now that v1/v4 give humpback
  real support (n=40/47) and it drops to ~0.55.
- Macro F1 (table column) is still nudged up by ship=1.0; don't compare it across versions
  (different eval sets/DBs). Per-class is the honest view.

**Findings.**
- **orca_call — strong and stable:** F1 opt ≈ 0.95 in all three, but always needs a
  *positive* threshold (+1.2 to +1.9); at 0.0, precision is only 0.75–0.84. Confirms
  raising the orca inference threshold (#5, #8).
- **humpback_song — weakest credible class (~0.55):** once it has real support it's the
  problem child, not a star, and an optimal threshold near 0 doesn't rescue it — this is a
  model/label-quality ceiling. **Direct evidence for the gray-whale-contamination
  hypothesis (#13):** humpback labels likely mixed with gray whale calls, blurring the
  class. Re-annotation + a `gray_whale_moan` class is the most promising lever to lift it.
- **dolphin_call — ceiling ~0.71–0.77:** consistent across models, high optimal threshold
  (~+2.05). Model-quality limited, like humpback but less severe.
- **other:** small support (n=7–11); F1 opt 0.78–0.89 but directional only.

**Threshold takeaway (feeds #8).** Optimal thresholds span +0.16 to +2.47 across classes
and models; the inference default 0.0 is uniformly too permissive (recall ≈ 1.0, precision
poor). A single global threshold cannot serve all classes → per-class inference thresholds.
For orca specifically, ~+1.5 buys precision 0.75→0.93 at unchanged recall (interim #5 lever).

---

## Annotation State (July 9 2026)

| DB | Annotations |
|---|---|
| MARS_20180401_20180430_32kHz_norm | 294 orca (219 + 75 extended-April review Jul 19) + 195 dolphin + 41 humpback + 24 ship + 51 other + 54 neg |
| MARS_20201001_20201031_32kHz_norm | 209 humpback + 5 dolphin |
| MARS_20260401_20260430_32kHz_norm | 25 humpback (hard negatives) + 23 negative + 39 humpback + 3 dolphin + 4 other + 5 ship |
| MARS_20180501_20180531_32kHz_norm | **181 orca** (May 12 confirmed ✅) + 34 negative + 2 dolphin + 4 ship + 6 other |
| MARS_20201001_20201031_32kHz_norm | 209+259=468 humpback + 6 negative + 7 dolphin + 39 other + 2 ship |

---

## Standard Commands

```bash
# Embed a full month
nohup python3 phase1_embed_torch.py \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --device cuda --compile \
    > /mnt/PAM_Analysis/perch-hoplite/logs/embed_april2018.log 2>&1 &

# Train classifier
time python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier-out /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --num-steps 256 --train-ratio 0.8

# Run inference
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20180401_20180430_v2_detections.csv \
    --logit-threshold 0.0

# Review annotations (Chrome incognito, viridis mel is preferred display)
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --target-label orca_call \
    --num-results 25 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7861 \
    > /mnt/PAM_Analysis/perch-hoplite/logs/review_7861.log 2>&1 &

# Kill Gradio (by port to avoid killing inference)
pkill -f "port 7861"

# Monthly plot
python3 tools/plot_monthly.py \
    --input /mnt/PAM_Analysis/perch-hoplite/results/MARS_20180401_20180430_v2_detections.csv \
    --output-dir /mnt/PAM_Analysis/perch-hoplite/results \
    --title "April 2018 MARS Hydrophone v2"

# t-SNE
python3 tools/plot_tsne.py \
    --db-dir \
        /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
        /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
    --output /mnt/PAM_Analysis/perch-hoplite/results/tsne_combined.png \
    --title "Perch V2 Embeddings — April 2018 + October 2020"
```

---

## Spectrogram Preferences

**Preferred display:** `--spectrogram-type mel --colormap viridis`

| Mode | Flag | Best for |
|---|---|---|
| Linear STFT | `--spectrogram-type linear` | Orca, dolphin (default) |
| Mel | `--spectrogram-type mel --colormap viridis` | Humpback, general review |
| Perch frontend | `--spectrogram-type perch` | Model inspection |
| Gray mel | `--spectrogram-type mel --colormap gray` | Publication figures |

---

## Figure Provenance (IMPORTANT — Duane's standing workflow)

**Every figure Duane captures (Gradio screenshots, plots) MUST be archived with full
provenance. This matters a lot to Duane and is a long-term project requirement — do not
skip it, and do not invent a parallel format. Always use the existing system below.**

**The system:** `tools/register_figure.py` writes a per-figure JSON sidecar
(`figures/<saved-name>.json`) and updates the master `figures/manifest.json`. Inspect
`figures/manifest.json` for the schema and prior examples before writing captions.

**Naming convention** (match existing entries, no spaces):
`gradio_<event>_<label>_<detail>.png`, e.g. `gradio_apr18_2018_orca_195s_wid202720.png`.
Plots: `tsne_*`, `<Month>_MARS_Hydrophone_v*_*` (monthly/heatmap).

**Standard registration command (Gradio screenshot):**
```bash
python3 tools/register_figure.py \
    --saved-name gradio_apr18_2018_orca_195s_wid202720.png \
    --original-name "Screenshot 2026-07-19 at 4.40.17 AM.png" \
    --computer DuaneEM1 \
    --type gradio_screenshot \
    --wav MARS_20180418_113912_resampled_32kHz.wav --offset 195 \
    --spectrogram mel --colormap viridis \
    --classifier orca_v4.pt --db MARS_20180401_20180430_32kHz_norm \
    --score 3.724 --label orca_call \
    --caption "..." --notes "..." --command "<full review command used>"
```
`--type` ∈ {gradio_screenshot, tsne_plot, monthly_plot, heatmap_plot, matplotlib_plot, other}.
`--computer` ∈ {ICEFISH, DuaneEM1, spark-ae0e, spark-0626, other}.
Then `git add` the .png + its .json + manifest.json, and commit together.

**macOS screenshot filename gotcha (both scp AND mv):** screenshot names contain spaces
(and sometimes non-ASCII space characters that will NOT byte-match a retyped quoted
string). ALWAYS glob on the unique timestamp instead of quoting the name:
```bash
# scp from Mac:
scp ~/Desktop/Screenshot\ 2026-07-19\ at\ 4*.png duane@134.89.11.107:~/perch-hoplite/figures/
# rename on spark (glob, don't retype the spaces):
mv ~/perch-hoplite/figures/Screenshot*4.40.17*.png ~/perch-hoplite/figures/gradio_apr18_2018_orca_195s_wid202720.png
```
`--original-name` is just a metadata string, so a quoted approximation there is fine — the
glob only matters when a command touches the actual file.

**Attribution rule:** Duane (D. Edgington) is the orca annotator/reviewer — orca IDs are
HIS expert call, recorded as expert-confirmed, never "pending review." Loop in J. Ryan
(PI) only for flagged QA sessions, chiefly humpback-vs-gray-whale (#13). Caption confirmed
orca clips as "Expert-confirmed orca (D. Edgington)".

**Co-authorship note (July 2026):** J. Ryan is a **co-author** on the IEEE OCEANS 2026 poster
submission (accepted — `docs/oceans_2026_acceptance_record.md`), not merely a reviewer credited
in figure captions. Distinguish the two roles: attribution-rule captions credit his QA review
of specific clips; the OCEANS poster's author line/acknowledgements must credit him as
co-author. The PyTorch Conference poster is Duane solo (per `docs/PyTorch_poster.md`).

**Pre-click screenshot quirk:** a Gradio screenshot may show `unlabeled` (or a stale
selection) if grabbed a beat before the label click registers. The displayed radio button
is NOT proof of the final label — the DB is the source of truth. Confirm the intended
label with Duane before captioning; note it in `--notes` if the screenshot shows pre-click state.

**Registered so far (examples):** `gradio_30s_context_orca2` (Apr 13 2018, v2, linear);
`gradio_apr18_2018_orca_195s_wid202720` + `_405s_wid202762` (Apr 18 2018, v4, mel/viridis);
five `gradio_apr25_2018_orca_*` clips (Apr 25 2018, v4, mel/viridis, scores 1.33–1.82) —
all expert-confirmed, finding #14.

---

## Known Issues / Pending Work

1. **Mel spectrogram banding** — minor horizontal artifacts in mel/pcen/perch modes (partially fixed)
2. **plot_monthly deduplication** — ✅ fixed July 12 2026: now deduplicates on `(idx, label)`
3. **Inference /tmp write speed** — 266K row CSV takes ~33 min to write
4. **May 2014 secondary event** — 19 (v4) / 58 (v6) detections, not yet reviewed
5. **April 2026 orca FPs** — Apr 14, 16 still elevated humpback FPs; need 2026 orca examples to resolve. Interim mitigation: raise orca inference threshold to ~+1.5 (v2 held-out: precision 0.75→0.93 at unchanged recall 0.975).
6. **May 2026 embedding** — not yet done
7. **Abstract update deadline** — July 26 2026; UPDATE NOW — May 12 2018 confirmed July 16 2026
8. **Option A inference** — add `--output-format full` flag to output all 5 logits per window. Now also motivated by per-class F1: optimal thresholds span +0.36 to +2.47, so a single global 0.0 is too permissive — per-class inference thresholds needed.
9. **Negative labels** — only April 2018 + May 2018 have negatives; October 2020 (6) and April 2026 (23) added July 16 2026
10. **Per-class F1 scores** — ✅ implemented July 17 2026: `src/f1_metrics.py` computes per-class precision/recall/F1 (fixed-0 + F1-optimal thresholds) on the same held-out split as cmap/ROC-AUC, folded into `eval_scores` → `.metrics.json` every training run. v1/v2/v4 measured (all reproduced their cmap exactly). See "Per-class F1 — v1/v2/v4" note under the classifier table. Remaining: retrain v0/v3/v5–v8 to populate; ship_noise (n=3 everywhere) needs more labels and small classes need per-class stratification before their F1 is trustworthy.
11. **4-season ship_noise inflation** — v6/v7/v8 all inflate ship_noise; root cause: May+April orca acoustically different enough to distort boundaries
12. **Ship_noise label review** — review April 2018 ship_noise labels (24 clips) in Gradio to confirm no errors (low priority)
13. **Gray whale annotation review** — some `humpback_song` labels may be gray whale calls. Pull all humpback-labeled clips in Gradio and have J. Ryan re-annotate as `humpback_song`, `gray_whale_moan` (new class), or `other`. Gray whales are seasonally present in Monterey Bay and can overlap spectrally with humpback at low frequencies. Then retrain with gray whale as a new species class. **Now supported by evidence:** with real held-out support (n=40/47 in v1/v4), humpback_song is the weakest credible class (F1 opt ~0.55) — consistent with label contamination blurring the class. Highest-priority lever for lifting overall model quality.
    - **Class name:** `gray_whale_moan` — gray whales don't "song"; moans (S3, low-freq, migration context) are the sound most likely to overlap humpback and be mislabeled. If John flags knocks (S1) / croaks (S4) instead, may need a broader `gray_whale` or a second class. Renaming later is a trivial SQL `UPDATE annotations SET label=…` (no re-embedding), so `gray_whale_moan` is safe to commit to now. **John's ear makes the reclassification — not the model side (attribution rule).**
    - **Where to hunt:** April, not October. Gray whales migrate past central CA in winter/spring (northbound Feb–May); October is peak humpback / near-zero gray whale. Humpback label counts: **April 2018 = 41, April 2026 = 39, Oct 2020 = 259**. Start with the 80 April clips (higher contamination yield); the 259 Oct labels are likely clean.
    - **Workflow (no-GPU prep, then one review):** `tools/export_labels.py` writes per-(month,species) JSON to `json_labels/labels_{month}_{species}.json` → `tools/labels_json_to_review_csv.py --month 2018_04 --species humpback_song` converts it to a review CSV (7-col `idx,project,filename,window_start,window_end,label,logits`; review locates clips by filename+window_start+`--audio-dir`, `idx` is cosmetic; `--window-len` default 5.0) → `phase2_classify.py review --target-label humpback_song --detections-csv … --classes humpback_song,gray_whale_moan,other,unlabeled`. One `--audio-dir` per session, so April 2018 and April 2026 run separately. (Apr 2018 CSV built July 20: `results/review_apr2018_humpback_graywhale.csv`, 41 clips.)
    - **Status (July 20 2026):** April 2018 41-clip review is **prepped and paused** — CSV built, launch command staged (port 7866), waiting on J. Ryan's availability; will launch when he has time. April 2026 (39 clips) is a one-command repeat afterward. Separately, the Oct 2020 orca-*survivor* review (10 clips ≥1.16) is **complete** — all humpback, 0 orca (see Label Class Definitions table); that was orca-FP characterization, not the gray-whale humpback review, which is still to come.
14. **Extended April 2018 orca activity — CONFIRMED (D. Edgington, July 19 2026).** v4 cross-month validation (threshold sweep, `tools/score_orca_regions.py`) surfaced strong, threshold-robust orca detections across **Apr 13–25 2018**, not just the confirmed Apr 13 event. At logit ≥ 1.16: Apr 18 = 173 (rivals Apr 13's 251; 105 survive +2.0), Apr 23–25 cluster = 118, full Apr 13–25 window ≈ 569. **Expert review complete for Apr 18 + Apr 25: 75 detections labeled (25 Apr 18 + 50 Apr 25), 100% orca, 0 false positives at ≥1.16** — orca_call annotations 219→294. April 2018 is now established as a **sustained ~2-week orca presence**, not a one-day event. Still to review: Apr 18 mid-logit band CSV exists (69 windows, port 7863, unreviewed), Apr 23/24. Counter-example note (revised July 21 2026): the sweep shape once made "May 14 reads FP-like (19→1 across the sweep)" look like a non-event, but expert review confirmed **all 4 May 14 detections at ≥1.16 as orca** — the collapse-shape heuristic gave a false FP signal here; ear review is authoritative. The genuine specificity counter-examples are Oct 2020 / Apr 2026, whose ≥1.16 residuals reviewed as humpback. Confirmed-orca example figures registered: `gradio_apr18_2018_orca_195s_wid202720`, `_405s_wid202762`, and five Apr 25 clips (`gradio_apr25_2018_orca_*`). **External corroboration (non-acoustic):** KSBW Action News 8 (Monterey) reported "record orca sightings" in April–May 2018 (one tour group ~50 orcas in a day), orcas drawn to hunt gray whales, and biologists identified **two pods that spring — one Alaskan, one Californian ("Emma's pod", the CA140s — matriarch CA140 "Emma", a Bigg's/transient matriline known for hunting gray-whale calves)**. Figure: `ksbw_news8_orca_invasion_monterey_spring2018` (© KSBW, reference only). Two independent methods (hydrophone + visual sightings) agreeing on a sustained spring-2018 presence.
    - **CA140 matriline ("Emma's pod") — family structure, recurrence, gray-whale link (July 21 2026):** CA140 "Emma" is the matriline matriarch; **CA140B "Louise" is her daughter** (with her own offspring — Stinger, Bee, Buzz, Bumble). Not a naming conflict: Emma leads the matriline, Louise leads a sub-group within it. The **same matriline recurs across event windows** — "Emma's pod" in the spring-2018 KSBW report, and CA140B "Louise" in the Oct 2020 whale-watch reports — i.e. the same Bigg's family appears in both the 2018 and 2020 data. The CA140s are documented **gray-whale-calf hunters**, so their presence is a biological reason to expect gray-whale calls in the recordings — an independent line pointing at the #13 humpback/gray-whale contamination. (Source: California Killer Whale Project.)
    - **May 2018 is likewise multi-day (D. Edgington, July 21 2026).** Beyond the confirmed May 12 event (181/181), review of the non-May-12 orca detections at ≥1.16 confirmed orca on **May 13 (1), May 14 (4), May 16 (3)** — **8 new orca labels**, orca_call 181→189 (2 clips left unlabeled as too faint, incl. a May 7 singleton). Two May 16 clips from the *same* recording 25 s apart sound audibly different ("a bit different, but clearly orca") — within-encounter call variation, relevant to the by-day/per-encounter t-SNE thread. **Spring 2018 now shows sustained Bigg's presence across BOTH months** (Apr 13/18/25 + May 12/13/14/16), matching the KSBW "record sightings April–May" report. Tooling: `tools/export_labels.py` now includes May 2018 (`2018_05`) so these labels are captured in the JSON snapshot. Caveat: May 13 is a single-clip day (rests on one detection); May 14's 4-clip morning cluster is the more robust new day. **v5 retrain queue:** these +8 May orca labels join the extended-April orca (+75) and Oct 2020 humpback relabels (+4), pending John's gray-whale/humpback work before retraining.

---

## Label Class Definitions

**`negative` (label_type=2, weak negative):**
Background/ambient ocean noise — clips explicitly labeled as "not any target species."
Used as weak negatives during training (`weak_neg_weight=0.050`). Does NOT appear
as a detection class in inference output. Includes April 2018 genuine orca-negative
clips (54) plus background clips from other months. These are windows where nothing
biological is happening.

**`other` (label_type=1, positive class):**
A positive detection class — clips with real acoustic content that doesn't fit any
of the four named species. Includes mixed signals (humpback + dolphin simultaneously),
unusual sounds, ambiguous calls, and background clips relabeled from `negative`.
DOES appear in inference output.

**Key distinction:** `negative` = silence/background (weak negative for training only).
`other` = real sound but unclassifiable (positive class, appears in inference).

**`humpback_song` (label_type=1, positive class):**
Despite the name, this denotes **humpback vocalization generally, not strictly full complex
song** (J. Ryan, July 20 2026): "there are periods when non-song social humpback sounds will
be present, so we don't always need full complex song as a humpback identifier." A 5 s focal
window counts as humpback if it is part of a humpback sequence. Relevant to the gray-whale
review (#13): the class is "humpback vocalization," and gray-whale moans are the expected
contaminant to separate out.

In t-SNE: gray (negative) forms a loose cluster in lower center.
Orange-red (other) sits at the humpback/orca/negative boundary.

| Month | Key finding |
|---|---|
| April 13 2018 | 289 orca detections — confirmed Bigg's orca hunting event ✅ |
| April 18 2018 | 173 orca @≥1.16 (v4); 25 reviewed = 100% orca — CONFIRMED bout (D. Edgington) ✅ |
| **April 21 2018** | **40 total, 25 @≥1.16 (v4) — ⚠️ ACTION ITEM: real signal, NOT YET REVIEWED.** Found Aug 19 2026 building poster FIG 4 data; never named in `orca_region_scores_v4.csv`, no Gradio session run. Do not present as confirmed or dismissed. |
| April 23–25 2018 (cluster total) | 118 orca @≥1.16 (v4) — cluster total, NOT a single-day count. Per-day breakdown: Apr 23 = 39 @≥1.16, Apr 24 = 19 @≥1.16, **Apr 25 = 211 total / 60 @≥1.16** (Apr 25: 50 reviewed = 100% orca — CONFIRMED, D. Edgington) ✅; Apr 23/24 individually pending |
| May 12 2018 | 181 orca labels — full review 181/181 confirmed, Bigg's orca event ✅ |
| May 13 2018 | 1 orca @≥1.16 (v4) — confirmed orca by ear (D. Edgington) ✅ (single clip) |
| May 14 2018 | 4 orca @≥1.16 (v4) — all 4 confirmed orca (D. Edgington), ~06–07h morning cluster ✅ (real secondary event) |
| May 16 2018 | 4 reviewed @≥1.16 (v4) — 3 confirmed orca (D. Edgington), ~15:09 cluster ✅ (1 too faint, unlabeled); figs wid9642/wid9647 |
| May 7 2018 | 1 @≥1.16 (v4) — too faint to confirm, left unlabeled |
| October 2020 | Oct 5-12 cluster confirmed — zero orca vocalizations (Bigg's orca silent during hunts) ✅. **Threshold-review July 20 2026 (J. Ryan):** the 10 detections surviving ≥1.16 are all humpback (4 newly labeled, 6 already-known humpback → 263; 1 left unknown on ambiguous 30 s context), **0 orca labels in the month**. Even at the operating threshold the residual "orca" hits are windows already known to be humpback — Oct 2020 confirmed orca-silent (true-negative / specificity result). |
| April 2026 | Apr 21 dominant (101 v4 detections) — all reviewed clips are humpback FP; consistent with Bigg's orca acoustic silence. Apr 17-24 CA51A/CA50B event window shows 129 detections but no confirmed orca vocalizations. |

**Orca cross-month threshold validation (July 19 2026):**
- Tools: `tools/run_orca_validation.sh` (v4 inference over Apr/May 2018, Oct 2020, Apr 2026 @ floor 0.0) + `tools/score_orca_regions.py` (threshold sweep vs known regions). Summary: `results/orca_region_scores_v4.csv`. **CAUTION (Aug 19 2026):** this file's `region`/`truth` labels ("SUSPECTED", "probable") are a snapshot from the original hypothesis-generating sweep and are now STALE — Apr 18/23-25/May 14 have since been expert-confirmed, and May 13/16 aren't in the file at all (discovered in a later review). Trust this file's raw `det_T*` count columns; do NOT trust its status labels — cross-check confirmation status against the Validated Events table above. Also: some rows are multi-day CLUSTERS (e.g. "Apr 23–25... = 118 @1.16"), not single days — don't misread a cluster total as one day's count (this happened once, corrected in `agile_modeling_history.md` Aug 19 2026).
- Inference CSV is **per-label** (one row per window per class), so T=0.0 counts are inflated by also-ran windows — read the NEGATIVE regions at higher thresholds, not at 0.0.
- False positives collapse under thresholding: Oct 2020 144→1, April 2026 323→6 (T=0.0→+2.0). Confirmed events retain: Apr 13 99%→74%, May 12 95%→40%.
- **Operating threshold: +1.16 (v4 F1-optimal) primary, +1.5 conservative.** At +1.16 FPs are ~90% gone and Apr 13 still holds 87%; the default 0.0 is unusable (144 / 323 FPs). Residual FPs never quite reach 0 (single digits at +2.0) — pull those specific windows in Gradio to characterize.
- Surfaced the extended-April-2018 finding (#14).

**Expert observations from the extended-April review (D. Edgington, July 19 2026):**
- **Ship noise co-occurs with orca on event days** — whale-watch / California Killer Whale Project boats arrive to view orcas once spotted and record them, so `ship_noise` and `orca_call` are positively correlated in time during events. Consequence: `ship_noise` is NOT an orca-absent cue here — any future FP-suppression logic keying on ship noise would be exactly wrong for event days.
- **Call variation across days** — Apr 13 / 18 / 25 calls sound somewhat different (possibly different pods, individuals, or call types/meaning). Consistent with the extended window being multiple encounters rather than one continuous event. Future refinement: acoustic pod/individual ID.
- **Clean orca ⇒ quiet background** — confirmed orca calls, even faint ones near the +1.16 threshold, tend to occur in acoustically sparse windows (little humpback/other-animal energy). Testable hypothesis: orca/humpback confusion is driven by the classifier firing on humpback energy in *busy* windows, not by orca acoustics — which would mean the FPs cluster where humpback is present (links to #13) and score lower (why the higher threshold works). Possible future feature: background energy / co-occurrence as a disambiguator.

**By-day orca t-SNE — Apr 25 within-month separation (July 19 2026):**
- Tool: `tools/plot_tsne_orca_by_day.py` (confirmed orca_call embeddings, colored by day; `--style analysis` light / `presentation` dark). `tools/archive_tsne_by_day.sh` archives the full perplexity(10/30/50) × style × {April, 4-day} matrix (12 figs) with provenance.
- **Apr 25 2018 orca separates from the Apr 13 Bigg's event in embedding space, within the same month.** Robust across perplexity 10/30/50; not a single-recording artifact (`tools/orca_day_recording_spread.py`: Apr 25 = 50 windows across 10 recordings spanning ~3.5 h, an **evening** encounter vs Apr 13's morning event). Apr 18 partially distinct. May 12 separates too but cross-month → confounded, not interpreted.
- **Confound-clearing template (reusable for any "calls look different" claim):** (1) same-month comparison (kills season/background), (2) distinct-recording spread (kills single-recording/boat artifact), (3) perplexity sweep 10/30/50 (kills t-SNE artifact). Apr 25 clears all three. This corroborates the "call variation across days" expert observation above.
- **Limit:** Perch V2 embeds species and collapses within-orca variation, so this is a strong lead, not proof. Interpretation (pod / individual / call-type / evening-vs-morning acoustic context) is **pending D. Edgington's direct listening** — the model can't resolve it. Figures: `tsne_orca_by_day_{april2018,4days}_px{10,30,50}[_pres]`.
- **Candidate biological explanation (July 20 2026):** KSBW news (fig. `ksbw_news8_orca_invasion_monterey_spring2018`) reports **two distinct pods in Monterey Bay spring 2018 — one Alaskan, one Californian (the CA140s / "Emma's pod")**. Different pods carry different call repertoires, so this is a plausible reason the days separate in embedding space. It is **consistent with, not proof of** the separation — detections are not yet assigned to pods, and morning-vs-evening acoustic context remains a confound. Turns "the calls look different" into a testable question: *do the separating clusters correspond to the two known pods?*

**Context embedding experiments (July 15 2026):**
- 30s Gaussian-weighted t-SNE: orca completely separated from humpback (zero overlap) — colleague-suggested method ✅
- Context DB (v5 classifier): ROC-AUC dropped 0.959→0.930, cmap 0.830→0.595 — context averaging hurts training
- Context post-processing filter (orca/humpback ratio): suppressed April 13 2018 orca — filter fails because Bigg's orca calls are brief discrete bursts, not sustained bouts
- **Conclusion:** Raw embeddings + v4 classifier remain best. Temporal sequence modeling is the right path for disambiguation.

---

## Repo Structure

```
perch-hoplite/
├── src/                    — modular Python library
│   ├── spectrogram.py      — 4-mode spectrogram generation
│   ├── audio.py            — audio encoding + 30s context
│   ├── torch_model.py      — TF mock + model loading
│   ├── train.py            — PyTorch classifier training
│   ├── infer.py            — inference + NFS-safe CSV writing
│   ├── review.py           — Gradio labeling GUI
│   └── paths.py            — canonical path definitions
├── tools/                  — standalone scripts
│   ├── merge_annotations.py
│   ├── merge_dbs.py
│   ├── plot_monthly.py
│   ├── plot_tsne.py
│   ├── plot_tsne_orca_by_day.py    — confirmed orca t-SNE by day (--style analysis/presentation)
│   ├── archive_tsne_by_day.sh      — generate+register perplexity×style t-SNE matrix (12 figs)
│   ├── orca_day_recording_spread.py — per-day distinct-recording confound check
│   ├── run_orca_validation.sh      — multi-month v4 orca inference (validation)
│   ├── score_orca_regions.py       — threshold sweep vs known ground-truth regions
│   ├── labels_json_to_review_csv.py — export_labels JSON → review CSV (re-review existing labels)
│   ├── extract_example_clips.py
│   └── review_example_clips.py
├── docs/                   — documentation and analysis
│   ├── pytorch_port_summary.md   — PyTorch Conference 2026 poster
│   ├── PyTorch_poster.md   — CURRENT PyTorch Conf abstract (updated Jul 17; supersedes PyTorch_abstract.md — real throughput ~635 clips/s w/ torch.compile, 2.5x vs ONNX; ~2.1M embeddings; update deadline Jul 26)
│   ├── PyTorch_abstract.md   — original Jul 12 submission, superseded by PyTorch_poster.md (kept for history)
│   ├── oceans_2026_acceptance_record.md   — IEEE OCEANS 2026 poster ACCEPTED; authors D. Edgington + J. Ryan (co-author, not just reviewer); poster-only, no manuscript
│   ├── october_2020_analysis.md
│   ├── FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md
│   └── PROGRESS_2026-07-09.md
├── figures/                — plots and screenshots
├── phase2_classify.py      — main CLI
├── phase1_embed_torch.py   — embedding pipeline
└── README.md
```
