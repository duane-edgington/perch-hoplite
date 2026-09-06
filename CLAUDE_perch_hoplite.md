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

## Roadmap (as of Aug 23 2026)

Options discussed and prioritized. Not a committed schedule — a menu of directions, roughly
ordered short-term → long-term. Highest-leverage items flagged.

### Short term (finishable, mostly solo, closes current loops)

1. **Finish `orca_v10`, then test it against v4 on the May 2018 hold-out.** **[May hold-out test DONE Aug 23 2026 — v10 wins decisively, see finding #20; remaining: humpback split + ship_noise depth to further improve v10, and a precision check before formal production swap.]** This is the
   near-term anchor. Improve v10 with the best available data (items 2-3 below), then judge
   v10 vs. v4 on **May 2018 — the permanent held-out test month (finding #18)** — specifically:
   is v10 better than v4 at picking up orca calls on data neither model trained on? May is the
   honest referee; this is the experiment that validates whether today's labeling work actually
   improved the model, not just changed the eval set.

2. **[HIGH-YIELD] Split `humpback_song` → `humpback_song` + `humpback_vocalization`, and
   reannotate.** The gray-whale hypothesis for humpback's weak F1 is dead (finding #13 closed);
   the real cause is that the class lumps true song with other (non-song) humpback
   vocalizations. This is the diagnosed, highest-yield model improvement available — the class
   with the most existing data (321 labels) and the clearest fix. **Requires John Ryan** (the
   humpback expert) to help define the song/non-song boundary crisply before reannotating.
   Panel 11 of the poster already promises this as the next classifier step.

3. **More ship_noise labels (April 2018 first, other non-May months as needed).** Depth review
   past the top-25 (rank ~26-75), no threshold gate, to push ship_noise eval support from n=10
   toward dolphin's n=44 (needs ~+150-180 labels; supply is not the constraint, quality-at-depth
   is the open question). NOT from May 2018 (held-out). Lower ceiling than the humpback split
   (ship_noise is already at 0.80 F1) but a clean, self-contained solo task.

4. **Decide `orca_v10`'s production status.** **[Aug 23 2026: RESOLVED in favor of v10. May hold-out test (#20, recall) + precision check (#21, 14/14 real orca, zero false positives at threshold) both confirm v10 > v4 on unseen data. v10 recommended for production. Optional final step: a formal precision sweep over more of May if a published precision number is wanted, but the threshold-region check is already clean.]** It was trained but never formally designated as
   v4's replacement. A decision, not labor — but a real loose end (which model does future
   inference use?). Resolve deliberately after the May hold-out test (item 1) gives evidence.

### Longer term (research thrusts, higher ceiling, higher effort/risk)

5. **[HIGHEST CEILING] Analyze the external public orca dataset — expand to multiple data
   sources and multiple ecotypes.** Palmer et al. 2025, *"A Public Dataset of Annotated
   Orcinus orca Acoustic Signals for Detection and Ecotype Classification"* (Scientific Data,
   doi:10.1038/s41597-025-05281-5; PDF in `references/s41597-025-05281-5.pdf`). 225,000+
   annotations, 23 locations, 11 years, Alaska/BC/WA, multiple hydrophone systems, archived at
   NOAA NCEI. **Includes Bigg's/West Coast Transients (our ecotype) plus Residents and
   Offshores** — so it both validates our MARS Bigg's work against an external Bigg's population
   AND opens ecotype classification (Resident vs. Bigg's vs. Offshore) as a published benchmark.
   Why highest-ceiling: transforms the story from "a detector for our bay" to "Perch V2
   embeddings transfer across populations and equipment." First experiment is small: pull the
   collated ecotype CSV, embed a balanced Resident/Bigg's/Offshore sample with Perch V2, linear
   probe — do ecotypes separate? Wrinkle: heterogeneous sample rates (9-250 kHz, some
   low-passed) need resampling to our 32 kHz standard; annotations are Raven selection tables
   (we have `csv_to_raven.py`). Scouting done Aug 23 2026; not yet started.

6. **Add temporal / sequence analysis to the Perch V2 pipeline.** Current windowing treats each
   5s window independently, but real encounters are call *sequences* — "we don't get one call
   in a 5s window, we get lots." Modeling sequences (bout structure, call-type transitions,
   possibly pod/individual signatures in temporal pattern) is the most scientifically
   interesting thrust and the highest-risk (new architecture on frozen embeddings, weeks of
   work, no guarantee). Best attempted *after* item 5 provides multiple populations to find
   sequence differences between — sequence modeling is far more compelling with multi-population
   data. File as "the ambitious next paper," not the next sprint.

7. **Expand to more dolphin species and more humpback vocalization types.** Partly downstream of
   the humpback split (item 2, already covers song/non-song) and the external data (item 5,
   which notes Pacific white-sided dolphins as a confounder). Let data acquisition drive these
   rather than starting cold.

8. **Try to detect gray whale moans.** Gray whales are essentially absent from current MARS
   data (the whole point of finding #13 — zero contamination found in two April batches). So
   this needs *new data where gray whales actually occur* — loops back to item 5 (external
   dataset) or new MARS months chosen for gray-whale season. Do not start cold; let data drive.

### Cross-cutting

- **New curated public repo before the conference.** Present a clean v0→v4→v10 lineage without
  the v5-v8 detour needing constant caveats (noted in finding #18). Also the natural home for
  external-dataset work if item 5 proceeds.

---

## System Architecture

| Machine | Role | Notes |
|---|---|---|
| **ICEFISH** (Mac M1 Max) | **RETIRED Aug 2026** — backed up to portable SSD, held through PyTorch Conf (Oct) per IT agreement, then returned | 134.89.114.25 / VPN 134.89.74.134 |
| **PERCH** (MacBook Pro M5 Max, 128GB, Tahoe 26.6.1) | **Current dev workstation, scp gateway** — replaces ICEFISH | LAN (en9, wired/dock — same subnet as sparks): **134.89.11.172** · Wi-Fi (en0): 134.89.112.169. Get via `ifconfig | grep "inet " | grep -v 127.0.0.1` (macOS `hostname -I` does NOT work, unlike Linux). |
| **spark-ae0e** (134.89.11.107) | Primary compute — NVIDIA GB10 DGX | Working dir `~/perch-hoplite/` |
| **spark-0626** (134.89.11.174) | **Validated second compute host (Aug 20 2026)** — NVIDIA GB10 DGX, same clone/setup via `scripts/clean_install.sh`. Confirmed: Gradio review renders identically to spark-ae0e (spectrograms + audio) against the same DBs. Package versions differ slightly (torch 2.13.0 vs 2.12.1, numpy 2.5.2 vs 2.4.4, usearch 2.26.0 vs 2.25.3, gradio 6.25.0 vs 6.15.1) — perch-hoplite's own pins kept librosa/scipy/pandas identical; the differences are believed low-risk (see clean_install.sh output Aug 20) but haven't been stress-tested beyond one review session. Good for a second parallel review (e.g. John on one host, Duane on the other) or as overflow when ae0e's GPU is busy with training. |
| **thalassa** | NFS server | thalassa.shore.mbari.org |

**macOS screenshot filenames — ALWAYS glob on the timestamp, NEVER retype/escape the name.**
This has caused repeated real failures (July 19, Aug 21 x2) — worth reading carefully.
Screenshot filenames contain spaces, and sometimes a non-ASCII space character that is
**invisible on screen** but will NOT byte-match anything you retype — including a fully
quoted string (`"Screenshot 2026-08-21 at 7.09.31 PM-1.png"`) or `$HOME/"..."`. **Quoting
does not fix this.** Backslash-escaping most of the name with only a short trailing wildcard
(e.g. `Screenshot\ 2026-07-16\ at\ 5*.png`) is ALSO unreliable — it still tries to literally
match several of the spaces. **The only reliable pattern is a wildcard on both sides of just
the unique timestamp digits, with zero literal spaces anywhere in the command:**
```bash
# scp from Mac to spark (glob the whole name away except the time):
scp ~/Desktop/Screenshot*7.09.31*.png duane@134.89.11.107:~/perch-hoplite/figures/
# mv/rename, same rule, on either machine:
mv ~/Desktop/Screenshot*7.09.31*.png ~/Downloads/gradio_apr2026_humpback_1.png
```
`--original-name` in `register_figure.py` is just a metadata string, so a typed
approximation there is fine — the glob-only rule matters solely when a command needs to
actually touch the real file (mv, scp, cp, etc.).

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

**Gradio version 6.15.1 was the original validated version; 6.25.0 also confirmed working**
(spark-0626, Aug 20 2026 — identical rendering/audio against the same DBs). Not known to be
version-sensitive within this range; if a NEW version misbehaves, incognito (below) is still
the first thing to try before suspecting the package.

**Browser for Gradio: Chrome (incognito) — ALWAYS, not just for Safari's audio issue.**
Symptom if you forget: the page loads but shows spinning wheels / never renders, even though
the server is fine — normal (non-incognito) Chrome windows can serve stale/cached state. **This
has bitten us before** (rediscovered painfully Aug 20 2026 after a month away from the
project) — if a Gradio session looks broken, try incognito FIRST, before debugging the server,
the DB, or the classifier. Safari has a separate, additional problem: audio playback fails
with data: URIs, so Safari is out regardless.

---

## Classifier Versioning (new era — July 9 2026)

Old classifiers (v1_clean through v8_clean) are retired. New versioning starts at v0:

> **What `_clean` meant (clarified Aug 25 2026 from docs/agile_modeling_history.md).** The
> `_clean` classifiers were the **Phase 1 bootstrap models (v1_clean–v8_clean, June 2026),
> trained on labels AUTO-SEEDED from the Google Multispecies Whale Model score CSVs with NO
> human review** (~4,002 high-confidence orca positives + 242,708 negatives, selected by
> thresholding the Google model's scores). "Clean" most plausibly refers to that
> high-confidence / cleanly-separated auto-label tier — though the history doc never literally
> states why the word was chosen, so this is well-supported inference, not a documented
> definition. They were retired for TWO reasons that are easy to conflate: (1) trained on
> **un-normalized embeddings** (the July 9 low-amplitude bug), and (2) superseded by the
> **human-reviewed agile loop**. Note the naming break: `_clean` numbered v1–v8; the new era
> deliberately restarted at **v0** (not v9) as a clean break. **There was never a `v0_clean`**
> — v0 was already the first post-normalization, human-labeled model (584 hand-reviewed clips,
> July 9). Common misconception: "clean" was the human-verified era — it was the OPPOSITE, the
> pre-human auto-seeded era. (The Colab/TF → spark/PyTorch migration was a real concurrent
> change but is NOT what `_clean` referred to.)

> **Note (Aug 24 2026) — `orca_v4_clean.pt` does NOT exist; do not use it.** The current
> canonical models are `orca_v4.pt` and `orca_v10.pt` (no `_clean`), in
> `/mnt/PAM_Analysis/perch-hoplite/models/`. `phase1_embed_torch.py`'s "Next step" hint
> formerly referenced `orca_v4_clean.pt` — that stale hint was fixed Aug 25 2026 to print the
> correct models + output naming. Always run inference with `orca_v4.pt` / `orca_v10.pt`.

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
- **ship_noise was n=3 in *every* model** (only ~24-35 ship labels total, mostly April 2018).
  Its 1.0 was an artifact across the board. **Aug 21 2026 labeling campaign addressed this
  directly** (see #15 below) — total confirmed ship_noise went 35→81 project-wide (52 within
  the 3-season Option A training recipe specifically — see #16). Needs a retrain to
  actually populate a trustworthy ship_noise F1; not yet done.
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

**macOS screenshot filename gotcha** — see the consolidated rule + examples earlier in this
doc (System Architecture section, "ALWAYS glob on the timestamp"). Applies equally to `mv`
on spark as to `scp` from the Mac.

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


## HANDOFF — project state as of Aug 27 2026 EOD (for a fresh chat)

**Where things stand.** orca_v10 validated across every accessible month; poster accepted and at
v44; pipeline fully documented; John has redirected priorities (finding #26). **The full-archive
campaign (priority #1) has STARTED — July 2015 is done, zero orca (finding #29).**

### MACHINES — four of them; ALWAYS `git pull` first (Duane keeps all four synced by hand)

| Machine | Role | perch-hoplite clone |
|---|---|---|
| **DuaneEM1** | home, M1 MacBook Pro (DuaneE + M1) | `/Users/duane/perch-hoplite` |
| **PERCH** | work MacBook Pro | `/Users/duane/Projects/perch-hoplite` |
| **spark-ae0e** | GB10 compute, NVIDIA DGX | `/home/duane/perch-hoplite` |
| **spark-0626** | GB10 compute, NVIDIA DGX — near-identical to ae0e | `/home/duane/perch-hoplite` |

Both laptops are portable, so **Duane will say which machine he is on** — the clone paths differ
and a copy-paste command for the wrong laptop fails. **Always `git pull` on the machine in use
before any repo action**; Duane syncs all four by hand and says so himself.
- **Screenshots** originate in `~/Desktop` on whichever laptop, as
  `Screenshot YYYY-MM-DD at H.MM.SS AM/PM.png` (macOS default). Staged files land in `~/Downloads`.
  Yesterday's July figures were `mv`'d (not `cp`'d), so **the repo copy IS the archival copy** —
  each sidecar preserves `original_name` + parsed `capture_timestamp`, and git history is the only
  recovery path if a figure file is ever lost.
- **`--computer` in `register_figure.py` records where the SCREENSHOT was taken**, not where the
  analysis ran (Gradio serves from a spark; the grab happens on a laptop). The spark side is
  captured in the sidecar's `--command` field. Choices now include PERCH (added Aug 27 2026).
- **VERSION SEAM CHECK — RESOLVED Aug 28 2026, no seam exists.** Both sparks:
  **torch 2.12.1+cu130, CUDA 13.0**, `perch_hoplite==1.0.2` pinned on all four machines, and the
  `/mnt` mounts are identical. So embeddings from either spark are interchangeable and there is no
  need to record which box embedded which month. (`smb://thalassa.shore.mbari.org/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz`
  is the same store the sparks see at `/mnt/PAM_Analysis/...`.) Residual venv divergence is in
  other pip packages only; torch was the one that could have affected embedding numerics.
- **Suggested convention (not a rule):** **spark-0626 for resampling** (unattended, CPU/wall-clock
  bound, ~1 day per full month) and **spark-ae0e for embed / infer / review / train** (where Duane
  is at the keyboard, and where contention would actually be noticed). Keeps a long SoX run from
  competing with a training job or making a Gradio session sluggish. If run concurrently, keep the
  two boxes on different months so output paths never collide.

### FULL-ARCHIVE CAMPAIGN — running progress (start here next session)

The loop, one month at a time, per `CLAUDE_embed.md`:
**resample (Stage 1) -> verify (Stage 1.5) -> embed (Stage 2) -> coverage (Stage 2.5) ->
infer v4+v10 (Stage 3) -> score-band triage -> Gradio review -> record finding -> delete bulk WAV.**

| Month | Status | Result |
|---|---|---|
| **2015-07** | ✅ COMPLETE (Aug 27 2026) | Zero orca. 469 files, 56,130 windows. 2 dolphin_call, 4 other. Finding #29. |
| **2015-08** | ✅ COMPLETE (Aug 28 2026) | 3,793 files, **453,123 windows**, 629.34 h (84.6%), 8/16 absent, 5 long dropouts. Reviewed: **13 dolphin_call + 1 UNCONFIRMED orca candidate** (`MARS_20150828_212219` @325 s). 12 figures registered. Findings #31, #33. |  [UPDATED Sep 1 2026: candidate reclassified dolphin — ZERO orca]
| **2015-09** | ✅ COMPLETE (Aug 30 2026) | 4,323 files, **517,984 windows**, 719.43 h (**99.9%**). **TWO ORCA ENCOUNTERS, 18 confirmed calls** + 17 dolphin + 12 ROV_noise. 22 figures registered. Findings #34-37. |
| **2015-10** | ✅ COMPLETE (Aug 31 2026) | 4,293 files, **502,871 windows**, 698.38 h (93.9%), Oct 18 absent. **31 CONFIRMED ORCA — largest month yet**, incl. **10 calls in one recording**. Both episodes are **Oct 26 local**. 5 dolphin, 1 humpback, 8 unlabeled (multi-species overlap). Findings #41-42. |
| **2015-11** | ✅ REVIEW COMPLETE (Sep 2 2026) | **3,685 files, 441,598 windows**, 613.33 h (85.2%). **236 orca confirmed** (DB authoritative; pass-2 chunks 8-24 skipped — diminishing returns). Episode A: Nov 22-23 local (~215 night). Episode B: Nov 28. Campaign high v10=4.662. Findings #45-51. |
| **2015-12** | ✅ COMPLETE (Sep 2 2026) | **4,403 files, 526,642 windows**, 731.45 h (98.3%). **4 confirmed orca** — sharp seasonal decline from Nov's 236. Two overnight episodes (Dec 6 and Dec 19 local). Finding #52. |
| **2016-01** | ✅ COMPLETE (Sep 4 2026) | **4,023 files, 479,020 windows**, 665.33 h (89.4%). **163 confirmed orca** (pass-2 through chunk 5). Jan 21-22 absent. Campaign high v10=4.692. Jan 15: 300 KW sighted, 0 acoustic. Findings #53, #55. |
| **2016-02** | ✅ COMPLETE (Sep 4 2026) | **3,133 files, 342,029 windows**, 68.3% (Feb 4-8 + Feb 13 absent). **ZERO orca** despite 6 MBWW sighting days with full coverage — crypsis during gray whale hunt. Finding #54. |
| **2016-03** | ✅ COMPLETE (Sep 4 2026) | **4,460 files, 533,460 windows**, 740.93 h (99.6%). **4 confirmed orca + 4 other** (unknown ecotype). Mar 29 sighting correlation. Mar 7 mystery call. Finding #56. |
| **2016-04** | 🔄 PASS-1 COMPLETE, pass-2 pending (Sep 5 2026) | **4,299 files, 514,843 windows**, 715.07 h (99.3%). **182 confirmed orca** pass-1. New high v10=4.698. Apr 19-21 sighting correlation. Finding #57. |
| **2016-05** | ✅ COMPLETE (Sep 5-6 2026) | **4,474 files, 535,379 windows**, 743.59 h (99.9%). **141 confirmed orca**. New high v10=5.139. Finding #59. |
| **2016-06** | ✅ COMPLETE (Sep 6 2026) | **3,597 files, 431,277 windows**, 599.00 h (83.2%). **48 confirmed orca**. New high v10=5.141. Jun 19-20: 400-500 KW sighted, 0 acoustic — strongest crypsis case. Finding #60. |
| **2016-07** | 🔄 RESAMPLING (Sep 6 2026, spark-0626) | PDT (UTC-7) throughout; MBWW: zero KW sightings |
| **2016-08** | ⬅️ NEXT after Jul | |
| **2015-11** | ⬅️ NEXT after Oct | |

**Canonical Stage 1 command** (day-range args; month WITHOUT leading zero):
`./tools/resample_sox_32k_batched_vol.sh <year> <month> [start_day] [end_day] [max_jobs]`
**NEVER** use `tools/resample_sox_32k_batched_novol.sh` — identical args, identical output path,
no filename marker, but omits the mandatory `vol 3` volts calibration (J. Ryan: always vol 3,
clipping is not a concern for signals of interest).

**STORAGE — corrected measured figures (Aug 31 2026; supersedes the Aug 30 numbers).**
thalassa `/mnt/PAM_Analysis`: **51 TB total, 47 TB used, 4.3 TiB free (92%)**.

⚠️ **The Aug 30 figures were wrong.** "873 GB across 7 months → ~125 GB/month" came from a `du`
crawl that had not finished (the parallel one that later died with the VPN). A completed
`du --max-depth=2` on Aug 31 gives **1.2 TB across 8 months**, and:
- **0.215 GB per HOUR RECORDED** — constant to three digits across all eight months, as expected
  for 32 kHz 16-bit mono. **Predict any month with `hours x 0.215 GB`.**
- **~160 GB for a full 31-day month** (2018-05 = 160 GB / 744 h; 2020-10 = 160 GB / 743 h).
- Partial months scale down exactly: 2015-07 = **17 GB** for 78 h; 2024-09 = **97 GB** for 450 h.
- **Revised projection: ~20 TB for 130 months**, not 16 TB. Still does not fit in 4.3 TiB, so the
  conclusion stands: **analyze-then-archive is what makes the campaign possible, not merely tidy.**
- **Permanent artifacts remain trivial:** ~1.6 GB embeddings/month → **~210 GB for all 130 months**.
  Storage will never constrain the outputs, only the transient WAV.

**Current live set (Aug 31 2026): 458 GB across four 2015 months** — Jul 17 GB, Aug 136 GB,
Sep 155 GB, Oct 150 GB. **July, August and September are all COMPLETE** (findings written, figures
registered, coverage CSVs committed) and therefore all qualify for WAV deletion under the standing
rule. July was kept deliberately as the reference month while the loop was new; **that rationale is
thinner now the loop has run four times.** Deleting Aug + Sep would free **291 GB** and take the
live set back to two months. Not urgent at 4.3 TiB free, but **the discipline is easier to keep on
schedule than to restart under pressure.**

**What the 24 kHz / 48 kHz deletions actually bought (Aug 30 2026):** the share moved from **96% to
92% used**. Both trees covered only a few selected months rather than the full archive, so at 51 TB
scale the reclaim was modest — a few hundred GB, not TB. Worth knowing before assuming a similar
cleanup could free campaign-scale space again: **there is nothing comparable left to delete.** From
here, headroom comes from the 32 kHz purge rule, not from retiring old model artifacts.
- **Reclaimed Aug 30 2026:** the 24 kHz tree (GoogleMultiSpeciesWhaleModel2) and the 48 kHz tree
  (OrcAI) were deleted via `find ... -depth -delete`. Both covered only **a few selected months,
  not the full archive**, so the freed space was small at this scale. Everything else under
  `/mnt/PAM_Analysis/perch-hoplite/` stays: `db/`, `provenance/`, `labels/`, `json_labels/`,
  `models/`, `results/`, `example_clips/` are small and hard to reproduce. **If storage pressure
  ever forces the issue, the plan is rsync to a 2 TB external SSD** rather than spend time on
  selective review — 2 TB holds all of it many times over.
- **Selective 32 kHz purge rule** (for the ~130 repetitions): delete a month's resampled WAV once
  its finding is written, figures registered, and coverage CSV committed — at that point
  everything needed to interpret the month survives, and the WAV is regenerable from the raw
  PAM_Archive anyway. **July 2015 is a deliberate exception**, kept as the reference month while
  the loop is new.

**Storage discipline is the binding constraint, not GPU time.** Duane is
reclaiming space by deleting old 24 kHz / 48 kHz resamples for other models (frees more than
Jul+Aug 2015 consume). **July 2015's resampled WAV is being KEPT** as the reference month while
the loop is new. From August onward: analyze-then-archive — keep embeddings (usearch.index +
hoplite.sqlite TOGETHER, #28) + crown jewels, delete bulk resampled WAV.

**Canonical facts a new chat needs:**
- Models: `orca_v4.pt` (prior production / poster worked-example), `orca_v10.pt` (current best /
  poster result), at `/mnt/PAM_Analysis/perch-hoplite/models/`. NO `orca_v4_clean.pt` (see
  Classifier Versioning; `_clean` = retired pre-normalization bootstrap era, v1-v8).
- May 2018 = permanent held-out test month (finding #18). Training recipe = Apr2018 + Oct2020 +
  Apr2026 (three-season). v10 beats v4 on the May hold-out: recall + precision, 4 new orca days
  (findings #20/#21/#27).
- Two dirs both named perch-hoplite: the GIT REPO clone (e.g. ~/perch-hoplite on spark) vs the
  NFS DATA area /mnt/PAM_Analysis/perch-hoplite/ (holds db/, json_labels/, results/, models/,
  logs/). json_labels/ + models/ + labels/ + provenance/ + example_clips/ = the <10 MB crown
  jewels. Embeddings = usearch.index files (~200 GB for 11 yr); sqlite = metadata only (#28).
- Three git clones: EM1 (/Users/duane/perch-hoplite), spark-ae0e (~/perch-hoplite, runs code),
  Perch work machine (/Users/duane/projects/perch-hoplite). All push to
  github.com/duane-edgington/perch-hoplite (PUBLIC).

**Working method (hard-won):** (1) Claude stages files to /mnt/user-data/outputs, canNOT push;
Duane pushes via a clone (download -> cp -> grep-verify new text -> commit -> push). (2) After
push, grep the pushed file on GitHub for the new text before believing it landed. (3) pull
before edit; one working copy, not stale clones. (4) On spark, run scripts via quoted-heredoc-
to-file, never fragile inline `python3 -c` (finding: Known Issues working-method note).

**Priority #1 (John, finding #26): full-archive seasonal/interannual orca analysis** — run best
model(s) across the ENTIRE MARS archive, characterize seasonal/interannual variance, cross-verify
with California Killer Whale Project sightings, correlate with warming/prey migration. Chunk it
to respect thalassa at 96% (analyze-then-archive; embeddings are ~1% of audio size so keep
embeddings, delete bulk resampled WAV after analysis). De-prioritized: other ecotypes, external
datasets (Palmer 2025). Focus = transient/Bigg's (CA140-associated) calls.

**Other open threads (all no-urgency / tabled):**
- Public repo build: take CLAUDE_repo.md + CLAUDE_release_plan.md + this file to a FRESH chat.
- Storage cleanup: backup disk -> rsync --dry-run -> verify -> remove ~723 GB bulk resamples
  (thalassa_storage_survey.md). Keep embeddings (usearch+sqlite together) + crown jewels.
- Embeddings archive for John (3 Perch users): ~200 GB for 11 yr — see #28.
- Sept 1-19 2024: humpback-heavy, no orca clustering — confirm with ~10-min ear-review of top
  >=2.31 clips (finding #25). 9/27 encounter NOT in acoustic record (real outage).
- **Blind-review strategy (DEFERRED by Duane + John, Aug 27 2026 — action item retained, not dropped).**
  Explicitly de-prioritized for now; the campaign focus is Duane reviewing Gradio sessions looking
  for ORCA. A blind-review / two-annotator design still needs to be devised later (findings #23/#26).
  Note: the Aug 24 April 2026 review server (13 clips, `ge231`) was left running 3 days and was
  killed Aug 27; it had NO `--annotator-id`, so anything labeled in it went in as generic `analyst`.
- **Reviewer stance for the campaign (Aug 27 2026):** Duane is the reviewer; the operative question
  is **orca / not-orca**. A not-orca call of "humpback" is a **working label, not authoritative** —
  John's humpback labels are the class ground truth. **ALWAYS set `--annotator-id duane`** so the two
  are distinguishable in the pool. **At some point every humpback detection gets reviewed and a
  classification scheme decided** (open action item; interacts with the song/vocalization split below,
  and the humpback pool grows with every campaign month reviewed).
- **ASK JOHN (Monday, week of Aug 31 2026): listen to the July 2015 Gradio set**, especially the
  **four `other` clips Duane could not identify** — Duane is a software engineer / neurobiologist,
  not (yet) a marine-mammal sound expert, and wants an expert ear on them. The two `dolphin_call`
  calls are probably solid; "real sound, not orca, don't know what" is where John's ear is needed.
  Highest value: `MARS_20150731_222345` @335 s — **both models' top hit and concordant** (v4 1.548 /
  v10 2.002), which Duane judged not-orca. If John agrees, it is a clean calibration point for a
  high-scoring non-orca in 2015-era audio; if not, finding #29 changes. Full clip table and context
  in `docs/multi_annotator_design.md` §8.
  **ALSO ASK HIM ABOUT THE AUGUST 2015 ORCA CANDIDATE** — `MARS_20150828_212219` @325 s, finding
  #33. Spectrogram already posted to Slack (Aug 30 2026). This is the higher-stakes ask: it is the
  campaign's first non-spring orca candidate, and Duane's ear is the only evidence for it.
  Figure: `figures/gradio_aug28_2015_ORCA_325s_wid255405.png`.
  **A COMPLETE WRITTEN BRIEF FOR JOHN COVERING ALL THREE MONTHS IS AT
  `docs/BRIEF_john_ryan_2015_07-09.md`** — headline results, exact UTC timestamps for all 18
  September orca calls (for navigating his Soundscape Visual Browser), the ROV finding, the ~17%
  recall number, the August candidate, the four July unknowns, coverage caveats, the two recorder
  artifacts needing his confirmation, and six questions for the meeting.
  **PREREQUISITE — now satisfied:** the Gradio label DELETE was unscoped and would have DESTROYED
  Duane's 6 existing labels when John relabeled. Fixed Aug 28 2026 (scoped by provenance); see
  finding #32.
- **Also with John at that meeting:** how to handle multi-annotator blind + reconciling reviews.
  Recommendation written up in `docs/multi_annotator_design.md` — one DB, additive rows, third
  `consensus:` row for reconciliation, training selects a view under a recorded policy, and
  inter-annotator agreement (Cohen's kappa) as the payoff.
- **SEPTEMBER 2015 PRODUCED NO HUMPBACK LABELS** — the expectation did not materialise in the
  reviewed set (18 orca, 17 dolphin, 12 ROV_noise, 0 humpback). Note this only means no humpbacks
  appeared among windows the ORCA classifier scored highly; it says nothing about humpback presence
  in the month. The concern below still stands for future months.
- **[ORIGINAL NOTE, still applicable] Duane expects humpback-heavy months, with a consequence worth
  acting on BEFORE the review.** The orca classifier fires on humpback vocalizations, so a humpback-heavy
  month yields more above-threshold windows that are not orca. **That is the known class confusion,
  not model failure** — and September could be the month that makes the case concrete. But it also
  means the pool of Duane's **working-grade** humpback labels grows fast, and the eventual
  reannotation with John gets correspondingly larger. **SUGGESTED ASK AT THE MONDAY MEETING:**
  does John want to set the song/vocalization scheme *before* another few hundred humpback labels
  accumulate, rather than after?
- Humpback song/vocalization split (tabled, needs John).
- Cross-hydrophone comparison in Monterey Bay (later, finding #26).
- Danelle: sent 3 annotation charts (total / training-only / F1-with-training-support, incl. the
  54 orca hard-negatives). Awaiting her response.
- Metric wording: use "nearest-neighbour retrieval" (or "inner product"), NOT "cosine" (#28).

**Context docs in the repo (all CLAUDE_*-prefixed, kept deliberately):**
CLAUDE_perch_hoplite.md (this, main findings), CLAUDE_embed.md (resample+embed pipeline, all
TODOs closed), CLAUDE_inference.md (inference/review workflow), CLAUDE_inference_april_2026.md
(session record), CLAUDE_release_plan.md (FAIR release blueprint), CLAUDE_repo.md (public-repo
build handoff), thalassa_storage_survey.md, poster_v42_review.md, docs/agile_modeling_history.md.


## Known Issues / Pending Work

> **Spark environment drift (Aug 27 2026).** The two GB10 sparks run slightly different Python
> package sets. Both have **perch-hoplite 1.0.2** and **sox v14.4.2** (resampling is byte-
> consistent across both). But their venvs drifted: **spark-0626 is newer** (torch 2.13.0+cu130,
> usearch 2.26.0, numpy 2.5.2, gradio 6.25.0) vs **spark-ae0e = the pinned set** in
> `requirements-spark.txt` (torch 2.12.1+cu130, usearch 2.25.3, numpy 2.4.4, gradio 6.15.1),
> because 0626's venv was installed unpinned at a different time. Both work; the drift is minor.
> `requirements-spark.txt` documents the ae0e snapshot. spark-0626's perch-hoplite venv is
> **pipeline-only** (no rfdetr/sahi — safe to rebuild). **To standardize in the future:** rebuild
> the target spark's venv from the pinned file — `rm -rf ~/perch-hoplite/venv` then
> `bash scripts/clean_install.sh` (it reads requirements-spark.txt and re-verifies 1.0.2 + TF-free).
> Or, if you'd rather track the newer set, update requirements-spark.txt to 0626's versions and
> rebuild ae0e. Open decision; not urgent since both work. `clean_install.sh` self-logs (tee) and
> guards against clobbering an existing venv, so a rebuild is safe and diagnosable.

> **Working method — capturing command output so you can find warnings later (Aug 27 2026).**
> Long installs/runs scroll warnings off-screen. Always capture output. Cheat-sheet (copy, don't
> memorize):
> - **See live AND save to file:** `bash some_script.sh 2>&1 | tee run.log`
>   (`2>&1` = merge errors into normal output; `tee` = write to file while still printing)
> - **Background long job + log (nohup):** `nohup bash some_script.sh > run.log 2>&1 &`
>   (order matters: `> file` before `2>&1`; the trailing `&` backgrounds it; `nohup` = keep
>   running if you log out). Check progress: `tail -f run.log`. See it's still running: `jobs` or
>   `ps aux | grep some_script`.
> - **Then find warnings/errors:** `grep -iE "warn|error|fail" run.log`
> - `clean_install.sh` now SELF-LOGS via `tee` to a timestamped `clean_install_*.log` in
>   PERCH_HOME — no need to remember the above for that script; just run it and check the log it
>   names. Review with `grep -i warning <that log>`.

> **Working method — running scripts on spark (Aug 27 2026).** For anything beyond a trivial
> one-liner, do NOT paste multi-line `python3 -c "..."` with nested quotes — it mangles (drops
> into `>` mode, breaks on quotes). Write to a file with a QUOTED heredoc and run it:
> ```
> cat > /tmp/x.py << 'PYEOF'
> ...python; use %-formatting to avoid nested quotes...
> PYEOF
> python3 /tmp/x.py
> ```
> Companion to push discipline (one working copy / pull-before-edit / grep-after-push).

> **Poster status (Aug 25 2026): current version is v44.** John Ryan reviewed it, has NO
> concerns; the extended abstract is uploaded and ACCEPTED by the conference. The v35 review
> must-fixes were resolved by v42 (header logos/MBARI logo; stat card + panel-11 orca F1 both
> 0.95; "foundation model" defined; 10→14-days reconciliation in the panel-7 figure caption);
> v44 continues layout/figure work (adds AWS). Poster is "just for show" — one conference day,
> no lasting presence — so polish is low-stakes; do not over-invest. Reviews on record:
> `poster_v35_review.md`, `poster_v35_replacements.md`, `poster_v42_review.md`. CMYK full-size
> print (OCEANS2026_orca_poster_v44_FULLSIZE_72x48_CMYK.pdf style name) is for the plotter;
> Melissa makes the real final poster once text/graphics/layout are locked. OCEANS 2026:
> Sept 21–24.

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
12. **Ship_noise label review** — ~~review April 2018 ship_noise labels (24 clips) in Gradio to confirm no errors (low priority)~~ **DONE / SUPERSEDED Aug 21 2026:** the ship_noise campaign (finding #15) went well beyond this — reviewed top-25 by score, re-confirmed the existing labels AND added 21 new ones (April 2018: 24→45), plus caught one "lots of bands" outlier correctly relabeled `other`. This task is fully absorbed into finding #15.
13. **Gray whale annotation review** — some `humpback_song` labels may be gray whale calls. Pull all humpback-labeled clips in Gradio and have J. Ryan re-annotate as `humpback_song`, `gray_whale_moan` (new class), or `other`. Gray whales are seasonally present in Monterey Bay and can overlap spectrally with humpback at low frequencies. Then retrain with gray whale as a new species class. **Now supported by evidence:** with real held-out support (n=40/47 in v1/v4), humpback_song is the weakest credible class (F1 opt ~0.55) — consistent with label contamination blurring the class. Highest-priority lever for lifting overall model quality.
    - **Class name:** `gray_whale_moan` — gray whales don't "song"; moans (S3, low-freq, migration context) are the sound most likely to overlap humpback and be mislabeled. If John flags knocks (S1) / croaks (S4) instead, may need a broader `gray_whale` or a second class. Renaming later is a trivial SQL `UPDATE annotations SET label=…` (no re-embedding), so `gray_whale_moan` is safe to commit to now. **John's ear makes the reclassification — not the model side (attribution rule).**
    - **Where to hunt:** April, not October. Gray whales migrate past central CA in winter/spring (northbound Feb–May); October is peak humpback / near-zero gray whale. Humpback label counts: **April 2018 = 41, April 2026 = 39, Oct 2020 = 259**. Start with the 80 April clips (higher contamination yield); the 259 Oct labels are likely clean.
    - **Workflow (no-GPU prep, then one review):** `tools/export_labels.py` writes per-(month,species) JSON to `json_labels/labels_{month}_{species}.json` → `tools/labels_json_to_review_csv.py --month 2018_04 --species humpback_song` converts it to a review CSV (7-col `idx,project,filename,window_start,window_end,label,logits`; review locates clips by filename+window_start+`--audio-dir`, `idx` is cosmetic; `--window-len` default 5.0) → `phase2_classify.py review --target-label humpback_song --detections-csv … --classes humpback_song,gray_whale_moan,other,unlabeled`. One `--audio-dir` per session, so April 2018 and April 2026 run separately. (Apr 2018 CSV built July 20: `results/review_apr2018_humpback_graywhale.csv`, 41 clips.)
    - **Status (Aug 20 2026, UPDATED — review complete):** April 2018 gray-whale review is **done** (J. Ryan). Result: **0 clips reclassified to `gray_whale_moan`** — J. Ryan confirmed all reviewed clips are humpback. He also reported **faint dolphin_call audible in the background** of some clips — not separately labeled, since `dolphin_call` wasn't in this session's `--classes` list; recorded as a QA observation only, not a DB change.
        - **Song vs. non-song distinction (D. Edgington's observation, Aug 20 2026 — attribution matters, keep separate from J. Ryan's call):** J. Ryan confirmed species ID (humpback); **Duane separately observes that many of these clips are NOT song** — they read as a different, distinct vocalization type. This is Duane's own judgment, not something J. Ryan stated. **Humpback whales have a genuinely diverse repertoire beyond song** (per NOAA/NPS: sighs/cries/squeaks/moans in long evolving patterns = song, sung by males, breeding/feeding grounds; separately: train-whistle feeding calls, whistles, screeches, grunts/groans/moans, whoops, megapclicks/buzzes, calf chatter = social/functional non-song calls). **Reporting rule going forward (poster, docs, everywhere):** when presenting a humpback clip, be explicit about which it is — "humpback song" only for clips that are actually (or plausibly) part of a song sequence; "humpback vocalization" or "humpback call (non-song)" otherwise. Do not caption a non-song clip as "song" just because the DB label is `humpback_song` — the label name is broader than the acoustic category it's often assumed to mean. A spectrogram of a literal song excerpt is fine to call "song"; a generic humpback call should not be.
        - **Count reconciliation:** the DB shows 19 `humpback_song` labels post-review, not the "41" figure quoted earlier. This is NOT a data-loss event — `export_labels.py`'s JSON export contained duplicate rows (10 of 19 distinct (file, offset) windows had been exported 2-3× each, likely from an earlier multi-pass labeling artifact); the true distinct count was always 19, confirmed by `sort | uniq`. **`export_labels.py` should be checked for a de-dup bug** (or the underlying `annotations` table audited for genuine duplicate rows) — logged as a to-do, not yet root-caused.
        - April 2026 (39 clips) is still a one-command repeat of this workflow, not yet run. Separately, the Oct 2020 orca-*survivor* review (10 clips ≥1.16, completed July 20) was orca-FP characterization, unrelated to this gray-whale pass.
        - **#13 net effect so far:** no evidence of gray-whale contamination in the April 2018 batch reviewed. Humpback's weak F1 (~0.55) is NOT explained by gray-whale mislabeling in this sample — the root cause remains open. The song/non-song heterogeneity noted above is a plausible independent contributor (a class covering acoustically distinct song vs. non-song calls could itself be harder to learn cleanly) — worth considering as a future class-split candidate alongside the gray-whale work.
        - **#13 FULLY CLOSED / gray-whale hypothesis DEAD (D. Edgington + J. Ryan, Aug 21-22 2026).** Both April batches reviewed (April 2018: 19 clips; April 2026: 39 clips) — **zero gray-whale sounds in either.** J. Ryan confirms the sounds are humpback vocalizations, not gray whale. The gray-whale-contamination explanation for humpback's weak F1 is ruled out, not merely unsupported. **Root cause reframed, no longer open:** `humpback_song` is a misnomer — it lumps true humpback *song* together with other humpback *vocalizations* (non-song calls). That single heterogeneous class is the likely reason it's hard to learn cleanly. **Concrete next step (D. Edgington can do this):** split the class into `humpback_song` + `humpback_vocalization` and reannotate the existing humpback labels into the two. `gray_whale_moan` stays defined for any genuinely future case but is not populated and is no longer a working hypothesis.
14. **Extended April 2018 orca activity — CONFIRMED (D. Edgington, July 19 2026).** v4 cross-month validation (threshold sweep, `tools/score_orca_regions.py`) surfaced strong, threshold-robust orca detections across **Apr 13–25 2018**, not just the confirmed Apr 13 event. At logit ≥ 1.16: Apr 18 = 173 (rivals Apr 13's 251; 105 survive +2.0), Apr 23–25 cluster = 118, full Apr 13–25 window ≈ 569. **Expert review complete for Apr 18 + Apr 25: 75 detections labeled (25 Apr 18 + 50 Apr 25), 100% orca, 0 false positives at ≥1.16** — orca_call annotations 219→294. April 2018 is now established as a **sustained ~2-week orca presence**, not a one-day event. Still to review: Apr 18 mid-logit band CSV exists (69 windows, port 7863, unreviewed), Apr 23/24. Counter-example note (revised July 21 2026): the sweep shape once made "May 14 reads FP-like (19→1 across the sweep)" look like a non-event, but expert review confirmed **all 4 May 14 detections at ≥1.16 as orca** — the collapse-shape heuristic gave a false FP signal here; ear review is authoritative. The genuine specificity counter-examples are Oct 2020 / Apr 2026, whose ≥1.16 residuals reviewed as humpback. Confirmed-orca example figures registered: `gradio_apr18_2018_orca_195s_wid202720`, `_405s_wid202762`, and five Apr 25 clips (`gradio_apr25_2018_orca_*`). **External corroboration (non-acoustic):** KSBW Action News 8 (Monterey) reported "record orca sightings" in April–May 2018 (one tour group ~50 orcas in a day), orcas drawn to hunt gray whales, and biologists identified **two pods that spring — one Alaskan, one Californian ("Emma's pod", the CA140s — matriarch CA140 "Emma", a Bigg's/transient matriline known for hunting gray-whale calves)**. Figure: `ksbw_news8_orca_invasion_monterey_spring2018` (© KSBW, reference only). Two independent methods (hydrophone + visual sightings) agreeing on a sustained spring-2018 presence.

15. **Ship_noise labeling campaign (D. Edgington, Aug 21 2026).** Addressed panel 10's admitted weakness — ship_noise had only n=3 held-out support in every model, its 1.0 F1 an acknowledged artifact. No validated ship_noise operating threshold exists (unlike orca's +1.16), so this pass deliberately skipped thresholding: took the top-25 highest-scoring `ship_noise` detections per month (raw model rank, no cutoff) and reviewed by ear, separately per DB.
    - **April 2018:** 24→45 confirmed (+21 new; 3 of the top-25 were pre-existing correct labels, re-confirmed — expected, since a real positive should score highly). **1 clip labeled `other`**, not ship_noise: "very different, lots of bands" — possibly a smaller motor/different vessel type than typical broadband ship noise; worth a future `ship_noise` subtype split (analogous to the humpback song/non-song distinction) if this recurs.
    - **May 2018:** 4→29 confirmed (+25, **100% clean, zero ambiguity**). The strongest, cleanest ship_noise batch to date.
    - **October 2020:** 2→2 (**0 new — all 25 candidates left deliberately unlabeled as genuinely ambiguous**, not forced into a category). D. Edgington's note: "all contaminated — there was something else in the background along with the ship noise... did not look like any of the pure ship noise" of April/May. **Leading hypothesis:** October is peak humpback season in this dataset (263 confirmed humpback labels, by far the richest month) — real ship engine noise plausibly co-occurring with audible humpback vocalization in the same 5s window, a genuinely mixed acoustic scene rather than a labeling error. This is a plausible structural contributor to the v6/v7/v8-era "ship_noise inflation" mystery (adding a 4th, humpback-heavy season degraded ship_noise calibration) — not proven, but a concrete lead.
    - April 2026 not reviewed — only 4 raw ship_noise detections total, not worth a session.
    - **Net: ship_noise confirmed labels 35→81 project-wide** (24→45 Apr 2018, 4→29 May 2018, 2→2 Oct 2020 unchanged, 5→5 Apr 2026 unreviewed) — a >2x increase in the weakest class's support. **CORRECTED Aug 21 2026: originally miscounted as "35→76" here (arithmetic error, dropped Apr 2026's unchanged 5 from the sum) — now fixed.** Note: the 3-season Option A training recipe (April 2018 + Oct 2020 + April 2026, no May) uses only **52** of these 81 (45+2+5) — see finding #16's `orca_v10.pt` result and eval n=10 for ship_noise. **Not yet reflected in v4** (the still-current production classifier) — folding these in is what finding #16's retrain did for the 3-season subset; May's labels remain unused pending the deferred "Option B" 4-season attempt.

16. **"Option A" retrain — 3-season re-merge in progress (D. Edgington, Aug 21 2026).** With #13 and finding #15 both closed, this is the retrain of v4's exact 3-season recipe (April 2018 + Oct 2020 + April 2026) on today's fully updated labels — a clean before/after F1 comparison against v4, no season-mix change (that's a separate, deliberately deferred "Option B" — see below).
    - **Step 1 (done):** `tools/merge_dbs.py` — April 2018 (518,400 windows, 685 annotations) + Oct 2020 (535,278 windows, 317 annotations) → **`MARS_combined_3month_32kHz_norm_v3`** (1,053,678 windows, 1,002 annotations). Positive: dolphin_call 203, humpback_song 282, orca_call 319, other 97, ship_noise 47; negative orca_call 54. All numbers reconcile exactly against the source DBs' Aug 21 2026 confirmed totals.
    - **Naming note:** the merged-DB version suffix (`_v3`, `_v3_apr2026`) is independent of the classifier version (`orca_v4.pt`) — this project's convention has never kept them 1:1 (v4 the classifier was trained on `..._v2` the DB). Deliberately avoided naming the next merge stage `..._v4` to prevent confusion with `orca_v4.pt`.
    - **Step 2 (done):** merged `MARS_combined_3month_32kHz_norm_v3` + April 2026 (`MARS_20260401_20260430_32kHz_norm`) → **`MARS_combined_3month_32kHz_norm_v3_apr2026`**. Result: 1,559,308 windows/vectors (**identical to v4's original training DB window count** — same underlying audio archive, only labels changed since v4 was trained), 1,076 annotations (up from v4's 803, **+273 new labels**). Positive: dolphin_call 206, humpback_song 321, orca_call 319, other 124, ship_noise 52; negative orca_call 54. Reconciles exactly against the component DBs. **Took ~18 minutes on spark-ae0e with no other load** (mostly the 1.5M-vector USearch index merge).
    - **Step 3 (done, Aug 21 2026) — with an incident:** trained on `MARS_combined_3month_32kHz_norm_v3_apr2026` (1,076 labels, `--num-steps 256 --train-ratio 0.8`, seed 42) and initially saved to **`orca_v5.pt` — this OVERWROTE the weights of the July 15 2026 context-embedding experiment**, which had already been assigned the name `orca_v5.pt` (❌ DO NOT USE, ROC-AUC 0.9303/cmap 0.5945, documented above). The July 15 provenance JSON survived untouched (`train_20260715_133632_orca_v5.json`), so the metrics/config are not lost, but **the actual trained weights file for that historical experiment is gone** — a real but low-consequence loss (the model was already marked unusable and nothing depended on the live weights). Renamed today's run's `.pt` to `orca_v5b.pt` immediately after discovery. **Known loose end:** the `.metrics.json` sidecar for today's run still carries the old `orca_v5.metrics.json` name (didn't get renamed alongside the `.pt`) — cosmetic mismatch, harmless, clean up when convenient.
    - **NEW RULE (Aug 21 2026): the next classifier trained is `orca_v10.pt`, not v5/v6/v7/v8/v9 or any letter-suffixed variant of those.** v0–v9 are ALL considered taken/retired (v5 by the July 15 experiment, now further complicated by today's overwrite; v6/v7/v8 by the 4-season ship_noise-inflation experiments; v9 used in README worked examples). Starting fresh at v10 removes any ambiguity. Update README's example commands away from `orca_v9.pt` to `orca_v10.pt`-style guidance if/when next touched.
    - **Result (`orca_v5b.pt`, aggregate):** ROC-AUC 0.9318, cmap 0.6587, macro_f1 0.5283, top1_acc 0.8824 — both **lower** than v4 (0.9590/0.8297) at first glance.
    - **Result, per-class (the real story — read this before the aggregate numbers above):** eval set is now 459 held-out examples, larger and more diverse than v4's eval ever was, so aggregate ROC-AUC/cmap are **not directly comparable** to v4's — a harder/bigger eval set naturally pulls aggregates down even when the model improves per-class. Per-class F1 (opt threshold):
      - **orca_call: 0.946 (n=61)** — matches v4's ~0.95, orca held steady.
      - **humpback_song: 0.616 (n=67)** — **improved** vs. the long-standing ~0.55 baseline. Real gain, credible support.
      - **ship_noise: 0.762 (n=10)** — **first-ever credible ship_noise F1**, on real held-out support (n=10, vs. the old n=3 artifact that produced a fake 1.0). This is the direct payoff of finding #15's labeling campaign.
      - **dolphin_call: 0.674 (n=44)** — slightly down vs. the ~0.71–0.77 historical range. Worth watching, not alarming alone.
      - **other: 0.591 (n=22)** — no clean prior baseline; appears to be the current weak point, plausibly because `other` is an inherently heterogeneous catch-all that grew a lot (52→124) across the merges.
    - **Bottom line: orca held, humpback improved, ship_noise became measurable and looks decent, dolphin dipped slightly, "other" is the residual weak spot.** A legitimately positive first-pass result — NOT yet promoted to production, NOT yet reflected on the poster (decision pending). `--num-steps 512` undertraining test optional/pending given orca's already near-ceiling.
    - **"Option B" (deferred, not yet started):** re-adding May 2018 to the season mix, like v6/v7/v8 attempted. Those attempts inflated ship_noise false positives; finding #15's Oct 2020 contamination hypothesis (real ship noise co-occurring with peak-season humpback vocalization) is a candidate mechanism worth testing for *before* blindly repeating the 4-month mix. Only attempt after Option A's results are in hand.
    - **CA140 matriline ("Emma's pod") — family structure, recurrence, gray-whale link (July 21 2026):** CA140 "Emma" is the matriline matriarch; **CA140B "Louise" is her daughter** (with her own offspring — Stinger, Bee, Buzz, Bumble). Not a naming conflict: Emma leads the matriline, Louise leads a sub-group within it. The **same matriline recurs across event windows** — "Emma's pod" in the spring-2018 KSBW report, and CA140B "Louise" in the Oct 2020 whale-watch reports — i.e. the same Bigg's family appears in both the 2018 and 2020 data. The CA140s are documented **gray-whale-calf hunters**, so their presence is a biological reason to expect gray-whale calls in the recordings — an independent line pointing at the #13 humpback/gray-whale contamination. (Source: California Killer Whale Project.)

17. **Ship_noise depth review — planned for Aug 22 2026 (D. Edgington).** Question: how many more confirmed ship_noise labels are needed to bring its eval support (n=10 in the 3-season DB) up toward dolphin_call's (n=44)? At the observed ~19–21% per-class eval fraction, matching n≈44 needs **~206–229 total ship_noise labels** — currently 52 in the 3-season DB, so **roughly +150–180 more**, on the order of 3–4× the entire Aug 21 ship_noise campaign (finding #15).
    - **Supply is not the constraint.** April 2018 has 1,523 raw ship_noise-scored windows, May 2018 has 1,113 — only the top 25 from each have been reviewed so far. Plenty of untapped candidates.
    - **The open question is whether quality holds going deeper into the ranked list.** The top-25 batches were exceptionally clean (May: 25/25; April: 24/25 + one distinct "other"). Confidence typically degrades below the highest-scoring detections — unknown whether rank 26–100 stays this clean or starts resembling October's contaminated batch (0/25 usable).
    - **Plan:** pull the next tier (rank ~26–75, no threshold gate, same method as Aug 21) from April and May 2018 and review by ear. If it stays mostly clean, parity with dolphin is realistically achievable in a few more sessions; if quality drops sharply, that establishes the real ceiling rather than chasing an unachievable number.
    - **Explicitly NOT expected to hit the same wall October did** — October's problem was peak-humpback-season contamination specific to that month; April/May's likely limiting factor (if any) is simply declining detection confidence, not a parallel species-contamination issue.
    - **May 2018 is likewise multi-day (D. Edgington, July 21 2026).** Beyond the confirmed May 12 event (181/181), review of the non-May-12 orca detections at ≥1.16 confirmed orca on **May 13 (1), May 14 (4), May 16 (3)** — **8 new orca labels**, orca_call 181→189 (2 clips left unlabeled as too faint, incl. a May 7 singleton). Two May 16 clips from the *same* recording 25 s apart sound audibly different ("a bit different, but clearly orca") — within-encounter call variation, relevant to the by-day/per-encounter t-SNE thread. **Spring 2018 now shows sustained Bigg's presence across BOTH months** (Apr 13/18/25 + May 12/13/14/16), matching the KSBW "record sightings April–May" report. Tooling: `tools/export_labels.py` now includes May 2018 (`2018_05`) so these labels are captured in the JSON snapshot. Caveat: May 13 is a single-clip day (rests on one detection); May 14's 4-clip morning cluster is the more robust new day. **v5 retrain queue:** these +8 May orca labels join the extended-April orca (+75) and Oct 2020 humpback relabels (+4), pending John's gray-whale/humpback work before retraining.

---


18. **May 2018 is now a permanent held-out test month — policy decision (D. Edgington, Aug 21 2026).** Triggered by a poster-accuracy question: the OCEANS poster states "neither v4 nor `orca_v10` has trained on May 2018 recordings" — true for those two, but v6/v7/v8 DID train on a May-inclusive 4-season DB, so an unscoped version of that claim would be checkable-false against the public repo's own history. Rather than keep patching poster wording around this, **the simpler, permanent fix: May 2018 is deliberately never trained on, by any future model, going forward.**
    - **Rationale:** a true, clean held-out month — never touched during training — lets any future classifier be validated against May with a clear conscience, no caveats needed. Also resolves the poster-wording problem at the source rather than per-sentence.
    - **v6, v7, v8 are explicitly discarded** (already ❌ DO NOT USE for the ship_noise-inflation reason; now also formally out of the "models trained here" lineage going forward). **v5 (July 15 context-embedding experiment) was already discarded, then its weights were accidentally overwritten Aug 21** (see finding #16) — also out of the lineage. Current usable models: v0–v4 (April 2018/Oct2020/Apr2026 only, matching the new policy already), `orca_v5b.pt` and `orca_v10.pt` (today's Option A retrain, same recipe).
    - **Practical effect:** #17's ship_noise depth review sources ONLY from April 2018 (and other non-May months as needed) — not May, even though May has abundant untapped ship_noise supply (1,113 raw candidates). The ship_noise data gap gets closed by pulling more from already-in-scope months, not by reopening May.
    - **Also affects finding #14's May orca work:** May's confirmed orca labels (181 May 12 + 8 more) remain valid, useful, *documented* science — not deleted or discredited — but will not be folded into any future training DB. May-2018 orca confirmations are held-out **validation** evidence for any future model, not training fuel.
    - **New public repo, pre-poster:** Duane plans to build a curated public repo before the conference, separate from this working repo — a chance to present a clean v0→v4/v10 lineage without the v5–v8 detour needing extensive caveats.

19. **Exact total annotation count — verified via SQL (D. Edgington, Aug 22 2026).** Triggered by a poster-accuracy question about the stat card's "~1,450 labels" claim vs. the trajectory chart's bars summing to exactly 873. Direct query (`SELECT COUNT(*) FROM annotations`) against all four DBs today:

    | DB | Annotations |
    |---|---|
    | April 2018 | 685 |
    | May 2018 | 260 |
    | October 2020 | 317 |
    | April 2026 | 74 |
    | **TOTAL** | **1,336** |

    **The verified total is 1,336, NOT ~1,450** — a confirmed 114-label gap between the poster's current claim and the actual database contents. Origin of "~1,450" is unknown (possibly an earlier draft's approximation, or written before some of Aug 21's final counts settled) — not worth chasing further; **1,336 is now the number of record, exact and re-derivable by anyone from this same four-line query.**
    - **873 vs. 1,336 — both are correct, they answer different questions:** 873 = labels that built the presented model (v4)'s trajectory. 1,336 = total labels across all months/classes today, including everything confirmed after v4 was trained (extended-April/May orca, the ship_noise campaign, gray-whale reviews, etc.). Poster instruction: state both explicitly with this distinction, don't reconcile via approximation.

20. **May 2018 hold-out test — orca_v10 beats v4 on unseen data (D. Edgington, Aug 23 2026).** The experiment the May-holdout policy (#18) was designed for: since neither v4 nor `orca_v10` trained on May 2018, it is a fair referee for "is v10 actually better, or did the retrain just shift the eval set?" Method: ran both models' orca inference over all of May 2018 at logit floor 0.0 (`phase2_classify.py infer`, outputs `MARS_20180501_20180531_v{4,10}_orcaval.csv`), then scored each model on the **196 confirmed-orca windows** (May 12/13/14/16, all ear-reviewed) via `tools/compare_may_holdout.py`. Result — v10 wins decisively on recall/confidence for known orca:
    - **Mean score on confirmed orca: 2.613 (v10) vs 1.646 (v4)** — nearly a full logit higher.
    - **Head-to-head: v10 scored higher on 192 of 195 shared windows** (v4 higher on 3); mean(v10−v4) = **+0.958**.
    - **Recall at the +1.16 operating threshold: 79.6% (v10) vs 60.7% (v4)** (+19 pts); at +2.0, 59.2% vs 39.3%.
    - **Misses (confirmed orca scored <0.0):** v4 = 0, v10 = 1. The one caveat — v10's more selective net dropped exactly 1 of 196 known orca that v4's broader net caught. Against 192 windows where v10 scored higher, this is negligible.
    - **Total May detections at floor 0.0: v10 = 271, v4 = 241** — v10 is slightly more, not runaway, so no sign its higher confidence comes from indiscriminate over-firing.
    - **Interpretation:** a genuine, defensible improvement on held-out data — the +100 orca labels and retrain produced real gains in orca discrimination, not a metrics artifact. Strongest evidence yet that **v10 should replace v4 as production** (resolves roadmap item 4, pending a precision check). Honest bound: this measures RECALL on known orca cleanly; it does NOT fully measure May precision (false-positive rate), since May is not exhaustively negative-labeled — a full "v10 better overall" claim would want a false-positive review too, but the near-equal total detection counts are reassuring. Scorer committed as `tools/compare_may_holdout.py`.

21. **v10 precision check on May hold-out — perfect, and 4 NEW orca days discovered (D. Edgington, Aug 23 2026).** Closes the open caveat from #20 (recall was measured cleanly; precision was not). Method: took v10's 271 May detections, removed the 196 confirmed-orca windows, leaving 76 non-confirmed; of those, **14 scored >= +1.16** (the operating threshold) — the precision-critical set. Reviewed all 14 by ear (6 min, incl. capturing example framegrabs; `review_may2018_v10_precision_ge116.csv`, port 7876). **Result: all 14 confirmed real orca. ZERO false positives.** orca 196->210, no other class changed (nothing was a misfire reclassified to humpback/ship/etc).
    - **v10 now wins on BOTH axes:** recall (#20 — scores known orca ~1 logit higher, +19pts recall at threshold) AND precision (this check — 14/14 real orca at >=1.16, its high confidence is earned not inflated). The production case for v10 replacing v4 is now essentially airtight (roadmap item 4).
    - **Bonus discovery — 4 NEW May orca days that v4 missed entirely:** the 14 included clips on May 2, 3, 7, and 29 — days NOT previously confirmed. All confirmed orca. **May 2018 confirmed orca days: 4 -> 8** (was 12/13/14/16; now +2, +3, +7, +29). v10 didn't just score better, it *surfaced orca activity invisible to v4*. Full May by-day (orca_call positive): May 2=1, 3=1, 7=1, 12=181, 13=8, 14=10, 16=7, 29=1.
    - **NOTE on May as held-out set:** these 14 new confirmations were added to the May DB as labels. May REMAINS held-out for training (finding #18) — labeling ground truth in a test month is fine and expected (it's how you measure a model on it); the policy is that May is never used to *train*, which is unchanged. The confirmations improve May as a validation reference; they are not training data.
    - **Method fully validated:** v10 (retrained via the agile-labeling loop) tested honestly on never-trained May data both scores known orca far better AND finds real orca v4 missed, with zero threshold false positives — a clean demonstration of the poster's core thesis on held-out data.

22. **Oct 2020 v10 scan — no orca found; "seen but not heard" confirmed under the better model (D. Edgington, Aug 23 2026).** Motivation: John's core goal is orca detection across seasons/years; v4 had found NO confirmed orca in Oct 2020 (only humpback false positives, heavy in peak-humpback October). Question: does the more-sensitive v10 (which found real orca v4 missed on May, finding #21) surface orca that v4 missed here? Ran v10 inference over all Oct 2020 (535,278 windows, ~3s): 113 detections at floor 0.0; 16 at >= +1.16 (3 at >= +2.31, clustered Oct 4-5 with two consecutive windows in one recording — the profile of a real call sequence, which made them look promising). Reviewed all 16 by ear.
    - **Result: NO identifiable orca. ~14 humpback, 2 other, 0 orca.** The promising Oct 4-5 high-score cluster turned out to be humpback, not an orca sequence. Possible multiple overlapping sound sources noted. (An orca_call=1 remains in the Oct 2020 DB — a pre-existing stray label, NOT a confirmation from this session; verify/clear later like the April strays.)
    - **This is the STRONGER outcome for the science, not a disappointment.** v4 hearing no orca could be a sensitivity limit; v10 — demonstrably able to find faint orca v4 missed (finding #21) — *still* hearing no orca in Oct 2020 means the absence is real, not a detection gap. Absence measured with a good instrument is real absence. Directly strengthens the poster's panel-9 "seen but not heard / Bigg's hunt silently" claim: the more-sensitive detector confirms acoustic silence despite boat sightings.
    - **NOTE on provenance:** Oct 2020 is a TRAINED month (part of the 3-season recipe), not held-out — but its training labels were humpback/other/ship, essentially no orca, so "does v10 now surface orca here?" was still a meaningful question. Only May 2018 supports a clean generalization claim; this Oct result is about biological presence, not generalization.
    - **Still pending:** catalog the Oct 2020 review framegrabs (humpback/other examples — documents what the false-positive candidates actually are). April 2026 (the other trained month) has 49 candidates >= +1.16, NOT yet reviewed — lower priority (trained month, less informative than a clean test).

23. **April 2026 v10 scan — AMBIGUOUS candidates, NOT confirmed; needs John's blind review (D. Edgington, Aug 24 2026, session 20260824_07xxxx_duane).** Completeness scan of the fourth month. Ran v10 over all April 2026 (505,630 windows): 278 detections at floor 0.0, 49 at >= +1.16, 13 at >= +2.31. The >= +2.31 set had a strong structural signature — **April 21 2026 dominated (6 of 13 in one recording `MARS_20260421_130000`, incl. a near-consecutive 515-530s bout; top score 4.17, higher than anything in the validated May set; activity across multiple April-21 recordings over hours), plus a secondary consecutive pair April 24.** On paper the most orca-looking candidate set generated. Reviewed the 13 (>= +2.31) by ear.
    - **Result: GENUINELY AMBIGUOUS — recorded as tentative, NOT confirmed.** Several clips sounded clearly orca in the 5 s window, BUT the 30 s context revealed surrounding humpback vocalizations (moans, upsweeps, mixed repertoire). Cannot distinguish from audio alone between (a) orca AND humpback both present, and (b) humpbacks producing orca-like sounds within their own vocal mix. This is exactly John Ryan's caution: "humpbacks can sound like any animal in the ocean." D. Edgington marked ~2-3 as orca (DB orca_call in April 2026 = 2) but explicitly regards these as UNCONFIRMED.
    - **Required next step — BLIND second-expert review:** John Ryan to review this same 13-clip batch independently, WITHOUT seeing D. Edgington's annotations (blind protocol, to avoid contaminating his judgment). Agreement → strong; divergence → the clips are genuinely ambiguous and must NOT be called confirmed orca. This is the correct method for ambiguous ground truth with mixed sound sources.
    - **Weaker than May/Oct results, on two counts:** (1) April 2026 is a TRAINED month (not a clean test — cf. #18); (2) the audio itself is ambiguous (humpback mimicry / mixed sources). So these are NOT new confirmed orca days in the sense May 2/3/7/29 (#21) were. Do NOT add April 2026 to any confirmed-orca-day count or the poster on this evidence.
    - **Scientifically interesting regardless:** the humpback-orca acoustic confusion here is a concrete instance of the panel-3 spectral-overlap problem and motivates the panel-9 "context matters" point (5 s alone is insufficient; 30 s context changed the call). Batch CSV: `review_apr2026_v10_orca_ge231.csv`. Framegrabs captured, to be cataloged with the Oct 2020 set.

24. **September 2024 flagged as a strong test-month candidate — documented multi-matriline Bigg's encounter (recorded Aug 24 2026; resampling underway).** External sighting report (California Killer Whale Project) documents a **confirmed, precisely-dated, all-day Bigg's presence in Monterey Bay on 27 Sept 2024** — an unusually rich, unambiguous event that makes Sept 2024 an excellent candidate to test v4 vs v10 detection.
    - **Three Bigg's matrilines present almost the entire day (9/27/24):** CA140s ("Emma's pod," 4 members incl. calf CA140E), CA39As ("Hopper's pod"), CA51As ("Aurora's pod," incl. newest calf CA51A5). CA51As off Moss Landing all day into after sunset; CA140s (not seen in the Bay since June 2024) arrived later; CA39As observed actively hunting sea lions with acrobatic predation. CA140s and CA39As joined for a hunt (CA140 Emma + CA140C Ben leading), CA140s fed on the carcass. CA39As also had a notable interaction with humpbacks and sea lions (humpbacks charged them).
    - **Why this is a strong test case:** all-day presence = high chance of substantial acoustic activity; three matrilines = potentially diverse vocalizations; and the humpback interaction means humpback vocalizations likely co-occur — a real test of v10's orca/humpback discrimination (cf. the April 2026 ambiguity, finding #23) on an event with independent visual ground truth.
    - **CA140 "Emma's pod" recurs AGAIN — now 2018, 2020, AND 2024.** The same matriline appears in the spring-2018 KSBW report, the Oct-2020 whale-watch reports (CA140B "Louise"), and now Sept 2024. Strong throughline: the same Bigg's family is repeatedly present in Monterey Bay across the years spanned by the MARS archive — directly relevant to John's cross-year orca-detection goal.
    - **Status:** D. Edgington resampling Sept 2024 MARS audio to 32 kHz (SoX pipeline, ~1 day on spark) so it's ready for v4/v10 inference once poster/John feedback clears. When ready: embed → infer with both models → compare detections around 9/27/24 against this visual ground truth (does either model hear the confirmed all-day encounter?). A clean external-ground-truth test, unlike the label-only months.
    - **Source:** California Killer Whale Project encounter report, 27 Sept 2024.

25. **September 2024 — DATA OUTAGE past Sept 19; the 9/27 encounter is NOT in the acoustic record (D. Edgington, Aug 24 2026).** Resampled + embedded Sept 2024 (2698 files, 323,760 embeddings, clean). **Coverage check BEFORE inference revealed the month ends on Sept 19:** days 1-18 complete (144 files/day), day 19 partial (106 files, cuts ~17:37), and NO data Sept 20-30 — including 9/27. So the documented three-matriline Bigg's encounter of 9/27/24 (finding #24), the whole reason this was an exciting ground-truth test, **falls in the gap and cannot be acoustically tested with this data.**
    - **Consequence:** the clean external-visual-ground-truth test is OFF THE TABLE for now. A "no orca on 9/27" inference result would be meaningless (no data =/= no orca) — caught before running inference precisely because coverage was checked first. Lesson reinforced: ALWAYS check day-by-day coverage before interpreting (or even running) a month's inference. MARS has outages.
    - **Open question for tomorrow (D. Edgington + J. Ryan / archive):** is this a true MARS recording outage, or just a RESAMPLING gap (raw files exist in /mnt/PAM_Archive/2024/09/ past the 19th but weren't resampled)? If raw exists past the 19th -> resample the rest, re-embed, and the 9/27 ground-truth test becomes possible. If MARS was genuinely down -> the encounter simply isn't in the acoustic record and the test is unrecoverable from this deployment.
    - **Outage CONFIRMED REAL (Aug 24 2026, D. Edgington):** the RAW archive itself stops at MARS_20240919_173756.wav; nothing in /mnt/PAM_Archive/2024/10; recording only restarts ~MARS_20241119_195257.wav for a day or so. Cause: a power-connector failure — the hydrophone ran on internal battery until it died, then a longer gap until the physical fix. So 9/27 is genuinely not in the acoustic record and is NOT recoverable by resampling. The single-date external-ground-truth test is off permanently for this deployment.
    - **BUT Sept 1–19 is still worth testing (corrected view).** The value was never only 9/27. Established pattern in this project: when CA-pod Bigg's are in the Bay they are present for MULTIPLE days and show up acoustically across them (April 2018's 13–25 run; May's clusters; recurring CA140 matriline). Sept 2024 is a month when that population was demonstrably active in the region (the 9/27 sighting proves it), so "was there Bigg's acoustic presence in the covered Sept 1–19 window?" is a legitimate, interesting detection question — not a throwaway scan. DB already built (323,760 embeddings). RUN v4/v10 inference on Sept 1–19.
    - **Inference run (Aug 24 2026):** v4 = 2765 detections @floor 0.0, v10 = 2417 (v10 more selective, as everywhere). These totals are ~10x a real orca month (May 2018 had v4=241/v10=271), which by itself signals a NOISY month, not an orca bonanza — the floor-0.0 total is never the signal. v10 @>=2.31 = ~60 detections, but **smeared evenly across ALL days 1-19 (1-8/day, no clustering).** That even daily background is the OPPOSITE of a real orca encounter signature (cf. April 21 2026: 6 hits in one recording, consecutive bouts) and is the classic profile of a persistent background source firing weakly — most likely PEAK HUMPBACK SEASON (September), same failure mode as Oct 2020. **Provisional read: humpback-heavy month, no obvious orca-encounter signature; NOT confirmed either way.** Confirm with a short ear-review of the top ~10-15 >=2.31 clips (across the higher-count days 5/8/9): if humpback -> hypothesis confirmed, done; if real orca -> go deeper. Low priority / no urgency (the ground-truth 9/27 test died with the outage). Do NOT treat the big detection count as orca presence without the listen.
    - Coverage query (recordings table, `substr(filename,6,8)` = day) recorded in CLAUDE_embed.md Stage 2.5 as the standard pre-inference check.

26. **Strategic direction from John Ryan (meeting Aug 25 2026) — full-archive seasonal/interannual analysis is the new #1 priority.** A significant redirect. John's strong bias: **run the best current model(s) across the ENTIRE MARS archive** and characterize **seasonal and interannual variance** in orca detection, cross-verified against California Killer Whale Project sightings. Look for which months/periods correlate with other events (ocean warming, prey/food-source migration, etc.) — i.e., find ecological patterns in when orcas are acoustically present.
    - **Explicitly DE-prioritized for now (John):** do NOT pursue other ecotypes (Resident/Offshore) yet; do NOT pursue external datasets (e.g. Palmer 2025) yet. Focus stays on the transient (Bigg's) calls we currently detect — even though they're largely CA140-associated. This overrides the earlier roadmap's "highest-ceiling external ecotype dataset" item — reprioritized below full-archive analysis at John's direction.
    - **(2) Cross-hydrophone comparison (later):** when ready, test/compare against OTHER hydrophones deployed in Monterey Bay — do we detect orcas best at MARS, or better at others? A deployment-siting question. Not now, but on the list.
    - **(3) Publication strategy:** John sees publishing the full-archive seasonal/interannual results as HIGH priority. Target BOTH science venues and technical venues — possibly a PAIR of publications (peer-reviewed conferences/workshops count). The ecological-pattern paper is the science story; the method/pipeline is the technical story.
    - **(4) Poster:** John reviewing v42 this afternoon; **he has NO concerns.** The extended abstract is uploaded and ACCEPTED by the conference. The poster is "just for show" — one day at the conference, no lasting presence after. So poster polish is genuinely low-stakes from here; do not over-invest.
    - **(5) April 2026 blind review — TABLED for now.** Multi-annotator strategy not discussed with John. Duane + Claude to design, implement, and document a two-annotator approach separately (still worth doing, just not urgent).
    - **(6) Humpback song/vocalization split — TABLED for now.** Keep focus on orcas; continue distinguishing humpback-from-orca (and vice versa) as needed for orca precision, but don't invest in the song/non-song reannotation project yet.
    - **(7) FAIR public release — John is fine with it.** His view: a notebook that runs the SoX resampling on the public AWS raw audio may itself suffice as "releasing the data" (reproducibility-not-dataset, matches CLAUDE_release_plan.md). He's ALSO open to releasing resampled data in an appropriate format/venue if we want. Either path acceptable.
    - **(8) thalassa / IT storage discipline (IMPORTANT operational constraint):** John is concerned about the data load our activity places on MBARI IT infrastructure. Directive: **monitor and keep our usage clean.** Do NOT keep all 32kHz resampled files on thalassa — analyze a year (or whatever unit) then DELETE all but key files supporting the work. Back up (then remove) old log files and other cruft. This directly shapes the full-archive plan (#1): we can't resample-and-keep the whole archive at once; we process in chunks, extract/keep only what supports findings (models, labels, confirmed clips, key figures), and delete the bulk resampled WAV after analysis. The reproducibility bundle (script + manifest + checksums) is what makes deleting the bulk resampled audio safe — it can be regenerated on demand.
    - **Roadmap impact:** #1 (full-archive seasonal/interannual + sightings correlation) is now the top near-term thrust, ahead of the external-ecotype and temporal-architecture items. Storage discipline (#8) is a hard constraint on how it's executed (chunked, clean-as-you-go).

27. **May 2018 per-day v4-vs-v10 breakdown (Aug 26 2026) — data artifact for a talk slide.** Produced per-day detection counts for all 31 days of May (held-out month), both models, thresholds 0.0/+1.16/+2.31, plus confirmed-clips-per-day. Tool: `tools/may_per_day_v4_v10.py` (reads the two existing May inference CSVs + DB confirmed windows — no re-inference). Output: `results/may2018_per_day_v4_v10.csv`. Reconciles exactly with #21 (confirmed total 210, 8 confirmed days).
    - **The 8 confirmed days (v4@1.16 / v10@1.16 / confirmed):** 05-02 (0/1/1 NEW), 05-03 (0/1/1 NEW), 05-07 (1/1/1), 05-12 (111/144/181), 05-13 (1/5/8), 05-14 (4/10/10), 05-16 (4/7/7), 05-29 (0/1/1 NEW). Totals v4@1.16=121, v10@1.16=170, confirmed=210. **v10 >= v4 on every single day**, and catches 3 days v4 misses entirely (05-02/03/29).
    - **Verified:** the 4 new days each carry EXACTLY ONE +1.16 detection (05-02/03/07/29 = 1 each) — previously inferred from "4 of 14," now DB-confirmed.
    - **Charting guidance (important for the slide):** plot the **+1.16** series, NOT +2.31 — at v10's own optimum (+2.31) five of the eight confirmed days collapse to zero detections (only 05-12/14/16 survive), which would read as "no orca" on real orca days. State +2.31 in text if wanted. Also: "detections at threshold" != "confirmed orca present" (e.g. 05-12: 181 confirmed but 144 detected @1.16 — 37 confirmed clips sit below cutoff). The stronger, more honest slide is RECALL PER DAY (confirmed-recovered vs confirmed-present), not raw detection counts. Answer-back doc: `may_per_day_ANSWER.md`.

28. **Embeddings live in `usearch.index`, NOT in the sqlite — storage/archive architecture pinned (Aug 27 2026).** Investigated for John's question "how big to archive all 11 years of embeddings" (3 Perch 2.0 users: Duane, Carlos, Danelle). Definitive schema check: the `hoplite.sqlite` contains ONLY metadata tables — `deployments, recordings, windows(offsets), annotations(label,label_type,provenance), hoplite_metadata`. **There is NO embedding-vector column in the sqlite.** The 1,536-dim float16 vectors live entirely in the companion **`usearch.index`** file. Arithmetic confirms: ~500K windows x 1536 x 2 bytes float16 ~= 1.5 GB, matching the observed ~1.6 GB usearch.index (vs. ~40 MB sqlite).
    - **Correction of an earlier mistake in this project's notes/discussion:** it was previously (wrongly) suggested that the sqlite holds the embeddings and usearch is a cheaply-regenerable index to skip when archiving. WRONG. **usearch.index IS the embeddings** — regenerating it requires re-embedding the audio on the GPU (not a cheap rebuild-from-sqlite). Archive unit = **usearch.index (vectors) + hoplite.sqlite (the window-id -> recording/time/label key) TOGETHER**; neither is useful alone (sqlite without usearch = labels with no vectors; usearch without sqlite = anonymous vectors).
    - **Similarity metric = INNER PRODUCT, and CONFIRMED != cosine (Aug 27 2026, Danelle's question, resolved by experiment).** `hoplite_metadata` usearch_config: `metric_name":"IP"`, and `idx.metric_kind = MetricKind.IP`. IP == cosine ONLY if vectors are L2-normalized — **VERIFIED they are NOT:** sampled 200 vectors from the April 2018 usearch.index (518,400 x 1536, F16), L2 norms ranged **2.75-4.78 (mean 3.30, std 0.34), nowhere near 1.0.** So retrieval is genuinely dot-product on non-normalized vectors — magnitude matters, unlike cosine. (The audio peak-norm-to-0.25 fix is AUDIO amplitude norm, NOT embedding L2-norm.) Norm-check script: read via `usearch.index.Index.restore(...).get(keys)` then `np.linalg.norm`. **Source-code confirmation (perch_hoplite v1.0.2, current & effectively only PyPI version, installed pip not git; usearch 2.25.3):** `perch_hoplite.db.score_functions.get_score_fn` offers `numpy_dot` (raw np.dot, NO normalization), `numpy_cos` (L2-normalizes both sides THEN dots), `numpy_neg_euclidean`. Our `phase2_classify.py` search selects via `--score-fn` whose **default is `"dot"`** (line 693), so the raw dot path is used. FOUR consistent confirmations that retrieval = raw inner product, NOT cosine: (1) usearch config metric_name=IP; (2) vectors not unit-norm (2.7-4.8); (3) score_functions numpy_dot is raw; (4) --score-fn default=dot. Cosine IS available (`--score-fn cos`) but unused. **For poster/paper, say "nearest-neighbour retrieval" (metric-agnostic, unattackable); if naming the metric, say "inner product" / "dot product," NEVER "cosine." Implication worth a methods sentence: embedding magnitude influences ranking, not just direction.**
    - **11-year archive estimate (assume all months ~= April 2018 full month):** ~132 nominal months (~120-130 with outage gaps). Embeddings (usearch): 132 x 1.6 GB ~= **~200-215 GB**. Metadata (sqlite): 132 x 40 MB ~= **~5 GB**. **Full runtime package ~= ~210 GB.** For scale, resampled 32kHz audio for 11 yr ~= 157 GB/mo x 132 ~= **~20 TB** — so embeddings are ~1% of the audio. Headline for John: archiving ALL 11 years of embeddings is **~200 GB, not terabytes** — Zenodo-feasible (~4 records) or a small S3 bucket.
    - **Regeneration cost if NOT archived:** resample (SoX, CPU, cheap compute but ~1 day/mo wall-clock + ~20 TB transient disk, must chunk) + embed (GB10 GPU: Sept 2024 = 324K embeddings in 17.9 min -> ~30 min/full month -> **~66 GPU-hours for 11 years**; usearch index built during this step). Trade-off for John: archive ~200 GB once (a few $/mo on S3) so three Perch users never re-spend ~66 GPU-hr + 20 TB transient resampling.
    - **Archive-set recommendation (per-DB):** KEEP `hoplite.sqlite` + `usearch.index`; SKIP `hoplite.sqlite-wal`/`-shm` (transient — run `sqlite3 <db>/hoplite.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"` with no process attached BEFORE archiving so the sqlite is complete), SKIP `logs/`. Across the db/ tree: keep the 5 per-month `_norm` DBs (Apr2018, May2018, Oct2020, Apr2026, Sep2024); SKIP non-`_norm` (superseded pre-normalization), SKIP `_ctx` (v5 context dead-end), SKIP redundant `combined_*` iterations (regenerable by recombining per-month `_norm`; keep at most the one v10 actually trained on — confirm which via train.py args). Full db/ survey: 15 dirs, sqlite total 1.1 GB, usearch total 43.2 GB currently on thalassa.
29. **July 2015 — FIRST MONTH OF THE FULL-ARCHIVE CAMPAIGN (#26). ZERO ORCA. Pipeline end-to-end validated on 2015-era audio (D. Edgington, Aug 27 2026).**
    The MARS archive campaign starts at the beginning: July 2015 is the first month the hydrophone
    was deployed. Run end-to-end (resample -> verify -> embed -> coverage -> infer v4+v10 -> review)
    in a single session. **Result: no orca.** A clean, unremarkable null — and exactly the expected
    outcome for late July, which sits well outside the spring window where every confirmed Monterey
    Bay Bigg's event to date has landed.
    - **Data:** deployment begins **2015-07-28 18:05:24**; the month is only ~78 h. 469 files,
      469 raw = 469 resampled (exact match), **56,130 embeddings**, 4.3 min on spark-ae0e GB10 at
      219.4 windows/s. Coverage (DB) = 36 / 144 / 145 / 144 for Jul 28/29/30/31, matching disk.
    - **Two recorder restarts:** `MARS_20150729_162524` = 242 s and `MARS_20150730_031011`
      = 205 s. **CORRECTED Aug 28 2026 by `tools/coverage_histogram.py`: total real gap is 62 s
      (53 s + 9 s), NOT the 753 s originally recorded here.** 753 s was the duration *deficit*
      vs 469 nominal 600 s files, wrongly reported as lost wall-clock time; the recorder
      restarted promptly so almost nothing was lost. The predicted next-start `163019` (not the
      cadence-implied `163524`) is why — **a restart shifts the filename cadence.** Also **one
      8 s timestamp overlap** exists (`163019` -> `164011`), below one 5 s analysis window and
      consistent with a clock nudge; the original "no duplicated audio" claim came from
      arithmetic that could not see overlaps at all. Span reconciliation now exact:
      Jul 28 18:05:24 -> Aug 1 00:03:45 = 280,701 s = 280,647 recorded + 62 gap - 8 overlap.
      The 145-file day (7/30) is a restart artifact plus midnight spill, not an overlap.
    - **Inference (floor 0.0):** v4 = 42 detections, v10 = 27. **Nothing at >=2.31 (v10's own
      optimal cutoff) for either model.** Exactly one detection each in [1.16, 2.31) — and it is
      **the same window** in both: `MARS_20150731_222345`, 335-340 s, **v4 = 1.548, v10 = 2.002**.
    - **THE CONCORDANT WINDOW IS NOT ORCA.** Reviewed by ear: not orca. This is a useful
      calibration point — the strongest evidence the month had, two independently-trained
      classifiers agreeing, and it still resolves negative. Worth remembering when a single
      concordant above-threshold window turns up in a future month with no bout structure.
    - **Floor-rate caution (recorded because it briefly misled this session):** 42/56,130 = 7.5e-4
      per window vs Oct 2020's v4 rate of 144/535,278 = 2.7e-4 — i.e. July 2015 fires ~2.8x more
      often per window than the month confirmed orca-silent. **This means nothing.** The mass is
      entirely in the bottom band (v4: 36 of 42 below 0.5; v10: 23 of 27), collapsing under
      threshold exactly as Oct 2020 did. Floor-0.0 rates are not comparable across months;
      re-confirms the standing rule that T=0.0 counts are unusable.
    - **v10's >=0.5 set is a strict SUBSET of v4's.** v4 showed three hits on the evening of 7/31
      (22:13 = 0.587, 22:23 = 1.548, 23:23 = 0.711) which superficially resembled an evening
      cluster (cf. Apr 25 2018, an evening encounter). **v10 drops both flanking windows below
      0.5**, so the better model does not support the cluster. Correctly read as scattered
      single windows, not a bout — no consecutive windows anywhere in the month.
    - **Review session (D. Edgington, annotator-id `duane`, 4 minutes):** 6 clips = everything
      either model scored >=0.5 (v4's superset). Saved 6 labels, 0 unlabeled skipped.
      **Outcome: 2 `dolphin_call`, 4 `other`, 0 orca.** "other" = real acoustic content Duane
      could not identify and was confident was not orca. NOTE: the labeling session ID was not
      captured this session — recoverable from
      `/mnt/PAM_Analysis/perch-hoplite/logs/review_jul2015_v10.log` if needed for audit.
    - **Figures registered** (both PERCH framegrabs, mel/viridis, sidecars + manifest committed):
      `gradio_jul30_2015_dolphin_450s_wid4531.png` (score 0.603 v4 / 0.895 v10) and
      `gradio_jul31_2015_dolphin_425s_wid20726.png` (score 0.711 v4 / **v10 below 0.5**).
      The second is notable: **a window orca_v10 essentially dismissed proved to be a real dolphin
      call by ear** — a `dolphin_call` recall observation, not an orca-performance one.
    - **Review-CSV score provenance (avoid re-deriving this):** the review set was built from the
      **v4** detections CSV (the superset at >=0.5) while the session ran `--classifier orca_v10.pt`.
      So **the scores displayed in the Gradio pane and recorded in the figure sidecars are v4
      scores**, not v10. Both figure sidecars document this explicitly.
    - **MONTH-BOUNDARY CAVEAT (carry into August 2015):** the record ends at midnight on 7/31, so
      any encounter beginning that evening is truncated by the month boundary. August 2015
      continues the same deployment — **check the early hours of Aug 1** when that month is
      embedded. A real event spanning the boundary would surface there. (This is a general
      artifact of month-at-a-time processing across an 11-year archive, not specific to July 2015.)
    - **Pipeline/docs hardening produced by this run** (all pushed): `CLAUDE_embed.md` gained a
      canonical Stage 1 script designation, a new **Stage 1.5 resample-verification** procedure,
      and the **`ceil(duration/5)`** window-count reconciliation rule. `register_figure.py` gained
      **PERCH** as a `--computer` choice (ICEFISH is retired; PERCH is the current screenshot source).

30. **Per-day recording-effort records are now MANDATORY per month — `tools/coverage_histogram.py` (Aug 28 2026).**
    The full-archive campaign deletes bulk resampled WAV after each month is analyzed, so **hours
    recorded per day is unrecoverable once the audio is gone.** August 2015 forced the issue:
    coverage ranges from **2.7 h to 24 h per day**, so Aug 19 (16 files) and Aug 20 (145 files)
    cannot be compared as raw counts. **Every seasonal/interannual figure must be detections per
    hour of effort**, with hours-recorded carried alongside every count. Tool writes
    `results/coverage/<YYYY>-<MM>_coverage.csv` (date, files, seconds, hours, expected_windows,
    pct_of_day, short_files, note) — **committed to the repo, run at Stage 1.5 before embedding.**
    Backfilled for all 7 months processed to date (see the coverage table in `CLAUDE_embed.md`).
    - **Two 2018 months are perfect:** Apr 2018 = 720.00 h / 100.0%, May 2018 = 744.00 h / 100.0%,
      zero short files. Fitting for the reference and held-out months.
    - **October 2020 has 47 short files** — `files × 120` overcounts its windows by **5,185 (~1%)**.
      Anyone reasoning from file counts in the orca-silent specificity month is off by that much.
      Total time lost is only 0.50 h, so the specificity conclusion stands. One 761 s gap falls on
      10/5 18:49, inside the confirmed Oct 5-12 silent cluster — immaterial but now on record.
    - **April 2026 lost 17.7 h in a single dropout** after 4/13 00:20 — **outside** the Apr 17-24
      CA51A/CA50B event window, so finding #17 is unaffected.
    - **Sept 2024 reproduces finding #25 exactly:** 449.67 h, 62.5%, 9/20-9/30 absent.
    - **WINDOWING RULE CORRECTED (Aug 28 2026): it is `max(1, floor(duration/5))`, NOT `ceil`.**
      This finding originally recorded `ceil` (from July 2015, which has only 2 short files and
      matched it by coincidence). **August 2015, with 21 short files, discriminates cleanly:**
      `ceil` predicts 453,137, `floor` predicts 453,119, DB holds **453,123** = `max(1,floor(...))`.
      The adapter DROPS the final partial window but never emits zero for a file. New tool
      `tools/audit_window_counts.py` verifies this per file every month and reports disagreements,
      skipped files, and padded windows. **Do not reintroduce `ceil`.**
      - **PADDED WINDOWS — a real inference concern.** Because of the `max(1, ...)`, a file shorter
        than one 5 s window still yields one window that is mostly empty. August 2015 has three
        (1 s, 2 s, 1 s). **Nothing stops the classifier scoring one of these high**, and across
        ~130 months of a restart-prone recorder there could be hundreds. OPEN QUESTION: exclude
        sub-5 s files at inference, or just check any detection that lands on one?
      - **KNOWN ANOMALY (1 file in 3,793, unresolved):** `MARS_20150817_155951` (301.0 s) holds 61
        windows where the rule predicts 60, while `MARS_20150803_153345` (476.0 s, same 1 s
        remainder) correctly holds 95. `soxi -D` prints 3 decimals only — check `soxi -s`.
    - **METHOD LESSON (cost real time; do not repeat): a duration DEFICIT is not a GAP.** A short
      file means that file ended early, not that time was lost — what matters is when the NEXT file
      STARTED. And **never infer the next start from the filename cadence: a restart shifts it.**
      Both errors were made on July 2015 (see #29, corrected) and are why the tool measures gaps
      and overlaps directly from start-time + true-duration rather than by arithmetic.
    - **Recorder CLOCK RESYNC produces small timestamp overlaps every month — NOT duplicated
      audio.** Apr 2018: `075914` -> `080911`, 3 s overrun, on Apr **1/8/15/22/29** (strictly
      weekly). May 2018: 2 s on May **6/13/20/27**. A weekly event at the same second of day is a
      clock correction — oscillator drifts ~2-3 s/week, gets resynced, filename stamps compress
      while the audio stays contiguous. At 2-3 s these cannot fill one 5 s window. Tool tiers them:
      **<5 s negligible · <60 s minor · >=60 s MATERIAL** (genuine re-record; stop and investigate).
      **ACTION: confirm the weekly resync with J. Ryan** — MARS clock discipline is his domain.
      If confirmed, it is a one-line methods note.
    - **A day's summed duration can legitimately exceed 24 h** — files are binned by START
      timestamp, so the last file of a date runs past midnight (max spill = one file = 600 s).
      An earlier version of the tool used a 24.05 h ceiling as an overlap proxy and **false-alarmed
      on 2015-07-30**; replaced with the real timeline walk.

31. **August 2015 embedded — first FULL month of the campaign (Aug 28 2026).**
    3,793 files, **453,123 windows**, 33.5 min, 225.6 windows/s on spark-ae0e. DB
    `MARS_20150801_20150831_32kHz_norm`. Stage 2.5 coverage matches disk exactly (30 dates; no
    8/16). Zero skips or errors in the embed log; all 3,793 files present in `recordings`.
    - **Coverage 629.34 h = 84.6% of nominal**, with **five long dropouts** rather than scattered
      loss: 58.6 h after 8/15 04:35, 22.6 h after 8/18 22:44, 16.2 h after 8/12 23:45, 11.9 h after
      8/7 06:38, 3.6 h after 8/21 17:03. Partial days therefore hold *contiguous* blocks, which is
      far easier to interpret than swiss cheese. **8/16 has zero data.**
    - **Aug 1 is COMPLETE (144 files)** — so the July 2015 month-boundary question (#29) is
      answerable: anything spanning midnight 7/31 will appear here.
    - **RECORDER THRASHING on Aug 5:** eight files in under four minutes — 13:15:03 (90 s),
      13:16:45 (17 s), 13:17:09 (6 s), 13:17:24 (11 s), 13:17:44 (9 s), 13:18:04 (7 s),
      13:18:18 (14 s), 13:18:42 (20 s). This is why 8/5 shows 152 files. Not overlap — the tiered
      timeline check found no material overlap. A recorder repeatedly restarting.
    - **THROUGHPUT REVISED:** 225.6 win/s here vs 219.4 for July. The earlier claim that July's 219
      was `torch.compile` warmup was **wrong** — ~220-226 win/s is the sustained GB10 rate.
      **Sept 2024's ~302 win/s is the outlier needing explanation.** Budget 30-35 min/full month.
    - **GPU CONTENTION — the first embed attempt DIED.** `torch.AcceleratorError: CUDA error: out
      of memory` at model load, caused by a forgotten Gradio review server (July 2015's, on port
      7881) holding 228 MiB. **The review server and the embedder share the same GPU.** Killed via
      `pkill -f "port 7881"`, GPU returned to "No running processes found", relaunch succeeded.
      Second review-server incident in two days (Aug 27 it blocked a port; Aug 28 the GPU).
      **`nvidia-smi` before every embed** is now in `CLAUDE_embed.md` Stage 2.

32. **Gradio label saving was DESTRUCTIVE across annotators — FIXED (Aug 28 2026).**
    `phase2_classify.py` review autosave ran `DELETE FROM annotations WHERE recording_id=? AND
    offsets=?` with **no provenance scope**, then inserted. So a second annotator did not add an
    opinion — they **deleted the first annotator's label**. Worse, the review UI is **blind by
    default**: the radio is hardcoded `value="unlabeled"` and never reads existing labels from the
    DB, so an annotator overwrote labels they could not even see.
    - **FIX (shipped):** scope the DELETE — `... AND provenance=?`, passing
      `gradio_gui:<annotator_id>`. Changing your own mind still replaces your own label; another
      annotator's label survives. **No schema change** — `provenance` already carried the annotator.
      Applied via `tools/patch_scoped_delete.py` (idempotent, backs up, self-reverts on syntax error).
    - **Caught just in time:** John was about to review July 2015, which would have destroyed
      Duane's 6 labels.
    - **Blind review is therefore already the default behavior**, accidentally. The missing feature
      is the opposite — a `--show-existing-labels` mode for *reconciliation*. If added, show prior
      labels as text, **not** as a pre-selected radio button, so a reviewer is not nudged into
      agreement.
    - **Design recommendation written up in `docs/multi_annotator_design.md`:** one DB with one row
      per (window, annotator); never separate DBs per annotator (duplicates ~1.6 GB usearch index
      per month and makes agreement uncomputable); reconciliation as a third `consensus:duane+john`
      row rather than an edit; training selects a view under an explicitly recorded policy
      (John's label wins where present, Duane's elsewhere); payoff is computable **Cohen's kappa**
      plus a per-class confusion matrix, which reviewers ask for in expert-labeled acoustics work.
    - **REMINDER: always pass `--annotator-id`** — the default is a generic `analyst`. The Aug 24
      April 2026 review session lacked it.

33. **August 2015 REVIEWED — 13 dolphin, 1 UNCONFIRMED ORCA CANDIDATE (Aug 28 2026).**
    - **★ UPDATE Sep 1 2026 — CANDIDATE RECLASSIFIED AS DOLPHIN. AUGUST IS NOW ZERO ORCA.**
      On re-review after hearing ~250 confirmed orca calls across Sep/Oct/Nov 2015, D. Edgington
      relabeled `MARS_20150828_212219` @325s (wid 255405, v10=1.406) from orca_call to
      **dolphin_call**. Quote: *"I see why I thought it could be orca, but after all the orcas of
      the last few days, now I think not."* This is calibration improving, not error correction —
      the candidate was honestly ambiguous in August; a trained ear now resolves it. **August 2015
      final: 20 dolphin, 2 ROV_noise, 2 other, 0 orca.** The 8/28 KW sighting (4 animals, a.m.)
      thus becomes another daytime-sighting / no-acoustic-detection case (finding #46 pattern).
    14 clips reviewed by ear (D. Edgington, `--annotator-id duane`): the union of **orca_v4 >=1.16**
    (6 windows) and **orca_v10 >=1.00** (8 more), presented **score-descending**. Saved 14 labels,
    0 unlabeled skipped. **Outcome: 13 `dolphin_call`, 1 `orca_call`.**
    - **THE CANDIDATE: `MARS_20150828_212219` @325-330 s (wid=255405).** v10 **1.406** (5th
      month-wide), v4 **0.512** (~20th; below the 1.16 cutoff, so it entered the set via v10).
      **Below BOTH models' operating thresholds.** First non-spring orca candidate of the campaign —
      every confirmed Monterey Bay Bigg's event to date is April-May.
      - **Duane's call:** looks and sounds like an orca but **lacks the higher frequencies** and is
        **isolated from other calls**. Spectrogram confirms: narrow harmonic stack **~2-4 kHz** with
        a gentle upward inflection, ~1.8-3.0 s in, **nothing above ~5 kHz** — visually distinct from
        all 11 dolphin clips in the same session, which sit at **4-16 kHz** with steep sweeps. A
        different acoustic object, not a weaker version of the same thing.
      - **Both readings remain open.** A DISTANT orca (range strips highs first; and **Bigg's are
        acoustically cryptic** — long silences then a few isolated calls, so an isolated call is not
        evidence against them) or a distant dolphin high-passed by the same propagation.
      - **LOCAL CONTEXT** (from a `--logit-threshold -10` diagnostic run): the most anomalous window
        in 3 hours. Within its own recording v10's next-highest is **-1.155** (median ~-5); across
        20:00-23:00 (2,160 windows) v10 **p99 = -1.99**, next-highest **-0.452**. Neighbours at
        320 s (-5.767) and 330 s (-3.378) are ordinary background — **the isolation is real, not an
        artifact of the 0.0 cutoff.**
      - **CAVEAT that keeps this honest:** local-outlier magnitude partly measures how quiet the
        water was, and Aug 28 21:00 was quiet. **Month-wide the candidate is mid-pack** among
        windows already labelled dolphin. Do NOT present it to John as "the anomaly of the month" —
        the signal here is Duane's ear, not the score.
      - **STATUS: UNCONFIRMED, pending J. Ryan.** Spectrogram posted to the Slack channel Aug 30 2026.
    - **v4 vs v10 at their own operating points: v4 says 6 candidates, v10 says ZERO.** Nothing
      reached v10's 2.31; its month-wide top is 1.993. **But all six of v4's >=1.16 windows also
      appear in v10's top 14** — so the models substantially AGREE on which windows are interesting
      and disagree only on whether any clears a bar. This is **threshold placement, not sensitivity**;
      neither model is missing what the other found. Beyond the top few the ORDERINGS diverge
      substantially, which matters: **the choice of model changes WHICH clips a fixed review budget
      covers.**
    - **Aug 8 is the month's acoustic hotspot** — 3 recordings (01:55, 12:45, 15:25) contribute 6 of
      v4's top 15. All reviewed as dolphin. `MARS_20150808_124545` contributed two windows to v10's
      top 14 (530 s, 550 s) — the only sub-bout structure in the month.
    - **12 figures registered** (`figures/gradio_aug*_2015_*`), incl.
      `gradio_aug28_2015_ORCA_325s_wid255405.png`. 15 grabs were taken for 14 clips; **3 were
      duplicate pairs** where the first shot cut off the audio control (labels #6, #7, #10) — the
      scrolled versions were kept. Labels #4 and #5 were never screenshotted. Every button
      reconciled against the DB with no discrepancy.
    - **⚠️ SCORE-SCALE WARNING for this review set:** it mixes **v4 scores** (the 6 windows v4
      flagged) and **v10 scores** (the 8 v10 contributed), so the scores displayed in Gradio and
      recorded in the 12 sidecars are **NOT on a single scale**. Each sidecar says which model's
      score it carries. "Score-descending" is therefore not a single coherent ranking.
    - **METHOD — the review protocol Duane wants (his reasoning, adopt it):**
      - **Score-DESCENDING, not randomized.** If a session has orcas, the strongest clip establishes
        what *this* encounter sounds like — this hydrophone, this range, this propagation, this
        animal's repertoire that day — and every weaker/fractured clip is then heard against that
        reference instead of a generic expectation. Sounds vary session to session for reasons not
        yet understood. This calibration gain outweighs the ordering-bias concern.
      - **Low scores are sometimes orcas and high scores are sometimes false positives.** Thresholds
        exist to keep review tractable across 130 months, not because sub-threshold windows are
        uninteresting.
      - **ZOOM-IN on any confirmation.** Once a day/time has a verified orca the prior shifts
        completely — an encounter means many calls, most scoring low. Second pass: drop the
        threshold far, restrict to a time window around the confirmation. Two-stage design:
        broad-and-shallow to find candidates, then narrow-and-deep around anything real.
    - **⚠️ COST WARNING — do NOT rerun a full month at `--logit-threshold -10`.** It took **64
      minutes per model** (vs **3.8 s** at the 0.0 floor) because writing and copying 453,123 CSV
      rows dominates; the scoring is the same work. **Scope diagnostic runs to the one recording or
      time range of interest.** The two 453K-row scratch CSVs were written to `/tmp` (deliberately
      NOT `results/`, since they are diagnostic, not archival) and deleted after use.

34. **SEPTEMBER 2015 — TWO ORCA ENCOUNTERS, 18 CONFIRMED CALLS. First confirmed orcas of the
    campaign (D. Edgington, Aug 30 2026).**
    Cleanest month yet: **719.43 h = 99.9% coverage**, 4,323 files, **517,984 windows**, 38.6 min
    embed at 223.8 win/s. Only 0.57 h missing, essentially all in one 32-min gap on 9/1 15:57.
    DB `MARS_20150901_20150930_32kHz_norm`. Floor detections: v4 186, v10 182 — near-parity, unlike
    August's 172 vs 99.
    - **v10 CLEARED ITS OWN THRESHOLD FOR THE FIRST TIME:** 4 windows >= 2.31 (one >3.00), where
      July and August produced zero. v4: 8 windows >= 1.16.
    - **EPISODE A — 2015-09-16 23:47 -> 09-17 06:45, NINE calls.** `20150916_234019` @435 s (1), then
      `20150917_062020` @320/370 s (2), then **SIX calls in `20150917_064020`** @10/65/80/95/100/295 s.
      The @80 s window scored **3.128 — the highest anywhere in the campaign — and is a confirmed orca.**
    - **EPISODE B — 2015-09-28 05:05 -> 07:54, NINE calls.** `050349` @95/170 s, `060349` @380 s,
      **FOUR in `071349`** @290/400/435/560 s, `072349` @145 s, `075349` @25 s. Dolphins present the
      same morning (06:23, 06:33, 07:43, 08:53, 09:03, 09:43, 10:52) — worth noting given Bigg's prey.
    - **CALL SIGNATURE:** narrow harmonic stack ~1-4 kHz, gentle upward inflection, **no energy above
      ~5 kHz.** Visibly distinct from same-session dolphins at 8-16 kHz with steep sweeps. Matches the
      August candidate's signature (#33), which strengthens that candidate.
    - **⚠️ RECALL AT THE OPERATING POINT IS ~17%.** Only 4 windows in the month cleared v10's 2.31 —
      and **one of those four was a DOLPHIN** (`20150928_063349` @355 s, v10 2.349, the month's
      2nd-highest score). So **3 true orca calls out of 18 present, at 75% precision.**
      **More than half the confirmed calls came from the low-threshold second pass**; the lowest
      confirmed call scored **v10 = 0.205**. The threshold is well suited to FINDING ENCOUNTERS (one
      call above threshold locates the event) but would **badly undercount calls** for call-rate
      statistics. Frame seasonal analysis accordingly.
    - **TWO-PASS PROTOCOL VALIDATED.** Pass 1 (27 clips, 14 min): union of v4>=1.16, v10>=2.31, the
      full `20150916_181020` bout at v10>=0.50, and the Sept 28 cluster at v10>=1.00. Pass 2 (21
      clips, 15 min): every previously-unheard window at **v10 >= 0.20** inside the two episodes.
      **11 of 21 pass-2 clips were orca.** Without the zoom-in the month reads as 7 scattered calls
      instead of two encounters.
    - **22 figures registered** (`figures/gradio_sep*_2015_*`), 16 of them ORCA. 27 grabs taken; 5
      dropped (2 pre-click, 2 duplicates, 1 superseded by a clean re-grab of the 3.128 whose header
      had been covered by a browser tooltip). One figure records a **deliberate `unlabeled` skip**
      (`20150928_065349` @315 s) — verified against an empty DB row, i.e. a documented "listened,
      could not tell", not a mistimed grab.
    - **⚠️ SCORE-SCALE WARNING:** the pass-1 review set mixes v4 and v10 scores; pass-2 is all v10.
      Sidecars record which model's score each figure carries.
    - **PER-DAY RATES (effort-normalised, from `tools/label_summary.py`):** only four days in
      September carry labels at all.

      | Date | ROV_noise | dolphin | orca | hours | **orca/h** |
      |---|---:|---:|---:|---:|---:|
      | 2015-09-02 | 0 | 1 | 0 | 24.0 | 0.000 |
      | 2015-09-16 | **12** | 2 | 1 | 24.0 | 0.042 |
      | 2015-09-17 | 0 | 1 | **8** | 24.0 | **0.333** |
      | 2015-09-28 | 0 | **13** | **9** | 23.9 | **0.376** |

    - **DOLPHINS AND ORCAS WITHIN SECONDS OF EACH OTHER (Episode A).** `MARS_20150917_062020`
      holds an **orca call at 06:26:30 (@370 s) and a dolphin call at 06:26:40 (@380 s)** —
      adjacent windows, 10 s apart, both confirmed by ear. In Episode B the two interleave across
      the morning instead: **13 dolphin calls against 9 orca calls on 2015-09-28**, from 06:06 to
      10:56. Given Bigg's prey on marine mammals this may be worth more than a footnote; we have no
      view on whether it is coincidence. **Raised with J. Ryan.**
    - **Sept 28 is denser than the orca count alone suggests:** 22 labels on one day (9 orca,
      13 dolphin), spanning 05:05-10:56. Two dolphin calls at 06:39:44 and 06:39:49 are adjacent
      windows.

35. **`ROV_noise` IS NOW A LABEL CLASS — J. Ryan's warning, confirmed in the data (Aug 30 2026).**
    John had warned that **ROV servicing of the MARS science node produces a signature broad-band
    "screech" across many frequencies**. September 2015 is the first month Duane recognised one in
    Gradio. **All 12 `ROV_noise` labels fall in ONE recording: `MARS_20150916_181020`** (18:10-18:20
    UTC), which shows dense horizontal banding across the whole spectrum.
    - **That recording alone produced 11 above-floor detections and initially looked like BOUT
      STRUCTURE** — five minutes of near-continuous "activity". Without John's warning it would have
      been read as a biological cluster. It is the single largest false-positive source found so far.
    - **`ROV_noise` added to `--classes`** (placed next to `ship_noise`; both anthropogenic). Use
      `--classes orca_call,humpback_song,dolphin_call,ROV_noise,ship_noise,other,unlabeled` from now on.
    - **THE SCREECH IS TIGHTLY BOUNDED: 11 of the 12 labels fall between 18:14:55 and 18:16:50 —
      under two minutes — with one straggler at 18:20:00.** A sharp target for a log check.
    - **ASK JOHN: confirm an ROV service visit on 2015-09-16, 18:14-18:20 UTC** from ship logs, and
      get the other service dates for 2015-2026.
    - **⚠️ EARLIER MONTHS MAY CONTAIN UNRECOGNISED ROV NOISE** filed as `other` or `ship_noise`.
      Revisit once service dates are known.
    - **CROSS-CHECK AVAILABLE:** the MBARI Soundscape Visual Browser
      (https://www.mbari.org/data/soundscape-visual-browser/) — John's tool — lets Duane navigate to
      any Y/M/D/time and inspect spectrograms independently of this pipeline. `tools/label_summary.py`
      emits exact UTC timestamps for every confirmed label to drive that navigation.

36. **⚠️ GRADIO AUTOSAVE DOES NOT SURVIVE A CONNECTION LOSS — 14 minutes of work lost (Aug 30 2026).**
    The review pane states *"Labels are also auto-saved on each click — reload is safe."* **This is
    misleading.** Duane's VPN dropped ~14 minutes into a 27-clip session; on reconnect the server
    was still running (it was under `nohup`) but **`SELECT COUNT(*) FROM annotations` returned 0** and
    **no JSON appeared** in `provenance/labels/`. Every clicked label was gone.
    - The per-session JSON is written on **save/exit**, not per click, so it cannot capture a partial
      session either.
    - **MITIGATION: press "Save Labels to DB" every ~8-10 clips.** Duane's fair objection: the button
      is at the top of the page, so this means scrolling up and back each time — real friction across
      130 months. **TOOL WISHLIST: make the save button sticky, or add a keyboard shortcut.**
    - Cost accounting: September's review was 29 min of listening **plus 14 min lost** — the drop cost
      about a third of the month's review effort.

37. **REVIEW-EFFORT TIMINGS — for agile loop planning (running record).**
    | Session | Clips | Time | Rate |
    |---|---|---|---|
    | July 2015 | 6 | 4 min | ~40 s/clip |
    | August 2015 | 14 | 4 min | ~17 s/clip |
    | Sept 2015 pass 1 | 27 | **14 min** (LOST to VPN drop) | ~31 s/clip |
    | Sept 2015 pass 2 | 21 | 15 min | ~43 s/clip |
    Steady state ~30 s/clip; **marginal/faint clips run slower** (~43 s) because the dolphin-vs-orca
    judgement is genuinely harder there. A month with an encounter needs ~45-50 clips across two
    passes = **~25-30 min of listening**. Across ~130 months that is roughly **50 hours of review** —
    tractable, but it is **the one pipeline stage that cannot be automated or split across the two
    sparks.** (The Sept pass-1 rerun was faster than 14 min, but that number is not comparable —
    Duane had already heard those clips.)

38. **`tools/label_summary.py` — per-day label histogram + exact UTC timestamps (Aug 30 2026).**
    Emits, for one month's DB: (a) confirmed labels per day by class, **effort-normalised against the
    coverage CSV** (`orca/h`), and (b) an **exact UTC timestamp for every confirmed label**, derived
    as recording start (from the `MARS_YYYYMMDD_HHMMSS` filename) + the window offset unpacked from
    the `annotations.offsets` blob. `--markdown` emits tables.
    - **Purpose: let J. Ryan navigate directly to any confirmed call in the MBARI Soundscape Visual
      Browser** (https://www.mbari.org/data/soundscape-visual-browser/) — his own tool, independent
      of this pipeline — to inspect the spectrogram himself. Cross-checking our labels against that
      browser is now a standing option, and is how the ROV_noise hypothesis can be verified.
    - **Verified Aug 30 2026:** all 18 September orca timestamps produced by the tool match
      hand-computed values to the second.
    - **Offset blob recipe confirmed working:** `struct.unpack('<2d', blob)` on `annotations.offsets`.
    - Note the `orca/h` column reads `0.000` (not blank) on days with labels but no orca — correct,
      but do not misread it as a measurement of absence across the month.

39. **SIGHTING-RECORD CORRELATION — a strong LEAD for the Sept 2015 encounters, UNVERIFIED
    (Aug 30 2026). Do not treat as established.**
    The three September acoustic detection days (**09-16, 09-17, 09-28**) coincide with claims of
    notable Monterey Bay Bigg's activity in the same window — specifically **CA95A1 / the CA95A
    matriline** around Sept 16-17, and a separate Sept 28 encounter described as Bigg's **hunting
    dolphins**. Also claimed: the fish-eating **Southern Resident L-Pod** moving along the coast in
    the same window.
    - **⚠️ NONE OF IT IS SOURCED.** The material surfaced with **citations that did not match their
      claims** (an orca "sighting" reference pointed at an unrelated news story; another at a
      fashion-show page) — the signature of generated rather than retrieved text. A later CKWP-styled
      record for CA95A1 was formatted like an AI info-card, with a "Notable Tracking Window" field
      matching the query dates exactly. **Note also that CA95 was first attached to Sept 28 and then
      CA95A1 to Sept 16-17** — the drift is itself a warning sign.
    - **Claude could not verify any of it** — no web search in that session, and browsing restricted
      to an allowlist that excludes news sites and CKWP. **Pod identifications are therefore kept OUT
      of this record** until J. Ryan confirms them.
    - **WHY IT MATTERS IF IT HOLDS.** (a) Three acoustic encounters matching three documented
      sighting days would be **strong external validation of the whole pipeline** — every
      confirmation so far rests on Duane's ear alone. (b) The **Sept 28 co-occurrence** (13 dolphin
      calls interleaved with 9 orca calls, 05:05-10:56) would become a **documented predation event**
      rather than an open question. (c) Same for the Sept 17 dolphin call **10 s after an orca call
      in the same recording**.
    - **⚠️ THE L-POD CLAIM NEEDS PARTICULAR SCRUTINY.** Southern Residents in Monterey Bay would be
      notable in itself, and it matters technically: a **fish-eating ecotype vocalises very
      differently from Bigg's, and `orca_v10` is trained on Bigg's.** If there is anything to it, it
      changes how these detections read. Probably nothing, but not ruled out.
    - **⚠️ 2015 MAY NOT BE A REPRESENTATIVE YEAR.** It was the peak of the North Pacific marine
      heatwave ("the warm blob"); anomalous warm water and unusual marine mammal distribution in
      Monterey Bay that autumn would need stating in any interannual comparison across the archive.
    - **ACTION (top ask for the Monday meeting): get JOHN to connect Duane with whoever holds the
      actual sighting logs.** Nancy Black / Monterey Bay Whale Watch / the California Killer Whale
      Project are the authoritative record for Monterey Bay Bigg's, but **the logs are not findable
      from outside** — Duane tried. **A warm introduction is worth more here than any amount of
      searching.**
    - **THE REAL PRIZE IS THE DATASET, NOT THE THREE DATES.** Ad-hoc per-date lookups do not scale to
      ~130 months. **A joinable table of dated sightings is the JOIN KEY for the entire seasonal /
      interannual analysis** (roadmap #1). Confirming three days is useful; having the whole log is
      transformative. Ask for it as a dataset.
    - Written up for the meeting in `docs/BRIEF_john_ryan_2015_07-09.md` §9, framed as questions
      rather than claims.

40. **★ SIGHTING CORRELATION CONFIRMED — Sept 16 2015 orca calls fall 17 MINUTES after a logged
    predation event (Aug 31 2026). First external validation of the pipeline.**
    Duane located the **Monterey Bay Whale Watch Marine Mammal Sightings List, September 2015**
    (`montereybaywhalewatch.com/sightings/slst1509/`) — a real, citable, contemporaneous expert
    record. It **supersedes the unverified material in finding #39**, which should not be repeated.
    - **⚠️ COPYRIGHT — DO NOT COMMIT THE SIGHTINGS DATA.** The source states it "may not be used
      without permission from Nancy Black" and that unauthorised use violates federal copyright law.
      **`perch-hoplite` is PUBLIC.** `docs/sighting_correlation_sep2015.md` deliberately contains
      only our own detection times plus the minimum date references needed to state the result.
      **Get written permission before committing any transcription or derived CSV.**
    - **⚠️⚠️ TIME BASE — THE THING THAT MADE THIS LEGIBLE. Our timestamps are UTC; sighting records
      are LOCAL (PDT = UTC−7 in September; PST = UTC−8 otherwise).** Converting moved an entire
      episode across a date boundary: Episode A looked like "09-16 23:47 then a cluster on 09-17
      morning" in UTC, but in local time it is **one call at 16:47 and a cluster at 23:25-23:45,
      ALL on Sept 16** — which is what aligns with the sightings. **ALWAYS CONVERT TO LOCAL BEFORE
      COMPARING WITH ANY SIGHTING SOURCE.** This is now a standing step in the monthly loop.
    - **THE HIT.** MBWW logged killer whales on **every trip on 9/16** — 7 animals at 8am, 9am, 1pm,
      2pm — rising to **14 at 4:30 p.m., annotated "predation on Common Dolphins"**, with ~2,000
      Long-beaked Common Dolphins present. **Our first confirmed orca call is 16:47:34 PDT — 17
      minutes later.** Eight more follow at 23:25-23:45 PDT the same night.
    - **The 10-second orca→dolphin co-occurrence now has context.** The dolphin call at 23:26:40 PDT
      sits 10 s after an orca call in the same recording, on a day with documented predation on
      common dolphins. Suggestive; not proof of predation at that moment.
    - **EPISODE B IS THE STRONGER ARGUMENT FOR THE METHOD.** All nine calls fall **22:05-00:54 local
      on the night of Sept 27→28**. **No killer whales are logged on 9/27 or 9/28 — but whale-watch
      trips run in DAYLIGHT.** This is an apparent encounter the visual record could not have caught.
      **Absence of sighting at night is not evidence of absence**, and passive acoustics is exactly
      what fills that gap. (The log does show **2,500 common dolphins on both 9/28 trips**, the
      month's largest counts, consistent with our 13 dolphin calls that night.)
    - **THE FALSE NEGATIVES MATTER AS MUCH AS THE HIT. Killer whales were sighted on FOUR days
      (9/1, 9/11, 9/14, 9/16); we detected them acoustically on ONE.** Coverage was ~24 h on all
      four, so these are genuine non-detections. Three non-exclusive explanations:
      (a) **RANGE** — MARS is ~25 km offshore in the canyon at ~890 m; the list covers the whole
      region and boats often work inshore, so animals can be "in the bay" and out of detection range;
      (b) **SILENCE** — Bigg's are acoustically cryptic; present ≠ audible;
      (c) **RECALL** — already measured at ~17% at v10's operating point (#34).
      **TESTABLE, AND WORTH DOING: run the pass-2 low-threshold zoom-in on 9/1, 9/11 and 9/14.**
      Calls found there → (c) dominates and recall is worse than estimated. Nothing found → (a)/(b)
      dominate and the pipeline is behaving correctly. **This is the single highest-value follow-up
      the campaign has.**
    - **WHAT IT DOES NOT ESTABLISH:** pod identity (the list gives counts, not IDs — CKWP photo-ID
      needed); that the animals heard are the animals seen; ecotype (the predation-on-dolphins entry
      is *consistent with* Bigg's, which is what v10 is trained on, but is not stated).
    - **ASKS (for J. Ryan):** (1) permission from Nancy Black; (2) **the sightings record as a
      DATASET** — MBWW appears to publish monthly at `/sightings/slst<YYMM>/`, so ~130 monthly pages
      form a scrapable, joinable series, which is the **join key for the entire campaign**;
      (3) CKWP photo-ID for 9/16 2015; (4) whether ecotype is recorded anywhere.
    - Full write-up: `docs/sighting_correlation_sep2015.md`.

41. **OCTOBER 2015 — 31 CONFIRMED ORCA CALLS, the campaign's largest month. One recording holds
    TEN (Aug 31 2026).**
    4,293 files, **502,871 windows**, 698.38 h = **93.9% coverage**, 38.0 min embed at 220.5 win/s.
    DB `MARS_20151001_20151031_32kHz_norm`. **Oct 18 absent entirely**; Oct 17 and 19 partial (one
    clean 44.2 h dropout from 19:31 on the 17th). Floor detections: v4 151, v10 134.
    - **v10 cleared its threshold 11 times (3 above 3.00)** vs September's 4 — and **v4 16 times**.
      The two models agreed closely on ranking; both put `20151026_084928` @320 s first
      (**v10 3.732 — highest score in the campaign**).
    - **PASS 1: 16 clips, 6 min → 15 orca + 1 dolphin (94% precision).** PASS 2 (v10 ≥0.20 on the
      episode dates, 28 clips, 20 min) → **16 more orca**, 4 dolphin, 1 humpback, **8 unlabeled**.
      Total **31 orca, 5 dolphin, 1 humpback_song** (the campaign's first humpback label).
    - **⚠️⚠️ BOTH EPISODES ARE THE SAME DATE IN LOCAL TIME.** The UTC filenames split them across
      Oct 26 and Oct 27, but in PDT: `20151026_07…09…` = **00:14–02:13 PDT Oct 26**, and
      `20151027_055928` = **23:03–23:05 PDT Oct 26**. So Oct 26 local holds **29 of the 31 calls**,
      in two bouts ~21 h apart. Only one call (00:15 PDT Oct 27) falls on the 27th.
      **This is the second time the UTC/local distinction has reshaped an episode — see #40.**
    - **THE MAIN ENCOUNTER — Oct 26, 00:14–02:13 PDT, 25 calls over 2 h**, with a clear profile:
      1 call at 00:14, 1 at 00:29, 1 at 00:31, then **4 at 01:11–01:18**, **4 at 01:30–01:34**,
      **3 at 01:51–01:54**, **TEN at 02:03–02:05**, then 1 at 02:13 and silence. Approach, peak,
      departure. **`MARS_20151026_085928` holds 10 orca calls in 3 minutes — the densest
      vocalisation the campaign has found** (September's best was 6 in one recording).
      Note `073928`/`074928`/`075928` are empty: a quiet interval mid-encounter, or the animals
      moved out of range and back.
    - **MULTI-SPECIES SCENE.** `085928` also holds **2 dolphin_call** (02:01, 02:04 PDT) among its
      10 orca calls; a **humpback_song** sits at 01:30 PDT in `082928` during the build-up. Three
      species vocalising inside a two-hour window.
    - **⚠️ TOOL LIMITATION FOUND: THE REVIEW UI CANNOT EXPRESS MULTI-LABEL WINDOWS.** Duane left
      **8 clips unlabeled** because they appeared to contain **more than one vocalisation** (orca +
      dolphin, or orca + humpback) in the same 5 s window. The radio button forces one class, so the
      only honest options were "pick one, which is wrong" or "skip, which loses the information."
      He chose to skip. **The specific recordings were not noted and are NOT worth re-listening to
      recover** — the structural picture is established, and those windows become worth revisiting
      only once multi-label support exists.
      - **The DB can already represent this.** `annotations` is one row per label, and the scoped
        DELETE fix (#32) means multiple rows per window are now safe. **The blocker is the UI**, not
        the schema. Options: checkboxes, a `mixed` class, or simply allowing two clicks.
      - This matters beyond bookkeeping: during a predation event on dolphins, **overlapping
        vocalisations are exactly what you would expect**, so the tool is blindest precisely where
        the science is most interesting.
    - **Two isolated singles** outside the episodes: Oct 7 03:15 PDT (v10 2.475) and a dolphin on
      Oct 5. **The Oct 7 window is at offset 5 s, near a file start** — worth a glance for boundary
      artefact, though it scored well above threshold.
    - **THE 80 PADDED SUB-5s WINDOWS SCORED NOTHING.** Oct 5's recorder thrashing produced 80 files
      shorter than one analysis window (79 of them in a 90-minute block), each yielding one mostly
      empty window. **Zero detections from either model in that block.** So the open question from
      #30 has a first data point: near-empty windows tend to score LOW, and excluding them at
      inference is not urgent.
    - **Review effort: 26 min total** (6 + 20). Pass 2 ran ~43 s/clip, same as September's pass 2 —
      marginal clips are consistently slower than clear ones.

42. **OCTOBER 2015 SIGHTING CORRELATION — both detection days fall in the small hours FOLLOWING a
    sighting day (Aug 31 2026).**
    Source: Monterey Bay Whale Watch, *Marine Mammal Sightings List October 2015*
    (`montereybaywhalewatch.com/sightings/slst1510/`). **⚠️ SAME COPYRIGHT RESTRICTION AS #40 — do
    NOT commit the sightings data to this PUBLIC repo without Nancy Black's written permission.**

    | Sighting (local) | Our confirmed orca |
    |---|---|
    | **10/6 all day — 15 Killer Whales** | **Oct 7, 03:15 PDT** (the following night) |
    | 10/11 a.m. + p.m. — 6 KW, **"predation event"** | none |
    | 10/14 a.m. — 6 KW; p.m. — 1 KW (**"Stumpy"**) | none |
    | 10/23 p.m. — 3 KW | none |
    | **10/25 all day — 1 Killer Whale (CA49B)** | **Oct 26, 00:14–02:13 PDT** (the following night), then again 23:03 PDT |

    - **NEITHER of our detection days is itself a sighting day — both are the NIGHT AFTER one.**
      Consistent with animals remaining in the area overnight and vocalising when no boats are out.
      Combined with September's Episode B (also overnight, also unsighted), **the campaign is
      repeatedly finding activity in exactly the window the visual record cannot cover.**
    - **CA49B is the first individually-identified animal in either month's log**, named on 10/25 —
      the day before our largest encounter. **Worth asking Josh McInnes / CKWP whether CA49B or
      associates were photographed again on 10/26.**
    - **THREE sighting days produced no acoustic detection — including the 10/11 predation event.**
      Same three candidate explanations as #40 (range / silence / recall), still unseparated.
      **Across September and October combined: killer whales sighted on NINE days, acoustically
      detected on ZERO of them** — every one of our five detection days is a non-sighting day.
      That is a strong and slightly uncomfortable pattern, and it sharpens the follow-up: **run the
      low-threshold pass-2 protocol on the sighting days that produced nothing** (9/1, 9/11, 9/14,
      10/6, 10/11, 10/14, 10/23, 10/25). If calls turn up, recall is the dominant explanation; if
      not, range and silence are.
    - **Photo captions in the October list are dated Oct 5, Oct 11 and Oct 25** ("Blackfin and Killer
      Whales", "Killer Whale tail throw" — both Oct 5), **but the 10/5 sightings rows list no killer
      whales.** Either the captions are dated differently from the trips or the row is incomplete.
      **Worth querying**, since 10/5 would otherwise be another sighting day.
    - **October ecotype note:** the 10/11 "predation event" and the 10/14 "Stumpy" and 10/25 "CA49B"
      identifiers are consistent with Bigg's, which is what `orca_v10` is trained on — but as in #40,
      **ecotype is not stated in the source.**

43. **`coverage_histogram.py` was computing `expected_windows` with the WRONG RULE — fixed
    Aug 31 2026.**
    It used `ceil(duration/5)`; the confirmed adapter rule is **`max(1, floor(duration/5))`** (#30).
    A full 600 s file gives 120 either way, so the error only appears on months with short files —
    but every coverage CSV written Aug 30–31 2026 has a slightly high target:

    | Month | tool said (ceil) | correct | DB actually held |
    |---|---:|---:|---:|
    | 2015-09 | 517,992 | **517,983** | 517,984 |
    | 2015-10 | 502,893 | **502,869** | 502,871 |

    - This is why October's embed "looked" 22 windows short when it was actually **+2** — the
      discrepancy was in the reference, not the data. **`tools/audit_window_counts.py` is the
      authority**; the coverage tool's column is a convenience.
    - Fix: a documented `windows_for()` helper, validated against all twelve known duration cases
      including sub-5 s files. **Regenerate the 2015-07…2015-10 coverage CSVs after pulling.**

44. **★ RANGE, NOT RECALL — a public X post locates the 10/11 predation event ~24-28 km from the
    hydrophone. First hard evidence separating the false-negative explanations (Aug 31 2026).**
    **@MBayWhaleWatch (GoWhales) on X, posted Oct 12 2015:** *"Another dolphin toss! Killer Whales
    hunted Common Dolphins just a few miles from Monterey Harbor 10/11/15 #Monterey"*, with video.
    **This source is PUBLIC and citable** — unlike the MBWW sightings lists (#40, #42), it carries no
    redistribution restriction, so it CAN go in this repo with attribution.
    - **THE GEOMETRY.** MARS node = 36°42.75′N 122°11.21′W, ~891 m depth. Monterey Harbor =
      ~36.605°N 121.888°W. **Harbor → MARS = 29.2 km (15.8 nmi / 18.2 statute miles).** So "a few
      miles from Monterey Harbor" puts the animals **~24-28 km from the hydrophone**; even a generous
      8 miles offshore still leaves ~16 km.

      | miles from harbor | km to MARS (toward) | km to MARS (away) |
      |---:|---:|---:|
      | 1 | 27.6 | 30.8 |
      | 3 | 24.4 | 34.0 |
      | 5 | 21.2 | 37.3 |
      | 8 | 16.3 | 42.1 |

    - **CONCLUSION FOR 10/11: RANGE, not recall and not silence.** The animals were actively hunting
      and vocal enough to be worth filming, but they were **inshore in the southern bay while MARS
      sits far out in the canyon.** Of the three candidate explanations in #40/#42 — (a) range,
      (b) Bigg's acoustic crypsis, (c) detector recall — **(a) is demonstrated for this day.**
    - **⚠️ THIS REFRAMES THE WHOLE NINE-DAYS-ZERO-DETECTIONS PATTERN.** It is not obviously a
      pipeline failure; it may be largely **geometry**. The MBWW list covers the **whole Monterey Bay
      region**, including waters MARS cannot plausibly reach, and whale-watch boats work grounds that
      are typically far closer to shore than the node. **A single hydrophone 29 km from the boats'
      usual grounds should not be expected to hear most inshore activity.**
    - **THE QUESTION CHANGES.** Not *"why did we miss nine sighting days?"* but **"what is the
      effective detection radius, and which sightings fall inside it?"** That is answerable if
      location data exists per sighting — and it converts the false negatives from an embarrassment
      into a **calibration dataset for detection range.**
    - **IT ALSO STRENGTHENS THE OVERNIGHT DETECTIONS.** Animals heard at 00:14 PDT from a node 29 km
      off the harbor were either much closer than the boats' usual grounds or calling loudly. Either
      way the acoustic record is carrying information the visual record did not have.
    - **ACTIONS:**
      1. **Ask J. Ryan and Josh McInnes for sighting LOCATIONS, not just dates.** Even a rough
         position per encounter lets us compute range for every sighting day and test this properly.
         **This is now a more valuable ask than the dates themselves.**
      2. **Search @MBayWhaleWatch / GoWhales on X for other located encounters.** If they routinely
         posted position detail, that is a **public, unrestricted source of exactly the missing
         variable** — and it sidesteps the copyright problem entirely.
      3. Re-frame the planned pass-2 test on sighting days: it is now a test of **(b) vs (c)** for
         days where the animals were plausibly in range, not a blanket recall audit.
    - **CAVEAT:** one located event does not establish that all nine misses were range-limited.
      10/11 is demonstrated; the rest are inferred by analogy until locations exist.

45. **NOVEMBER 2015 — LARGEST ENCOUNTER YET, PASS 1 COMPLETE (Sep 1 2026). Pass 2 pending.**
    3,685 files, **441,598 windows**, 613.33 h = **85.2% coverage**. Nov 15 absent; five long
    dropouts. One 1-second padded file (Nov 27, SoX EOF warning). Audit: perfect +0 match.
    DB `MARS_20151101_20151130_32kHz_norm`. Floor detections: v4 847, v10 909 — BOTH far larger
    than any previous month (Sep had 186/182, Oct 151/134). v4>v10 pattern REVERSED: v10 leads.
    - **DOMINANT SIGNAL IS HUMPBACK.** 90% of the 847/909 are humpback song firing the orca
      classifier — the known class confusion. **The key discriminator: date distribution.**
      83 of 90 v10 above-threshold windows fall on a single day, **Nov 23**, which is encounter
      structure not humpback structure. Humpback would be spread across many days.
    - **PASS 1 RESULTS (chunks 1-4 + 9, 117 clips, ~52 min total):**
      - Chunk 1 (1.21–4.66, ~12 min): **21 orca, 3 humpback, 1 unlabeled**
      - Chunk 2 (2.63–2.98, ~12 min): **25 orca, 0 humpback**
      - Chunk 3 (2.40–2.63, ~12 min): **23 orca, 0 humpback, 2 unlabeled**
      - Chunk 4 (2.18–2.39, ~8 min): **2 orca, 23 humpback** ← HUMPBACK CROSSOVER ~v10=2.4
      - Chunk 9 (1.18–3.48, ~8 min): **17 orca, 0 humpback** (Nov 28 + Nov 26 + Nov 23 tail)
      - **Chunks 5–8 SKIPPED** (scores 1.16–2.17, all Nov 23) — at the crossover threshold,
        expected to be predominantly humpback. Can be reviewed if humpback labels are needed.
      - **DB totals after pass 1: 88 orca_call, 3 humpback_song** — campaign's largest month.
    - **PASS 2 FINAL TOTALS (Sep 2 2026) — REVIEW COMPLETE FOR NOW:**

      | Session | Clips | Time | Orca | Hump | Dolphin | Other | Skip |
      |---|---|---|---|---|---|---|---|
      | Pass 1 chunks 1-3 | 75 | 36 min | 69 | 3 | 0 | 0 | 3 |
      | Pass 1 chunk 4 | 25 | 8 min | 2 | 23 | 0 | 0 | 0 |
      | Pass 1 chunk 9 | 17 | 8 min | 17 | 0 | 0 | 0 | 0 |
      | Pass 2 Nov 28 | 24 | 10 min | 21 | 1 | 0 | 1 | 1 |
      | Pass 2 Nov 22-23 chunk 1 | 25 | 10 min | 23 | 2 | 0 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 2 | 25 | 11 min | 23 | 1 | 0 | 0 | 1 |
      | Pass 2 Nov 22-23 chunk 3 | 25 | 9:45 | 23 | 2 | 0 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 4 | 25 | 8:15 | 25 | 0 | 0 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 5 | 25 | 6:17 | 24 | 1 | 0 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 6 | 25 | ~15 min* | 15 | 1 | 6 | 0 | 3 |
      | Pass 2 Nov 22-23 chunk 7 | 25 | ~12 min | 15 | 3 | 3 | 0 | 5 |
      | **TOTAL** | **341** | **~134 min** | **257** | **37** | **9** | **1** | **13** |

      *chunk 6 was reviewed accidentally via a stale Nov server during the Aug re-review incident (#48).
      The labels are correct (reviewed by ear); only the session context was confused.

      **DB final (pass-2 stop point): 236 orca_call, 12 humpback_song, 9 dolphin_call, 1 other.**
      Wait — DB shows 236 orca which is 21 fewer than the table total of 257. The difference is
      the 21 annotations that were overwritten during the Aug/Nov server confusion (#48) — those
      windows were re-labeled as dolphin. The DB count of 236 is authoritative.

      **⚠️ PASS 2 CHUNKS 8-24 SKIPPED (scores 0.20-1.47).** By chunk 7, the soundscape had
      become a dense multi-species cocktail party — humpback, dolphin, and orca vocalizing
      simultaneously, with 5-second windows often containing multiple species. Further review
      at lower scores would add calls within the same known time window (Nov 22-23 18:00-01:00
      local) without revealing new temporal structure. Decision: stop here. Chunks 8-24 remain
      available if humpback labels are needed for classifier training.
    - **CAMPAIGN'S NEW HIGH SCORE: v10 = 4.662** (`MARS_20151123_061447` @480s), beating Oct's
      3.732. Top 8 windows all exceed 3.7. `MARS_20151123_062447` alone contributes 4 windows
      scoring 3.541–4.118 — a single 10-minute recording with extraordinary density.
    - **EPISODE A — Nov 22–23 local (PST=UTC-8), overnight:** `032447` through `083447` UTC
      = **19:24 PST Nov 22 to 00:34 PST Nov 23**. More than 5 hours of continuous high-scoring
      activity. 83 of 90 above-threshold windows on this one day.
    - **EPISODE B — Nov 28:** `114409` through `161409` UTC = **03:44–08:14 PST**. Six windows
      above threshold, scores 2.367–3.478. Clean second encounter.
    - **PASS 2 PARTIAL — IN PROGRESS (Sep 1 2026).** Sessions completed so far:

      | Session | Clips | Time | Rate | Orca | Hump | Other | Skip |
      |---|---|---|---|---|---|---|---|
      | Pass 1 chunks 1-3 | 75 | 36 min | 29 s/clip | 69 | 3 | 0 | 3 |
      | Pass 1 chunk 4 | 25 | 8 min | 19 s/clip | 2 | 23 | 0 | 0 |
      | Pass 1 chunk 9 | 17 | 8 min | 28 s/clip | 17 | 0 | 0 | 0 |
      | Pass 2 Nov 28 (24 clips) | 24 | 10 min | 25 s/clip | 21 | 1 | 1 | 1 |
      | Pass 2 Nov 22-23 chunk 1 | 25 | 10 min | 24 s/clip | 23 | 2 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 2 | 25 | 11 min | 26 s/clip | 23 | 1 | 0 | 1 |
      | Pass 2 Nov 22-23 chunk 3 | 25 | 9:45 | 23 s/clip | 23 | 2 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 4 | 25 | 8:15 | 20 s/clip | 25 | 0 | 0 | 0 |
      | Pass 2 Nov 22-23 chunk 5 | 25 | 6:17 | 15 s/clip | 24 | 1 | 0 | 0 |
      | **Total so far** | **266** | **~106 min** | | **227** | **10** | **1** | **5** |

      **DB total: 227 orca_call, 10 humpback_song, 1 other** as of Sep 1 2026.
      **⚠️ PARTIAL — chunks 6-24 of Nov 22-23 pass-2 (scores 0.20-1.65, ~450 clips) still pending.**
      At 96% orca rate through chunk 5, expect another 50-100 confirmed orca when complete.
      Precision has held remarkably: 96% orca through v10=1.66, with chunks 4-5 at 96-100%.
    - **HUMPBACK-ORCA COCKTAIL PARTY (D. Edgington's observation, Sep 1 2026):** Some Nov 23
      spectrograms show both humpback harmonics AND orca low-frequency elements simultaneously
      in the same 5-second window. These are left unlabeled (radio button cannot express
      multi-label). **Not useful for training but ecologically significant:** multiple species
      vocalizing simultaneously from a hydrophone 900m deep in Monterey Canyon. This problem
      recurs and sharpens the case for multi-label UI support (finding #41).
    - **ANNOTATION TIMINGS (running record):**
      - Chunks 1–3: ~12 min each (~29 s/clip) — high-confidence orca, some careful listening
      - Chunk 4: ~8 min (~19 s/clip) — humpback, faster to dispatch
      - Chunk 9: ~8 min (~28 s/clip) — mixed, Nov 28 episode required care
    - **Some calls were weak** — detectable by trained ear only, below any publishable
      threshold. Reinforces that operational recall (~17% at v10's threshold) understates
      actual presence. The zoom-in pass exists precisely to recover these.

46. **★ FOUR-MONTH SIGHTING PATTERN — every acoustic encounter is overnight, every daytime
    sighting produces no detection (Sep 1 2026).**
    Across September, October, and November 2015 — three consecutive months with both acoustic
    data and MBWW sighting lists:

    | Month | Sighting days | Acoustic detection on sighting days | Overnight acoustic |
    |---|---|---|---|
    | Sep 2015 | 4 | 1 (9/16, 17 min after predation event) | 2 (9/27–28) |
    | Oct 2015 | 5 | 0 | 2 (10/7, 10/26) |
    | Nov 2015 | 3 (11/5, 11/12, 11/13) | 0 | 2 (11/22-23, 11/28) |

    **November sighting days:** 11/5 (20+5 KW), 11/12 (7 KW), 11/13 (7 KW). No acoustic
    detection on any of them. 11/5 had only 5.67 h coverage — partly a gap; 11/12 and 11/13
    had good coverage (~18 h and ~24 h) and are genuine misses.

    **Nov 2015 source:** MBWW sightings list `montereybaywhalewatch.com/sightings/slst1511/`.
    ⚠️ SAME COPYRIGHT RESTRICTION — do not commit data to the public repo.

    **What this pattern means:**
    - **Bigg's acoustic crypsis during daylight hunting** — they go quiet when prey has
      acute hearing; a hydrophone may simply not hear them even when boats can see them
    - **Range geometry** — Oct 11 demonstrated animals can be "in the bay" and 24–28 km from
      MARS (finding #44); the daytime sighting grounds may simply be out of range
    - **The acoustic record fills the overnight gap the visual record structurally cannot cover**
      — this is the clearest statement of what passive acoustics adds to the survey program

    **TESTABLE:** run pass-2 low-threshold zoom-in on the nine sighting-day non-detections
    (9/1, 9/11, 9/14, 10/6, 10/11, 10/14, 10/23, 10/25, 11/12, 11/13). If calls appear,
    recall is the dominant explanation; if not, range and crypsis dominate.

47. **⚠️ PROCESS BUG — never feed a raw `_orcaval.csv` directly to the review tool (Sep 1 2026).**
    The raw inference CSV from `phase2_classify.py infer` is in WINDOW ORDER (by wid), not score
    order. `--num-results N` takes the first N rows as-is, so pointing the review tool at a raw
    `MARS_..._v10_orcaval.csv` shows the first N windows by wid — early-month low-score clips — NOT
    the top-scoring candidates. This happened twice on Aug 1 2026 re-reviewing August: the actual
    candidate (wid 255405) sat deep in the file and never appeared; the tool showed 14 unrelated
    low-score windows instead.
    - **RULE: always build the review set through a sorting script** (`build_pass2.py` or the inline
      chunking code), which does `sort(key=score, reverse=True)`. Every month built that way
      (Sep/Oct/Nov) was correct. Only the August shortcut — raw CSV — was wrong.
    - To review one specific window, filter by wid explicitly (as done for the Aug candidate).

48. **⚠️ TWO GRADIO HAZARDS confirmed the hard way (Sep 1 2026).**
    - **A crashed review server leaves the browser on the OLD session.** An August review launch hit
      CUDA out-of-memory and never started, but the browser was still connected to a live November
      chunk-6 server on port 7878. 25 clips got labeled into the NOVEMBER DB while Duane believed he
      was reviewing August. No data was corrupted (the 25 were real Nov 23 clips, correctly labeled
      by ear), but the intent/reality mismatch is dangerous. **MITIGATION: the review launcher should
      refuse to start if 7878 is already bound; and ALWAYS `nvidia-smi` before launch (a crashed
      server holds GPU memory — that OOM was a stuck prior server).**
    - **The review GUI never loads existing labels** — buttons are always blank even for
      previously-labeled windows. You cannot see your prior call without querying the DB. This makes
      re-review error-prone. **WISHLIST: `--show-existing-labels`.**
    - **CHECK THE HEADER before labeling.** The Gradio header shows the filename/date. A glance
      ("does this say the month I expect?") would have caught the Aug/Nov mix-up immediately. Add
      to the review checklist.

49. **`tools/plot_diel_vs_sightings.py` — time-of-day scatter with per-day civil-twilight night
    band (Sep 2 2026).** Diel plot: day-of-month (x) vs. local time-of-day (y), confirmed orca
    calls in green, surface sightings in blue (sized by count), night region shaded using civil
    twilight computed PER DAY from the NOAA solar algorithm (embedded, no network/deps). Built
    because the per-day bar version was unreadable to J. Ryan; this makes the diel pattern (#46)
    legible at a glance.
    - **Reads confirmed calls STRAIGHT FROM THE DB** (recording-start + window-offset, converted to
      local). This is the authoritative source — see the correction below.
    - **Sightings are passed on the command line** (`--sighting DAY HOUR COUNT`, repeatable), NOT
      read from any committed file, because MBWW/CKWP sighting data is copyright. The tool is safe
      for the public repo; the FIGURES it produces plot copyrighted counts and are INTERNAL ONLY.
    - **`--utc-offset` is required, not guessed** — pass -7 (PDT) or -8 (PST) per month so the DST
      boundary can't silently corrupt a plot.
    - Location defaults to the MARS node (36.7125 N, 122.1868 W).

50. **⚠️ TIMESTAMP CORRECTION — the DB is authoritative; earlier hand-transcribed call times were
    approximate (Sep 2 2026).** When first building the diel plots, Sep/Oct/Nov call times were
    typed from memory/notes and the November panel was worse — it plotted 43 FABRICATED
    placeholder points instead of the real calls. Pulling times straight from each DB corrected all
    of it. The confirmed COUNTS were always right (from the DB); only the plotted TIMES were off.

    | Month | Confirmed orca (DB) | Night | Day | Notes |
    |---|---:|---:|---:|---|
    | Aug 2015 | 0 | – | – | candidate reclassified dolphin (#33) |
    | Sep 2015 | 18 | 17 | 1 | the 1 day call = Sep 16 16:47, 17 min after logged predation (#40) |
    | Oct 2015 | 31 | 31 | 0 | Oct 26 runs ~00:14-02:13 AND ~23:05 local, a few crossing to Oct 27 |
    | Nov 2015 | 236 | ~230 | ~6 | review stopped at chunk 7; 6 day calls are Nov 28 post-dawn; Nov 26 cluster of 3 |

    - **Correction to #41:** the October encounter is NOT a single early-morning block. Local times
      are ~00:14-02:13 on Oct 26, then again ~23:05 on Oct 26, with a couple after midnight into
      Oct 27. All still night.
    - **Correction to #45:** November has a small **Nov 26** cluster (3 calls ~01:00 local) in
      addition to the Nov 22-23 and Nov 28 episodes — not previously noted.
    - **RULE (reinforces #47):** for any per-call timing, read the DB via the diel tool or
      `label_summary.py`. Never hand-transcribe call times into a figure or finding.

51. **DIEL PATTERN QUANTIFIED — 264 of 270 confirmed orca calls (Aug-Nov 2015) are at night
    (Sep 2 2026).** Combining all four months from the DB: Aug 0, Sep 18, Oct 31, Nov 221 = 270
    confirmed orca. **Night: 263 (17+31+215). Day: 7 (1 Sep + 6 Nov-28-post-dawn).** ~97.4% of all
    confirmed calls fall in civil-twilight night. Every MBWW killer-whale sighting (Aug-Nov) is
    daytime. **The single strongest daytime overlap — Sep 16, orca call 17 min after a logged
    predation event — is the exception that anchors the correlation.** This is the headline diel
    result for the J. Ryan / Nancy Black collaboration: passive acoustics and daytime visual survey
    are sampling almost disjoint parts of the diel cycle, so pooling the records is strictly
    additive. (Nov still partial; ratio expected to hold as chunks 7-24 are reviewed.)

52. **DECEMBER 2015 — COMPLETE. 4 confirmed orca calls — the seasonal decline (Sep 2 2026).**
    4,403 files, **526,642 windows**, 731.45 h = **98.3% coverage** — cleanest month of the 2015
    campaign. Dec 11 partial (79.9%), Dec 31 partial (80.0%). 7 sub-5s padded files on Dec 18
    (recorder glitch block, same pattern as Oct 5). Audit: perfect +0 match. PST (UTC-8) throughout.
    DB `MARS_20151201_20151231_32kHz_norm`. Floor detections: v4 210, v10 202.
    - **Only 6 windows above the v10 operating threshold (2.31)** — 1 above 3.00, 5 between 2.31-3.00.
      Two dates: Dec 7 (2 windows, scores 3.628 and 2.588) and Dec 20 (4 windows, scores 2.744/2.508/2.508/2.337).
    - **Pass-1 review: 6 clips, 5 min → 4 orca_call, 2 humpback_song.**
      DB final: 4 orca_call, 2 humpback_song.
    - **EPISODE A — Dec 6 local (PST):** `075815` UTC Dec 7 @570s → **23:58 PST Dec 6** (score 3.628);
      `063815` UTC Dec 7 @275s → **22:38 PST Dec 6** (score 2.588). Two calls, both overnight.
    - **EPISODE B — Dec 19 local:** `072505`/`073505` UTC Dec 20 → **23:25–00:35 PST Dec 19-20**.
      4 candidate windows, 2 confirmed orca. The Dec 20 @540s window was **orca call in a 30-second
      context of humpback song** — another multi-species co-occurrence (finding #41, #45).
    - **THE SEASONAL DECLINE IS CONFIRMED.** Nov 2015: 236 orca. Dec 2015: 4.
      The 2015 campaign is complete:

      | Month | Orca | Coverage | Notes |
      |---|---:|---:|---|
      | Jul 2015 | 0 | 77.96 h | partial deployment |
      | Aug 2015 | 0 | ~630 h | candidate reclassified dolphin |
      | Sep 2015 | 18 | 99.9% | two episodes; first external validation |
      | Oct 2015 | 31 | 93.9% | campaign high v10=3.732 (at time) |
      | Nov 2015 | 236 | 85.2% | new campaign high v10=4.662; peak month |
      | Dec 2015 | 4 | 98.3% | sharp seasonal decline |
      | **2015 total** | **289** | | **first complete campaign year** |

    - **2016 is next** — expected to be the first full 12-month year in the archive.

53. **JANUARY 2016 — 41 confirmed orca, new campaign high score 4.692 (Sep 3 2026).**
    4,023 files, **479,020 windows** (audit +4 vs floor rule — known fractional-second boundary
    cases), 665.33 h = **89.4% coverage**. Jan 21-22 ABSENT (72.4 h gap — largest single dropout
    in the campaign). Jan 8 partial (82.2%). 63 short files. 1 sub-5s padded file (Jan 8).
    PST (UTC-8) throughout. DB `MARS_20160101_20160131_32kHz_norm`.
    Floor detections: v4 284, v10 368.
    - **44 above threshold (v10 ≥ 2.31):** 40 on Jan 12, 3 on Jan 9, 1 on Jan 29.
    - **Pass-1: 44 clips, 22 min → 41 orca_call, 0 other, 3 unlabeled (too faint).** 93% precision.
      All orca, no humpback labels — clean encounter. Dolphins present in 30-second context on some
      clips but not in the 5-second window. **t-SNE of these against other months' orca detections
      would be interesting** — do they cluster with Nov 2015, or occupy a different embedding region?
    - **NEW CAMPAIGN HIGH SCORE: v10 = 4.692** (`MARS_20160112_034820` @250s), beating Nov 2015's
      4.662. Score 4.220 and 4.194 also in the top 5.
    - **JAN 12 ENCOUNTER — spans ~15 hours UTC:** `001820` through `145820` UTC Jan 12 =
      **16:18 PST Jan 11 through 06:58 PST Jan 12**. Longest single encounter in the campaign.
      High-score cluster at 02:58–06:18 UTC (18:58–22:18 PST Jan 11). Later cluster at 11:58–14:58
      UTC (03:58–06:58 PST Jan 12). Both overnight in local time.
    - **JAN 9 episode:** `020820` UTC (18:08 PST Jan 8) — 3 windows, scores 3.050/2.920/2.683.
      Overnight.
    - **JAN 29 single:** `011707` UTC (17:17 PST Jan 28) — score 2.427. Overnight.
    - **PASS-2 PENDING** on Jan 9, 12, 29 at v10 ≥ 0.20.
    - **SEASONAL CONTEXT:** Dec 2015 had 4 orca; January rebounds to 41. The Dec→Jan jump is sharp
      and suggests the animals returned to the area in early-mid January.

54. **FEBRUARY 2016 — ZERO ORCA ACOUSTICALLY; SIX SIGHTING DAYS WITH FULL COVERAGE (Sep 4 2026).**
    Inference: v4 30, v10 55. **Zero windows above operating threshold (v10 ≥ 2.31).** No review
    needed. Coverage: 68.3%, 342,029 windows. Six absent days (Feb 4-8, Feb 13) — largest single
    gap 126.8 h. PST (UTC-8) throughout.
    - **MBWW February 2016 sightings (© Nancy Black — do not commit):**

      | Local date | KW count | Coverage that day | Acoustic |
      |---|---:|---:|---|
      | Feb 12 | 13 | 66% (partial) | 0 |
      | Feb 20 | 9 | 100% | **0** |
      | Feb 21 | 6+6 | 100% | **0** |
      | Feb 24 | 8+8 | 100% | **0** |
      | Feb 25 | 6 | 100% | **0** |
      | Feb 28 | 6+6 | 100% | **0** |

    - **The late-February false negatives (Feb 20-28) cannot be explained by coverage** — all
      those days were fully recorded. Range and Bigg's acoustic crypsis are the leading
      explanations. **In late February the first gray whale calves are arriving in Monterey Bay
      — precisely the foraging context in which Bigg's orca go acoustically silent during hunts.**
    - **The interannual contrast within the season is striking:** January 2016 had 41 confirmed
      orca (new campaign high 4.692); February has zero despite orcas being repeatedly present
      and visually numerous. This is not absence of animals — it is acoustic silence of animals
      that are present.
    - **This month strengthens the crypsis hypothesis** over the range hypothesis. The Oct 11 2015
      false negative was explained by range (finding #44). Feb 2016 animals were being observed
      from boats — at inshore distances — and still produced nothing at the hydrophone 29 km out.
      Bigg's go quiet when hunting marine mammals with acute hearing. Gray whale calves fit exactly.
    - Source: MBWW Feb 2016 sightings list `montereybaywhalewatch.com/sightings/slst1602/`.
      ⚠️ SAME COPYRIGHT RESTRICTION — do not commit data to public repo.

55. **JANUARY 2016 — FINAL: 163 confirmed orca; sighting correlation complete (Sep 4 2026).**
    Pass-2 complete through chunk 5 (scores 1.02–2.71). Chunks 6-10 skipped — signals faint,
    diminishing returns below score 1.02. 5 "other" labels relabeled to orca_call after temporal
    analysis confirmed all inside orca bouts (finding #53 — now updated).

    **PASS-2 SESSION TABLE:**
    | Session | Clips | Time | Orca | Hump | Dolphin | Notes |
    |---|---:|---:|---:|---:|---:|---|
    | Pass 1 | 44 | 22 min | 41 | 0 | 0 | scores 2.31–4.69 |
    | Pass 2 chunk 1 | 25 | 6 min | 24 | 0 | 0 | scores 2.01–2.71 |
    | Pass 2 chunk 2 | 25 | 10 min | 21+5* | 0 | 0 | scores 1.77–2.00 |
    | Pass 2 chunk 3 | 25 | 7 min | 24 | 0 | 0 | scores 1.54–1.76 |
    | Pass 2 chunk 4 | 25 | 7 min | 24 | 1 | 0 | scores 1.20–1.43 |
    | Pass 2 chunk 5 | 25 | 18 min | 24 | 1 | 1 | scores 1.02–1.18 |
    | **Total** | **169** | **~70 min** | **163** | **2** | **1** | |
    *5 "other" relabeled to orca_call — all inside confirmed orca bouts

    **DB final: 163 orca_call.** 144 of 163 (88%) are at night (civil twilight PST).

    **SIGHTING CORRELATION — Jan 2016 (MBWW):**
    | Local date | KW sighted | Acoustic | Coverage | Notes |
    |---|---|---|---|---|
    | Jan 1 | 5 | 0 | 100% | 1:30 PM daytime |
    | Jan 11 PM | 100 (mother+calf) | 0 | 100% | Daytime — no acoustic |
    | Jan 15 AM | **300** (mother+calf) | **0** | **100%** | Largest single-day count — total silence |
    | Jan 16 AM | 7 (females) | 0 | 100% | Daytime — no acoustic |
    | Jan 23 AM | unknown | 0 | 100% | Daytime — no acoustic |
    | **Jan 8-9 night** | 0 sighted | **3** | 100% | Overnight only |
    | **Jan 11-12 night** | 0 sighted | **153** | 100% | 15-hour encounter — overnight |
    | **Jan 28-29 night** | 0 sighted | **7** | 100% | Overnight only |

    **Jan 15 with 300 KW sighted + zero acoustic** is the most dramatic false-negative in the
    campaign. Full coverage, massive aggregation visible from boats, complete acoustic silence
    at the hydrophone. **The animals were present and visibly numerous — they simply were not
    vocalizing.** Bigg's acoustic crypsis during gray whale calf hunting (first calves arrive
    Monterey Bay mid-January) is the leading hypothesis.

    Source: MBWW Jan 2016 sightings list. ⚠️ Copyright Nancy Black — do not commit to public repo.

56. **MARCH 2016 — 4 confirmed orca, 4 unconfirmed "other"; Mar 29 sighting correlation (Sep 4 2026).**
    4,460 files, **533,460 windows** (audit +2), 740.93 h = **99.6% coverage** — excellent month.
    3 MATERIAL overlaps on Mar 22 (recorder re-recorded wall-clock time — flagged CHECK-OVERLAP,
    embedded and noted). 1 padded file (Mar 2). DST starts Mar 13 — PST (UTC-8) before, PDT (UTC-7) after.
    DB `MARS_20160301_20160331_32kHz_norm`. Floor: v4 186, v10 219.
    - **8 above threshold (v10 ≥ 2.31):** Mar 7 (1), Mar 13 (4), Mar 29 (3).
    - **Pass-1: 8 clips, 10 min → 4 orca_call, 4 other. "This was hard."**
      D. Edgington: *"Sort of orca-like but different. I only scored orca when I was sure."*

    **LABELS BY RECORDING:**
    | Recording (UTC) | Local time | Label | Notes |
    |---|---|---|---|
    | `20160307_122233` @295s | 04:27 PST Mar 7 | **other** | Unknown ecotype — "Dolphin school?" John to advise |
    | `20160313_124227` @260s | 05:47 PDT Mar 13 | orca_call | Just after dawn, DST change day |
    | `20160313_124227` @355s | 05:46 PDT Mar 13 | orca_call | |
    | `20160313_124227` @400s | 05:47 PDT Mar 13 | orca_call | |
    | `20160313_124227` @415s | 05:47 PDT Mar 13 | **other** | Ambiguous — inside orca bout |
    | `20160329_212448` @545s | 14:33 PDT Mar 29 | orca_call | **DAYTIME — afternoon** |
    | `20160329_232448` @80s  | 16:26 PDT Mar 29 | **other** | Afternoon, uncertain |
    | `20160329_232448` @445s | 16:32 PDT Mar 29 | **other** | Afternoon, uncertain |

    - **MAR 29 SIGHTING CORRELATION:** MBWW logged 6 KW at 10 AM PDT Mar 29. Our acoustic
      record shows a confirmed orca call at 14:33 PDT — same day, 4.5 hours later, in daylight.
      This is a rare daytime acoustic detection coinciding with a sighting day. The animals may
      have been actively hunting (late March = gray whale calf season) and vocalizing in daylight.
    - **MAR 7 "OTHER":** Tonal call with different character from confirmed Bigg's orca. Time is
      04:27 PST (overnight). Could be: resident orca (different call structure), Pacific
      white-sided dolphin, or another cetacean. **Flag for John Ryan's expert ear on Tuesday.**
    - **MBWW MARCH 2016 SIGHTINGS:** Mar 1 (5 KW), Mar 10 (6 KW), Mar 23 (7), Mar 24 (5+6),
      Mar 25 (4+6), Mar 26 (6), Mar 29 (6 — correlates), Mar 30 (5). All morning trips.
      The Mar 23-28 cluster — multiple sighting days with no acoustic detection — is consistent
      with the crypsis pattern during active gray whale calf predation.
    - Source: MBWW Mar 2016 sightings list. ⚠️ Copyright Nancy Black — do not commit.

57. **APRIL 2016 — 182 confirmed orca pass-1; STRONGEST SIGHTING-ACOUSTIC CORRELATION IN CAMPAIGN (Sep 5 2026).**
    4,299 files, **514,843 windows** (audit +0 — perfect), 715.07 h = **99.3% coverage**.
    No absent days. PDT (UTC-7) throughout. DB `MARS_20160401_20160430_32kHz_norm`.
    Floor: v4 1,072, v10 1,292 — second largest month at floor level.
    New campaign high score: **v10 = 4.698** (`MARS_20160404_150039` @590s).

    **PASS-1 SESSION TABLE (all above-threshold dates):**
    | Session | Dates | Clips | Time | Orca | Notes |
    |---|---|---:|---:|---:|---|
    | Apr 19 | 20160419 | 41 | 18:30 | 37 | Overnight + 2 afternoon |
    | Apr 20 | 20160420 | 39 | 15:00 | 38 | Overnight continuation |
    | Apr 21 | 20160421 | 30 | ~12 min | 30 | Pure midday (11:41–12:41 PDT) |
    | Apr 4-5 | 20160404-05 | 38 | 13:00 | 37 | Dawn + midday; new high 4.698 |
    | Rest | 20160413,16,17,18,27,30 | 40 | 8:00 | 40 | 100% precision |
    | **Total** | | **188** | **~67 min** | **182** | |

    **DB after pass-1: 182 orca_call, 2 dolphin_call, 1 humpback_song.**
    "The orca calls sometimes sound different." — D. Edgington. April 2016 may be a different
    pod or behavioral context (active gray whale calf hunting) producing call variants.
    **t-SNE of April 2016 vs April 2018 orca embeddings is a priority comparison.**

    **ENCOUNTER STRUCTURE (UTC → PDT = UTC-7):**
    - **Apr 4:** 09:50 UTC (02:50 PDT overnight) + 15:00-15:10 UTC (08:00-08:10 PDT dawn). New high 4.698.
    - **Apr 13:** 18:30 + 21:40 UTC (11:30 + 14:40 PDT midday)
    - **Apr 16:** 09:00-10:40 UTC (02:00-03:40 PDT overnight)
    - **Apr 17:** 00:20 UTC (17:20 PDT Apr 16 evening)
    - **Apr 18:** 01:30 UTC (18:30 PDT Apr 17 evening)
    - **Apr 19-20:** 00:07-10:37 UTC (17:07 PDT Apr 18 – 03:37 PDT Apr 20) — multi-day encounter
    - **Apr 21:** 18:41-19:41 UTC (11:41-12:41 PDT) — pure midday
    - **Apr 27:** 09:34-17:44 UTC (02:34-10:44 PDT)
    - **Apr 30:** 01:56-02:16 UTC (18:56-19:16 PDT Apr 29 evening)

    **SIGHTING CORRELATION (MBWW Apr 2016 — © Nancy Black):**
    | Sighting date | KW count | Acoustic same day | Notes |
    |---|---|---|---|
    | Apr 1 | 8 | 0 | No acoustic |
    | Apr 3 | 11 | 0 | No acoustic |
    | Apr 5 | 6 | 3 | ✓ Apr 5 acoustic |
    | Apr 10 | 2-3 | 0 | No acoustic |
    | Apr 11 | 2-400 | 0 | No acoustic |
    | Apr 13 | 9-12 | 6 | ✓ Apr 13 acoustic |
    | Apr 16 | 4-15 | 17 | ✓ Apr 16 acoustic (overnight PDT) |
    | Apr 17 | 15-1520 | 2 | ✓ Apr 17 acoustic |
    | Apr 18 | 1-15 | 1 | ✓ Apr 18 acoustic |
    | **Apr 19** | **4+15** | **41** | **★ STRONG CORRELATION** |
    | **Apr 20** | **8+100** | **39** | **★ STRONG CORRELATION** |
    | **Apr 21** | **KW at sunset** | **30** | **★ STRONG CORRELATION — midday acoustic** |
    | Apr 23 | 2-7 (hunting) | 0 | No acoustic |
    | Apr 26 | 8 | 0 | No acoustic |

    **Apr 19-21 is a 3-day sighting-acoustic correlation** — both visual and acoustic records show
    orca on the same dates. The acoustic fills in the overnight hours; the sightings cover daytime.
    Together they document a near-continuous 3-day encounter. This is the most compelling example
    yet of the complementary nature of the two survey methods.

    **PASS-2 NOT DONE (by design):** Pass-2 was applied to earlier months (Sep-Dec 2015, Jan 2016)
    where above-threshold windows were sparse (6-90) and pass-1 recall was low — zoom-in recovered
    the majority of real calls. April 2016 already has 182 confirmed from 195 above-threshold windows
    (93% precision). The encounter structure is fully characterized from pass-1. Pass-2 across 15
    dates would add ~2,000+ clips for marginal gain. This selective application of pass-2 will be
    noted in the future paper methods section.
    Source: MBWW Apr 2016 sightings list. ⚠️ Copyright Nancy Black — do not commit to public repo.

58. **★ DIEL PATTERN IS CONTEXT-DEPENDENT — April 2016 is 50% day / 50% night vs 97% night in fall 2015 (Sep 5 2026).**
    April 2016: 182 confirmed orca — **91 at night, 91 in daylight**. This is a fundamental
    departure from the fall 2015 pattern where 97% of confirmed calls fell in civil-twilight night.

    | Period | Month | Confirmed orca | Night | Day | % Night |
    |---|---|---:|---:|---:|---:|
    | Fall hunting | Sep 2015 | 18 | 17 | 1 | 94% |
    | Fall hunting | Oct 2015 | 31 | 31 | 0 | 100% |
    | Fall hunting | Nov 2015 | 236 | ~230 | ~6 | 97% |
    | Winter decline | Dec 2015 | 4 | 4 | 0 | 100% |
    | Winter rebound | Jan 2016 | 163 | 144 | 19 | 88% |
    | Crypsis | Feb 2016 | 0 | — | — | — |
    | Transition | Mar 2016 | 4 | 3 | 1 | 75% |
    | **Gray whale hunt** | **Apr 2016** | **182** | **91** | **91** | **50%** |

    **The interpretation:** Bigg's orca diel calling pattern reflects prey behavior and hunting
    strategy, not a fixed biological rhythm:
    - **Fall (Sep-Nov 2015):** hunting dolphins and sea lions — prey with acute hearing, hunted
      cryptically at night → almost all calls at night
    - **Spring (Apr 2016):** hunting gray whale calves — prey with different sensory ecology,
      hunted actively in daylight during migration → calls spread across the full diel cycle

    **This is a testable prediction:** months with documented gray whale calf predation events
    should show higher daytime call rates than months dominated by dolphin/sea lion hunting.
    The sighting notes confirm: Apr 2016 MBWW logs explicitly note "hunting" on multiple dates.

    **For the future paper:** the diel analysis is not just "orca call more at night" but
    "orca diel calling reflects prey type and hunting strategy." This is a richer finding
    and more ecologically meaningful.

59. **MAY 2016 — 141 confirmed orca; new campaign high v10=5.139 (Sep 5-6 2026).**
    4,474 files, **535,379 windows** (audit +2), 743.59 h = **99.9% coverage** — best month
    in campaign. No absent days, no material overlaps. PDT (UTC-7) throughout.
    DB `MARS_20160501_20160531_32kHz_norm`. Floor: v4 929, v10 1,095.
    - **145 above threshold (v10 ≥ 2.31):** May 12 dominant (76), May 8 (31), May 5 (13),
      May 25 (8), plus scattered singles on 8 other dates.
    - **NEW CAMPAIGN HIGH: v10 = 5.139** (`MARS_20160512_065934` @95s) — significant margin
      over previous high of 4.698 (Apr 2016).
    - **Pass-1: 145 clips, ~45 min → 141 orca, 4 dolphin. Zero unlabeled.**
      May 12 reviewed in 3 chunks (25+25+26); VPN dropped during chunk34 first attempt —
      re-reviewed all 26; all saved correctly on second pass.
      "Fabulous. All orca. Very clean spectrograms and recordings." — D. Edgington (May 12)
      One clip on May 7 noted: "sounds different, almost like a rooster" — labeled orca.

    **PASS-1 SESSION TABLE:**
    | Session | Dates | Clips | Time | Orca | Dolphin |
    |---|---|---:|---:|---:|---:|
    | May 12 chunk 1 | 20160512 | 25 | 8 min | 25 | 0 |
    | May 12 chunk 2 | 20160512 | 25 | 10 min | 25 | 0 |
    | May 12 chunk 34 | 20160512 | 26 | 6 min | 25 | 1 |
    | Rest chunk 1 | 20160504-25 | 25 | 7 min | 25 | 0 |
    | Rest chunk 2 | 20160504-25 | 25 | 7:20 | 24 | 1 |
    | Rest chunk 3 | 20160504-25 | 19 | 7 min | 17 | 2 |
    | **Total** | | **145** | **~45 min** | **141** | **4** |

    **DB final: 141 orca_call, 4 dolphin_call.**
    **PASS-2 NOT DONE (by design)** — same rationale as April 2016 (finding #57).

    **MBWW MAY 2016 SIGHTINGS:** Orca sighted almost daily — May 4 (20-30), May 5 (15+),
    May 6 (22-25), May 8 (39), May 12 (40), May 13 (40), May 19 (80), May 22-27 multiple days.
    May 12 acoustic (76 clips, score 5.139) coincides with the May 12 sighting day (40 KW).
    Source: MBWW May 2016 sightings list. ⚠️ Copyright Nancy Black — do not commit.

60. **JUNE 2016 — 48 confirmed orca; new campaign high v10=5.141; Jun 19-20 crypsis (Sep 6 2026).**
    3,597 files, **431,277 windows** (audit +1), 599.00 h = **83.2% coverage**.
    Jun 4-5-7 absent (largest gap 244,238 s = 67.8 h after Jun 3). Jun 15-30 complete.
    PDT (UTC-7) throughout. DB `MARS_20160601_20160630_32kHz_norm`. Floor: v4 343, v10 327.
    - **51 above threshold (v10 ≥ 2.31):** Jun 30 dominant (31), Jun 17 (10), Jun 1 (7),
      Jun 8 (2), Jun 25 (1).
    - **NEW CAMPAIGN HIGH: v10 = 5.141** (`MARS_20160630_231742` @385s) — beating May's 5.139.
    - **Pass-1: 51 clips, 20 min → 48 orca, 3 dolphin, 1 other→orca.**
      "Soundscape sometimes complicated with ship noise, dolphins, orcas, even humpbacks."
      1 "other" label (`20160630_234742`) relabeled to orca_call — inside the Jun 30 bout.
    - **DB final: 48 orca_call, 3 dolphin_call.**

    **ENCOUNTER STRUCTURE (UTC → PDT = UTC-7):**
    - **Jun 1:** `071357`–`092357` UTC (00:13–02:23 PDT) — overnight
    - **Jun 8:** `225449` UTC (15:54 PDT) — afternoon
    - **Jun 17:** `105746`–`133746` UTC (03:57–06:37 PDT) — pre-dawn overnight
    - **Jun 25:** `120743` UTC (05:07 PDT) — pre-dawn
    - **Jun 30:** `223742`–`234742` UTC (15:37–16:47 PDT) — **afternoon daylight**

    **★ JUN 19-20 CRYPSIS — MBWW logged 400 KW (Jun 19) and 500 KW (Jun 20), both with
    full acoustic coverage, yet ZERO above-threshold windows on either day.** The Jun 19-20
    sightings noted active behaviors (tail throwing, breaching, spyhopping) — orcas were
    visually prominent and inshore. Complete acoustic silence with 400-500 animals present
    is the most extreme crypsis case in the campaign. Gray whale calf hunting in daylight
    remains the leading explanation.

    **JUN 30 AFTERNOON ENCOUNTER:** 31 above-threshold windows (15:37–16:47 PDT) — all
    daylight. Consistent with the April 2016 pattern of daytime calling during active hunting.

    **MBWW JUNE 2016 SIGHTINGS:** Jun 19: 400 KW (tail throwing, breaching, spyhopping);
    Jun 20: 500 KW (male breach); Jun 21: 7 KW; Jun 25: 5 KW (named: "Liner" male,
    "Younger"). Source: MBWW Jun 2016. ⚠️ Copyright Nancy Black — do not commit.
    July 2016: ZERO KW sightings — seasonal departure confirmed by both visual and (likely)
    acoustic records.

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
| April 23–25 2018 (cluster total) | 118 orca @≥1.16 (v4) — cluster total, NOT a single-day count. Per-day breakdown: Apr 23 = 39 @≥1.16, Apr 24 = 19 @≥1.16, **Apr 25 = 211 total / 60 @≥1.16** (Apr 25: 50 reviewed = 100% orca — CONFIRMED, D. Edgington) ✅. **Apr 23 & 24 CONFIRMED (D. Edgington, Sat Aug 23 2026):** all 58 ≥1.16 detections across both days reviewed (39 Apr23 + 19 Apr24) in 18 min — **55 confirmed orca, 2 humpback, 1 orca/humpback mix left unlabeled** (~95% orca). orca_call 319→374. Strong ship_noise in backgrounds throughout (whale-watch boats following the orcas — consistent with KSBW "record sightings" report). **April 2018 now shows SIX consecutive-ish confirmed orca days: 13, 18, 21, 23, 24, 25** — Apr 23/24 fill the gap between 21 and 25, making it a genuinely continuous multi-day presence. Earlier "318 vs 319" stray label (Apr 23 `MARS_20180423_230912`, 515-520s) is now subsumed into this full confirmation. **Poster panel 7's "Apr 23/24 detected but never listened to" is now STALE — they've been listened to and confirmed.** |
| May 12 2018 | 181 orca labels — full review 181/181 confirmed, Bigg's orca event ✅ |
| May 13 2018 | **UPGRADED (D. Edgington, Sat Aug 23 2026):** was 1 orca @≥1.16 (single-clip caveat). Reviewed all 11 detections down to floor 0.0 (scores 1.24 → 0.11) in 5.5 min — **8 confirmed orca** (1 at ≥1.16 re-confirmed + 7 new sub-threshold), **2 dolphin_call**, 1 "can't tell" left unlabeled. orca 189→196, dolphin +2. No longer a fragile single-clip day — now a properly-supported secondary event, comparable to May 14's cluster. Clips were genuinely hard (needed 5s + 30s context + replays). |
| May 14 2018 | 4 orca @≥1.16 (v4) — all 4 confirmed orca (D. Edgington), ~06–07h morning cluster ✅ (real secondary event) |
| May 16 2018 | 4 reviewed @≥1.16 (v4) — 3 confirmed orca (D. Edgington), ~15:09 cluster ✅ (1 too faint, unlabeled); figs wid9642/wid9647 |
| May 7 2018 | 1 @≥1.16 (v4) — too faint to confirm, left unlabeled |
| October 2020 | Oct 5-12 cluster confirmed — zero orca vocalizations (Bigg's orca silent during hunts) ✅. **Threshold-review July 20 2026 (J. Ryan):** the 10 detections surviving ≥1.16 are all humpback (4 newly labeled, 6 already-known humpback → 263; 1 left unknown on ambiguous 30 s context), **0 orca labels in the month**. Even at the operating threshold the residual "orca" hits are windows already known to be humpback — Oct 2020 confirmed orca-silent (true-negative / specificity result). |
| **July 2015** | **ZERO ORCA (finding #29).** First month of MARS operation (deployed 7/28 18:05; only ~78 h). v4 = 42 detections @0.0, v10 = 27; **none at >=2.31**; one concordant window @[1.16,2.31) (`MARS_20150731_222345` 335 s, v4 1.548 / v10 2.002) **reviewed and NOT orca**. 6 clips >=0.5 reviewed in 4 min (D. Edgington) = **2 dolphin_call, 4 other, 0 orca**, none unlabeled. Expected null: late July is outside the spring Bigg's window. Caveat: record ends midnight 7/31 — check early Aug 1. |
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
