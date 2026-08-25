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
