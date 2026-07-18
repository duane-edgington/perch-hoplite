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
  class. Re-annotation + a `gray_whale_call` class is the most promising lever to lift it.
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
| MARS_20180401_20180430_32kHz_norm | 219 orca + 195 dolphin + 41 humpback + 24 ship + 51 other + 54 neg |
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
13. **Gray whale annotation review** — some `humpback_song` labels may be gray whale calls. Pull all humpback-labeled clips in Gradio and have J. Ryan re-annotate as `humpback_song`, `gray_whale_call` (new class), or `other`. Gray whales are seasonally present in Monterey Bay and can overlap spectrally with humpback at low frequencies. Then retrain with gray whale as a new species class. **Now supported by evidence:** with real held-out support (n=40/47 in v1/v4), humpback_song is the weakest credible class (F1 opt ~0.55) — consistent with label contamination blurring the class. Highest-priority lever for lifting overall model quality.

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

In t-SNE: gray (negative) forms a loose cluster in lower center.
Orange-red (other) sits at the humpback/orca/negative boundary.

| Month | Key finding |
|---|---|
| April 13 2018 | 289 orca detections — confirmed Bigg's orca hunting event ✅ |
| May 12 2018 | 190 orca detections — probable event, expert review pending |
| May 14 2018 | 45 orca detections — secondary event, expert review pending |
| October 2020 | Oct 5-12 cluster confirmed — zero orca vocalizations (Bigg's orca silent during hunts) ✅ |
| April 2026 | Apr 21 dominant (101 v4 detections) — all reviewed clips are humpback FP; consistent with Bigg's orca acoustic silence. Apr 17-24 CA51A/CA50B event window shows 129 detections but no confirmed orca vocalizations. |

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
│   ├── extract_example_clips.py
│   └── review_example_clips.py
├── docs/                   — documentation and analysis
│   ├── pytorch_port_summary.md   — PyTorch Conference 2026 poster
│   ├── october_2020_analysis.md
│   ├── FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md
│   └── PROGRESS_2026-07-09.md
├── figures/                — plots and screenshots
├── phase2_classify.py      — main CLI
├── phase1_embed_torch.py   — embedding pipeline
└── README.md
```
