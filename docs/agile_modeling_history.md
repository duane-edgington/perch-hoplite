# Agile Modeling Training History — MBARI Perch-Hoplite Pipeline

This document captures how the agile modeling active learning loop worked
in practice, from initial bootstrap through classifier v6.

**Author:** Duane R. Edgington, MBARI
**Period:** June – July 2026

---

## Overview: The Agile Modeling Loop

Each classifier version represents one iteration of the loop:

```
Embed audio → Search DB → Human review via Gradio → Label →
Train classifier → Infer → Review detections → Add labels → Retrain
```

The key insight: embeddings are computed once (~37 min/month). Every
subsequent train/infer/review cycle takes minutes. A new classifier
can be trained in 30 seconds after a labeling session.

---

## Phase 1 — Bootstrap from Google Multispecies Whale Model (June 2026)

Before any Gradio review, labels were seeded from the Google Multispecies
Whale Model score CSVs (`scores_gpu/`) using `tools/convert_scores_to_labels.py`.

| Label source | Count | Notes |
|---|---|---|
| Orca POSITIVE (high-confidence) | 4,002 | From 2018, 2020, 2021 score CSVs |
| NEGATIVE (other whale species) | 242,708 | Humpback, fin, blue used as contrast class |
| Labeling tiers | 3 | High-confidence / review / negative |

**Status:** These bootstrap classifiers (v1_clean through v8_clean) were
trained on **un-normalized embeddings** — a critical bug discovered July 9 2026.
All v*_clean classifiers were retired. They are not comparable to v0–v6.

---

## Critical Fix: Low-Amplitude Normalization (July 9 2026)

**Discovery:** MARS hydrophone recordings at 891m depth have typical peak
amplitudes of 0.001–0.003. Without per-window peak normalization before
the Perch V2 model, PyTorch embeddings diverged from TF reference at
cosine 0.43–0.94 on real MARS audio.

**Fix:** Pre-normalize each 5-second window to peak 0.25 before embedding.

**Impact:** All embeddings were regenerated. All classifiers below (v0–v6)
use normalized embeddings. The v*_clean classifiers are not usable.

---

## New Era: Normalized Classifier Versions (July 2026)

### v0 — Baseline (July 9 2026)

**Training DB:** April 2018 only (`MARS_20180401_20180430_32kHz_norm`)

**How labels were obtained:**
- Gradio review of top-scoring orca detections from April 2018
- Expert review of April 13 2018 known event
- Dr. John Ryan (MBARI) confirmed April 2018 humpback labels

| Class | Count | How obtained |
|---|---|---|
| orca_call | 219 | Gradio review, April 13 2018 event |
| dolphin_call | 195 | Gradio review |
| humpback_song | 41 | Gradio review, confirmed by J. Ryan |
| ship_noise | 24 | Gradio review |
| other | 51 | Gradio review |
| negative (orca) | 54 | Gradio review — confirmed background |
| **Total** | **584** | |

**Metrics:** ROC-AUC 0.9773 · top1_acc 0.9405 · cmap 0.8810
**Training time:** 22 seconds on GB10

---

### v1 — Cross-Season (July 9–10 2026)

**Training DB:** April 2018 + October 2020 (`MARS_combined_apr2018_oct2020_32kHz_norm`)

**New labels added — October 2020:**
- Gradio review of October 2020 humpback and dolphin detections

| Class | New in Oct 2020 | Notes |
|---|---|---|
| humpback_song | 209 | October 2020 peak humpback season |
| dolphin_call | 5 | October 2020 |
| **New total** | **214** | Combined DB: 778 annotations |

**Metrics:** ROC-AUC 0.9533 · top1_acc 0.9559 · cmap 0.7999
**Note:** Lower ROC-AUC than v0 — cross-season generalization is harder

---

### v2 — Expanded April 2018 (July 10 2026)

**Training DB:** April 2018 expanded (`MARS_20180401_20180430_32kHz_norm`)

**Additional labels:** More dolphin and other examples added to April 2018 DB
to improve multi-class balance.

**Metrics:** ROC-AUC 0.9654 · top1_acc 0.9438 · cmap 0.8930
**Best for:** April and May 2018 inference

---

### v3 — Three Seasons, First Pass (July 11 2026)

**Training DB:** Apr 2018 + Oct 2020 + Apr 2026
(`MARS_combined_3month_32kHz_norm`)

**New labels added — April 2026:**
Gradio review of top-25 highest-scoring orca detections in April 2026.
All reviewed clips turned out to be humpback song misclassified as orca
— these became valuable hard negatives.

| Class | New in Apr 2026 | Notes |
|---|---|---|
| humpback_song | 17 | High-scoring orca false positives — hard negatives |
| Combined DB | 796 | annotations |

**Metrics:** ROC-AUC 0.9467 · top1_acc 0.9481 · cmap 0.7370
**Result:** April 2026 orca false positives dropped from 6,489 → 304 (−95%)

---

### v4 — Three Seasons, Best Cross-Season (July 11 2026) ✅

**Training DB:** Apr 2018 + Oct 2020 + Apr 2026 v2
(`MARS_combined_3month_32kHz_norm_v2`)

**Additional labels — April 2026:**
Second Gradio review session, top-25 v3 orca detections.
Again all humpback false positives — 8 additional hard negatives.

| Class | New in Apr 2026 | Notes |
|---|---|---|
| humpback_song | 25 total | All high-scoring orca false positives |
| Combined DB | 803 | annotations |

**Metrics:** ROC-AUC 0.9590 · top1_acc 0.9650 · cmap 0.8297
**Best cross-season classifier to date ✅**

---

### v5 — Context Embedding Experiment (July 15 2026) ❌ DO NOT USE

**Training DB:** 3-season context DB (`MARS_combined_3month_32kHz_ctx`)

**Experiment:** Replace raw 5-second embeddings with 30-second
Gaussian-weighted context averages (σ=1.5 windows, ±15s).

**Result:**
- t-SNE showed beautiful orca/humpback separation — promising
- Classifier metrics dropped significantly: ROC-AUC 0.9303, cmap 0.5945
- Context filter post-processing suppressed April 13 2018 true orca
- **Conclusion:** Bigg's orca calls are brief discrete bursts, not sustained
  bouts — context averaging hurts rather than helps

**Metrics:** ROC-AUC 0.9303 · top1_acc 0.9301 · cmap 0.5945 ← worse than v4

---

### v6 — Four Seasons, May 2018 Validated (July 16 2026) ✅

**Training DB:** Apr 2018 + Oct 2020 + Apr 2026 + May 2018
(`MARS_combined_4month_32kHz_norm`)

**Major new labels — May 2018 (all July 16 2026):**
Full review of 181 May 12 2018 orca detections — all confirmed.

| Class | New in May 2018 | How obtained |
|---|---|---|
| orca_call | 181 | **Full review of all 181 May 12 detections** — all confirmed |
| negative | 34 | Random background clips, clearly ambient ocean noise |
| dolphin_call | 2 | Identified during review |
| ship_noise | 4 | Identified during review |
| other | 6 | Mixed or ambiguous clips |
| **May 2018 total** | **227** | |

**New background labels (all July 16 2026):**

| DB | New negatives | Other new labels |
|---|---|---|
| October 2020 | 6 | 259 humpback, 7 dolphin, 39 other, 2 ship |
| April 2026 | 23 | 39 humpback, 3 dolphin, 4 other, 5 ship |

**Combined 4-month DB:**
2,094,988 windows · 1,030 annotations

| Class | Total positive | Months |
|---|---|---|
| orca_call | 395 (219+176) | April 2018 + May 2018 |
| humpback_song | 275 | April 2018 + Oct 2020 + Apr 2026 |
| dolphin_call | 193 | All months |
| other | 51 | April 2018 |
| ship_noise | 28 | All months |
| negative | 88 (54+34) | April 2018 + May 2018 |

**Metrics:** ROC-AUC 0.9499 · top1_acc 0.9409 · cmap 0.7763
**Note:** Slight drop from v4 — expected with increased cross-season diversity.
Real-world orca detection on May 2018 is the key validation test.

---

## Summary: Annotation Effort and Active Learning Value

| Version | Sessions | New labels | Key insight |
|---|---|---|---|
| v0 | 2–3 Gradio sessions | 584 | April 13 2018 event identified |
| v1 | 1 session | +214 | October 2020 humpback added |
| v2 | 1 session | +~50 | More dolphin/other balance |
| v3 | 1 session | +17 | April 2026 FPs → hard negatives; 95% FP reduction |
| v4 | 1 session | +8 | More hard negatives; best cross-season |
| v5 | 0 (experiment) | 0 | Context embedding — negative result |
| v6 | 6 sessions | +227 May + ~350 bg | May 12 2018 fully confirmed (181/181 orca) |

**Total human annotation time to v6:** Approximately 8–10 hours across
multiple sessions over 5 weeks. Total labels: ~1,030.

**Key agile modeling observation:** Each iteration was fast because embeddings
were pre-computed. The labeling sessions themselves were the bottleneck —
not computation. A new classifier took 30 seconds to train after each session.

---

## Validated Events

| Date | Detections | Review | Conclusion |
|---|---|---|---|
| April 13 2018 | 289 (v2) | Gradio + J. Ryan | Confirmed Bigg's orca hunting event ✅ |
| May 12 2018 | 181 (v4) | Full Gradio review, 181/181 | Confirmed Bigg's orca event ✅ |
| May 14 2018 | 19 (v4) | Not yet reviewed | Probable secondary event |
| October 2020 | 144 (v4) | Gradio review | Confirmed zero orca vocalizations — Bigg's orca acoustic silence during documented hunt ✅ |
| April 2026 | 323 (v4) | Gradio review top-25 | All humpback false positives — acoustic silence consistent with documented CA51A/CA50B visits |

---

*Duane R. Edgington — MBARI — July 16 2026*
*github.com/duane-edgington/perch-hoplite*
