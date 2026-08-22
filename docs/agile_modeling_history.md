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

## IMPORTANT — Seed Source Transition: Google CSV scores → Perch V2 agile detections

**This is the single most important thing to understand about how candidates were selected
for review, and it changed very early:**

- **Google Multispecies Whale Model score CSVs were used for ONE thing only: the very first
  100-clip screen** (50 high-scoring-orca + 50 low-scoring-background candidates — see the
  pre-v0 section below). That one hand-screened set of 100 trained the first real classifier.
- **From that point on — from v0 onward — candidate selection came ENTIRELY from the Perch V2
  classifier's own detections, NOT from the Google model.** Every "Gradio review of
  top-scoring detections" entry in this document, for every version v0 and later, means
  top-scoring under **the Perch V2 classifier being iterated**, run over the audio and ranked
  by its own scores. This is the agile-modeling loop: train a classifier → run it → review
  its highest-scoring (and, for hard negatives, its confidently-wrong) detections → relabel →
  retrain.
- **The Google CSVs were never used for seeding again after that initial bootstrap.** They got
  the very first model off the ground when no Perch-based classifier existed yet; once one
  existed, the project fed on its own detections exclusively.

So when reading any version's "how labels were obtained" below: unless it explicitly says
"Google Multispecies" (which only the Phase 1 bootstrap and the initial 100-clip screen do),
the candidates were surfaced by the in-progress Perch V2 classifier itself.

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

### Pre-v0 — First Human-Reviewed Gradio Sessions (D. Edgington's recollection, Aug 22 2026)

**Distinct from Phase 1's automated bootstrap above** (which seeded labels directly from
Google Multispecies Whale Model scores, no human review, used only for the doomed
un-normalized `v1_clean`–`v8_clean` classifiers) — this is the actual start of hand-verified
ground truth, all on April 2018 data.

**The very first screening — 100 annotations that trained the first model.** In the initial
Gradio tool, D. Edgington screened **100 clips: 50 high-scoring orca candidates** (top of the
Google Multispecies Model's orca scores) **and 50 low-scoring candidates** (low orca score →
material for the background/negative class). These 100 hand-screened annotations were the
original training set for the very first model.

**Then other classes were added.** Subsequent Gradio sessions expanded beyond
orca/background to the additional classes (dolphin, ship_noise, humpback).

**Then the 30-second context window was added — and with it, two expert annotators for the
first time.** The wider context view was built because a single 5-second window is too short
to identify humpback song structure. From this point the labeling split by expertise:
**D. Edgington annotated orca, ship_noise, and dolphin; John Ryan annotated humpback** (John
as the humpback expert — D. Edgington is now developing humpback expertise as well).

**Session sizes:** initial screens were **50 clips per session**. These were later **reduced
to 25 clips per session**, with a stepping increment added so that when a batch had more than
100 clips to review, it could be stepped through in **25-clip chunks** (each clip a 5-second
window).

**Total estimated effort: ~4 hours** across this pre-v0 / early labeling phase. The initial
100 grew, through the subsequent multi-class sessions, into v0's documented 584 labeled
examples (July 9 baseline below).

### v0 — Baseline (July 9 2026)

**Training DB:** April 2018 only (`MARS_20180401_20180430_32kHz_norm`)

**How labels were obtained:**
- Gradio review of top-scoring orca detections from April 2018 (**top-scoring under the Perch
  V2 classifier itself — NOT the Google model; the Google CSVs were used only for the initial
  100-clip bootstrap screen, see the seed-source transition note above**)
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
| orca_call | 400 (219+181) | April 2018 + May 2018 |
| humpback_song | 275 | April 2018 + Oct 2020 + Apr 2026 |
| dolphin_call | 193 | All months |
| other | 51 | April 2018 |
| ship_noise | 28 | All months |
| negative | 88 (54+34) | April 2018 + May 2018 |

**Metrics:** ROC-AUC 0.9499 · top1_acc 0.9409 · cmap 0.7763
**Note:** Slight drop from v4 on held-out eval — expected with increased cross-season diversity.

**v6 Inference — May 2018:**

| Class | v4 | v6 | Change |
|---|---|---|---|
| orca_call | 241 | **438** | +82% ↑ |
| dolphin_call | 5,912 | 4,429 | −25% |
| humpback_song | 511 | 444 | −13% |
| ship_noise | 1,113 | 7,597 | +582% ↑ ⚠️ |
| other | 5,638 | 7,889 | +40% ↑ |

**May 12 orca detections: 181 (v4) → 237 (v6)** — classifier is more sensitive
because it was trained on 181 confirmed May 12 calls. May 14 also grew 19→58.

**⚠️ Ship noise issue:** 1,113→7,597 is a red flag. The `negative` label being
treated as a 6th class during training likely miscalibrated the ship_noise boundary.
Needs investigation before using v6 in production.

**Status:** v6 is a May 2018 specialist. v4 remains production classifier for other months.

**Pending investigation:**
- Review v6 ship_noise detections — understand miscalibration
- Run v6 on April 2018 — validate April 13 orca preserved, dolphins correct
- Run v6 on April 2026 — check if known CA51A event window improves
- Run v6 on October 2020 — validate Oct 5-7 orca cluster preserved
- Consider retraining with `negative` as label_type=2 only (not a named class)

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
| April 13 2018 | 289 (v2) | Gradio + J. Ryan | Confirmed Bigg's orca hunting event (morning) ✅ |
| April 18 2018 | 173 @≥1.16 (v4) | Gradio 25/25 (D. Edgington) | Confirmed orca bout (late morning) ✅ |
| **April 21 2018** | **40 total, 25 @≥1.16 (v4)** | **NOT YET REVIEWED** | **⚠️ ACTION ITEM — real signal (comparable to confirmed days), never named in `orca_region_scores_v4.csv`, never Gradio-reviewed. Found Aug 19 2026 while building poster FIG 4 calendar data. Do not present as confirmed OR dismissed until reviewed.** |
| April 25 2018 | **211 total, 60 @≥1.16 (v4)** | Gradio 50/50 clips reviewed (D. Edgington) | Confirmed orca (evening); separates from Apr 13 in embedding space ✅. **CORRECTED Aug 19 2026** — was mislabeled "118," which was actually the Apr 23–25 3-day cluster total (see `orca_region_scores_v4.csv`), not Apr 25 alone. 60 is the per-day ≥1.16 detection count; 50 is the number of clips actually reviewed/confirmed by ear — these are two different (correlated, not identical) numbers, don't conflate them. |
| May 12 2018 | 181 (v4) | Full Gradio review, 181/181 | Confirmed Bigg's orca event ✅ |
| May 13 2018 | 1 @≥1.16 (v4) | Confirmed by ear (D. Edgington) | Confirmed orca — single clip, weakest evidence ✅ |
| May 14 2018 | 4 @≥1.16 (v4) | Confirmed by ear (D. Edgington) | **CORRECTED July 21 2026** — all 4 confirmed orca, ~06:00-07:00 morning cluster; real secondary event, not merely "probable" ✅ |
| May 16 2018 | 3 confirmed + 1 unlabeled @≥1.16 (v4) | Confirmed by ear (D. Edgington) | Confirmed orca, ~15:09 cluster; 2 clips from the same recording sound audibly different (repertoire variation) ✅ |
| October 2020 | 144 (v4) @0.0 floor; 10 survivors @≥1.16 | J. Ryan: all 10 = humpback | Confirmed zero orca vocalizations — Bigg's orca acoustic silence during documented hunt; specificity confirmed even at operating threshold ✅ |
| April 2026 | 323 (v4) | Gradio review top-25 | All humpback false positives — acoustic silence consistent with documented CA51A/CA50B visits |

---

*Duane R. Edgington — MBARI — July 16 2026*
*github.com/duane-edgington/perch-hoplite*

---

## v6, v7, v8 — Four-Season Experiments (July 16–17 2026)

### Root cause of ship_noise inflation

Adding May 2018 to the training set caused ship_noise to inflate from ~1,278 (v4)
to ~4,500+ across all 4-season classifiers. Three fixes were attempted:

| Fix | Result |
|---|---|
| v6: rename `negative` label string → `orca_call|2` | No change — same metrics |
| v7: same as v6, SQL rename | Identical to v6 |
| v8: background clips → `other|1` positive class | Still inflated |

All three 4-season classifiers rank the April days as:
Apr 25 (365), Apr 18 (335), Apr 13 (303).
This was originally read as "April 13 buried by false positives." **Corrected July 19
2026:** expert review (D. Edgington) confirmed Apr 18 (25/25 reviewed) and Apr 25 (50/50
reviewed) as **genuine orca**, not FPs — April 2018 held a multi-day Bigg's presence, not a
one-day event (orca labels 219 → 294; see finding #14). The 4-season models do over-*rank*
the later days relative to a calibrated threshold (at v4 logit ≥ 1.16, Apr 13 is still first:
251, vs Apr 18 173, Apr 25 60 [corrected Aug 19 2026 — was miscited as "~118," the Apr 23–25
3-day cluster total, not Apr 25 alone]), but they were surfacing real activity, not fabricating it.
The earlier "buried by FPs" wording was speculation and is superseded.

### Root cause hypothesis

May 2018 and April 2018 orca calls are acoustically different enough (different
pods, different call types) that training on both simultaneously degrades
precision on each individually. The two spring events spread the orca embedding
cluster (visible in the 4-season t-SNE) and shift the ship_noise decision boundary.

**Refined July 19 2026:** the by-day t-SNE (`tools/plot_tsne_orca_by_day.py`) shows this
heterogeneity is finer than per-season — even *within April 2018*, the Apr 25 evening
encounter separates cleanly from the Apr 13 morning event in embedding space (robust across
perplexity 10/30/50, 10 recordings over ~3.5 h, confound-checked via
`tools/orca_day_recording_spread.py`). So the cluster-spreading that hurts multi-event
training is **per-encounter**, not merely per-season — a stronger version of this hypothesis.

### Classifier comparison

| Classifier | ROC-AUC | Apr 13 rank | ship_noise | Status |
|---|---|---|---|---|
| v4 | **0.9590** | **1st (289)** | **1,278** | Production through Aug 21 2026 |
| v6 | 0.9499 | 3rd (306) | 4,496 | ❌ DO NOT USE |
| v7 | 0.9499 | 3rd (306) | 4,496 | ❌ DO NOT USE |
| v8 | 0.9463 | 3rd (303) | 4,831 | ❌ DO NOT USE |
| `orca_v10.pt` (Aug 21 2026) | 0.9372* | — | — | Best current model — see "August 20-22 2026" section below for full per-class results and the *eval-set caveat |

*`orca_v10`'s aggregate ROC-AUC looks lower than v4's, but its eval set is larger/harder
(459 examples vs. v4's 296) — not a real regression. Per-class F1 is the metric that matters
here; see below. Apr-13-rank and ship_noise-count columns aren't directly comparable for
`orca_v10` since its training recipe (3-season, no May) differs from the 4-season v6/v7/v8
rows above it in this table.

### Final classifier strategy (updated Aug 21 2026)

- **`orca_v10.pt`** — current best model, trained on the fully updated Aug 21 label set
  (1,076 annotations). Not yet a formal "production" designation — see the August 20-22 2026
  section below before treating it as v4's replacement everywhere.
- **v4** — prior production classifier (April 2018 + Oct 2020 + April 2026, 803 labels);
  numbers still cited on the OCEANS poster for panels 1-9, kept internally consistent there.
- **v2** — best for April 2018 specifically (superseded in practice by v4/`orca_v10` for
  general use).
- **May 2018 orca** — use v4 inference for reference; May is now a permanent held-out test
  month, never a training input (policy decision, Aug 21 2026 — see below).
- **v5, v6, v7, v8** — all formally retired from the model lineage (v5's weights were also
  accidentally overwritten Aug 21, see below); informative experiments only, not
  production-ready.

### Pending investigation

- ~~Review April 2018 ship_noise labels (24 clips) in Gradio — confirm no errors~~ **DONE /
  SUPERSEDED Aug 21 2026:** absorbed into the ship_noise campaign (Aug 20-22 section below) —
  April 2018 ship_noise went 24→45, not just re-confirmed.
- Long-term fix: separate per-season classifiers, or larger/more balanced training set
- Gray whale annotation review (some humpback labels may be gray whale)

---

## July 17–19 2026 — Diagnosis & Validation Era

After v6/v7/v8 exhausted the "add more data" approach, the work shifted from
*training* classifiers to *diagnosing and validating* the best one (v4). Three
threads, all on normalized embeddings, all reproducible.

### Per-class F1 (July 17 2026)

Added `src/f1_metrics.py` — per-class precision/recall/F1 on the same held-out split
as cmap/ROC-AUC, folded into `eval_scores` → `.metrics.json` on every training run.
Measured for v1/v2/v4 (each reproduced its table cmap exactly):

| Class | F1 (v1/v2/v4, F1-optimal threshold) | Read |
|---|---|---|
| orca_call | ≈ 0.95 | Strong — but needs a **positive** threshold (+1.16 to +1.9); at logit 0.0 precision is only 0.75–0.84 |
| dolphin_call | ≈ 0.71–0.77 | Model-quality ceiling |
| humpback_song | **≈ 0.55** | **Weakest credible class** |
| ship_noise | (n=3 held-out) | Insufficient support — its 1.0 is an artifact |

**Key diagnosis:** once humpback had real held-out support (n=40/47 in v1/v4, vs n=5 in
v2), it emerged as the weakest class — consistent with gray-whale contamination of
humpback labels blurring the class. This makes the gray-whale re-annotation (below) the
highest-value model-quality lever, replacing the earlier assumption that dolphin was the
problem.

### Cross-month orca validation + operating threshold (July 19 2026)

`tools/run_orca_validation.sh` (v4 inference, 4 ground-truth months at logit floor 0.0)
+ `tools/score_orca_regions.py` (threshold sweep vs known regions). Result:

- False positives collapse under thresholding: Oct 2020 (confirmed silent) 144 → 1,
  April 2026 (confirmed silent) 323 → 6, across logit 0.0 → +2.0.
- Confirmed events retain: Apr 13 99% → 74%, May 12 95% → 40%.
- **Operating threshold = +1.16** (v4 F1-optimal) primary, +1.5 conservative. The default
  0.0 is unusable (144 / 323 FPs on silent months).

### Extended April 2018 — finding #14 (July 19 2026)

The by-day sweep surfaced strong, threshold-robust orca beyond Apr 13. Expert review
(D. Edgington) confirmed **Apr 18 (25/25) and Apr 25 (50/50) as genuine orca, 0 FP at
≥1.16** — orca labels 219 → 294. April 2018 was a **sustained ~2-week Bigg's presence**,
not a one-day event.

### By-day / per-encounter t-SNE (July 19 2026)

`tools/plot_tsne_orca_by_day.py` (+ `archive_tsne_by_day.sh`, `orca_day_recording_spread.py`):
Apr 25 (evening) separates from Apr 13 (morning) in Perch V2 embedding space **within the
same month** — robust across perplexity 10/30/50, spanning 10 recordings over ~3.5 h (not a
single-recording artifact). Interpretation (pod / individual / call-type / evening-vs-morning
context) is **pending direct expert listening** — Perch embeds species and collapses
within-orca variation, so this is a lead, not proof. Confound-clearing template established:
(1) same-month, (2) distinct-recording spread, (3) perplexity sweep.

### Updated pending work (superseded — see Aug 20-22 2026 section below for current status)

- ~~**Gray-whale re-annotation (#13, highest priority):** re-review humpback-labeled clips
  with J. Ryan, adding `gray_whale_call`; retrain and check whether humpback F1 lifts from
  ~0.55.~~ **CLOSED Aug 21 2026** — both April 2018 (19 clips) and April 2026 (39 clips)
  reviewed, zero gray-whale contamination found in either. Humpback F1 improved anyway
  (~0.55→0.62) via the retrain, for reasons unrelated to gray-whale contamination — see below.
- Per-class inference thresholds (from the F1 sweep): orca ~+1.5 vs ship ~+0.4 — one global
  threshold can't serve both.
- Re-score example spectrogram clips under v4 for current README caption numbers.

---

## August 20-22 2026 — Poster Push, Data Campaign, and First Retrain Since v4

Triggered by IEEE OCEANS 2026 poster preparation (Sept 21-24, Monterey). Three PIs: D.
Edgington, J. Ryan, working with an MBARI poster-production collaborator.

### April 21 2018 — the last "pending review" orca day, resolved (Aug 21)

Flagged Aug 19 while building poster calendar data: 40 total detections, 25 at ≥1.16, never
reviewed. Full 25-clip Gradio review (D. Edgington): **all 25 confirmed orca, no ambiguity, no
other sounds present.** orca_call 294→319. April 2018 now shows confirmed orca on **four**
days — 13, 18, 21, 25 — not three. Recording-spread check found only 2 recordings in a
10-minute window (23:19-23:29), unlike the other three confirmed days (5-15 recordings each
over 1-4 hours) — a genuinely different pattern, flagged as relevant if this day's t-SNE
separation is ever used as evidence of a distinct encounter (it should NOT be, on current
evidence — see below).

### Ship_noise labeling campaign — the weakest class gets real support (Aug 21)

Panel-10-style review of the poster's own admitted weak point: ship_noise had only n=3 held-out
support in every prior model, its reported F1 an acknowledged artifact. Top-25-by-score,
no threshold gate, reviewed per month:
- **April 2018:** 24→45 confirmed (+21; one clip was distinctly different — "lots of bands,"
  possibly a different vessel/motor type — labeled `other` instead).
- **May 2018:** 4→29 confirmed (+25, **100% clean**).
- **October 2020:** 2→2 (**0 new** — all 25 candidates showed audible contamination,
  correctly left unlabeled rather than forced into a category; leading hypothesis is real ship
  noise co-occurring with October's peak humpback activity in the same window).
- Project-wide ship_noise: 35→81. Within the 3-season "Option A" recipe specifically (below):
  52.

### May 2018 becomes a permanent held-out test month (Aug 21, policy decision)

Triggered by a poster-accuracy question: an early poster draft claimed "no model here trained
on May 2018," true for v4 but not for the discarded v6/v7/v8 (which DID include a May-inclusive
4-season DB — the same recipe that caused the historical ship_noise inflation). Rather than
keep qualifying that sentence, **May 2018 is now permanently excluded from all future
training**, kept purely as held-out validation. v6/v7/v8 (and the accidentally-overwritten July
15 v5 experiment) are formally retired from the model lineage. May's existing orca
confirmations (181 + 8 more) remain valid science — just never training input again.

### "Option A" retrain — first classifier trained since v4 (Aug 21)

3-season recipe matching v4 exactly (April 2018 + Oct 2020 + April 2026), retrained on the
fully updated Aug 21 labels (1,076 annotations vs. v4's 803). Two runs, 256 vs. 512 steps
(`orca_v5b.pt`, `orca_v10.pt` — v5-v9 all considered taken/retired after the naming
collision below, next model is v10+). Aggregate ROC-AUC/cmap came in lower than v4's
(0.937/0.678 vs. 0.959/0.830) — **but this reflects a larger, harder eval set (459 examples,
up from v4's 296), not a real regression.** Per-class F1 (512-step run) tells the real story:

| Class | F1 (opt) | vs. prior |
|---|---|---|
| orca_call | 0.945 | held steady (~0.95) |
| humpback_song | 0.619 | **improved** from ~0.55 |
| ship_noise | **0.800** | first-ever credible score, up from a fake 1.0 on n=3 |
| dolphin_call | 0.687 | slightly down from ~0.71-0.77 |
| other | 0.591 | now the visible weak point (grew 52→124, merging 3 seasons' worth of catch-all clips for the first time — plausibly needs its own song/non-song-style split, same idea as humpback) |

**Incident, recorded for the record:** the first run was accidentally saved as `orca_v5.pt`,
overwriting the July 15 context-embedding experiment's weights (already marked unusable;
provenance/metrics survived, only the live weights file was lost). Renamed to `orca_v5b.pt`;
new rule going forward is the next classifier is `orca_v10.pt`, no v5-v9 reuse.

### Exact label-count reconciliation (Aug 22)

A poster stat card claimed "~1,450 total labels" against a trajectory chart whose bars summed
to 873. Direct SQL count across all four current DBs: **1,336** exact — not ~1,450 (origin of
that figure unknown, likely a stale approximation). 873 (built v4) and 1,336 (everything
confirmed to date) are both correct; they answer different questions and should be stated as
such, not reconciled by rounding.

### Tooling fixes accompanying the poster push (Aug 22)

- `plot_tsne_orca_by_day.py` and `plot_tsne.py` both gained a `--dpi` flag (previously
  hardcoded at 150) for print-quality (300 dpi) figure exports.
- `plot_tsne_orca_by_day.py` refactored to accept `--confirmed-april-days` /
  `--confirmed-may-days` on the command line, replacing the old pattern of editing the
  `APRIL_DAYS`/`MAY_DAYS` constants directly in source — that editing pattern had caused a
  real figure-mismatch incident (an exploratory day list change silently altered what
  "panel 8's figure" meant for every subsequent run).
- `register_figure.py` gained a dedicated `--script` field in its provenance sidecar schema,
  separate from the free-text `--command` field, after a recurring difficulty pinning down
  exactly which script/version produced a given archived figure.

---



The agile-modeling story as a sequence of deliberate expansions, each cheap because
embeddings are computed once and every train/infer cycle takes minutes:

1. **Binary bootstrap (June 2026).** Seed labels from the Google Multispecies Whale
   (Kaggle) model scores — effectively *orca_call vs. other*. No manual review yet.

2. **Enabling fix — low-amplitude normalization (July 9).** Per-window peak-normalize to
   0.25; without it the PyTorch Perch V2 port diverged from TF on quiet MARS audio
   (cosine 0.43–0.94). This unlocked everything downstream and retired the un-normalized
   `_clean` era.

3. **First true multi-class — v0.** Five classes (orca, dolphin, humpback, ship, other)
   on April 2018; April 13 Bigg's event identified.

4. **Expand in waves (the agile core):**
   - **+ season →** v1 (add October 2020 humpback/dolphin).
   - **+ balance →** v2 (more dolphin/other; best April/May 2018).
   - **+ hard-negative mining →** v3/v4: April 2026's top orca detections were all humpback
     FPs → relabeled as hard negatives → **95% April-2026 FP reduction**. v4 = best
     cross-season.
   - **+ event/season →** v6: full review of May 12 2018 (181/181 confirmed orca).

5. **Hit the ceiling, then diagnose it (July 17–19) — the research payoff.** Adding a
   4th season (v6–v8) inflated ship_noise and reshuffled April day-rankings, so the method
   pivoted from *more data* to *measurement*: per-class F1 exposed **humpback (~0.55) as the
   real weak point** (gray-whale contamination); cross-month validation set a defensible
   operating threshold (+1.16); and embedding visualization revealed the model surfacing
   **biology** — a multi-day April Bigg's presence (finding #14) and per-encounter acoustic
   structure (Apr 25 evening vs Apr 13 morning) — while honestly bounding what it cannot yet
   resolve (pod/individual/call-type).

6. **Close the measured gaps, then retrain (Aug 20-22) — the payoff of the payoff.** The
   per-class weaknesses measured in step 5 became a punch list, not just a caveat: April 21's
   last unreviewed orca signal got resolved (25/25 confirmed), the gray-whale hypothesis for
   humpback's weak F1 got tested directly and closed (zero contamination in two full batches),
   and ship_noise — stuck at a fake 1.0-on-n=3 since the beginning — got a real, targeted
   labeling campaign. May 2018 was deliberately set aside as a permanent held-out test month
   rather than folded back in, learning from step 5's 4-season lesson instead of repeating it.
   The resulting retrain (`orca_v10.pt`) shows real per-class gains — ship_noise 0.80 F1 on
   genuine support, humpback up to 0.62 — even though its aggregate ROC-AUC looks lower on
   paper, because the eval set itself grew harder alongside the training set. The method's
   whole arc, end to end: bootstrap → normalize → multiply classes → diagnose the ceiling →
   close the specific gaps the diagnosis found.

**Poster thesis:** agile modeling on frozen Perch V2 embeddings let one expert, in ~8–10
hours of labeling over weeks, build a cross-season Bigg's-orca detector *and* use it as an
instrument that surfaces testable marine-mammal biology — with its limits measured, not
hidden, and — as of this update — those measured limits actively being closed one at a time.

---

*Duane R. Edgington — MBARI — updated Aug 22 2026*
*github.com/duane-edgington/perch-hoplite*
