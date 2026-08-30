# Brief for J. Ryan — MARS archive campaign, July–September 2015

**D. Edgington, 2026-08-30.** First three months of the full-archive pass through the
11-year MARS Pacific Ocean Sound record. All artifacts are in
`github.com/duane-edgington/perch-hoplite` (public).

---

## 1. Headline

**September 2015 contains two orca encounters — 18 confirmed calls by ear.** These are the
first confirmed orcas of the campaign. July returned zero; August returned a single
unconfirmed candidate.

| Month | Coverage | Windows | Confirmed by ear | Verdict |
|---|---:|---:|---|---|
| **2015-07** | 77.96 h (partial — deployment began 7/28 18:05) | 56,130 | 2 dolphin, 4 unidentified | **zero orca** |
| **2015-08** | 629.34 h (84.6%) | 453,123 | 13 dolphin, **1 orca (UNCONFIRMED)** | one candidate, needs your ear |
| **2015-09** | 719.43 h (**99.9%**) | 517,984 | **18 orca**, 17 dolphin, 12 ROV noise | **two encounters** |

---

## 2. The two September encounters — exact UTC times for the Soundscape Browser

Times are recording start + window offset. Each is a 5-second window Duane listened to and
called orca.

### Episode A — 2015-09-16 23:47 → 2015-09-17 06:45 (9 calls)

| UTC | Recording | Offset |
|---|---|---|
| 09-16 23:47:34 | `MARS_20150916_234019` | 435 s |
| 09-17 06:25:40 | `MARS_20150917_062020` | 320 s |
| 09-17 06:26:30 | `MARS_20150917_062020` | 370 s |
| 09-17 06:40:30 | `MARS_20150917_064020` | 10 s |
| 09-17 06:41:25 | `MARS_20150917_064020` | 65 s |
| **09-17 06:41:40** | `MARS_20150917_064020` | **80 s** |
| 09-17 06:41:55 | `MARS_20150917_064020` | 95 s |
| 09-17 06:42:00 | `MARS_20150917_064020` | 100 s |
| 09-17 06:45:15 | `MARS_20150917_064020` | 295 s |

**Six calls in the single 06:40 recording.** The 06:41:40 window scored **3.128** — the
highest score anywhere in the campaign so far, and confirmed orca. An isolated call seven
hours earlier at 23:47 may or may not be the same animals.

### Episode B — 2015-09-28 05:05 → 07:54 (9 calls)

| UTC | Recording | Offset |
|---|---|---|
| 05:05:24 | `MARS_20150928_050349` | 95 s |
| 05:06:39 | `MARS_20150928_050349` | 170 s |
| 06:10:09 | `MARS_20150928_060349` | 380 s |
| 07:18:39 | `MARS_20150928_071349` | 290 s |
| 07:20:29 | `MARS_20150928_071349` | 400 s |
| 07:21:04 | `MARS_20150928_071349` | 435 s |
| 07:23:09 | `MARS_20150928_071349` | 560 s |
| 07:26:14 | `MARS_20150928_072349` | 145 s |
| 07:54:14 | `MARS_20150928_075349` | 25 s |

Spread over ~2h50m, with four calls in the 07:13 recording. **Dolphins were also present
that morning** (06:23, 06:33, 07:43, 08:53, 09:03, 09:43, 10:52) — worth a look given
Bigg's prey preferences.

### What the calls look like
Narrow harmonic stack, roughly **1–4 kHz**, gentle upward inflection, **no energy above
~5 kHz**. Visually distinct from the dolphin clips in the same sessions, which sit at
**8–16 kHz** with steep sweeps. Figures: `figures/gradio_sep*_2015_ORCA_*.png`, each with a
JSON sidecar giving provenance.

---

## 3. Your ROV warning was right — and it explains a whole cluster

You warned that ROV servicing of the MARS science node produces a broad-band screech.
**All 12 ROV-noise labels fall in one recording: `MARS_20150916_181020`** (2015-09-16
18:10–18:20 UTC), showing dense horizontal banding across the full spectrum.

That recording alone produced 11 above-floor detections and initially looked like bout
structure. **Without your warning it would have been mistaken for a biological cluster.**

- A new label class **`ROV_noise`** now exists in the review tool (added mid-session).
- **Can you confirm an ROV service visit on 2015-09-16 around 18:10 UTC?** If the ship logs
  bear it out, that closes the cluster completely.
- **Prior months may contain unrecognised ROV noise** filed as `other` or `ship_noise` —
  September was the first time Duane recognised it. Worth revisiting if you can point at the
  service dates.

---

## 4. The number we'd most like your view on: detector recall

Of the **18 confirmed orca calls** in September, only **4 windows in the whole month** cleared
`orca_v10`'s F1-optimal operating threshold of +2.31 — and **one of those four was a dolphin**
(`MARS_20150928_063349` @355 s, v10 = 2.349, the month's second-highest score).

So at the operating point: **3 true orca calls detected out of 18 present ≈ 17% recall**, at
75% precision.

**More than half the confirmed calls were found only by a second, low-threshold pass** — the
lowest was v10 = 0.205 and is unambiguously a real call by ear.

Practical reading: the threshold is well suited to **finding encounters** (one call above
threshold is enough to know where to look) but would **badly undercount calls** if used for
call-rate statistics. That distinction matters for how we frame any seasonal analysis.

---

## 5. August's single candidate — the one clip we'd like you to hear

`MARS_20150828_212219` @325–330 s (**2015-08-28 21:27:44 UTC**), figure
`figures/gradio_aug28_2015_ORCA_325s_wid255405.png`.

- Duane: looks and sounds like an orca, but **lacks the higher frequencies** and is
  **completely isolated** — no other detection within three hours in either model.
- Scores: v10 1.406 (5th in the month), v4 0.512. **Below both operating thresholds.**
- A diagnostic low-threshold run confirmed the isolation is real: within its own recording
  the next-highest window is −1.155; across 20:00–23:00 (2,160 windows) the 99th percentile
  is −1.99.
- **Both readings remain open:** a distant orca (range strips high frequencies first, and
  Bigg's are acoustically cryptic), or a distant dolphin high-passed the same way.
- Caveat we're keeping honest about: it stands out locally partly because that evening was
  quiet. Month-wide it is mid-pack among windows that turned out to be dolphins.

**An August orca would be notable** — every confirmed Monterey Bay Bigg's event on record is
April–May.

---

## 6. July 2015 — four clips Duane couldn't identify

Zero orca, as expected for late July. Two clips were clearly dolphin; **four were real
acoustic content he could not identify** and would value your ear on:

| Recording | Offset | v4 | v10 |
|---|---|---|---|
| `MARS_20150731_222345` | 335 s | 1.548 | 2.002 |
| `MARS_20150730_095345` | 315 s | 0.589 | 0.537 |
| `MARS_20150731_064345` | 310 s | 0.572 | <0.5 |
| `MARS_20150731_221345` | 255 s | 0.587 | <0.5 |

The first is the interesting one: **both models' top hit for the month, concordant, and
judged not-orca.** If you agree, it's a clean calibration point for what a high-scoring
non-orca looks like in 2015-era audio.

---

## 7. Coverage — needed to interpret any per-day count

Per-day recording hours are committed at `results/coverage/<YYYY>-<MM>_coverage.csv` for
every processed month, because the resampled audio is deleted after analysis and the hours
would otherwise be unrecoverable.

- **2015-07:** 77.96 h. Deployment began 7/28 18:05:24; only ~78 h exist.
- **2015-08:** 629.34 h (84.6%). **8/16 has no data at all**, and five long dropouts
  (58.6 h after 8/15 04:35; 22.6 h after 8/18 22:44; 16.2 h after 8/12 23:45; 11.9 h after
  8/7 06:38; 3.6 h after 8/21 17:03). Partial days hold *contiguous* blocks, not scattered
  gaps.
- **2015-09:** 719.43 h (**99.9%**) — only 0.57 h missing, essentially all in one 32-minute
  gap on 9/1 at 15:57.

**August coverage ranges from 2.7 h to 24 h per day**, so raw per-day detection counts across
that month are not comparable. Everything seasonal must be **per hour of effort**.

---

## 8. Two recorder artifacts worth your confirmation

**Weekly clock resync.** April 2018 shows a 3 s filename-timestamp overlap at `07:59:14` on
**Apr 1, 8, 15, 22, 29**; May 2018 shows 2 s on **May 6, 13, 20, 27**. A weekly event at the
same second of day looks like a scheduled clock correction — the oscillator drifts, gets
reset, and the stamps compress while the audio stays continuous. **Is that right?** At 2–3 s
it cannot affect a 5 s analysis window, but we'd like to describe it correctly.

**Restart-and-resume.** September shows three 6–8 s overlaps each immediately followed by a
19–21 s gap (9/20, 9/24, 9/28) — a different signature from the 2018 weekly pattern.

---

## 9. Questions for Monday

1. **The August candidate** — orca or distant dolphin? (§5)
2. **The four July unknowns** — especially `MARS_20150731_222345` @335 s. (§6)
3. **ROV service on 2015-09-16 ~18:10 UTC** — confirmable from ship logs? And are there
   other service dates we should expect in 2015–2026? (§3)
4. **Humpback song/vocalisation split.** Duane's humpback labels are working-grade, not
   authoritative. Would you rather set the classification scheme **before** the campaign
   accumulates several hundred more, or after? (September produced none, but the seasonal
   expectation is that they are coming.)
5. **Multi-annotator review.** A design proposal is in `docs/multi_annotator_design.md` —
   one DB, one row per (window, annotator), reconciliation as a third `consensus:` row,
   with inter-annotator agreement (Cohen's κ) as the payoff. A destructive-delete bug that
   would have overwritten Duane's labels when you reviewed has been fixed.
6. **Recall vs precision at the operating point** (§4) — does 17% recall / 75% precision
   change how you'd want the seasonal analysis framed?

---

## 10. Reproducing any of this

```bash
git clone https://github.com/duane-edgington/perch-hoplite

# per-day labels by class + exact UTC timestamps for a month
python3 tools/label_summary.py \
    --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150901_20150930_32kHz_norm/hoplite.sqlite \
    --coverage results/coverage/2015-09_coverage.csv --markdown
```

Every figure has a JSON sidecar recording the source WAV, offset, window id, model, score,
spectrogram settings, the exact review command, and which model's score is displayed.
`CLAUDE_perch_hoplite.md` holds the numbered findings; `CLAUDE_embed.md` holds the pipeline.
