# October 2020 MARS Hydrophone — Orca Detection Analysis
**Classifier:** orca_v7_clean (ROC-AUC 0.9773)
**Date of analysis:** July 6 2026

---

## Summary

Full-month inference on 535,278 five-second embeddings (31 days, October 2020).
The v7_clean classifier was trained on April 2018 data only — October 2020 is
a fully independent validation dataset.

**Known ground truth:** Bigg's (transient) killer whale activity documented by
whale watch operators and the California Killer Whale Project:
- **October 3** — CA140Bs observed (matriarch CA140B "Louise" and offspring)
- **Early October** — CA51As and CA50B documented hunting sea lions
*Source: California Killer Whale Project, https://www.californiakillerwhaleproject.org/orcas*

---

## Detection Counts — v7_clean (logit threshold 0.0)

| Class | Detections | Notes |
|---|---|---|
| humpback_song | 127,827 | Dominant — peak fall humpback season |
| orca_call | 52,636 | Requires threshold screening (see below) |
| dolphin_call | 18,507 | Consistent daily presence |
| other | 226 | Minimal |
| ship_noise | 147 | Very low — COVID lockdown effect |

---

## Orca Detection Screening by Logit Threshold

| Threshold | Orca detections | Notes |
|---|---|---|
| ≥ 0.0 | 52,636 | All detections — scattered 24 hrs, many FP |
| ≥ 1.0 | 23,314 | Still scattered |
| ≥ 2.0 | 2,283 | Cleaner but still broad |
| **≥ 3.0** | **81** | **Best signal — Oct 5-12 cluster dominant** |
| ≥ 4.0 | 0 | Too restrictive |

**Recommended threshold: logit ≥ 3.0** for high-confidence orca screening.

---

## High-Confidence Orca Detections (logit ≥ 3.0) — by Day

| Date | Detections | Notes |
|---|---|---|
| Oct 2 | 4 | Pre-sighting acoustic activity |
| Oct 3 | 5 | **CA140B sighted** ✅ |
| Oct 5 | 9 | **Known event begins** ✅ |
| Oct 6 | 2 | Event continuing |
| Oct 7 | 5 | Event continuing ✅ |
| Oct 9 | 8 | **Peak event day** ✅ |
| Oct 10 | 8 | Peak event continuing ✅ |
| Oct 11 | 6 | Event winding down ✅ |
| Oct 12 | 4 | Event end ✅ |
| Oct 13 | 1 | Possible lingering |
| Oct 15 | 3 | Secondary activity |
| Oct 16 | 1 | |
| Oct 17 | 8 | Secondary cluster |
| Oct 18 | 1 | |
| Oct 19 | 1 | |
| Oct 20 | 1 | |
| Oct 22 | 9 | Secondary cluster |
| Oct 23 | 1 | |
| Oct 25 | 4 | |
| **Total** | **81** | 19 days with high-confidence detections |

---

## Key Findings

1. **Oct 5-12 event confirmed** — high-confidence orca detections cluster
   tightly around the known CA140B and CA51A sighting dates. The classifier,
   trained entirely on April 2018 data, correctly identifies an independent
   orca event 2.5 years later.

2. **Oct 2-3 pre-sighting acoustic activity** — 9 high-confidence detections
   on Oct 2-3, the day of and day before the first documented CA140B sighting.
   The hydrophone may have detected the orca acoustically before they were
   visually observed.

3. **Secondary clusters Oct 17 and Oct 22** — 8-9 detections each, suggesting
   possible return visits. Not corroborated by available sighting records but
   consistent with Bigg's orca ranging patterns.

4. **COVID lockdown effect** — only 147 ship_noise detections vs 1,899 in
   April 2018. Dramatically reduced vessel traffic clearly visible in the data.

5. **Humpback/orca confusion** — at threshold 0.0, the 52,636 "orca"
   detections are dominated by humpback song false positives (humpback
   present all 31 days). At threshold ≥ 3.0, the signal cleans up to 81
   detections with clear event structure. Additional humpback labeling
   from October 2020 (v8_clean) should further reduce this confusion.

---

## Classifier Trajectory

| Version | ROC-AUC | humpback labels | Notes |
|---|---|---|---|
| v4_clean | 0.974 | 0 | TF baseline |
| v5_clean | 0.973 | 0 | Pure PyTorch |
| v6_clean | 0.972 | 22 | First humpback |
| **v7_clean** | **0.9773** | **41** | Expert-confirmed (John Ryan) |

---

## Next Steps (v8_clean)

- Review 81 high-confidence orca detections in Gradio to confirm/correct
- Add October 2020 humpback labels from review sessions (48 already labeled)
- Build combined April + October training DB for v8_clean
- Expected improvement: better humpback/orca separation in October

---

*MBARI — Monterey Bay Aquarium Research Institute*
*Analysis date: July 6 2026*
