# CLAUDE_inference_april_2026.md — April 2026 v10 orca scan (session record)

Session-specific record of the April 2026 v10 inference + review. Companion to the general
workflow in `CLAUDE_inference.md`. Purpose: preserve exact settings, the selected clips, and the
outcome for this one session so it is fully reproducible and its provenance is unambiguous.

**Bottom line: AMBIGUOUS result — candidates only, NOT confirmed orca. Awaiting John Ryan's
blind review.** See CLAUDE_perch_hoplite.md finding #23.

---

## Session identity

- Date: Aug 24 2026 (~07:05 local)
- Labeling session ID (from the tool): `20260824_070516_duane`  *(annotator now set explicitly —
  see the general doc; earlier Aug-23 sessions defaulted to generic `analyst`)*
- Annotator: `duane` (orca/ship/dolphin expert). **Humpback ground truth for this batch pending
  John Ryan (blind).**
- Machine: spark-ae0e, `~/perch-hoplite`, venv active
- Model: `orca_v10.pt`
- Month/DB: April 2026 — `MARS_20260401_20260430_32kHz_norm` (505,630 embeddings)
- IMPORTANT provenance: **April 2026 is a TRAINED month** (part of the 3-season recipe), NOT
  held-out. Only May 2018 is the clean hold-out (finding #18). So this scan is a
  completeness/biological-presence check, not a generalization test.

---

## Step 1 — Inference (already run earlier Aug 23)

```bash
cd ~/perch-hoplite
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \
    --labels orca_call \
    --logit-threshold 0.0 \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20260401_20260430_v10_orcaval.csv
```

Output: **278 detections at floor 0.0.** Score bands:
- `>= 2.31`: 13
- `1.16 - 2.31`: 36
- `0.5 - 1.16`: 103
- `0.0 - 0.5`: 126

---

## Step 2 — Candidate selection (high-confidence tier only)

Selected the `>= +2.31` tier (v10's own optimal orca cutoff) for the first pass — 13 clips.

```bash
cd /mnt/PAM_Analysis/perch-hoplite/results
python3 << 'PYEOF'
import csv
SRC = "MARS_20260401_20260430_v10_orcaval.csv"
OUT = "review_apr2026_v10_orca_ge231.csv"
THRESH = 2.31

rows_out, header = [], None
for row in csv.reader(open(SRC)):
    if header is None:
        header = row; continue
    if row[5] != 'orca_call':
        continue
    if float(row[6]) >= THRESH:
        rows_out.append(row)

rows_out.sort(key=lambda r: float(r[6]), reverse=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows_out)
print(f"Wrote {len(rows_out)} rows to {OUT}")
for r in rows_out:
    print(f"  {float(r[6]):.3f}  {r[2]}  {r[3]}-{r[4]}s")
PYEOF
```

### The 13 selected candidates (score / file / offset)

| score | file | offset |
|---|---|---|
| 4.173 | MARS_20260421_130000 | 545-550s |
| 3.454 | MARS_20260421_130000 | 560-565s |
| 3.391 | MARS_20260404_035000 | 40-45s |
| 3.128 | MARS_20260421_130000 | 525-530s |
| 3.023 | MARS_20260424_052000 | 475-480s |
| 2.970 | MARS_20260424_052000 | 465-470s |
| 2.942 | MARS_20260421_122000 | 450-455s |
| 2.525 | MARS_20260421_130000 | 230-235s |
| 2.432 | MARS_20260421_130000 | 515-520s |
| 2.425 | MARS_20260411_045000 | 390-395s |
| 2.410 | MARS_20260421_023000 | 245-250s |
| 2.341 | MARS_20260405_155000 | 580-585s |
| 2.335 | MARS_20260421_120000 | 285-290s |

Pre-review structural read: **April 21 dominates — 6 of 13 in `MARS_20260421_130000` (incl. a
near-consecutive 515/520/525/530 + 545/560 bout), top score 4.17, and hits across multiple
April-21 recordings (02:30, 12:00, 12:20, 13:00) over hours.** Plus a consecutive April-24 pair
(465/475s). On paper this was the most orca-looking candidate set generated — the sequence/bout
structure that was ABSENT in Oct 2020's humpback clips. That is what made it worth an ear-review.

---

## Step 3 — Gradio review launch (exact command used)

```bash
cd ~/perch-hoplite
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \
    --target-label orca_call \
    --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_apr2026_v10_orca_ge231.csv \
    --detections-offset 0 --num-results 13 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --annotator-id duane \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2026/04 \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7878 \
    > /mnt/PAM_Analysis/perch-hoplite/logs/review_apr2026_v10.log 2>&1 &
sleep 3 && tail -5 /mnt/PAM_Analysis/perch-hoplite/logs/review_apr2026_v10.log
```

URL: `http://134.89.11.107:7878` (incognito Chrome). Both 5 s window and 30 s context inspected.

---

## Step 4 — Result (AMBIGUOUS)

Tool report: `Saved 13 labels (0 unlabeled skipped).`
DB totals after — positive: `{dolphin_call: 3, humpback_song: 50, orca_call: 2, other: 27, ship_noise: 5}`.

**Finding: genuinely ambiguous, recorded as tentative — NOT confirmed orca.**
- Several clips sounded clearly orca in the 5 s window, BUT the 30 s context revealed surrounding
  humpback vocalizations (moans, upsweeps, mixed repertoire).
- Cannot distinguish from audio alone: (a) orca AND humpback both present, vs (b) humpbacks
  producing orca-like sounds within their own mix. John Ryan's caution applies directly:
  "humpbacks can sound like any animal in the ocean."
- D. Edgington marked ~2-3 as orca (DB orca_call = 2) but explicitly regards these as
  UNCONFIRMED.
- The 30 s context was decisive in flipping several apparent-orca 5 s clips to ambiguous —
  concrete support for the "5 s is insufficient; context matters" point.

**These are NOT new confirmed orca days.** Unlike May 2/3/7/29 (finding #21, clean hold-out,
14/14 unambiguous), April 2026 is a trained month with ambiguous audio. Do NOT add to any
confirmed-orca-day count or the poster on this evidence.

---

## REQUIRED NEXT STEP — John Ryan blind review

- John reviews this **same 13-clip batch** (`review_apr2026_v10_orca_ge231.csv`) **independently
  and BLIND** — without seeing D. Edgington's annotations — to avoid contaminating his judgment.
- Suggested: launch a fresh review session with `--annotator-id john` on the same CSV, ideally
  against a copy/branch of the labels so his calls are stored separately from Duane's for
  comparison. (Confirm the tool's behavior for two annotators on the same clips before running;
  if it overwrites, review into a separate DB copy or export Duane's labels first.)
- Interpretation: agreement → strong; divergence → clips are genuinely ambiguous and must NOT be
  called confirmed orca.

---

## ACTION ITEMS

- [ ] **Get April 2026 framegrabs onto the repo** (captured this session; catalog + register
      together with the Oct 2020 framegrab set — same sweep/dedupe/rename/register/push pass).
      *Later today.*
- [ ] **John Ryan blind review** of the 13-clip batch (see above).
- [ ] Decide two-annotator storage approach before John's session (separate DB/branch so blind
      comparison is possible).
- [ ] Verify/clear any pre-existing April 2026 orca stray labels if the orca_call=2 doesn't
      reconcile to Duane's marks on re-review.
