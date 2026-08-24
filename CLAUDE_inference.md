# CLAUDE_inference.md — Inference, candidate selection, and Gradio review workflow

Reference for the standard loop: run a trained classifier over a month, select the
above-threshold candidates worth listening to, and launch a Gradio review session on exactly
those clips. Worked example uses orca_v10 on Oct 2020, but the pattern is month/model-agnostic.

Keep this as the canonical record of HOW clips are selected and the EXACT parameters used to
build a review session, so any session is reproducible.

---

## Environment

- Run on spark-ae0e, from `~/perch-hoplite`, with the venv active (`source venv/bin/activate`).
- DBs: `/mnt/PAM_Analysis/perch-hoplite/db/MARS_<YYYYMMDD>_<YYYYMMDD>_32kHz_norm/hoplite.sqlite`
- Models: `/mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt` (current best), `orca_v4.pt` (prior)
- Inference CSVs land in: `/mnt/PAM_Analysis/perch-hoplite/results/`
- Audio (for Gradio playback): `/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM>/`

Inference CSV columns: `idx, project, filename, window_start, window_end, label, logits`
(logits = the classifier's score for that window; higher = more target-class-like.)

---

## Step 1 — Run inference over a month (floor 0.0, captures every positive logit)

`phase2_classify.py infer` flags: `--db-dir/-d`, `--classifier/-c`, `--output-csv/-o`,
`--labels` (restrict to class(es)), `--logit-threshold` (default 0.0), `--plot-distribution`.

```bash
cd ~/perch-hoplite
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \
    --labels orca_call \
    --logit-threshold 0.0 \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20201001_20201031_v10_orcaval.csv
```

Fast (~3-5 s for ~500K windows). Prints total detection count.

### Quick score-band triage (decide if there's anything worth reviewing)

```bash
cd /mnt/PAM_Analysis/perch-hoplite/results
base=MARS_20201001_20201031
echo -n "total detections (floor 0.0): "
awk -F, 'NR>1 && $6=="orca_call"' ${base}_v10_orcaval.csv | wc -l
for lohi in "2.31:99" "1.16:2.31" "0.5:1.16" "0.0:0.5"; do
  lo="${lohi%%:*}"; hi="${lohi##*:}"
  n=$(awk -F, -v lo="$lo" -v hi="$hi" 'NR>1 && $6=="orca_call" && $7>=lo && $7<hi' ${base}_v10_orcaval.csv | wc -l)
  echo "  [$lo - $hi): $n"
done
```

Band meanings (orca_call, orca_v10):
- `>=2.31` — v10's own optimal orca cutoff (per-class F1). Highest-confidence; listen first.
- `1.16 - 2.31` — above the general operating threshold; review, but expect some humpback
  false positives here, especially in peak-humpback months (e.g. October).
- `< 1.16` — below operating threshold; mostly noise/humpback, low priority.

---

## Step 2 — Build the review CSV (select the clips to listen to)

Standard selection: all detections at or above the operating threshold (>=1.16), sorted by
score descending so the highest-confidence clips are reviewed first. Adjust the `>= 1.16`
threshold to taste (e.g. `>= 2.31` for only the highest-confidence set, or a lower value to go
deeper). Optionally exclude already-confirmed windows (see the precision-check variant below).

```bash
cd /mnt/PAM_Analysis/perch-hoplite/results
python3 << 'PYEOF'
import csv

SRC = "MARS_20201001_20201031_v10_orcaval.csv"
OUT = "review_oct2020_v10_orca_ge116.csv"
THRESH = 1.16          # operating threshold; raise/lower to change the review set

rows_out, header = [], None
for row in csv.reader(open(SRC)):
    if header is None:
        header = row; continue
    # cols: idx,project,filename,window_start,window_end,label,logits
    if row[5] != 'orca_call':
        continue
    if float(row[6]) >= THRESH:
        rows_out.append(row)

rows_out.sort(key=lambda r: float(r[6]), reverse=True)   # highest score first
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows_out)

print(f"Wrote {len(rows_out)} rows to {OUT}")
print()
print("Candidates (score / file / offset) — highest first:")
for r in rows_out:
    s = float(r[6])
    flag = "  <-- HIGH (>=2.31)" if s >= 2.31 else ""
    print(f"  {s:.3f}  {r[2]}  {r[3]}-{r[4]}s{flag}")
PYEOF
```

### What to look for in the candidate list BEFORE listening
- **Consecutive windows in the same recording** (e.g. 500-505s and 505-510s) suggest a real
  call *sequence/bout* — orca vocalize in bouts, not isolated windows. Strong sign of a genuine
  encounter vs. scattered single-window false positives.
- **Day clustering** (several high-score hits on adjacent days) also points to a real encounter.
- **Isolated single windows at low scores**, spread across many days, are more likely false
  positives (humpback in peak season, noise).

### Variant — precision check (review only NEW detections, excluding already-confirmed)
Used for the May 2018 v10 precision test. Same as above, but first build the set of confirmed
windows from the DB and skip any candidate already in it — so you only review detections that
are NOT yet ground-truth. Requires reading the annotations table:

```python
import sqlite3, struct
from pathlib import Path
DB = "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm/hoplite.sqlite"
con = sqlite3.connect(DB); cur = con.cursor()
rows = cur.execute("SELECT r.filename,a.offsets FROM annotations a "
                   "JOIN recordings r ON r.id=a.recording_id "
                   "WHERE a.label='orca_call' AND a.label_type=1").fetchall()
con.close()
confirmed = set()
for fn, blob in rows:
    s,_ = struct.unpack('<2d', blob)               # offsets blob = (start_s, end_s), little-endian doubles
    confirmed.add((Path(fn).name.replace('.wav',''), round(s,1)))
# then in the row loop, skip: key=(Path(row[2]).name.replace('.wav',''), round(float(row[3]),1)); if key in confirmed: continue
```

---

## Step 3 — Launch the Gradio review session on exactly those clips

`phase2_classify.py review` key flags: `--db-dir`, `--classifier`, `--target-label`,
`--detections-csv` (the review set from Step 2), `--detections-offset` (start index, 0),
`--num-results` (how many rows to load — set to the CSV row count), `--classes` (label buttons,
include `unlabeled` for "skip/ambiguous"), `--audio-dir` (playback WAVs for that year/month),
`--annotator-id <name>` (WHO is labeling — set it, don't leave the default generic `analyst`: use `duane` for orca/ship/dolphin, `john` for humpback since he is the humpback authority and his calls are the ground truth for that class), `--spectrogram-type mel --colormap viridis`, `--serve --port <N>`.

```bash
cd ~/perch-hoplite
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \
    --target-label orca_call \
    --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/review_oct2020_v10_orca_ge116.csv \
    --detections-offset 0 --num-results 16 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --annotator-id duane \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2020/10 \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7877 \
    > /mnt/PAM_Analysis/perch-hoplite/logs/review_oct2020_v10.log 2>&1 &
sleep 3 && tail -5 /mnt/PAM_Analysis/perch-hoplite/logs/review_oct2020_v10.log
```

Then open `http://134.89.11.107:<port>` in incognito Chrome.

Notes:
- `--num-results` should equal the number of rows in the review CSV (here 16).
- `--audio-dir` must point at the correct `<YYYY>/<MM>` for the month being reviewed.
- Pick an unused `--port` (7874-7877 used across Aug 23 sessions). One session per port.
- Labels auto-save on each click (reload/VPN-drop safe); "Save Labels to DB" also commits.
- Reviewer can inspect 5 s window AND 30 s context (built originally for John's humpback review —
  a 5 s window is too short to identify humpback song structure).

### When done
```bash
pkill -f "phase2_classify.py review"          # kill all review servers
ps aux | grep "[p]hase2_classify.py review"   # confirm empty
```

---

## Step 4 — Reconcile / record results

- The session prints "Saved N labels" and "DB totals — positive: {...}". Reconcile: labels
  saved should equal clips reviewed minus unlabeled-skipped; the per-class DB deltas tell you
  what was confirmed as what (orca vs humpback vs dolphin vs other).
- Record in CLAUDE_perch_hoplite.md: date, month reviewed, model, # clips, session time,
  outcome per class, any new days discovered. Always note the session time (ask if not stated).
- **Record the annotator** (`--annotator-id`) — who made the calls. Default is the generic
  `analyst`; always set it (duane / john / other). Matters for multi-annotator provenance,
  especially John's humpback ground-truth.
- **Record the labeling session ID** the tool prints (e.g. `20260824_070516_duane` —
  timestamp + annotator). It is the audit handle linking a label set back to one session.
- Remember where annotations came from (which months are in the training recipe vs held-out):
  training recipe = April 2018 + Oct 2020 + April 2026; **May 2018 is the permanent held-out
  test month** (finding #18). Hold-out status doesn't change how you review, but it changes how
  a result can be framed (only May supports a clean "generalizes to unseen months" claim).

---

## Worked selections on record (Aug 23 2026)

| Month | Model | Set reviewed | Result |
|---|---|---|---|
| May 2018 | v4 vs v10 | 196 confirmed windows (scored, not re-listened) | v10 recall 79.6% vs v4 60.7% @+1.16 (finding #20) |
| May 2018 | v10 | 14 non-confirmed @>=1.16 (precision variant) | 14/14 real orca, 0 false pos; 4 new days 2/3/7/29 (finding #21) |
| Apr 23-24 2018 | v4 | 58 @>=1.16 | 55 orca, 2 humpback, 1 unlabeled |
| Oct 2020 | v10 | 16 @>=1.16 (this doc's worked example) | pending review — Oct 4-5 cluster is the promising candidate |
