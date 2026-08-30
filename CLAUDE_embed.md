# CLAUDE_embed.md — Resampling + embedding pipeline (raw MARS → hoplite DB)

The exact process for turning raw MARS archive audio into a hoplite embedding DB ready for
`phase2_classify.py infer`. Two stages: (1) SoX resample to 32 kHz, (2) Perch V2 embedding.
This step is infrequent and easy to get subtly wrong (venv, normalization, naming), so it's
written down.

STATUS: complete (updated Aug 27 2026). Resampling script + exact nohup invocation + embed
stage all captured. Coverage-query table confirmed as `recordings`; SoX version pinned (v14.4.2).
The "fancier" concurrency-throttling resample variant is LOCATED (`tools/resample_sox_32k_batched_vol.sh`)
and is now the canonical script for the full-archive campaign. Stage 1 verification and the
window-count reconciliation rule added from the July 2015 run.

---

## Stage 1 — SoX resampling (raw archive → resampled 32 kHz WAV)

Raw MARS audio lives at `/mnt/PAM_Archive/<YYYY>/<MM>/MARS_*.wav`. Resampled output goes to
`/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM>/`.

### CANONICAL SCRIPT (full-archive campaign, Aug 27 2026 onward)

**`tools/resample_sox_32k_batched_vol.sh`** — in the repo, throttles concurrency, supports day
ranges. This is the script to use going forward.

```
./tools/resample_sox_32k_batched_vol.sh <year> <month> [start_day] [end_day] [max_jobs]
```
- `year month` required; month WITHOUT leading zero (`7`, not `07`)
- `start_day end_day` default 1–31; `max_jobs` defaults to `nproc`
- `SAMPLE_RATE` env var overrides 32000 (output dir + filename suffix follow the rate)
- Example (July 2015, the partial deployment month): `./tools/resample_sox_32k_batched_vol.sh 2015 7 28 31`

**LOG NAMING CONVENTION (adopted Aug 30 2026 — Duane's, and it is the right one):** put the
**sample rate** in the log filename, so a log read years later is self-describing and cannot be
confused with a resample at another rate:

```
/mnt/PAM_Analysis/perch-hoplite/logs/resample_<RATE>_<YYYY>_<MM>.log
```

**Canonical full invocation** — September 2015, throttled to 8 concurrent jobs (the actual command
run on spark-0626, Aug 30 2026):

```bash
cd ~/perch-hoplite
nohup ./tools/resample_sox_32k_batched_vol.sh 2015 9 1 31 8 \
  > /mnt/PAM_Analysis/perch-hoplite/logs/resample_32kHz_2015_09.log 2>&1 &

sleep 10 && head -20 /mnt/PAM_Analysis/perch-hoplite/logs/resample_32kHz_2015_09.log
```

**Always eyeball the banner** before walking away — it must read `Starting (vol 3):`. Passing
`1 31` for a 30-day month is fine; the script finds nothing on day 31 and moves on.

Progress checks while it runs:
```bash
tail -5 /mnt/PAM_Analysis/perch-hoplite/logs/resample_32kHz_<YYYY>_<MM>.log
ls /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM>/*.wav | wc -l
```
- Prints a `Starting (vol 3): ...` banner, a per-day file count, and a
  `Finished: N submitted, M failure(s)` report with any failed filenames listed. No `set -e`:
  one bad file does not kill the batch.

**`tools/resample_sox_32k_batched_novol.sh` was DELETED from the repo on Aug 30 2026.** It was
byte-for-byte identical except it omitted `vol 3`, took the SAME arguments, and wrote to the SAME
output directory with the SAME `_resampled_32kHz.wav` suffix — so **nothing in the filename or
path recorded which variant produced a file**, and the only in-band evidence was the log banner
(`Starting (vol 3):` vs `Starting (no vol):`). Across a ~130-month campaign that is a foot-gun
with no upside: **`vol 3` is the volts calibration and is mandatory** (J. Ryan — always vol 3,
clipping is not a concern for the signals of interest). Duane's reasoning for deleting rather than
keeping it: *"If I really want to have no vol in the future that should be a parameter, not two
almost identical scripts. And I don't think I ever will."* Recoverable from git history.

If a resample's provenance is ever in doubt (e.g. data produced before Aug 30 2026), compare peak
amplitude against the raw file — `vol 3` output should be ~3x raw:
```bash
sox <raw>.wav       -n stat 2>&1 | grep -i "maximum amplitude"
sox <resampled>.wav -n stat 2>&1 | grep -i "maximum amplitude"
```
Note the `_vol` script's own header comment block is a bad copy-paste of the (now deleted)
`_novol` one — it correctly says "This variant DOES apply the `vol` gain adjustment", then
continues with the novol rationale about clipping warnings. Do not trust that header; trust this
doc. (Lower priority now that there is no second script to choose between.)

### Original script (historical — produced the 2018/2020/2024/2026 data)

Script used for the earlier data (`new_32k_resample_sox.sh`), run as
`./new_32k_resample_sox.sh <year> <month>` (e.g. `./new_32k_resample_sox.sh 2024 09`).
Located in `~/gmwd/new3-12_whale_detection/gmwd/` on spark-ae0e. Takes only two args (any extra
positional args are silently ignored) and loops `seq 1 31` unconditionally, so its log carries
harmless `No such file` glob misses for days with no data. Launches one `sox` per file with no
concurrency cap. Same SoX effect chain, so **same output bytes** as the batched vol variant.

**Environment for the resampling run (pin these for the FAIR reproducibility bundle):**
- **SoX version: `SoX v14.4.2`** (`sox --version`), binary at `/usr/bin/sox`. SoX output can
  differ across versions/builds, so this version is part of the reproducibility record.
- Run from `~/gmwd/new3-12_whale_detection/gmwd/` with **that directory's own venv active**
  (`source venv/bin/activate` from within that dir) — a separate venv from both the perch-hoplite
  venv and the perch-pytorch (embedding) venv. Note: SoX itself is a system binary, so the venv
  matters for any surrounding Python tooling, not for sox's output.

**Exact nohup invocation used for the Sept 2024 run** (from `~/gmwd/new3-12_whale_detection/gmwd/`
on spark-ae0e; note the month is passed WITHOUT a leading zero — `9`, not `09`):
```bash
nohup ./new_32k_resample_sox.sh 2024 9 > logs/nohup_resample_2024_09.out &
```
Log went to `logs/nohup_resample_2024_09.out` (relative to the script's dir). Resampling a full
month takes ~1 day on spark.

Note: a "fancier version that throttles the number of concurrent processes" also exists somewhere
but was NOT the one used for the current set — the version below launches one `sox` process per
file concurrently (`&` ... `wait`), which on a large month is a lot of parallel processes. For
reproducibility: concurrency affects speed/system-load, NOT the output bytes.

```bash
#!/usr/bin/env bash
# Runs sox for resampling all days in a given year/month (both required args).
# Example:  ./new_resample_sox.sh 2018 11
# Each resample is launched in its own process.
set -ue

year=$1
month=$2
days=$(seq 1 31)  # 1–31 regardless of month, for convenience

audio_base_dir="/mnt/PAM_Archive"
decimated_base_dir="/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz"

days_line="$(echo "${days}" | tr '\n' ' ')"
in_dir=$(printf "%s/%04d/%02d" ${audio_base_dir} "${year}" "${month}")
out_dir=$(printf "%s/%04d/%02d" ${decimated_base_dir} "${year}" "${month}")
mkdir -p "${out_dir}"

printf "Starting resample_sox.sh: %04d-%02d days: %s\n" "${year}" "${month}" "${days_line}"

# SoX resample directly:
#   rate -v 32000   convert to 32 kHz, -v = very high quality
#   -b 16           16-bit depth (required by the Google model)
#   highpass 10     remove DC offset (10 Hz highpass)
#   vol 3           adjust volume 3x (correct the signal in Volts)
#   fade 0.1 -0 0.1 logarithmic 0.1 s fade in, full-duration hold (-0), 0.1 s fade out
for day in ${days}; do
  prefix=$(printf "%s/MARS_%04d%02d%02d" "${in_dir}" "${year}" "${month}" "${day}")
  for infile in "${prefix}"_*.wav; do
    basename=$(basename "${infile}" .wav)
    outfile="${out_dir}/${basename}_resampled_32kHz.wav"
    echo "infile = ${infile}"
    echo "outfile = ${outfile}"
    sox "${infile}" -b 16 "${outfile}" rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3 &
  done
done
wait
```

### Key SoX parameters (for the reproducibility bundle — these define the bytes)
- `rate -v 32000` — resample to 32 kHz, very-high-quality filter
- `-b 16` — 16-bit output (required by the Google/Perch model input spec)
- `highpass 10` — 10 Hz highpass to remove DC offset
- `vol 3` — **calibration to volts, NOT an optional gain.** Converts the raw hydrophone output
  to volts, the physical unit John wants to work in. Applied to EVERY 32k resample in
  PAM_Analysis, always, by design. Do not drop it.
- `fade 0.1 -0 0.1` — 0.1 s log fade in/out, full-duration hold

IMPORTANT — on `vol 3` and clipping (corrected understanding, per D. Edgington / J. Ryan,
Aug 24 2026): `vol 3` is a deliberate, standard volts-calibration step, NOT a risky gain to
hedge against. **Clipping is not a practical concern** — signals of true interest do not clip.
The real and opposite concern is that MARS signals are often **very low amplitude**; that is
solved downstream by the per-window peak-normalization-to-0.25 (Stage 2), as Perch V2's own
front end also does. So: always use `vol 3`; the low-amplitude problem, not clipping, is what
the pipeline is built around. (An earlier project note mislabeled a "no-vol variant" as a
clipping remedy — that framing was wrong. Every PAM_Analysis 32k resample uses `vol 3`.)

**For the FAIR bundle, pin the SoX version (`sox --version`) — output can differ across SoX
versions.**

### Output naming
`MARS_<YYYYMMDD>_<HHMMSS>_resampled_32kHz.wav` — the `_resampled_32kHz` suffix is added by the
script; downstream tools match on it.

---

## Stage 1.5 — VERIFY THE RESAMPLE before spending GPU time (run EVERY month)

### FIRST: run `tools/coverage_histogram.py` and COMMIT ITS CSV (mandatory, Aug 28 2026)

```bash
python3 tools/coverage_histogram.py \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM> \
    --out results/coverage/<YYYY>-<MM>_coverage.csv
```

**Why it is mandatory:** the bulk resampled WAV is DELETED after each month is analyzed, and once
it is gone there is no way to recover how many hours were recorded on a given day.
**Per-day detection counts are uninterpretable without this file** — August 2015 ranges from 2.7 h
to 24 h of coverage per day, so Aug 19 (16 files) cannot be compared with Aug 20 (145 files) as
raw counts. Every seasonal/interannual figure needs **detections per hour of effort**.
It also supersedes the hand-rolled duration scan below: it reads true durations, enumerates
**every** calendar date (so a fully-absent date appears as an explicit `ABSENT` row instead of
vanishing from a `uniq -c` histogram — 2015-08-16 is exactly this case), reports real gaps and
overlaps from the timeline, and prints the **TRUE expected window count** for Stage 2.

**Coverage of the 7 months processed as of Aug 28 2026** (`results/coverage/*.csv`):

| Month | Files | Hours | % nominal | TRUE windows | vs files×120 | Short files | Notes |
|---|---|---|---|---|---|---|---|
| 2015-07 | 469 | 77.96 | 10.5 | 56,130 | −150 | 2 | partial deployment month (starts 7/28 18:05) |
| 2015-08 | 3,793 | 629.34 | 84.6 | **453,137** | −2,023 | 21 | 8/16 absent; 5 long dropouts |
| 2018-04 | 4,320 | 720.00 | **100.0** | 518,400 | 0 | 0 | perfect month |
| 2018-05 | 4,464 | 744.00 | **100.0** | 535,680 | 0 | 0 | perfect month (held-out) |
| 2020-10 | 4,504 | 743.42 | 99.9 | 535,295 | **−5,185** | 47 | many restarts; 0.50 h lost |
| 2024-09 | 2,698 | 449.67 | 62.5 | 323,760 | 0 | 0 | 9/20–9/30 absent (finding #25) |
| 2026-04 | 4,215 | 702.27 | 97.5 | 505,632 | −168 | 3 | one 17.7 h dropout after 4/13 00:20 |

**Note October 2020:** 4,504 files but 47 short ones, so `files × 120` overcounts by **5,185
windows (~1%)**. Any reasoning from file counts in that month is wrong by that much.

**August 2015's 114.64 h of loss is FIVE LONG DROPOUTS, not scattered outages** — 58.6 h after
8/15 04:35, 22.6 h after 8/18 22:44, 16.2 h after 8/12 23:45, 11.9 h after 8/7 06:38, 3.6 h
after 8/21 17:03. So the partial days hold *contiguous* coverage blocks, which is much easier to
interpret than swiss cheese.

### Then the remaining checks

Cheap, and it establishes the true expected window count for the Stage 2 check. Added Aug 27 2026
after the July 2015 run, where a size-based check missed a short file and a start-to-start cadence
check found only one of two recorder restarts.

```bash
RS=/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM>
RAW=/mnt/PAM_Archive/<YYYY>/<MM>

# 1. the run's own failure report
grep -E "^Starting|^Finished:|^Failed files:|^WARNING:" -A20 <resample-log>

# 2. counts, resampled vs raw — must match exactly
ls $RS/*_resampled_32kHz.wav | wc -l
ls $RAW/MARS_*.wav | wc -l

# 3. per-date histogram (substr 6,8 = YYYYMMDD). Expect 144 files/day for a full day.
ls $RS | awk '{print substr($0,6,8)}' | sort | uniq -c

# 4. bookends
ls $RS | head -2; ls $RS | tail -2

# 5. DURATION SCAN — the important one. Finds restarts/truncations a size floor misses.
for f in $RS/*.wav; do
  d=$(soxi -D "$f"); [ "$d" = "600.000000" ] || echo "$d  $(basename $f)"
done

# 6. total duration, and the TRUE expected window count (ceil of each file / 5 s)
for f in $RS/*.wav; do soxi -D "$f"; done \
  | awk '{s+=$1; w+=int(($1+4.999999)/5)} END {printf "total %.1f s = %.2f h — expected windows %d\n", s, s/3600, w}'

# 7. format check (want 32000 Hz, 1 channel, 16-bit)
soxi $(ls $RS/*.wav | head -1)
```

**Why the duration scan and not `find -size`:** a 205 s 32 kHz 16-bit mono file is ~13 MB, so any
size floor coarse enough to catch true zero-byte failures sails right past a restart-truncated
file. Duration is the direct measurement.

**Why start-to-start cadence spacing is not sufficient:** a restart shifts the filename cadence,
so spacing between *starts* conflates "file was cut short" with "next file began late". On
2015-07-30 the start-to-start delta read 214 s while the file was actually 205 s — a real 9 s gap,
not 214. Use the durations to get gaps right:
`gap = next_start − (this_start + this_duration)`.

**⚠️ A DURATION DEFICIT IS NOT A GAP (corrected Aug 28 2026 — this doc got it wrong first time).**
A short file does not mean lost recording time; it means that file ended early. What matters is
**when the next file STARTED.** Summing `Σ(600 − short_durations)` gives a *deficit*, and reporting
it as missing audio is a mistake — July 2015's deficit is 753 s but its true gap is **62 s**.
Nor can you infer the next start from the filename cadence: **a restart SHIFTS the cadence**
(the file after `162524` is `163019`, not the predicted `163524`).
**Use `tools/coverage_histogram.py`** — it walks the timeline from start-time + true-duration and
reports real gaps and overlaps directly. Do not hand-derive either quantity.

### Recorder CLOCK RESYNC — expect small timestamp overlaps every month (Aug 28 2026)

`coverage_histogram.py` reports timestamp overlaps tiered by magnitude. **Small ones are normal
and are NOT duplicated audio.** Observed pattern in April/May 2018:

| Month | Pattern | Overrun | Dates |
|---|---|---|---|
| 2018-04 | `075914` → `080911` | 3 s | Apr 1, 8, 15, 22, 29 — **strictly weekly** |
| 2018-05 | `075914` → `080912` | 2 s | May 6, 13, 20, 27 — **strictly weekly** |

A weekly event at the same second of the day is a **clock correction**, not a recorder
re-recording audio: the oscillator drifts ~2-3 s/week, gets resynced, and the filename stamps
compress while the audio stream stays contiguous. At 2-3 s these cannot fill even one 5 s
analysis window (5 affected windows out of 518,400 in April 2018, worst case).
**Tiering:** `< 5 s` negligible · `< 60 s` minor · `>= 60 s` MATERIAL (a recorder genuinely
re-recorded wall-clock time — stop and investigate).
**TODO: confirm the weekly resync with J. Ryan** (MARS clock discipline is his domain). If
confirmed it is a one-line methods note, not a data problem.

### Worked example — July 2015 (partial deployment month, verified Aug 27 2026)
- MARS first deployed 2015-07-28; recording starts **18:05:24**, so the 28th has only 36 files.
- **469 resampled = 469 raw**, no failures, no truncated-to-zero files.
- Per-date: 28th = 36, 29th = 144, 30th = **145**, 31st = 144. The 145 is a restart artifact, not
  an overlap.
- Two short files, both recorder restarts (**figures corrected Aug 28 2026** by
  `tools/coverage_histogram.py`; the original hand-derived numbers were wrong):

| File | Duration | Ends | Next file starts | **Real gap** |
|---|---|---|---|---|
| `MARS_20150729_162524` | 242 s | 16:29:26 | `163019` | **53 s** |
| `MARS_20150730_031011` | 205 s | 03:13:36 | `031345` | **9 s** |

- **Total real gap: 62 s** across 78 h — i.e. essentially continuous. (The earlier "753 s
  missing" was the duration *deficit* vs 469 nominal 600 s files, wrongly reported as lost
  wall-clock time. The recorder restarted promptly, so almost no time was actually lost.)
- **One 8 s timestamp overlap** exists (`163019` overruns `164011`) — below one 5 s analysis
  window, consistent with a clock nudge, not duplicated audio. The earlier claim of "no
  overlap" came from arithmetic that could not see overlaps at all.
- **Span reconciliation (the check that actually works):** Jul 28 18:05:24 → Aug 1 00:03:45
  = 280,701 s. Recorded 280,647 + gap 62 − overlap 8 = 280,701. ✅ Exact.
- Measured total 280,647 s = **77.96 h**. Consistent with the standing "near-continuous,
  not gapless" description of MARS.
- Each restart also shifted the filename cadence (`:x5:24` → `:x0:11` → `:x3:45`), which is why
  the month's last file is `MARS_20150731_235345`.

---

## Stage 2 — Perch V2 embedding (resampled WAV → hoplite DB)

Tool: `phase1_embed_torch.py` (NOT `phase2_classify.py` — phase2 has no embed subcommand).
Pure-PyTorch Perch V2 on spark-ae0e GB10. **Requires the `~/perch-pytorch/venv`, NOT the
perch-hoplite venv.**

### CRITICAL: normalization + naming
- **Per-window peak normalization to 0.25 is applied automatically**, downstream in the
  perch-pytorch adapter (inside `embed()`, transparent to callers) — it is NOT a flag in
  `phase1_embed_torch.py`, and grep finds no `_norm`/normalize logic in that script. It is
  always on. (This normalization was the fix for the low-amplitude MARS divergence bug — see
  CLAUDE_perch_hoplite.md "Low-Amplitude Normalization".)
- **The `_norm` suffix on the db-dir is a MANUAL naming convention**, NOT auto-added by the
  tool. Always name the DB `MARS_<start>_<end>_32kHz_norm` so it's clear the normalized
  pipeline produced it and it matches the other validated months (May/Oct/April all `_norm`).

### Command (Sept 2024 example, as actually run Aug 24 2026)
```bash
cd ~/perch-hoplite
source ~/perch-pytorch/venv/bin/activate          # <-- perch-pytorch venv, not perch-hoplite

nohup python3 phase1_embed_torch.py \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2024/09 \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20240901_20240930_32kHz_norm \
    --device cuda --compile \
    > /mnt/PAM_Analysis/perch-hoplite/logs/embed_sep2024_norm.log 2>&1 &

sleep 10 && tail -15 /mnt/PAM_Analysis/perch-hoplite/logs/embed_sep2024_norm.log
```

Flags:
- `--audio-dir` — the resampled `<YYYY>/<MM>` dir from Stage 1
- `--db-dir` — output hoplite DB (name it `..._norm`)
- `--device cuda` — GB10 GPU
- `--compile` — torch.compile, ~2.5x faster (slow first batch, then accelerates); worth it
- defaults: `--hop-size-s 5.0` (non-overlapping 5 s windows), `--batch-size 8`
- `--date YYYYMMDD` — optional, embed only one day
- Idempotent: re-running skips already-embedded files (`handle_duplicates='skip'`)

### Quirk (harmless)
The auto-generated internal **Dataset name** may show the wrong end-date (Sept 2024 run logged
`MARS_20240901_20240919_32kHz`). This is an internal label only — the **db-dir path is what
inference queries**, so it doesn't matter. Don't be alarmed by the dataset field.

### ⚠️ BEFORE LAUNCHING: kill any Gradio review server (Aug 28 2026)

The review server and the embedder share the **same GPU** on a spark. A forgotten review holding
even ~228 MiB caused August 2015's first embed attempt to die at model load with
`torch.AcceleratorError: CUDA error: out of memory` — before touching any audio.

```bash
nvidia-smi                                       # any python3 holding memory?
ps aux | grep "[p]hase2_classify" | awk '{print $2, $12, $NF}'
pkill -f "port <PORT>"                           # targeted; do NOT blanket-kill if a colleague may be reviewing
sleep 3 && nvidia-smi                            # confirm "No running processes found"
```

This has now bitten twice in two days (Aug 27: a 3-day-old review blocked port 7878; Aug 28: a
review blocked GPU memory). Make the `nvidia-smi` check a habit before every embed.

If an embed does die, confirm it left no partial DB before relaunching:
```bash
ls -la /mnt/PAM_Analysis/perch-hoplite/db/<DBNAME>/
```
An empty dir is clean. A `hoplite.sqlite` or `usearch.index` means move it aside first.

### Sanity check after embedding

**⚠️ The printed `Expected` figure is an approximation — do not treat a shortfall as a failure.**
`phase1_embed_torch.py` computes it as `len(audio_files) * int(600 / hop_size_s)`, i.e. it
**assumes every file is exactly 600 s**. Any month containing a recorder restart will legitimately
come in *under* the printed number.

**Windowing rule — CORRECTED Aug 28 2026. The adapter DROPS the final partial window:**

```
windows_per_file = max(1, floor(duration / 5))
true expected windows = Σ max(1, floor(duration_i / 5))
```

⚠️ **An earlier version of this doc said `ceil()`. That was WRONG.** July 2015 has only two short
files (242 s, 205 s) and matched `ceil` by coincidence. August 2015, with 21 short files,
discriminates the rules cleanly: `ceil` predicts 453,137, `floor` predicts 453,119, and the DB
holds **453,123** = `max(1, floor(...))`. Verified per-file against the `windows` table.

The `max(1, ...)` matters: a file **shorter than one 5 s window still produces one window**, mostly
padding. August 2015 has three (1 s, 2 s, 1 s). Scores on those are not meaningful and nothing
stops them scoring high — see `tools/audit_window_counts.py`, which lists them.

**Verify every month with:**
```bash
python3 tools/audit_window_counts.py \
    --audio-dir /mnt/PAM_Analysis/.../resampled_32kHz/<YYYY>/<MM> \
    --db /mnt/PAM_Analysis/perch-hoplite/db/<DBNAME>/hoplite.sqlite
```
It reports predicted vs actual, every file that disagrees with the rule, any file on disk with no
DB recording (a skipped embed), and the padded sub-window files.

**KNOWN ANOMALY (unresolved, 1 file in 3,793):** `MARS_20150817_155951` (301.0 s) holds **61**
windows where the rule predicts 60, while `MARS_20150803_153345` (476.0 s — the same 1 s remainder)
correctly holds 95. Flagged, not explained. `soxi -D` prints only 3 decimals, so check `soxi -s`.

Reconcile against the rule, not against `files × 120`. A 600 s file gives exactly 120 either way,
which is why this only surfaces on months with restarts.

```bash
tail -12 /mnt/PAM_Analysis/perch-hoplite/logs/embed_<month>_norm.log   # "Windows in DB" vs "Expected"
```

### Verified runs

| Month | Files | Windows in DB | Printed "Expected" | Elapsed | Throughput | Notes |
|---|---|---|---|---|---|---|
| May 2018 | 2232 | 535,680 | 535,680 | — | — | all files 600 s |
| Sept 2024 | 2698 | 323,760 | 323,760 | 17.9 min | ~302 win/s | month ends 9/19 (real outage) |
| **August 2015** | **3,793** | **453,123** | 455,160 | **33.5 min** | **225.6 win/s** | first FULL month; 21 short files; rule `max(1,floor(d/5))` predicts 453,123 ✅ (one 301 s file off by +1) |
| **July 2015** | **469** | **56,130** | 56,280 | **4.3 min** | **219.4 win/s** | partial month (deployment 7/28 18:05); 2 restarts → 150-window shortfall is CORRECT: 467×120 + ceil(242/5)=49 + ceil(205/5)=41 = 56,130 ✅ |

**On throughput (revised Aug 28 2026):** July 2015 ran 219.4 win/s and August 2015 — a full
33.5-minute month — ran **225.6 win/s**. So the earlier guess that July's 219 was `torch.compile`
warmup was WRONG: ~220-226 win/s is simply the sustained rate on the GB10. **Sept 2024's ~302 win/s
is the outlier that needs explaining**, not the norm. Budget ~30-35 min per full month.

**Full-month planning figures (for the 11-year campaign):** ~4,464 files → ~535,680 windows →
~30 min GPU, ~1 day SoX wall-clock, ~157 GB resampled WAV. The WAV footprint is the binding
constraint (thalassa), not GPU time.

---

## Stage 2.5 — CHECK DATA COVERAGE before interpreting inference (IMPORTANT)

MARS has had outages; do NOT assume a month is fully recorded. A "no orca" result on a day with
NO DATA is meaningless. Before reading detections for any month, check which days actually have
embedded data:

```bash
sqlite3 /mnt/PAM_Analysis/perch-hoplite/db/MARS_20240901_20240930_32kHz_norm/hoplite.sqlite \
  "SELECT substr(filename,6,8) AS day, COUNT(*) FROM recordings GROUP BY day ORDER BY day;"
# Table is `recordings` (CONFIRMED Aug 27 2026 — this is also the query phase1_embed_torch.py
# prints in its own "next step" tip on completion). Earlier drafts of this doc said
# `hoplite_sources`, which was wrong. If in doubt: sqlite3 <db>/hoplite.sqlite ".tables"
# Note substr(filename,6,8) yields the full YYYYMMDD, not just the day-of-month.
```

Cross-check against an independent source (the raw archive `/mnt/PAM_Archive/<YYYY>/<MM>/`, or
ask a collaborator) — the DB tells you what was resampled+embedded; the archive tells you what
was actually recorded. For Sept 2024 specifically: a possible outage means confirm 9/27/24
(the visual-ground-truth encounter date) actually has data before concluding anything.

---

## Stage 3 — Inference (once DB exists and coverage is confirmed)

Standard v4 + v10 orca inference (see CLAUDE_inference.md for the full workflow):
```bash
cd ~/perch-hoplite
# (perch-hoplite venv is fine for infer; only embedding needs perch-pytorch venv)
for model in v4 v10; do
  python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20240901_20240930_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_${model}.pt \
    --labels orca_call --logit-threshold 0.0 \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20240901_20240930_${model}_orcaval.csv
done
```
Then score-band triage + candidate selection + review per CLAUDE_inference.md.

---

## Full chain summary

raw `/mnt/PAM_Archive/<YYYY>/<MM>/`
  → **Stage 1** SoX (`tools/resample_sox_32k_batched_vol.sh <year> <month> [start_day] [end_day]
    [max_jobs]`, rate -v 32000 / -b 16 / highpass 10 / vol 3 / fade)
  → resampled `/mnt/PAM_Analysis/.../resampled_32kHz/<YYYY>/<MM>/`
  → **Stage 1.5** verify: counts vs raw, per-date histogram, DURATION SCAN, true expected
    windows = Σ ceil(dur/5)
  → **Stage 2** `phase1_embed_torch.py` (perch-pytorch venv, --device cuda --compile,
    per-window peak-norm-0.25 auto-applied, db named `..._norm`)
  → hoplite DB `/mnt/PAM_Analysis/perch-hoplite/db/MARS_<start>_<end>_32kHz_norm/`
  → **Stage 2.5** coverage check (which days have data?)
  → **Stage 3** `phase2_classify.py infer` (v4, v10)
  → detections CSV → review (CLAUDE_inference.md)

## TODO to finalize this doc (D. Edgington, from office)
- [x] Exact nohup command + log path used for the Sept 2024 SoX resampling run. (Done Aug 25 2026:
      `nohup ./new_32k_resample_sox.sh 2024 9 > logs/nohup_resample_2024_09.out &` — note month
      passed as `9` not `09`.)
- [x] Locate the "fancier" concurrency-throttling resample variant. **Done Aug 27 2026:** it is
      `tools/resample_sox_32k_batched_vol.sh` in this repo, and it is now the CANONICAL Stage 1
      script (day-range args + concurrency cap + failure report). Its sibling
      `tools/resample_sox_32k_batched_novol.sh` DOES exist and omits `vol 3` — see the warning in
      Stage 1. Concurrency affects speed/load only, not output bytes.
- [x] Pin the SoX version — SoX v14.4.2, /usr/bin/sox (done Aug 25 2026).
- [x] Confirm the coverage-query table name. **Done Aug 27 2026: it is `recordings`**, not
      `hoplite_sources`. Stage 2.5 corrected.

## Open items (Aug 27 2026)
- [ ] Fix the copy-pasted header comment block in `tools/resample_sox_32k_batched_vol.sh` — it
      reproduces the `_novol` rationale (and has an unclosed paren). Doc-only; does not affect
      output. Lower priority now that there is no second script to choose between.
- [x] Delete `tools/resample_sox_32k_batched_novol.sh`. **DONE Aug 30 2026.** Identical arg
      signature + identical output path + no filename marker = a foot-gun across a ~130-month
      campaign, with no upside since `vol 3` is mandatory. If a no-vol mode is ever wanted it
      should be a **parameter on the one script**, not a second near-identical file. Recoverable
      from git history.
