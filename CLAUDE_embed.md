# CLAUDE_embed.md — Resampling + embedding pipeline (raw MARS → hoplite DB)

The exact process for turning raw MARS archive audio into a hoplite embedding DB ready for
`phase2_classify.py infer`. Two stages: (1) SoX resample to 32 kHz, (2) Perch V2 embedding.
This step is infrequent and easy to get subtly wrong (venv, normalization, naming), so it's
written down.

STATUS: partial draft (Aug 24 2026). Resampling script captured below; exact nohup invocation
+ log path to be added by D. Edgington from the office. Embedding stage captured from the
Sept 2024 run.

---

## Stage 1 — SoX resampling (raw archive → resampled 32 kHz WAV)

Raw MARS audio lives at `/mnt/PAM_Archive/<YYYY>/<MM>/MARS_*.wav`. Resampled output goes to
`/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/<YYYY>/<MM>/`.

Script used for the current data (`new_32k_resample_sox.sh`), run as
`./new_32k_resample_sox.sh <year> <month>` (e.g. `./new_32k_resample_sox.sh 2024 09`).
Located in `~/gmwd/new3-12_whale_detection/gmwd/` on spark-ae0e.

TODO (D. Edgington, from office): add the exact nohup command + log-file path used for the
Sept 2024 run. A "fancier version that throttles the number of concurrent processes" also
exists somewhere but was NOT the one used for the current set — the version below launches one
`sox` process per file concurrently (`&` ... `wait`), which on a large month is a lot of
parallel processes. Note for reproducibility: concurrency affects speed/system-load, NOT the
output bytes.

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

### Sanity check after embedding
The tool prints an "Expected windows" figure up front and a `Done. NNNNNN embeddings in <db>`
line at the end. Confirm they match and are in a sane range.
- May 2018: 2232 files → 535,680 embeddings
- Sept 2024: 2698 files → expected 323,760 windows (shorter/fewer-per-file than May; ~120
  windows/file ≈ ~10-min files). Confirm final count ≈ 323,760.
```bash
tail -8 /mnt/PAM_Analysis/perch-hoplite/logs/embed_sep2024_norm.log   # look for "Done. NNNNNN embeddings"
```

---

## Stage 2.5 — CHECK DATA COVERAGE before interpreting inference (IMPORTANT)

MARS has had outages; do NOT assume a month is fully recorded. A "no orca" result on a day with
NO DATA is meaningless. Before reading detections for any month, check which days actually have
embedded data:

```bash
sqlite3 /mnt/PAM_Analysis/perch-hoplite/db/MARS_20240901_20240930_32kHz_norm/hoplite.sqlite \
  "SELECT substr(filename,6,8) AS day, COUNT(*) FROM hoplite_sources GROUP BY day ORDER BY day;"
# (if hoplite_sources is not the table, run .tables and adjust)
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
  → **Stage 1** SoX (`new_32k_resample_sox.sh <year> <month>`, rate -v 32000 / -b 16 /
    highpass 10 / vol 3 / fade)
  → resampled `/mnt/PAM_Analysis/.../resampled_32kHz/<YYYY>/<MM>/`
  → **Stage 2** `phase1_embed_torch.py` (perch-pytorch venv, --device cuda --compile,
    per-window peak-norm-0.25 auto-applied, db named `..._norm`)
  → hoplite DB `/mnt/PAM_Analysis/perch-hoplite/db/MARS_<start>_<end>_32kHz_norm/`
  → **Stage 2.5** coverage check (which days have data?)
  → **Stage 3** `phase2_classify.py infer` (v4, v10)
  → detections CSV → review (CLAUDE_inference.md)

## TODO to finalize this doc (D. Edgington, from office)
- [ ] Exact nohup command + log path used for the Sept 2024 SoX resampling run.
- [ ] Locate the "fancier" concurrency-throttling resample variant (speed/load only — same
      output bytes; `vol 3` is standard and always used, there is no operational no-vol variant).
- [ ] Pin the SoX version (`sox --version`) for the reproducibility bundle.
- [ ] Confirm the `hoplite_sources` table name for the coverage query (adjust if different).
