# CLAUDE_release_plan.md — Public release & FAIR reproducibility plan

Plan for releasing the perch-hoplite orca work as a FAIR, reproducible package sized to a
poster + peer-reviewed abstract (with a full PyTorch paper to follow). Deliberately modest:
the raw audio is already public, so the job is to make the path from public raw → our results
deterministic and checkable — NOT to re-host terabytes or build a Palmer-scale data paper.

Design principle: **publish reproducibility, not a dataset.** The reusable artifacts are the
trained classifiers (tiny), the labels, the scripts, and a small confirmed-clip subset.

---

## FAIR mapping (the gold-star case, stated plainly)

- **Findable** — Zenodo record with a DOI; GitHub repo linked from the DOI and the poster QR.
  Descriptive metadata + keywords (Perch V2, passive acoustic monitoring, killer whale,
  Orcinus orca, Bigg's, agile modeling, MBARI MARS).
- **Accessible** — everything retrievable by standard tools over open protocols: `git clone`
  for code, Zenodo HTTPS download for the archived snapshot, and a documented `aws s3` /https
  path to the already-public raw audio (Pacific Ocean Sound, AWS Open Data). No credentials,
  no gatekeeping.
- **Interoperable** — open, standard formats: CSV/JSON for labels and metadata, WAV for the
  clip subset, PyTorch `.pt` for models, plain Python for code, Raven-compatible selection
  tables where relevant. No proprietary containers.
- **Reusable** — clear license (code vs data may differ — see below), provenance for every
  artifact, exact tool versions and flags pinned, and checksums so a re-runner can verify they
  reproduced our inputs. A model card documents intended use and limits.

---

## Three tiers (near-term release = Tier 1 + light Tier 2; skip Tier 3 for now)

### Tier 1 — GitHub repo (free, a few hours): the reproducible pipeline
A curated PUBLIC repo (clean version of the working perch-hoplite repo — present a tidy
v0→v4→v10 lineage without the v5–v8 detour needing constant caveats). Contents below.

### Tier 2 — Zenodo record (free ≤50 GB, gives a DOI): the citable snapshot
Archived, versioned, DOI'd snapshot of the DERIVED artifacts only — trained models, label
tables, the confirmed-clip audio subset (MB, not TB), and a frozen copy of the repo at release.
This is what a paper cites.

### Tier 3 — AWS public bucket (ongoing cost): SKIP for the poster/abstract
Only revisit if the full paper needs to host something too big for Zenodo (e.g. all resampled
months, or full embedding DBs). Not needed now — raw is already public; derived is Zenodo-sized.

---

## Concrete repo layout (Tier 1)

```
perch-hoplite-orca/                 (curated public repo)
├── README.md                       overview, quickstart, how to reproduce, links to Zenodo DOI + poster
├── LICENSE                         code license (e.g. MIT/Apache-2.0)
├── LICENSE-DATA                    data/labels license (e.g. CC-BY-4.0) — code and data often differ
├── CITATION.cff                    machine-readable citation (GitHub renders "Cite this repository")
├── environment/
│   ├── requirements.txt            pinned deps (torch==…, perch-hoplite==…, librosa, soundfile, gradio==…)
│   └── VERSIONS.md                 exact tool versions incl. SoX version + OS, Python, CUDA
├── data_access/
│   ├── SOURCE_MANIFEST.csv         which raw POS/MARS files → month/day, with AWS keys/paths + date ranges
│   ├── how_to_get_raw_audio.md     exact `aws s3 cp` / https commands to pull the public raw audio
│   └── CHECKSUMS.md                sha256 of a sample of resampled outputs, so re-runners can verify parity
├── resampling/
│   ├── new_32k_resample_sox.sh     the actual pipeline script (SoX; rate -v 32000, -b 16, highpass 10, vol 3, fade)
│   └── README.md                   what it does, exact flags, why 32 kHz, and why vol 3 (volts calibration)
├── models/
│   ├── orca_v4.pt                  prior production classifier (small — linear probe on frozen Perch V2)
│   ├── orca_v10.pt                 current best (the paper's model)
│   ├── orca_v10.metrics.json       per-class F1/thresholds/train args (already exists)
│   └── MODEL_CARD.md               intended use, training data, per-class limits, ecotype scope, cautions
├── labels/
│   ├── annotations_<month>.csv     confirmed labels per month (class, file, offset, annotator, session_id)
│   └── LABELS_README.md            schema, label classes, what "confirmed" means, annotator provenance
├── clips/                          the confirmed-clip SUBSET (MB) — only windows referenced in the paper
│   ├── orca/  humpback/  ...       a few exemplar WAVs per class + the confirmed orca-day clips
│   └── CLIPS_MANIFEST.csv          clip → source WAV, offset, score, model, label, annotator
├── src/                            inference / review / figure code
│   ├── phase2_classify.py          embed / train / infer / review
│   ├── compare_may_holdout.py      the held-out evaluation (v4 vs v10)
│   ├── plot_tsne*.py, register_figure.py, etc.
│   └── README.md
├── figures/                        figure-generation outputs + provenance sidecars (manifest.json)
└── docs/
    ├── agile_modeling_history.md   the method narrative (v0→v10)
    └── REPRODUCE.md                step-by-step: raw → resample → embed → infer → results
```

What STAYS PRIVATE / out of the public repo: the full working-repo scratch (dead-end
experiments, internal notes), any un-reviewed/ambiguous labels presented as confirmed
(e.g. April 2026 candidates — include only with the "pending blind review" caveat or omit),
full embedding databases (too big; regenerable from script), and anything with unclear
licensing.

---

## Reproducibility bundle spec (answers "is the resampling script sufficient?" → no, but this is)

The script alone reproduces the *method*, not the *bytes*. Sufficient bundle = 4 parts, all small:

1. **Source manifest** (`data_access/SOURCE_MANIFEST.csv`)
   Columns: `month, date, raw_filename, aws_bucket, aws_key, byte_size, source_sample_rate_hz`.
   Lets a re-runner pull the EXACT same raw files from the public bucket.

2. **Pinned environment** (`environment/VERSIONS.md` + `requirements.txt`)
   Exact SoX version + resampling flags, Python, torch, perch-hoplite, CUDA, OS. Resampling
   output can differ across SoX versions/dither settings, so pin them.

3. **The resampling script** (`resampling/…novol.sh`)
   The actual command run, with its flags documented — including that `vol 3` is a deliberate
   calibration to volts (the unit the science works in), applied to every PAM_Analysis 32k
   resample, NOT an optional gain. (Low signal amplitude, handled by downstream normalization,
   is the real concern; clipping is not.)

4. **Verification checksums** (`data_access/CHECKSUMS.md`)
   sha256 of a handful of representative resampled output WAVs. A re-runner resamples, hashes,
   and compares — confirming they reproduced our inputs rather than silently diverging.

Plus, downstream of inputs: the **trained models** (so results are reproducible without
retraining) and the **label tables + review CSVs** (so "confirmed orca" is auditable). With all
of this, someone can go public-raw → our exact inputs → our exact results. That is genuine
reproducibility at modest cost.

### Minimal checksum recipe (for CHECKSUMS.md)
```bash
# pick ~5 representative resampled files spanning months, then:
sha256sum MARS_20180413_*.wav MARS_20180512_*.wav MARS_20201005_*.wav ... > CHECKSUMS.txt
# document SoX version used:  sox --version
```

---

## Model card outline (models/MODEL_CARD.md)

- **What it is:** linear-probe classifier on frozen Google Perch V2 embeddings; 5 classes
  (orca_call, humpback_song, dolphin_call, ship_noise, other).
- **Training data:** MARS hydrophone, 3-season recipe (April 2018 + Oct 2020 + April 2026);
  May 2018 held out. N labels, per-class support.
- **Performance:** per-class F1 at tuned cutoffs (from metrics.json); held-out May 2018
  validation (v10 vs v4). Be explicit these are MARS/Monterey-Bay numbers.
- **Intended use:** detecting orca (Bigg's-dominant in this data) and contrastive classes in
  MARS-like PAM audio. Research use.
- **Limits / cautions:** orca here are predominantly Bigg's (not validated on Residents/
  Offshores); humpback/orca acoustic overlap causes ambiguity (the April 2026 case); per-class
  cutoffs required (one global threshold won't do); precision measured at/above threshold on
  the confirmed set, not a full month-wide false-alarm rate; ship_noise is the smallest class.
- **Ecotype scope:** ecotype classification (Resident/Bigg's/Offshore) is NOT provided — flagged
  as future work (cf. the external Palmer 2025 dataset).

---

## Licensing note (decide, don't default)

- **Code:** permissive OSI license (MIT or Apache-2.0). Apache-2.0 if patent-grant clarity is
  wanted.
- **Labels/data/clips:** a Creative Commons license (CC-BY-4.0 is common and FAIR-friendly —
  requires attribution, allows reuse). Keep code and data licenses SEPARATE files.
- **Models:** state a license too (often same as code, or CC-BY). Check any upstream Perch V2 /
  Google model terms for redistribution constraints before publishing derived weights.
- **Raw audio:** not yours to relicense — it's Pacific Ocean Sound / AWS Open Data under its own
  terms; just link and cite it, don't re-host or relicense.

---

## Suggested release sequence (when poster feedback clears; ~1–2 days of work)

1. Fork/curate the working repo → clean public `perch-hoplite-orca` (strip scratch/dead-ends).
2. Write README, MODEL_CARD, REPRODUCE.md, LABELS_README; add LICENSE + LICENSE-DATA + CITATION.cff.
3. Build SOURCE_MANIFEST.csv, VERSIONS.md, CHECKSUMS.md (the reproducibility bundle).
4. Assemble the confirmed-clip subset (MB) + CLIPS_MANIFEST.csv.
5. Snapshot to Zenodo → get DOI → put DOI in README + poster QR + the paper.
6. (Optional) HuggingFace model card for orca_v10 if pick-up-and-run discoverability is wanted.
7. Skip AWS bucket unless the full paper later needs it.

**President / FAIR pitch, one line:** "Raw audio was already open; we added a DOI'd,
checksum-verified reproducibility bundle (models + labels + scripts + source manifest) so anyone
can regenerate our results from the public data — a small, FAIR, exemplary release."
