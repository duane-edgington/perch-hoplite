# CLAUDE_orcas_MBNMS_release.md — the public release repo: state, invariants, open items

**Purpose:** a self-contained brief so a fresh chat can pick up
`github.com/duane-edgington/perch-hoplite-orcas-MBNMS` without the history of how it
was built. Supersedes `CLAUDE_repo.md`, which was the pre-build handoff and is now
obsolete. `CLAUDE_release_plan.md` remains useful as the FAIR rationale.

*Naming note: the file convention in this repo is underscores (`CLAUDE_embed.md`,
`CLAUDE_release_plan.md`), so this is `CLAUDE_orcas_MBNMS_release.md` rather than
mirroring the hyphenated repo name.*

**Status as of 7 September 2026.** The release is complete and published. Remaining
work is optional.

---

## Where things stand

| | |
|---|---|
| Release repo | `github.com/duane-edgington/perch-hoplite-orcas-MBNMS`, **public** |
| Release `main` | `dd69b06` — 144 files, 9.8 MB |
| Working repo `main` | `4d06762` |
| Zenodo concept DOI | **10.5281/zenodo.22574816** — always resolves to newest; use for the poster QR |
| Zenodo v1.0.0 DOI | 10.5281/zenodo.22574817 |
| Tags | `v1.0.0` at `ba192c3` (2 Sept). **v1.1.0 not yet cut** |
| GitHub Pages | `main` branch, `/docs` folder, `docs/.nojekyll` present |
| Listening page | <https://duane-edgington.github.io/perch-hoplite-orcas-MBNMS/listen/> |
| Poster | IEEE OCEANS 2026 Monterey, 21–24 September 2026 |

Zenodo archives automatically from GitHub Releases — the integration is enabled for
this repo only, and `.zenodo.json` in the repo root supplies the metadata
(`upload_type: dataset`, both authors with ORCIDs, Apache-2.0).

---

## What is released, and what deliberately is not

**In:** the resampling script and every flag's rationale; embedding and the main CLI;
`orca_v4.pt` and `orca_v10.pt` with both metrics sidecars; 1,405 confirmed
annotations; the reproducibility bundle (source manifest, pinned versions, sha256
checksums); the listening page; and the docs.

**Out, on purpose:** raw audio (already public on AWS), resampled audio (~723 GB,
regenerable), embedding databases (~45 GB, regenerable), the v5–v8 detour and the
`_clean` era, unreviewed labels, and 2016 onward.

**2016 and 2015 are a different paper.** Seasonal and interannual analysis across the
decade is its own research question with its own release. The poster covers four
fully analyzed months, which already exceeds what the accepted extended abstract
promised. Resist the pull to fold partial 2016 work into this release: `docs/DATA.md`
promises that every positive label was resolved by an expert, and pass-1 labels would
weaken a promise that currently holds without qualification.

---

## Numbers a fresh chat will need

### Labels — 1,405 distinct = 1,351 positive + 54 weak negative

| Month | Distinct | Role |
|---|---|---|
| April 2018 | 714 | training (660 positive, 54 weak negative) |
| May 2018 | 283 | **permanently held out** |
| October 2020 | 322 | training |
| April 2026 | 86 | training |

Verify against the databases with:

```bash
for db in MARS_20180401_20180430 MARS_20180501_20180531 \
          MARS_20201001_20201031 MARS_20260401_20260430; do
  printf "%-24s " $db
  sqlite3 /mnt/PAM_Analysis/perch-hoplite/db/${db}_32kHz_norm/hoplite.sqlite \
    "SELECT COUNT(DISTINCT recording_id || '|' || hex(offsets) || '|' || label) FROM annotations;"
done
```

**Four counts coexist and all are correct.** 873 built v4's trajectory. **1,076** rows
were in the frozen three-season merge v10 trained on, of which **1,048** are distinct
— the 28 extras are merge-artifact duplicates from April 2018. **1,405** is the
archive at release. The label files track the archive, not any model's training set,
so retraining from `labels/` will *not* reproduce `orca_v10` exactly.

### Thresholds — one per model, not interchangeable

| Class | `orca_v4` | `orca_v10` |
|---|---|---|
| ship_noise | 0.16 | 1.81 |
| humpback_song | 0.98 | 0.79 |
| other | 1.99 | 1.31 |
| dolphin_call | 2.05 | 1.94 |
| **orca_call** | **1.16** | **2.31** |

Both orca values are that model's F1-optimal threshold on its held-out split, later
confirmed by cross-month validation. Applying v4's 1.16 to v10 floods you with
humpback. **v4's ship_noise 0.16 is unreliable** — 3 held-out examples and an
artifact F1 of 1.0.

Operational practice: run **both** models on every month, score each at its own
threshold, compare, and extend review below the cutoff by ear — sometimes below 0.0
— where a month looks like it holds more. The threshold defines a review queue, not a
detection boundary.

### Models

```
orca_v4.pt             266c51fd…9ed9e2    ROC-AUC 0.9589885  cmap 0.8297407  n_eval 296
orca_v4.metrics.json   3ddbf966…e8227d
orca_v10.pt            3393a904…cc487a    ROC-AUC 0.9372342  cmap 0.6782188  n_eval 459
orca_v10.metrics.json  de5e4ff7…3068ae
```

The `.pt` files are **JSON with base64 arrays** despite the extension. `torch.load`
fails on them, and `perch_hoplite.agile.classifier` cannot be imported outside the
CLI because of a bare `import tensorflow`. Read them with `json` + `base64`.

v10 per-class F1 at optimal thresholds: orca 0.945, ship 0.800, dolphin 0.687,
humpback 0.619, other 0.591.

v10 training args: `num_steps 512, lr 0.001, weak_neg_weight 0.05, batch_size 128,
train_ratio 0.8, seed 42`. Note `batch_size` is **recorded but ignored** — see
Deferred.

### Raw audio

**One bucket per year**, region `us-west-2`:

```
s3://pacific-sound-256khz-<YYYY>/<MM>/MARS_<YYYYMMDD>_<HHMMSS>.wav
```

The year is in the bucket name, not the key — `s3://pacific-sound-256khz/2018/05/`
does not exist. Years 2016, 2018, 2020, 2024, 2026 all confirmed present.

Files are **10 minutes, 256 kHz, mono, 24-bit, 460,800,356 bytes**. Resampling
converts both rate and bit depth. 600 s ÷ 5 s = 120 windows per file, which is the
arithmetic behind the embedding counts. Recordings are **not hour-aligned** and the
offset varies by deployment period (May 2018 starts at `:0912`, October 2020 at
`:0001`) — list the bucket, never construct filenames.

Environment: **SoX 14.4.2**, Ubuntu 24.04.3 LTS, perch-hoplite 1.0.2 from PyPI.

### Timing, measured 6 September

On the merged three-season DB (1,559,308 embeddings, NFS): DB load 7 s,
**materialize + preload 8 m 45 s**, training loop 1.3 s at ~400 it/s, total
**9 m 47 s**. Two runs landed four seconds apart.

The documented `~37x faster` benchmark was a **single-day DB with 559 labels** at 16
seconds total. Do not repeat "16 seconds" for a full-archive database. Whether the
preload cost is NFS-specific or scales with database size is **not established** —
both benchmarks ran over the same appliance.

Weak-negative expansion: **2.06×** — 1,816 train IDs materialize to 3,736 rows,
22.0 MiB payload. Now logged on every run.

---

## Invariants worth checking before any release

### The copied-file audit

Nine `pipeline/` files and eight `tools/` scripts are **copies** of working-repo
files. They drift silently. Run this before cutting a tag:

```bash
# with both repos cloned side by side
for f in phase1_embed_torch.py phase2_classify.py; do
  diff -q release/pipeline/$f dev/$f >/dev/null && echo "$f identical" || echo "$f DIFFERS"
done
for f in $(cd release/pipeline/src && ls *.py); do
  diff -q release/pipeline/src/$f dev/src/$f >/dev/null && echo "src/$f identical" || echo "src/$f DIFFERS"
done
for f in $(cd release/tools && ls *.py); do
  [ -f dev/tools/$f ] || { echo "tools/$f release-only"; continue; }
  diff -q release/tools/$f dev/tools/$f >/dev/null && echo "tools/$f identical" || echo "tools/$f DIFFERS"
done
```

**Expected result: 17 identical, two differing, and both differences are known:**

- `phase2_classify.py` — 3 hunks, all redacted hostnames (`134.89.11.107` →
  `<gpu-host>`, `134.89.11.174` → `<gpu-host-2>`). Nothing else should differ.
- `export_labels.py` — 18 hunks, the release rewrite: DB paths as arguments, bare
  filenames in output, deduplication, no `INVENTORY.md` generator.
- `make_listen_page.py` — release-only, no dev counterpart.

**This check exists because it caught a real bug.** `phase2_classify.py` was copied
on 25 August; the working repo fixed annotation deletion on 28 August to scope by
`provenance`, so a second annotator adds an opinion instead of silently destroying
the first annotator's label. The release carried the unscoped form for three weeks.
Anyone running multi-annotator review from v1.0.0 would have lost labels invisibly.
Fixed in `f319a43`; **v1.0.0's Zenodo archive still contains the broken version**, which
is a reason to cut v1.1.

### Shared-code rule

Fix in the **working repo first**, push, then sync to the release. Never edit the
release copy of shared code directly. `spectrogram.py` gained `figsize`/`dpi`
parameters this way (`8ac90e6`), defaults unchanged so the review interface renders
identically.

### Listening page settings

`tools/make_listen_page.py` deliberately matches the Gradio review interface, because
those settings were arrived at painfully:

- **mel** spectrograms, **viridis** colormap (`colormap` is a parameter of
  `make_spectrogram_image` — pass it)
- 5-second clips peak-normalized to **−3 dBFS (0.708)**, per `make_audio_loader()`
- 30-second context peak-normalized to **0.5**, centered on the window with the
  window marked, per `load_30s_context()`
- rendered large — 988×486 clips, 1395×506 context — one per row
- **MP3** audio and palette-quantized PNGs: 18 cards is 8.5 MB, against 39 MB as
  lossless FLAC with full-color PNGs

Raw resampled audio is inaudible without normalization. Normalizing raises the noise
floor, which destroys FLAC's compression — hence MP3. The 30-second context is a
finding, not decoration: whether a call belongs to one animal's bout, differs from
its surroundings, stands alone, or is masked by ship noise are questions the
5-second window cannot answer.

Regenerate on spark with:

```bash
python3 tools/make_listen_page.py \
    --labels labels/ --selection tools/listen_selection.json \
    --audio-root /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz \
    --out docs/listen
```

---

## Claims that were corrected, and why

**October 2020 is not a clean absence result.** Earlier text said zero orca; there is
one confirmed call (`MARS_20201005_004726` at 230.0 s, annotation id 454). A later
draft then over-corrected, arguing that the single call proved the detector can hear
these animals and therefore that their silence was measured. Both were wrong.

The defensible statement: in a month when whale-watching vessels reported multiple
killer whale encounters, the detector surfaced **one** verifiable orca call out of
535,278 windows, and its remaining high-scoring candidates were **misclassified
humpback** — of which there were an abundance (265 confirmed windows). Two readings
fit equally well and the data cannot separate them: the animals were near-silent, or
the detector misses their calls against a humpback-dominated background, which is its
known dominant failure mode. Do not present either as a result.

**April 2026 is partly resolved.** The 24 April window (475.0 s, score 3.023) is
confirmed orca with humpbacks in the background. The 21 April window (245.0 s, score
2.410) stayed ambiguous and was deleted rather than forced into a class — project
discipline is that ambiguous windows are left unlabeled so a future, better-calibrated
ear meets them clean. What changed on the 24 April call was the listener, not the
model: comparable orca-over-humpback examples had accumulated from other years.

**Ship noise only sometimes correlates with orca presence.** Vessels arrive once orcas
are sighted, but Monterey Bay traffic is not conditional on whales. `RESULTS.md`'s
conservative form — ship noise is not an orca-absent cue — stands unqualified.

**Annotator provenance is stated in prose, not read from the database.** D. Edgington
annotated every label; J. P. Ryan verified every humpback. The four released months
were labeled with a generic `analyst` id and `merge_dbs.py` then appended `_merged`.
The `--annotator-id` flag is **not** broken — `src/review.py:320` writes
`gradio_gui:{annotator_id}` and newer databases show `gradio_gui:duane` correctly.

---

## Deferred — after the poster

**`--batch-size` is recorded but ignored.** `src/train.py` hardcodes
`batch_size = min(512, n_train)` while metrics sidecars record 128. Honoring the flag
would change every gradient step and produce a different model needing fresh May 2018
validation. Not a cleanup — a new model version.

**`merge_dbs.py` duplicates annotations and mangles provenance.** Each run re-inserts
existing annotations and appends `_merged`. This produced the 28 April 2018
duplicates. It will keep happening as the archive march continues, so fix it before
the next large merge.

**Empty-split guards sit after `nn.Linear(...).to(device)`** in `src/train.py`, so the
function allocates on the GPU before checking it has data. Small, inert, correct.

**`docs/RETRAIN.md` does not exist.** The release ships models but no documented
`train` command, so anyone wanting to retrain must reverse-engineer it. Natural home
for the 9 m 47 s preload warning.

**Benchmark doc.** First deliberate measurements were taken 6 September. Still needed:
peak CUDA memory, TF-versus-PyTorch under matched conditions with medians, and whether
preload cost is NFS-specific. PyTorch-Conference-paper material.

**Splitting `humpback_song`** into song and non-song. Identified as the leading
explanation for that class's weak F1; not attempted. Gray whale contamination was
tested and closed — zero found.

**Clip subset on Zenodo.** `docs/listen/` clips are MP3 for page weight. If Zenodo
should carry lossless versions, `tools/extract_example_clips.py` produces them.

---

## Working method

Duane runs three machines with two layouts. **Always ask or `pwd`; never assume.**

| Machine | Repos |
|---|---|
| DuaneEM1 (laptop, VPN) | `~/perch-hoplite`, `~/perch-hoplite-orcas-MBNMS`, scratch in `~/scratch` |
| PERCH (laptop) | `~/projects/perch-hoplite`, `~/projects/perch-hoplite-orcas-MBNMS` |
| spark-ae0e / spark-0626 | `~/perch-hoplite`, `~/perch-hoplite-orcas-MBNMS`; data under `/mnt/PAM_Analysis` |

Neither laptop has matplotlib, torch, or librosa. Analysis runs on spark; git work
happens on either.

**Flow:** Claude stages files to outputs → Duane downloads, extracts to `~/scratch`,
copies into the repo, commits and pushes from a laptop → spark pulls. Claude cannot
push. Generated output (`docs/listen/`) is committed from spark, where it is built.

**Habits that have each caught a real error:**

- Safari **renames** downloads (`amer_fixes-1.tar.gz`) and sometimes unpacks them into
  `~/Downloads/files-N/`. Ask where a file landed; never assume the path. A stale
  tarball extracted silently and nearly shipped a label-destroying bug.
- Clear `~/scratch` between deliveries. Three similar directories is how the wrong
  `make_listen_page.py` gets copied.
- After extracting, **grep for content unique to the new version** — not just
  `git status`. `tar` is silent about giving you an old file.
- Long commit messages: `git commit -F -` with a heredoc. A pasted `-m` string was
  truncated mid-sentence and needed amending.
- `lsof -i :PORT` then `kill <PID>`. A `pkill -f "port 7878"` terminated two review
  jobs when only one was intended.
- **`grep -c` counts lines, not occurrences.** Predicted counts were wrong four times
  in one session. Read the string out of the file instead of recalling it.
- Never delete without listing first, by explicit filename, no globs.
- `nvidia-smi` reports `Memory-Usage: Not Supported` on GB10 (unified memory). Use
  `free -h`.
- Unit tests in the working repo need `CUDA_VISIBLE_DEVICES="" pytest tests/ -v`;
  `train.py` grabs CUDA whenever it is available.
- A shell whose working directory has been deleted breaks relative paths *and*
  `os.getcwd()`. `cd ~` first.

**Prose:** American English only, always. `colors`, not `colours`. Concise
deliverables. Do not over-claim a result; state what the data supports and name what
would settle the rest.
