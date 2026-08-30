# thalassa `/PAM_Analysis` — storage summary for D. Edgington's directories

**Measured:** 2026-08-30, from spark-ae0e over the NFS mount (`du`).
**Volume:** `//thalassa.shore.mbari.org/PAM_Analysis` — **51 TB total, 46 TB used, 4.6 TiB free (92%)**.

---

## Top-level totals

| Directory | Size | What it is |
|---|---:|---|
| `GoogleMultiSpeciesWhaleModel2/` | **~880 GB** * | Resampled audio + model scores/logits |
| `perch-hoplite/` | **47 GB** | Orca detection project: embedding DBs, models, results, provenance |
| `duane_scratch/` | **31 GB** | Scratch |
| `perch_weights/` | **44 MB** | Perch model weights |
| **Total** | **~958 GB** | ~1.9% of the 51 TB volume |

\* The full crawl of `GoogleMultiSpeciesWhaleModel2/` did not complete (the VPN link dropped mid-`du`).
The figure is dominated by `resampled_32kHz/` at **873 GB**, measured separately and reliable. The
remaining subdirectories (`logits/`, `perch/`, `scores/`, `scores_gpu/`, `scores_rerun/`,
`scores_sinc/`) are unmeasured; the total above assumes they are small relative to the audio.
**This one number should be re-measured** (`nohup` it next time — the crawl outlives a VPN session).

---

## `GoogleMultiSpeciesWhaleModel2/` — the resampled audio

| Item | Size | Note |
|---|---:|---|
| `resampled_32kHz/` | **873 GB** | 7 months of MARS audio at 32 kHz — the active working set |
| `resampled_24kHz/` | *deleted* | Removed 2026-08-30. Was for the Google Multispecies Whale model |
| `resampled_48kHz/` | *deleted* | Removed 2026-08-30. Was for the OrcAI model |
| `logits/`, `perch/`, `scores*/` | *not measured* | Model outputs from earlier work |

**Per-month cost of resampled audio: ~125 GB** (873 GB ÷ 7 months). Both deleted trees covered only
a few selected months rather than the full archive, so reclaiming them freed relatively little at
this scale — the volume moved from 96% to 92% largely through those deletions.

**Resampled audio is transient by design.** Under the current full-archive campaign each month's WAV
is deleted once its analysis is complete and its findings, figures, and coverage record are
committed. The audio is always regenerable from `PAM_Archive` with SoX.

---

## `perch-hoplite/` — the orca detection project

| Subdirectory | Size | Note |
|---|---:|---|
| `db/` | **46 G** | Embedding databases — 98% of the project total |
| `logs/` | 538 M | |
| `results/` | 225 M | Inference CSVs, coverage records |
| `provenance/` | 3.5 M | Labeling session records |
| `example_clips/` | 3.2 M | |
| `models/` | 708 K | Trained classifiers (orca_v4, orca_v10, …) |
| `json_labels/` | 636 K | |
| `labels/` | 52 K | |

**Everything except `db/` totals under 800 MB** — and `models/`, `provenance/`, `labels/`, and
`json_labels/` together are under 5 MB while being the hardest things in the project to reproduce.

### Inside `db/` — where the 46 GB actually goes

| Category | Count | Size | Note |
|---|---:|---:|---|
| Single-month DBs (current `_norm`) | 7 | ~9.6 GB | The campaign working set |
| Combined multi-month DBs | 7 | **~32 GB** | Experimental iterations |
| Superseded single-month DBs | 3 | ~4.9 GB | Pre-normalization / `_ctx` variants |

**A single full month costs ~1.6–1.7 GB.** Partial months cost proportionally less — August 2015
(84.6% coverage) is 1.4 GB, September 2024 (62.5%) is 1019 MB, and July 2015 (78 h only) is 177 MB.
DB size tracks *windows embedded*, not calendar days.

**The 46 GB is experimental history, not per-month cost.** Seven combined DBs account for ~32 GB,
including four at 4.8 GB each that appear to be versioned iterations of the same 3-month set
(`_norm`, `_norm_v2`, `_norm_v3`, `_norm_v3_apr2026`), plus a 6.5 GB 4-month combination. Three
further DBs are pre-normalization or context-window duplicates of months that also exist in `_norm`
form. **Roughly 15–20 GB is likely reclaimable** if the superseded combinations are confirmed
retired — this needs review before deleting anything, not an automatic purge.

---

## Projection for the full-archive campaign

The campaign processes all ~130 months of the MARS Pacific Ocean Sound archive one month at a time.

| Quantity | Per full month | × 130 months |
|---|---:|---:|
| Resampled WAV (transient) | ~125 GB | ~16 TB — **does not fit** |
| Embedding DB (permanent) | ~1.6 GB | **~210 GB** |

**The permanent artifacts are not a storage problem.** All 130 months of embeddings would occupy
~210 GB, under 0.5% of the volume. **The transient WAV is the constraint:** resampling the whole
archive at once would need ~16 TB against 4.6 TiB free. Analyze-then-archive — keeping only a few
months live at a time — is therefore what makes the campaign feasible, not merely tidy.

At present three months are live (July, August, September 2015), and the current headroom would
allow roughly 36 concurrently if nothing were ever deleted.

---

## Notes

- **Backup asymmetry:** thalassa is backed up by IT; the two DGX Spark boxes are **not**. Anything
  produced locally on a spark must be copied to thalassa to be safe.
- If storage pressure ever forces the issue, the plan is to `rsync` the small-but-irreplaceable
  directories to a 2 TB external SSD rather than spend time on selective review — 2 TB holds all of
  `perch-hoplite/` many times over.
- The `du` crawl of `resampled_32kHz/` takes several minutes over SMB (~29,000 files); run it under
  `nohup` so it survives a dropped VPN session.
