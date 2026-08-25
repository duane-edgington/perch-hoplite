# thalassa storage survey — Aug 25 2026

Survey only (no deletions). Prompted by John's directive (finding #26 pt 8) to keep our
footprint clean, plus Duane's idea to back-up-then-remove earlier resamples and the
Google-multispecies / OrcAI datasets. Plan: procure a backup disk, rsync (--dry-run first)
what we might want, THEN remove from thalassa. Philosophy: for large regenerable derived data,
archive rather than delete — cheaper than the wasted work + regret of regenerating later.

## Volume pressure (URGENT context)
```
Filesystem                             Size  Used Avail Use%  Mounted on
thalassa:/PAM_Analysis                  51T   49T  2.5T  96%  /mnt/PAM_Analysis
thalassa:/PAM_Archive                  273T  261T   13T  96%  /mnt/PAM_Archive
```
Both at 96%. This is shared MBARI infrastructure — 96% is the danger zone (write failures,
degraded performance for everyone). Cleanup is needed soon, not "someday." BUT: our total
footprint is a small fraction of 49T — tidying ours is being a good tenant, not a fix for the
volume-level pressure (that's everyone's 49T, not mainly ours).

## Our resampled 32kHz footprint: ~723 GB (primary archive-then-remove target)
Regenerable from public raw via the SoX script (CLAUDE_embed.md) — safe to archive+remove once
the DB/labels/clips for a month are preserved.
| Year | Size | Months |
|------|------|--------|
| 2018 | 315G | Apr 155G + May 160G |
| 2020 | 160G | Oct 160G |
| 2026 | 151G | Apr 151G |
| 2024 |  97G | Sep 97G |

## Our perch-hoplite footprint (non-resample)
| Dir | Size | Disposition |
|-----|------|-------------|
| db/ (embedding DBs) | 45G | archive-worthy; regenerable (re-embed) but slow; each DB = inference-ready without WAVs |
| logs/ | 446M | back-up-then-trim old ones |
| results/ (inference CSVs) | 225M | KEEP (small, valuable) |
| provenance/ | 3.5M | KEEP (crown jewel) |
| example_clips/ | 3.2M | KEEP (crown jewel) |
| models/ | 708K | KEEP (crown jewel — the actual contribution) |
| json_labels/ | 636K | KEEP |
| labels/ | 52K | KEEP |

**Key insight: the irreplaceable artifacts (models + all labels + provenance + example_clips)
total <10 MB.** Everything that IS the intellectual work fits in a rounding error. The bulk is
regenerable: resamples (723G) + DBs (45G) + logs (446M).

## Other datasets
- GoogleHumpbackModel: 3.1G (small, low priority)
- GoogleMultiSpeciesWhaleModel2 (parent, holds the resamples + model data): NOT fully measured
  (du Ctrl-C'd) — likely the largest single item; measure before deciding.
- OrcAI dataset(s): NOT measured yet.

## Cleanup plan (when backup disk is ready — NOT today)
1. Measure the unmeasured items (GoogleMultiSpeciesWhaleModel2 parent, OrcAI) with a patient
   single `du -sh` each (NFS-slow; let it finish, don't Ctrl-C).
2. Procure backup disk.
3. `rsync -av --dry-run` the resampled_32kHz tree (and DBs, and anything else archive-worthy)
   to the backup disk; verify the dry-run listing.
4. Real rsync; verify with checksums or `rsync -c` spot-check.
5. Remove from thalassa: the full-month resampled WAVs (723G) — the big win.
6. KEEP on thalassa: DBs (or archive+remove per-month after analysis), results CSVs, and the
   <10MB crown-jewel artifacts (models/labels/provenance/example_clips) — always.
7. Reproducibility bundle (SoX script + source manifest + checksums, per CLAUDE_release_plan.md)
   is what makes removing bulk resamples SAFE — they can be regenerated on demand.

## Caveat — Gradio review needs the WAVs
The Gradio review tool reads full-month WAVs for playback (`--audio-dir`). If a month might be
re-reviewed by ear (e.g. pending Sept humpback listen; possible May/April re-listens), keep its
WAVs (or restore from backup) before reviewing. The confirmed-clip SUBSET is tiny — keep those
locally regardless.

## NFS note
`du -sh` on these NFS volumes is slow (walks every file, adds IT load). Run ONE at a time, let
it finish (it's slow, not frozen; Ctrl-C often ignored mid-walk). Prefer targeted single-dir
`du` over recursive sweeps. `ls` is instant if you just need to see structure.
