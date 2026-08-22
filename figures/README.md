# figures/ — notes on versioned/superseded files

This directory keeps full history rather than deleting superseded figures — consistent with
this project's practice elsewhere (e.g. classifier versions v5-v8 are kept in the repo even
though only v0-v4 are presented as results). When a figure is regenerated with corrected or
updated data, the old file is usually left in place rather than removed, so provenance and
project history stay intact.

**This means some filenames look similar but represent different points in time.** Check the
figure's own JSON sidecar (same basename + `.json`) for its caption, date, and notes before
assuming which one is current.

## By-day orca t-SNE (`tsne_orca_by_day_*`)

- **`tsne_orca_by_day_4days_*.png`** — SUPERSEDED (as of Aug 21 2026). Covers the 3 April days
  confirmed as of July 2026 (Apr 13, 18, 25) + May 12 — 4 confirmed days total. Predates April
  21's confirmation.
- **`tsne_orca_by_day_5days_*.png`** — CURRENT. Covers 4 confirmed April days (13, 18, 21, 25)
  + May 12 — 5 confirmed days total, reflecting the Aug 21 2026 resolution of the April 21
  "pending review" flag (finding #14).
- **`tsne_orca_by_day_april2018_*.png`** — April-only view. This filename is stable across
  updates (no day-count in the name) — always check its sidecar's date/caption to know which
  day-set it currently reflects, since the underlying image gets replaced in place when the
  April day count changes.

If you're looking for the figure that matches the poster's "eight days of orca" / current
finding #14 state, use the highest day-count / most recent date in the sidecar, not
necessarily the first match you find by filename.
