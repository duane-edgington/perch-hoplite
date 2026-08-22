# OCEANS 2026 poster — figure delivery manifest

For Melissa. Every figure on draft v12, at the highest resolution available.
**Sizes are final printed inches on the 72 × 48 in poster** (the .pptx is authored at 36 × 24,
i.e. half scale — print at 200% or rebuild at ×2).

**Use the PDF wherever one exists.** PDFs are true vector: they stay sharp at any size, and
they are the safe choice if a figure gets scaled up during layout. The PNGs are 300 dpi at the
sizes below and are fine if placed at or under that size.

| Figure | Panel | Printed size (in) | Best file | Effective dpi | Notes |
|---|---|---|---|---|---|
| MARS observatory | 2 | 9.4 × 6.18 | `oceans2026_fig1_mars_observatory.jpg` | 145 | MBARI illustration, cropped to slot. See "known soft spots" |
| Agile-modeling loop | 4 | 10.2 × 3.75 | **`oceans2026_fig2_agile_modeling_loop.pdf`** | vector | PNG also supplied, 4815 × 1770 |
| All-class t-SNE | 5 | 7.8 × 6.50 | `tsne_apr2018_oct2020_apr2026_norm_dpi300.png` | 325 | Updated 22 Aug: 1,076 windows (was 823), orca n=319 (was 219). The 823-window file is superseded — do not use |
| v0→v4 trajectory | 6 | 11.8 × 4.72 | **`oceans2026_fig3_classifier_trajectory.pdf`** | vector | PNG 4200 × 1680 |
| Class spectrograms | 3 | 16.05 × 4.10 | `poster_fig3_class_spectrograms.png` | 301 | Repo commit `d307d54`, plain "Frequency (Hz)" axis. Raster only — no vector version generated |
| April–May calendar | 7 | 16.05 × 6.60 | **`oceans2026_fig4_calendar_apr_may_2018.pdf`** | vector | PNG 4815 × 1980 |
| By-day t-SNE | 8 | 11.5 × 7.62 | `tsne_orca_by_day_4days_px30_pres_dpi300.png` | 310 | Apr 13/18/25 + May 12, n=473, no Apr 21. Commit `cce5da0`. The 155 dpi file is superseded |
| Threshold sweep | 9 | 16.05 × 4.60 | **`oceans2026_fig8_threshold_sweep_v4.pdf`** | vector | PNG 4815 × 1380 |
| Per-class detection scores | 10 | 16.05 × 4.20 | **`oceans2026_fig9_per_class_f1_v4.pdf`** | vector | PNG 4815 × 1260 |

## Known soft spots — one figure below 300 dpi

Every generated figure is vector plus 300 dpi raster, and both t-SNE maps were re-rendered at
300 dpi on 22 Aug. **One item remains:**

**MARS observatory — 1362 × 896 as cropped, about 145 dpi at 9.4 in wide.** The uncropped
original is included as `..._UNCROPPED_original.jpg` (1362 × 1002) in case MBARI comms can
supply a larger master; if they can, recrop from that rather than upscaling this one.
Acceptable as-is for large-format viewed at a metre or more.

**Superseded — do not use:** `tsne_orca_by_day_4days_px30_pres.png` (155 dpi),
`tsne_apr2018_oct2020_apr2026_norm.png` (155 dpi, and 823 windows rather than 1,076).
**Exploratory only, never for panel 8:** `tsne_orca_by_day_5days_px30_pres_dpi300.png` and
`tsne_orca_by_day_april2018_px30_pres_dpi300.png` — both include 21 April, which panel 8
deliberately excludes. The latter's sidecar caption claims "for panel 8"; disregard it.

## Extras

- `tsne_orca_by_day_april2018_px30_pres.png` — April-only variant of the by-day map. Not used on
  the poster; archived spare.
- A 5-day by-day t-SNE including 21 April exists in the repo and is **deliberately not used**.
  See the production doc, v11 entry, for the reasoning. Please do not substitute it.
- Regenerate any Claude-made figure with `python make_poster_figures.py <outdir>`; it needs
  `poster_fig4_calendar_apr_may_2018.csv` and `poster_fig8_threshold_sweep_v4.csv` alongside it.
