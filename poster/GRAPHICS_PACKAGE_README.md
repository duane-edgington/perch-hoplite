# OCEANS 2026 poster — figures, graphics and logos

Everything used to build **draft v44**, plus the sources it was made from and the candidates that
were considered and set aside. Prepared 24 August 2026.

**Sizes throughout are final printed inches on the 72 × 48 in sheet.** The working `.pptx` is
authored at 36 × 24 (half scale); the full-size PDFs are true 72 × 48.

---

## 01_on_poster_figures — the eleven graphics on v44

**Use the PDF wherever one exists.** Those are true vector: sharp at any size, and safe if a
figure gets scaled during layout. The PNGs are 300 dpi at the sizes below.

| File | Panel | Printed size (in) | Format |
|---|---|---|---|
| `oceans2026_fig1_mars_observatory.jpg` | 2 | 9.00 × 5.92 | JPG, ~145 dpi — see note |
| `poster_fig3_class_spectrograms.png` | 3 | 12.48 × 3.19 | PNG, ~380 dpi |
| `oceans2026_fig2_agile_modeling_loop` | 4 | 12.48 × 4.59 | **PDF** + PNG |
| `gradio_review_panel_crop.png` | 4 | 12.48 × 5.09 | PNG, screenshot crop |
| `tsne_apr2018_oct2020_apr2026_norm_dpi300.png` | 5 | 9.60 × 8.00 | PNG, ~265 dpi |
| `oceans2026_fig3_classifier_trajectory` | 6 | 12.48 × 4.99 | **PDF** + PNG |
| `oceans2026_fig4_calendar_apr_may_2018` | 7 | 12.48 × 5.13 | **PDF** + PNG |
| `tsne_orca_by_day_4days_px30_pres_dpi300.png` | 8 | 12.48 × 8.27 | PNG, ~285 dpi |
| `oceans2026_fig8_threshold_sweep_v4` | 9 | 12.48 × 3.58 | **PDF** + PNG |
| `oceans2026_fig10_v10_holdout` | 10 | 12.48 × 3.20 | **PDF** + PNG |
| `oceans2026_fig9_per_class_f1_v4` | 11 | 12.48 × 3.27 | **PDF** + PNG |

`.json` sidecars carry provenance for the two t-SNE maps.

**One soft spot:** the MARS observatory image is 1362 × 896, about 145 dpi at 9 in wide. Fine at
viewing distance, below the rest. The uncropped original is in `05_unused_candidates/` in case
MBARI comms can supply a larger master — recrop from that rather than upscaling.

## 02_logos

| File | Where | Printed size (in) |
|---|---|---|
| `Logo_OES_hi.png` | top-right block | 4.09 × 1.90 |
| `Logo_O26M_hi.png` | top-right block | 2.50 × 2.50 |
| `general_poster.png` | top-right block | 2.69 × 2.10 |
| `MBARI_logo_type.png` | **top-left corner** (template-mandated) | 7.40 × 2.74 |
| `MBARI_logo+type.eps` | source for the above | vector |

**On the MBARI EPS:** ImageMagick alone renders it at 331 × 123 — it silently falls back to the
file's embedded low-resolution preview. Ghostscript is required:

```
gs -dEPSCrop -sDEVICE=pngalpha -r1200 -sOutputFile=MBARI_logo_type.png MBARI_logo+type.eps
```

That gives 5502 × 2038 with transparency, ~740 dpi at printed size. The supplied PNG is already
that render — no need to redo it.

## 03_source_data_and_scripts

- `make_poster_figures.py` — regenerates every generated figure: `python make_poster_figures.py <outdir>`. Needs the two CSVs beside it.
- `poster_fig4_calendar_apr_may_2018.csv`, `poster_fig8_threshold_sweep_v4.csv` (+ README) — the data behind panels 7 and 9. Note the calendar CSV still carries stale statuses for 21, 23 and 24 April and 7 May; the script overrides them via `STATUS_OVERRIDE` with dated comments.
- `poster_fig3_spectrograms_v5.py` — generator for the class-spectrogram row.
- `build_poster.js` — builds the poster itself (`node build_poster.js`).
- `build_poster_v40_backup.js` — regenerates v40, which had MBARI in the footer rather than the top-left corner.
- `references_verified.bib` — the reference list, checked against sources. Holds five entries beyond the seven printed.

## 04_superseded_do_not_use

Older renders of figures that are still on the poster. **Do not substitute these.**

- `tsne_apr2018_oct2020_apr2026_norm.png` — 823 labeled windows and orca n=219; the poster uses the 1,076-window version.
- `tsne_orca_by_day_4days_px30_pres.png` — 155 dpi; superseded by the `_dpi300` render.
- `tsne_orca_by_day_april2018_px30_pres.png` — early April-only variant.

## 05_unused_candidates

Considered and set aside. Kept for the record.

- `tsne_orca_by_day_5days_px30_pres_dpi300.png` — includes 21 April. **Deliberately excluded from panel 8:** that day's separation rests on 2 recordings inside a 10-minute window, against 25 April's 10 recordings across 3.7 hours, so it does not carry the same evidentiary weight.
- `tsne_orca_by_day_april2018_px30_pres_dpi300.png` — April only, also includes 21 April. Its sidecar caption claims it is "for the OCEANS poster (panel 8)"; **that caption is wrong**, written during the same mix-up.
- `gradio_apr2324_2018_orca_{1,2,3}.png` — full review screenshots from the 23 April session. `_3` is the one cropped for panel 4.
- `gradio_may2018_v10_orca_may16_1.png` — the best-looking spectrogram of the set, held in case a second capture ever earns space. Only one fits: a second would force a v4-based panel below the v10 panel, which breaks the two-model separation.
- `apr23_orca_spectrogram_crop_UNUSED.png` — spectrogram-only crop, no interface.
- `oceans2026_fig1_mars_observatory_UNCROPPED_original.jpg` — source for the panel 2 image.

---

## Print files (not in this package)

The poster itself ships separately as `OCEANS2026_orca_poster_v44_FULLSIZE_72x48_CMYK.pdf`
(send this to the plotter — MBARI's plotters take CMYK only) and `..._RGB.pdf` (the editable
Illustrator source; opens with live vector text, save as `.ai` from there). Both are true
72 × 48 in, so no 200% scaling step.
