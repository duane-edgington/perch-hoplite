# OCEANS 2026 Poster — Pre-Print Checklist

**Poster:** "Application of Foundation Model and Agile Modeling to Passive Acoustic Detection of
Orcas in Monterey Bay National Marine Sanctuary" — D. R. Edgington and J. Ryan, MBARI.
**Venue:** IEEE OCEANS Monterey 2026, 21–24 September 2026, General Poster Track.
**Current draft:** `OCEANS2026_orca_poster_DRAFT_v39.pptx` (36 × 24 in, half of 72 × 48).
**Full state and rationale:** `CLAUDE_Oceans_poster_production.md`.

> **Versioning rule (from 20 Aug 2026):** every build that leaves this chat gets a **new**
> version number, even for a one-line change. Never re-issue a number that has already been
> sent. Two different files both called v5 existed briefly; the current draft is **v39** and the
> earlier v5s are superseded — delete them.
**Updated:** 20 August 2026 — v7: spectrogram slot, all-class t-SNE, v0→v4 timeline. Roughly one month of runway.

---

## Blocking — poster is not final until these are done

### 1. Spectrogram row — ✅ PLACED (20 Aug 2026)
`poster_fig3_class_spectrograms.png` is in panel 3 at 16.05 × 4.10 in, which works out to
almost exactly 301 dpi at print size. **Every figure on the poster is now real — no
placeholders remain.**

✅ Axis label resolved at v11: the corrected render (repo commit `d307d54`) reads plain
"Frequency (Hz)" and is now the one on the poster.

### 2. 21 April 2018 — ✅ RESOLVED (21 Aug 2026)
All 25 clips above the cutoff confirmed orca by ear (D. Edgington), no ambiguity. The bar is
mint, the "pending review" callout is gone, the calendar title reads **eight days of orca**, and
the headline stat card moved 7 → 8. Nothing in April or May 2018 is pending now.

⬜ **One housekeeping item:** `poster_fig4_calendar_apr_may_2018.csv` still carries the old
`UNREVIEWED_ACTION_ITEM` status for 2018-04-21. `make_poster_figures.py` overrides it via a
dated `STATUS_OVERRIDE` dict so the figure is correct either way. Regenerate the CSV from the
repo when convenient and the override can be deleted.

### 3. QR code — regenerate when the public repo lands ⬜
**Owners: Duane + Melissa.** Target is currently `github.com/duane-edgington/perch-hoplite`.
The URL is printed in small type beside the code, so **both must change together**: edit the
footer string in `build_poster.js` *and* have Melissa regenerate the QR image from the new URL.
This is the single most likely thing to ship stale.

### 4. Acknowledgements wording ⬜
**Owner: Duane.** Current text: *"MARS is operated by MBARI with support from the David and
Lucile Packard Foundation. Visual sighting context from the California Killer Whale Project and
Monterey Bay vessel reports."* Confirm the funder phrasing, and whether CKWP wants specific credit.

### 5. Gray-whale review (issue #13) — ✅ CLOSED (21 Aug 2026)
**No freeze date needed; there is nothing pending.** J. Ryan's expert judgement: none of the
ambiguous low-frequency calls are gray whale — **all of it is humpback vocalization.** Humpbacks
produce many sounds; song is easy to recognise from its pattern and duration, while the non-song
calls are what a naive listener mistakes for other species. The gray-whale-moan class is not
being added, and the re-review it was waiting on is moot.

**Terminology, poster-wide (applied at v12):** "humpback vocalization" as the general term;
"humpback song" only where song is actually meant; **no mention of gray whales anywhere**, the
single exception being that Bigg's orcas prey on gray whale calves, if prey comes up.

### 6. Logo block — ✅ COMPLETE (23 Aug 2026)
All four marks placed from `poster/logos/`. Top row right-aligned: IEEE OES, OCEANS 2026 Monterey,
General Poster marker. Below them, MBARI at 6.4 x 2.37 in, right-aligned to the same edge.

The MBARI file is an EPS (`MBARI_logo+type.eps`, Illustrator, 2014). ImageMagick alone fell back
to the file's embedded 331 x 123 preview; ghostscript was installed and the PostScript rasterised
properly at 1200 dpi to `MBARI_logo_type.png`, 5502 x 2038 with transparency — about 860 dpi at
printed size. **Use that PNG, not the EPS, and not an ImageMagick conversion without ghostscript.**

**QR decision: footer only.** The slot in the logo block is removed. The QR stays at bottom right
where its URL is printed beside it.

### 7. Scale ⬜
**Owner: Melissa.** The draft is authored at **36 × 24 in** because OOXML caps slide dimensions
at 56 in. Either **print at 200%**, or rebuild in the 72 × 48 template with every element ×2.
Figures are supplied as PDF vector and as PNG at 300 dpi relative to final size, so both routes
hold up.

### 8. References — ✅ VERIFIED (21 Aug 2026)
Checked against source material and returned as BibTeX (`references_verified.bib`, archived with
the figure set). **One real error found and fixed: orcAI is 2026, not 2025** — the volume/year
mismatch flagged in review was genuine. Confirmed clean: both arXiv IDs (2508.04665, 2512.03219),
the OCEANS 2016 DOI, the NOAA catalog DOI, and the Kaggle DOI.

Poster reference block updated at v18 with corrected years, the orcAI article number (e70083),
fuller titles, and DOIs on the three formal publications. The full BibTeX carries five further
entries not shown on the poster (Ghani 2023, Allen 2024 Biotwang, orcAI GitHub, CKWP, Pacific
Sound) — deliberately not added; the selected list stays at seven.

### 9. Full-size proof read ⬜
**Owner: Duane.** Read the entire poster at print size before it goes to paper. References,
captions and the footer are set at 17–19 pt, which looks fine on a laptop at 6% zoom — that is
exactly where a wrong e-mail, a stale URL or a typo survives every earlier check.

---

## Day count — settled at v34 ✅

**14 confirmed orca days**: April 13, 18, 21, 23, 24, 25 (six) and May 2, 3, 7, 12, 13, 14, 16,
29 (eight). The headline carries the attribution rather than hiding it — "ten from v4, four more
when the loop was re-run" — so the growth in the count *is* the agile-modeling argument instead
of an inconsistency with the v4 panels.

## FOR DUANE AND JOHN — headline framing question ⬜

With 23 and 24 April confirmed, April 2018 reads less like separated days and more like a
sustained bout. Project CLAUDE suggests the "N days of orca" framing may now undersell it, and
recommends "a sustained multi-week spring 2018 presence" instead. **Deliberately not acted on** —
this changes the poster's narrative spine, and is for the two of you to decide, not the poster
chat. The factual corrections are in; the framing is untouched.

Note when you discuss it: confirmed April days are 13, 18, 21, 23, 24, 25 — **six of the
thirteen days from 13 to 25 April**, with gaps at 14–17, 19, 20 and 22. "Sustained presence
across two weeks" is well supported; "continuous run" is not what the calendar shows, and a
sharp reviewer will read the gaps straight off the figure.

## Pending — a second ship-noise batch lands 22 Aug ⬜

Panel 10 currently shows **orca_v10** (21 Aug), with ship noise at n=10 / F1 0.80. A larger
ship-noise annotation round on 22 Aug targets n≈44. When the resulting classifier is trained,
update `CLASSES` in `make_poster_figures.py` and rerun — one command, and the sidebar caveat
("still our smallest test set") can soften or go. Nothing else on the poster depends on it.

## Confirm before print ⬜

1. **Label counts — ✅ RESOLVED at v19.** `SELECT COUNT(*) FROM annotations` across all four
   hoplite DBs gives **1,336** exactly (April 2018 685, May 2018 260, October 2020 317, April
   2026 74). The old "~1,450" was an unsourced approximation and is gone. Both numbers now appear
   with their meanings stated: **873** built v4 (the trajectory bars), **1,336** is today's total
   across all months and classes, including confirmations added after v4 was trained.
   ✅ **Effort claim restored at v22.** Duane confirmed 22 Aug that **8–10 h is total effort to
   date**, inclusive of the ~4 h pre-v0 sessions, with the 20–21 Aug annotation work adding
   30 minutes at most. So the figure covers all 1,336 labels and the card reads "in ~8–10 h of
   one expert's time — all of it".
2. **Figure numbers removed — ✅ approved by Duane, 21 Aug 2026.** Captions are descriptive;
   **panel numbers 1–11** are the reference scheme. Danelle and John may still weigh in.

## Resolution — ✅ RESOLVED (22 Aug 2026)

Both t-SNE maps are now 300 dpi. Panel 5 also gained updated data (1,076 windows, orca n=319).
Panel 8's replacement is the correctly-scoped 4-day render, `..._4days_..._dpi300.png`
(commit `cce5da0`), verified before swapping: 214 + 28 + 50 + 181 = 473, no 21 April.

⬜ **Only the MARS image is still below 300 dpi** (~145 dpi at 9.4 in). Fine at viewing
distance; if MBARI comms have a larger master, recrop from the uncropped original supplied in
the figure set.

## Optional — nice, not blocking

- ⬜ **A spectrogram or photo.** The poster is diagram-heavy. A confirmed Bigg's call spectrogram
  (`gradio_apr18_2018_orca_195s_wid202720`, or one of the five Apr 25 clips) or a CKWP photo with
  permission would add warmth to panel 6.
- ⬜ **Cross-reference the companion PyTorch Conf poster.** One line in panel 10 —
  the repo frames them as complementary: OCEANS = the marine-science story, PyTorch = the
  TF-free port that enables it.

---

## Settled — do not reopen

- ✅ **Authorship.** J. Ryan is a co-author and is in the byline; per-panel review credits are
  worded accordingly.
- ✅ **v4 is the presented model.** v5–v8 appear only as named negative results, never as "next".
- ✅ **95% false-positive reduction is v3** (17 hard negatives, 6,489 → 304) — not v3→v4.
- ✅ **Apr 25 2018 = 211 detections, 60 at ≥1.16** (118 was the Apr 23–25 cluster total). The 50
  elsewhere is a review count, not a detection count; the poster keeps them distinct.
- ✅ **Label scope: ~1,450 labels in ~8–10 h**, the whole campaign, matching the documented hours.
- ✅ **`retain_pct` in the sweep CSV** = `det_T1.16 / ref`, not `/det_T0.00`. Resolved, repo fixed
  (`a633399`). FIG 8 plots absolute counts on a log axis — correct regardless, since only 3 of 10
  rows carry a `ref`.
- ✅ **Contacts:** duane@mbari.org, ryjo@mbari.org. Project page:
  mbari.org/project/soundscape-listening-room.
- ✅ **FIG 1 credit:** "Image: MBARI" is in the caption.
- ✅ **No KSBW image** (copyright). Corroboration appears as text only.
- ✅ **Hedging held:** two-pod ↔ two-cluster is "consistent with, not proof of"; 13 May's
  single-clip weakness is stated; macro F1 is never compared across versions.

---

## Handing figures to Melissa — how this actually works

**The poster chat has no live access to the GitHub repo or any file store.** It can only use
what is pasted or uploaded into the conversation directly. Every figure in this set arrived by
manual staging, and future ones will too.

So the loop for any new or updated figure is:

1. **Melissa names what she needs** (a filename, a higher-resolution version, a variant).
2. **Duane fetches it from the repo** and uploads it into the poster chat.
3. **The chat places it, rebuilds, and produces the download** for Melissa.

She cannot ask the chat to "look up" or "pull" a file, and neither can it retrieve one on its
own — a named path is not a retrievable path here. Worth saying this to her explicitly at
handoff; it saves a round of confusion the first time she references a file by name and nothing
comes back.

**Highest-quality material available on request:** every generated figure exists as PDF vector
and PNG at 300 dpi (sized to final printed inches). Ask and they can be re-issued as downloads
at any time. Current figure set:

| File | Panel | Formats |
|---|---|---|
| `oceans2026_fig1_mars_observatory.jpg` | 2 | JPG (MBARI original, cropped) |
| `oceans2026_fig2_agile_modeling_loop` | 4 | PNG 300 dpi + PDF |
| `oceans2026_fig3_classifier_trajectory` | 5 | PNG 300 dpi + PDF |
| `oceans2026_fig4_calendar_apr_may_2018` | 6 | PNG 300 dpi + PDF |
| `tsne_orca_by_day_4days_px30_pres.png` | 7 | PNG (155 dpi at print size — see above) |
| `tsne_orca_by_day_april2018_px30_pres.png` | — | PNG, April-only variant, archived spare |
| `oceans2026_fig8_threshold_sweep_v4` | 8 | PNG 300 dpi + PDF |
| `oceans2026_fig9_per_class_f1_v4` | 9 | PNG 300 dpi + PDF |

---

## Rebuild commands

```bash
python make_poster_figures.py figs     # regenerates all six figures (PNG 300 dpi + PDF)
node build_poster.js                   # rebuilds the .pptx from those figures
python /path/to/validate.py OCEANS2026_orca_poster_DRAFT_v39.pptx
```
`make_poster_figures.py` reads `poster_fig4_calendar_apr_may_2018.csv` and
`poster_fig8_threshold_sweep_v4.csv` from the working directory. Panel geometry lives in the
four `COLUMN n` blocks of `build_poster.js`; each column's panel heights plus 0.6 in gaps must
sum to 33.2 in.
