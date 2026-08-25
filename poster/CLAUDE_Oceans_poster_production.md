# CLAUDE_Oceans_poster_production.md

**State file for the IEEE OCEANS Monterey 2026 poster.** Supersedes `CLAUDE_Oceans_Poster.md`
(that file remains useful background; this one is the production source of truth).
Hand this to a fresh Claude chat to resume without re-deriving anything.

**Last updated:** 23 August 2026 — draft **v44**. Five columns, twelve panels, ten figures,
four logos; no placeholders except the QR image, which awaits its final URL.
**Owner:** Duane R. Edgington, MBARI. **Co-author:** John Ryan, MBARI.
**Poster designer:** **Melissa Rowell** (correct spelling; `CLAUDE_Oceans_Poster.md` has it wrong).

---

## 0d. TERMINOLOGY — humpback, and no gray whales

**Settled 21 Aug 2026 on J. Ryan's expert judgement.** None of the ambiguous low-frequency calls
in this dataset are gray whale. **All of it is humpback vocalization.** Humpbacks produce a wide
range of sounds: song is easy to identify from its pattern and duration, while the non-song calls
are varied and are what gets mistaken for other species by a naive listener.

- **"humpback vocalization"** is the general term throughout the poster.
- **"humpback song"** only where song is specifically meant. The panel 3 spectrogram exemplar
  (MARS_20201025_175314 @ 135–140 s) is genuine song, so that panel title and its caption line
  are correct as written.
- **No mention of gray whales anywhere**, with one exception: Bigg's orcas prey on gray whale
  calves, so gray whales may legitimately appear in a prey/ecology context.
- The model's class is named `humpback_song` in the database but denotes vocalization broadly.
  Figures now display **"humpback vocalization"**; the class name survives only in the panel 5
  t-SNE legend, which is a fixed repo image — the caption notes the class covers vocalization
  generally, not only song.
- **Issue #13 is closed, not pending.** The gray-whale-moan class is not being added. Anything
  in older notes describing gray-whale contamination of humpback labels is superseded — including
  the speculation recorded at v7 about the rejected October 2020 exemplar, which under this
  reading is most likely simply a humpback non-song call rather than a mislabel.

**Why humpback still scores worst:** not label contamination. The class is broad by nature —
one label spanning song and a wide range of other calls. Splitting them is the real next step,
and the likeliest route to lifting the weakest number on the poster.

### Humpback repertoire — background supporting §0d
*Supplied by Duane, 21 Aug 2026, from public references (NOAA Sanctuaries, DOSITS, NPS,
WhaleSound, Whale Trust). Not independently verified here; recorded as context for why one
`humpback_song` label is doing so much work.*

Humpbacks (*Megaptera novaeangliae*) produce two broad categories:

**Song** — long, elaborate, repeating and evolving sequences, sung almost exclusively by males
on breeding grounds. Mixed tonal and pulsed units: high-frequency squeaks, moans and cries
through to deep low-frequency grunts, organised in predictable patterns. *This* is what makes
song easy to recognise — pattern and duration — and it is what the panel 3 exemplar shows.

**Non-song social calls** — used across their range, year-round:
- whistles: piercing high-pitched tones, often in competitive groups
- screeches and growls: harsh, loud, in aggressive or competitive male interactions
- grunts and moans: low-frequency communicative pulses between individuals
- calf chatter: softer squeaks, grunts and clicks

**Feeding calls** — the "train whistle": a loud, intense, single-note pulsed call of about
2.5 s, used to coordinate bubble-net feeding or herd schooling fish.

**Why this matters to the poster.** The repertoire spans high-frequency squeals to low-frequency
rumbles, tonal to pulsed, patterned to isolated — all under one class label. A 5-second window
of a low-frequency moan or an isolated grunt carries almost nothing in common with a song
sequence, which is exactly why the class scores worst and why a naive listener mistakes the
non-song calls for other species. It also retires the old contamination hypothesis on its own
terms: low-frequency moans are squarely inside the normal humpback repertoire, so hearing them
is not evidence of another species.

## 0f. TWO MODELS, TWO ROLES — the v4 / v10 framing

**v4 is the worked example. v10 is the result.** They must never read as rival production models.

- **Panels 1–9 stay on v4** and walk through *how* agile modeling builds a working detector. v4 is now described as "the model built in this walkthrough", not "the model used throughout this poster".
- **Panel 10 is the only place v10's hold-out result lives.** Header "Running the loop again".
- **Panel 11** carries v10's per-class scores, immediately after, and says so in one line.
- **May 2018 is the referee** that bridges them: neither model trained on it, so the comparison is fair. This is why the hold-out must stay a hold-out (§ v15/v17 entries).
- **The poster stays descriptive.** It says the method keeps improving the model; it makes no deployment or production-swap claim, even though the project's own roadmap now recommends v10 replace v4.
- **Do not scatter v10 numbers into v4 panels.** Where the two touch — the day count, the cutoffs — the difference is stated explicitly, never left implicit.

## 0e. SPELLING — US English

Duane is American, the venue is IEEE, the audience is largely US. **Use US spellings.** Fixed at
v24: colored / color, analyzed, neighbors, gray, labeled. Watch the usual drifters — -our vs
-or, -ise vs -ize, -lled vs -led, grey vs gray, centre vs center. "Analysis" is the same in both
and is fine. Applies to figure text too, which is where three of the five hid.

## 0c. AUDIENCE — IEEE OCEANS engineers, not ML people

Assume a capable engineer who has never used machine learning. **F1, logit, embedding,
perplexity, precision, specificity, ROC-AUC and cmap all need glossing or replacing** — an
unexplained "F1" reads as a motorsport reference. The rule is to explain in-line, in a few
words, at first use, rather than to dumb the content down:

- "score" not "logit"; "our cutoff" not "the operating threshold"; the software default of 0.0 is named so the +1.16 choice has meaning.
- "embedding — a 1,536-number fingerprint of the sound" at first use; "fingerprints" thereafter.
- F1 glossed wherever it appears: "0 = useless, 1 = perfect; folds missed calls and false alarms into one score".
- "hard negatives — the detector's own false alarms, labeled as counter-examples".
- t-SNE glossed as "each dot is one 5-second window, placed so sounds the model hears as similar sit close together".
- "false alarms" rather than "false positives" in body text; "seen but not heard" rather than "specificity result".
- ROC-AUC and cmap are left named but glossed in the figure as "0–1 accuracy scores on held-out data — higher is better".

Any new text should be checked against this before it goes on the poster.

## 0b. HARD RULE — experimental model versions stay off the results

**v5, v6, v7 and v8 must never appear as numbered stops on the development timeline, or
anywhere else as if they were part of the model's development sequence.** Duane's explicit
instruction, 20 Aug 2026, consistent with the project's own drafted Poster Narrative Arc.

- The presented arc is **v0 → v1 → v2 → v3 → v4**. v4 is the production model throughout.
- v5 = context-embedding averaging, a negative result. v6/v7/v8 = a 4-season retrain that
  inflated ship_noise and is not production-ready.
- They appear on the poster as exactly one muted line in panel 6: *"Two further experiments —
  averaging embeddings over context, and adding a fourth season directly — degraded specificity
  and are not presented here; detail in the project repository."* No metrics, no version
  numbers, no bars.
- `make_poster_figures.py`'s `fig3_trajectory()` slices `VERSIONS[:5]`; the v5–v8 rows remain in
  the data table for reference but are never plotted. The docstring says why.

## 0a. Versioning rule

**Every build that leaves this chat gets a new version number, even for a one-line change.**
Never re-issue a number already sent. Two distinct files were both labelled v5 (before and
after FIG 5 was placed), which is exactly the confusion this rule prevents. Current draft:
**v44**. Anything labelled v5 is superseded.

## 0. Pre-print checklist — resolve every line before Melissa prints

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | ~~FIG 5~~ | Duane | ✅ **done 20 Aug 2026** — `tsne_orca_by_day_4days_px30_pres.png` placed in panel 7 at native 1.51 aspect. Follow-up: PNG is ~155 dpi at print size; a 300 dpi or vector re-export would match the rest |
| 2 | **21 April 2018** — drawn as a hatched "pending review" bar (approved 20 Aug 2026). Review the day, or ship as-is. If it confirms: bar → mint, drop the caption clause, headline stat 7 days → 8 | Duane | ⬜ open, treatment approved |
| 3 | **QR code** — target is the current repo. When the dedicated public repo exists, change the footer string in `build_poster.js` **and** have Melissa regenerate the QR image. The printed URL sits beside the code, so both must move together | Duane + Melissa | ⬜ open |
| 4 | **Acknowledgements wording** — confirm funder phrasing and whether the California Killer Whale Project wants specific credit | Duane | ⬜ open |
| 5 | **Gray-whale review (issue #13)** — set a freeze date. If results land, panel 10's first bullet becomes a result instead of a plan | Duane + J. Ryan | ⬜ open |
| 6 | **Template block** — replace the dashed "RESERVED" rectangle with the conference logos and the "Regular Paper" marker from `POSTER_TEMPLATE_72x48.pptx`, plus the MBARI logo | Melissa | ⬜ open |
| 7 | **Scale** — poster is authored at 36 × 24 in. Print at 200%, or rebuild in the 72 × 48 template with every element ×2 (§2) | Melissa | ⬜ open |
| 8 | **Full-size proof** — read the whole poster at print size before it goes to paper; small-type items (references, captions, footer) are where errors survive | Duane | ⬜ open |

Optional, not blocking: a confirmed Bigg's spectrogram or a CKWP photo (with permission) in
panel 6 for warmth; a one-line cross-reference to the companion PyTorch Conf poster.

---

## 1. Current state

| Item | Status |
|---|---|
| Accepted abstract (long + short form) | ✅ in hand |
| Poster narrative / content spec | ✅ written (§4), verified against the repo docs |
| Draft poster artifact | ✅ **v5 built** — `OCEANS2026_orca_poster_DRAFT_v44.pptx` |
| Generator script (rebuildable) | ✅ `build_poster.js` (pptxgenjs) |
| Real figure files | ❌ **none embedded** — 5 dashed placeholder boxes, see §5 |
| Official conference template | ❌ not in this chat — logo/marker area is a reserved block, §3 |
| Open questions for Duane | §7 |

### Deliverables
- `OCEANS2026_orca_poster_DRAFT_v44.pptx` — the draft poster.
- `OCEANS2026_orca_poster_DRAFT_v44.pdf` — flat preview for review/e-mail.
- `build_poster.js` — regenerates the .pptx (`node build_poster.js`). Iterate here, not by hand in PowerPoint, while the draft is still moving.
- this file.

---

## 2. Critical technical constraint — poster is authored at HALF SCALE

OOXML caps slide dimensions at **56 in** (`sldSz/@cx ≤ 51206400 EMU`); a genuine 72-in-wide
`.pptx` fails schema validation. The draft is therefore **36 × 24 in — exactly half of 72 × 48,
same 3:2 aspect ratio.**

1. **Print at 200%** (36×24 → 72×48). Supply art at ≥ 600 dpi on the 36×24 canvas (= 300 dpi final).
2. **Or Melissa rebuilds in `POSTER_TEMPLATE_72x48.pptx`**, scaling every element ×2.

In `build_poster.js` a single constant `K = 0.5` scales geometry and font sizes at write time.
**Every coordinate and point size in the script is written in final 72×48 poster units** — read
them at face value. Final sizes: title 62 pt · stat numbers 60 pt · panel headers 34 pt ·
body 22–26 pt · captions 19–21 pt · references 17 pt.

---

## 3. Conference requirements

- IEEE OCEANS Monterey 2026, **21–24 September 2026**. General Poster Track (no full paper, not IEEE Xplore, one display day).
- 72 × 48 in preferred (max); 48 × 36 in minimum; anything between OK.
- **Fixed:** conference logos + poster-type marker ("Regular Paper") top-right as supplied.
- In the draft this is a dashed **"RESERVED — TEMPLATE BLOCK"** rectangle, top right, 14.2 × 5.6 in (poster units). Melissa drops the template block in and adds the MBARI logo.

---

## 4. Poster content (as built in v2)

Title band → 4 headline stat cards → **4 columns × 10 numbered panels**, read down each column.
Palette navy `0B2545` / deep `13315C` / teal `1C7293` / mint `34C6A8` / card `EEF3F7`;
headers Cambria, body Arial.

**Title band.** Title exactly as accepted: *"Application of Foundation Model and Agile Modeling to
Passive Acoustic Detection of Orcas in Monterey Bay National Marine Sanctuary."*
Authors: **Duane R. Edgington and John Ryan** · MBARI, Moss Landing, California, USA.
Strap: IEEE OCEANS 2026 Monterey · General Poster Track · September 21–24, 2026.

**Headline stats.** `6,489 → 304` April-2026 false positives after 17 hard-negative labels (v3) ·
`~870 labels` in ~8–10 h of one expert's time, classifier refits in ~30 s ·
`0.95` orca-class F1 (v4) at the +1.16 operating threshold ·
`7 days` of confirmed Bigg's orca across April–May 2018.

**Col 1** — 1 *The gap* · 2 *A continuous listening post* (MARS, Smooth Ridge 36°42.75′ N 122°11.21′ W, ~30 km offshore, ~900 m; icListen HF 10 Hz–100 kHz, 24/7; public Pacific Ocean Sound archive; 2018–2026) + **FIG 1** · 3 *Off-the-shelf CNNs confuse the bay* (Google multispecies whale + OrcAI: high-confidence orca detections *and* orca/Pacific white-sided dolphin confusion; confusion not sensitivity is the limit; their scores still bootstrap the first labeling round).

**Col 2** — 4 *Agile modeling on frozen embeddings* (Perch 2.0 → linearly separable embeddings; Perch-Hoplite vector DB; search→label→refit(~30 s)→inspect loop; **per-window peak normalization to 0.25** as the enabling fix; one annotator, ~8–10 h, foundation model never fine-tuned) + **FIG 2** · 5 *Labeling waves — and negative results* + **FIG 3** (v0 584 labels recovers 13 Apr 2018; v1 +214 adds Oct 2020, cross-season; v2 +~50 rebalances, still best for Apr/May 2018 at cmap 0.893; v3 +17 hard negatives → 6,489→304 April-2026 FPs; **v4 +8 = the production model**, ROC-AUC 0.959, cmap 0.830, orca F1 0.947 @ +1.16; **v5–v8 kept as negative results** — context averaging cmap 0.830→0.595, 4-season training inflated ship_noise 1,278→4,496).

**Col 3** — 6 *Spring 2018 was not one encounter* (known: 13 Apr only; v4 @+1.16 also flagged 18 Apr (173) and 23–25 Apr (118); 25 and 50 clips reviewed → 75 orca labels, 0 FP; sustained ~2-week April presence; May: 12 May 181/181 plus 13/14/16 May (+8, 13 May is a single clip); two independent methods agree — news/whale-watch record sightings and two identified pods, Alaskan + CA140 matriline) + **FIG 4** · 7 *Structure between encounters* + **FIG 5** (25 Apr evening separates from 13 Apr morning; stable at perplexity 10/30/50; not a single-recording or vessel artifact; two clusters **consistent with, not proof of** the two pods).

**Col 4** — 8 *Seen but not heard* (Oct 2020 and Apr 2026 visually documented, acoustically absent; FPs collapse under thresholding — Oct 2020 144→10 @+1.16, Apr 2026 323→6 @+2.0, while 13 Apr 2018 retains 74–99%; all 10 Oct 2020 survivors reviewed as humpback, zero orca labels in the month; Bigg's orcas are often silent while hunting) · 9 *Measured limits* (orca F1 ≈0.95 n=45 only above +1.16, precision 0.75–0.84 at the 0.0 default; humpback ≈0.55 with real support n=47 — weakest credible class and direct evidence for gray-whale contamination; dolphin ≈0.71–0.77 ceiling; ship_noise 1.00 on n=3 is an artifact; optimal thresholds span +0.16 to +2.47; **whale-watch vessels arrive once orcas are sighted, so ship noise correlates with orca presence and can never be an absence cue**) · 10 *Next* + Acknowledgements + 7 references + QR.

### Claim discipline (keep this if you rewrite)
- **v4 is the presented model.** v5–v8 appear only as named negative results — never as "next".
- The 95% figure is **v3** (17 hard negatives, 6,489 → 304), not v3→v4. v1 in the poster said "v3→v4"; corrected in v2.
- Gray-whale work is **staged/in progress**; no gray-whale numbers appear.
- Two-pod ↔ two-cluster is hedged "consistent with, not proof of".
- 13 May's single-clip weakness is stated in the FIG 4 caption so the calendar isn't read as equal-weight days.
- Macro F1 is not compared across versions (different eval sets); only per-class F1 is quoted.
- No KSBW image (© KSBW). Corroboration is text only: "regional news and whale-watch reports".
- J. Ryan is a **co-author** (byline), so review credits in panels are written as plain "expert review", not as a third-party attribution.

---

## 5. Figures still needed (the only real blocker)

Sizes are **final 72×48 poster inches**; supply vector/PDF where possible, else ≥300 dpi at that size.

| Box | Size (in) | Status | File |
|---|---|---|---|
| **FIG 1** | 10.6 × 6.97 | ✅ **placed** | `oceans2026_fig1_mars_observatory.jpg` — MBARI's MARS observatory illustration, cropped 6%/4.5% off top and bottom for the slot. **Credit "Image: MBARI" is in the caption** |
| **FIG 2** | 16.05 × 5.9 | ✅ **built** | `oceans2026_fig2_agile_modeling_loop` — the loop, panel 4 |
| **FIG 3** | 16.05 × 7.2 | ✅ **built** | `oceans2026_fig3_classifier_trajectory` — v0→v8 dated timeline, panel 5 |
| **FIG 4** | 16.05 × 6.6 | ✅ **built** | `oceans2026_fig4_calendar_apr_may_2018` — per-day calendar, panel 6 |
| **spectrograms** | 16.05 × 4.10 | ✅ **placed** | `poster_fig3_class_spectrograms.png`, panel 3, ~301 dpi at print size. Generator: `poster_fig3_spectrograms_v5.py`. **Open:** delivered render still labels the y-axis "Frequency (Hz, mel)" although the handoff notes say it was corrected to plain "Frequency (Hz)" |
| **all-class t-SNE** | 8.2 × 6.83 | ✅ **placed** | `tsne_apr2018_oct2020_apr2026_norm.png` — 823 windows, 6 classes, 3 seasons, panel 5. Argues "the classes separate"; the by-day t-SNE argues "and within orca, encounters do too" |
| **by-day t-SNE** | 11.5 × 7.62 | ✅ **placed** | `tsne_orca_by_day_4days_px30_pres.png` (4 days, perplexity 30, dark presentation style). April-only variant `tsne_orca_by_day_april2018_px30_pres.png` archived as a spare. Native aspect 1.51 preserved — column 3 was re-proportioned (panel 6 → 16.8 in, panel 7 → 15.8 in) rather than squashing a scatter plot |
| **FIG 8** | 16.05 × 4.6 | ✅ **built** | `oceans2026_fig8_threshold_sweep_v4` — threshold sweep, panel 8 (was four bullets) |
| **FIG 9** | 16.05 × 4.2 | ✅ **built** | `oceans2026_fig9_per_class_f1_v4` — per-class F1, panel 9 |

All generated figures come from **`make_poster_figures.py`** (matplotlib, Liberation Sans =
Arial metrics, poster palette, sized to final printed inches), written as **PNG at 300 dpi and
PDF vector**. It reads `poster_fig4_calendar_apr_may_2018.csv` and
`poster_fig8_threshold_sweep_v4.csv` from the working directory. The `.pptx` embeds the PNGs;
give Melissa the PDFs for vector placement. Suitable for `tools/register_figure.py`.

**All nine figures are real. No placeholders remain.**

**Panel structure changed at v7** — eleven panels now, so panel numbers have shifted. Column 1:
1 the gap, 2 listening post + MARS, 3 CNN confusion + spectrograms. Column 2: 4 method + loop,
5 all-class t-SNE, 6 v0→v4 trajectory. Column 3: 7 spring 2018 + calendar, 8 by-day t-SNE.
Column 4: 9 specificity + sweep, 10 measured limits + F1, 11 next. The MARS image was reduced
from 6.97 to 5.26 in tall to make room, as Duane approved.

---

## 6. Verified source data for FIG 3 and FIG 4

### FIG 3 — classifier trajectory (README.md + agile_modeling_history.md)

| Ver | New labels | ROC-AUC | top1 | cmap | macro F1† | Status / key insight |
|---|---|---|---|---|---|---|
| v0 | 584 (total) | 0.9773 | 0.9405 | 0.8810 | — | baseline, April 2018; 13 Apr event identified |
| v1 | +214 | 0.9533 | 0.9559 | 0.7999 | 0.799 | +Oct 2020 humpback; first cross-season |
| v2 | +~50 | 0.9654 | 0.9438 | 0.8930 | 0.897 | dolphin/other balance; **best for Apr/May 2018** |
| v3 | +17 | 0.9467 | 0.9481 | 0.7370 | — | hard negatives → **April-2026 FPs 6,489 → 304 (−95%)** |
| **v4** | **+8** | **0.9590** | **0.9650** | **0.8297** | **0.830** | **production — the presented model** |
| v5 | 0 (experiment) | 0.9303 | 0.9301 | 0.5945 | — | ❌ context embeddings (30 s Gaussian avg) hurt |
| v6 | +227 May, +~350 bg | 0.9499 | 0.9409 | 0.7763 | — | ❌ ship_noise inflated 1,278 → 4,496 |
| v7 | — | 0.9499 | 0.9409 | 0.7763 | — | ❌ identical to v6 (negative→orca_call fix) |
| v8 | — | 0.9463 | 0.9347 | 0.6489 | — | ❌ still inflated (background→other fix) |

† macro F1 at F1-optimal per-class thresholds; **do not compare across rows** (different eval sets, inflated by low-support classes).

Per-class F1 (v4, n_eval=296): orca 0.947 @ +1.16 (n=45) · dolphin 0.765 @ +2.05 (n=38) ·
humpback 0.548 @ +0.98 (n=47) · other 0.889 (n=10) · ship_noise 1.000 (n=3 ⚠ artifact).

Two beats that make FIG 3 a research story rather than a version list: **v3's 95% FP drop from
17 labels**, and **v5–v8 kept as published negative results**.

### FIG 4 — April/May 2018 (authoritative table, `CLAUDE_perch_hoplite.md`; the older
`agile_modeling_history.md` "Validated Events" table was stale and has been fixed)

| Day | Detections | Review | Outcome |
|---|---|---|---|
| Apr 13 2018 | 289 (v2) | Gradio + J. Ryan | ✅ confirmed event, morning ~06:49–08:49, 13 recordings |
| Apr 18 2018 | 173 @≥1.16 (v4) | 25/25 (D.E.) | ✅ confirmed bout, ~10:39–11:59, 5 recordings |
| Apr 23–25 2018 | 118 @≥1.16 (v4) | 50/50 on Apr 25 (D.E.) | ✅ confirmed, **evening** ~18:49–22:29, 10 recordings; Apr 23/24 pending |
| May 12 2018 | 181 (v4) | 181/181 | ✅ confirmed event, ~07:59–11:09, 15 recordings |
| May 13 2018 | 1 @≥1.16 | by ear (D.E.) | ✅ confirmed — **single clip, weakest of the set** |
| May 14 2018 | 4 @≥1.16 | 4/4 (D.E.) | ✅ real secondary event, ~06–07 h |
| May 16 2018 | 3 confirmed (+1 too faint) | (D.E.) | ✅ ~15:09; two clips from one recording sound audibly different — repertoire variation |
| Oct 2020 | 144 @0.0 → 10 @≥1.16 | J. Ryan 10/10 | ✅ acoustically silent — humpback FPs, 0 orca |
| Apr 2026 | 323 (v4) @0.0 → 6 @+2.0 | Gradio top-25 | ✅ acoustically silent — all humpback FP |

Totals: April orca labels 219→294 (+75, 0 FP @≥1.16); May 181→189 (+8).
Threshold sweep behaviour for the figure: FPs collapse (Oct 2020 144→1, Apr 2026 323→6 at T=0→+2.0)
while confirmed events retain (Apr 13 99%→74%, May 12 95%→40%).
**Do not equal-weight the days** — encode detection count, not just presence.

---

## 7. How to rebuild

```bash
node build_poster.js
python /mnt/skills/public/pptx/scripts/office/validate.py OCEANS2026_orca_poster_DRAFT_v44.pptx
python /mnt/skills/public/pptx/scripts/office/soffice.py --headless --convert-to pdf OCEANS2026_orca_poster_DRAFT_v44.pptx
pdftoppm -jpeg -r 60 OCEANS2026_orca_poster_DRAFT_v44.pdf slide   # then view slide-1.jpg
```
Panel geometry is in the four `COLUMN n` blocks; each column's panel heights + 0.6 in gaps must
sum to **33.2 in** (content runs y = 12.2 → 45.4 in poster units). `figBox()` draws a placeholder —
swap for `s.addImage({ path, x, y, w, h })` when art exists (the `K` wrapper does not scale
`addImage`, so route it through the wrapper or halve the numbers yourself).

---

## 8. Open questions for Duane

1. **Acknowledgements wording** — current: "MARS is operated by MBARI with support from the David and Lucile Packard Foundation. Visual sighting context from the California Killer Whale Project and Monterey Bay vessel reports." Confirm funder phrasing and whether CKWP wants specific credit.
2. **QR — regenerate before print.** Target is currently `github.com/duane-edgington/perch-hoplite`. Duane plans to stand up a **separate public repo**; when that URL exists, update the footer string in `build_poster.js` **and** have Melissa regenerate the QR image from the new URL. This is the single most likely thing to ship stale.
3. **Gray-whale re-review (issue #13)** — April 2018 41-clip review is prepped and paused pending J. Ryan's availability; April 2026 (39 clips) is a one-command repeat. If it lands before ~10 Sept it can turn panel 10 into a result. Suggest a hard freeze date.
5. **Cross-reference the companion PyTorch Conf poster?** `oceans_2026_acceptance_record.md` frames them as complementary (OCEANS = marine-science story, PyTorch = the TF-free port). One line in panel 10 would do it, if wanted.
6. **A spectrogram or orca photo?** The poster is diagram-heavy. A confirmed Bigg's call spectrogram (e.g. `gradio_apr18_2018_orca_195s_wid202720` or the Apr 25 clips) in panel 6, or a CKWP photo with permission, would add warmth. Say the word and it can share FIG 4's space.
7. **Does the official template impose fonts/colors** beyond logos and the marker? Worth a glance at slide 0.

### Data status — all clear

Every data request has been answered. Nothing outstanding except the FIG 5 image file and the
gray-whale review's outcome.

**Corrections absorbed 20 Aug 2026 (from `correction_and_fig_data_for_poster_chat.md`):**
- ✅ **Apr 25 2018 = 211 detections, 60 at ≥1.16** — not 118. (118 was the Apr 23–25 three-day cluster total.) Panel 6 and FIG 4 both use 60. The **50** elsewhere is a different quantity: clips expert-reviewed and confirmed by ear. Poster keeps them distinct — 60 is a detection count, 50 is a review count.
- ✅ **21 April 2018: 40 detections, 25 at ≥1.16, never reviewed.** Drawn in FIG 4 as a **hatched grey bar labelled "pending review"** — neither claimed as orca nor dismissed as noise. Taking the sanctioned second option rather than dropping the day, because a per-day bar chart with a silent gap at 21 April would imply the day was checked and found empty. **✅ Signed off by Duane, 20 Aug 2026** — this treatment is approved for the draft. It stays on the pre-print checklist (§0, item 2) because the underlying day is still unreviewed: if Duane reviews 21 April before print and it confirms, the bar becomes mint, the caption line drops, and the headline stat moves 7 days → 8. If it does not confirm, the hatched bar and its label ship as they are.
- ✅ Other per-day numbers now come from the CSV, superseding earlier prose: Apr 13 = 285/251 (docs previously said 289), May 12 = 181 total / 111 at ≥1.16.
- ✅ **`retain_pct` — resolved 20 Aug 2026, no poster change.** The percentages were never wrong: `retain_pct_T1.16` = `det_T1.16 / ref`, where `ref` is the original expert-labeled reference count, **not** `det_T0.00`. The `ref` column had been dropped when the poster CSV was simplified, orphaning the percentage from its denominator. With `ref` restored all three populated rows reconcile exactly: Apr 13 251/289 = 86.9%, May 12 111/190 = 58.4%, May 14 4/45 = 8.9%. Fixed in the repo (commit `a633399`) and the refreshed CSV is archived here. FIG 8 still plots **absolute counts on a log axis** — the right call regardless, since only 3 of 10 rows carry a `ref` and mixing "some rows have a meaningful percentage, most don't" into one figure buys nothing over the counts. The reasoning is now a comment in `make_poster_figures.py` so nobody re-derives it.
- ✅ FIG 8 includes 14 May with an explicit note: small-n, and its curve collapses like a false-positive curve, yet all four clips are orca by ear. The docs record that this shape heuristic once misled the project, so the figure says shape alone is not proof.

### Contacts and links (confirmed 20 Aug 2026)

- Duane R. Edgington — `duane@mbari.org` — https://www.mbari.org/person/duane-edgington/
- John Ryan — `ryjo@mbari.org` — https://www.mbari.org/person/john-ryan/
- Project page — https://www.mbari.org/project/soundscape-listening-room/
- Code — https://github.com/duane-edgington/perch-hoplite (QR target; **to be replaced by a dedicated public repo**)
- MARS technology page — https://www.mbari.org/technology/monterey-accelerated-research-system-mars/

## 9. Resolved since v1

- ✅ Authorship: John Ryan is a **co-author**, already in the byline; per-panel reviewer credits reworded accordingly.
- ✅ v5 is **not** "the next retrain" — it is a negative result. Poster corrected.
- ✅ 95% FP reduction attributed to **v3** with the real numbers (6,489 → 304).
- ✅ FIG 5 file identified.
- ✅ May 13/14/16 confirmed (the stale "not yet reviewed" table is superseded).
- ✅ Contacts confirmed for both authors; QR target set to the current repo pending the new public one.

---

## 10. Change log

- **v1 — 20 Aug 2026.** First draft: 36×24 in (half of 72×48), 10 panels, 4 headline stats, 5 figure placeholders, reserved template block. Title corrected to the accepted wording; "Melissa Rowell" spelling fixed.
- **v2 — 20 Aug 2026.** Repo verification pass applied. Rewrote panel 5 around v0→v4 plus v5–v8 negative results; re-attributed the 95% FP drop to v3 with absolute counts; added threshold-collapse numbers and the ship-noise-is-not-an-absence-cue insight; added review counts and the 13 May single-clip caveat; normalization stated as per-window peak to 0.25; headline stat 1 changed from "95%" to "6,489 → 304" and stat 2 to "~870 labels"; FIG 3 rescoped to v0→v8; FIG 5 filename named.

- **v3 — 20 Aug 2026.** Footer now carries both authors' e-mails, the MBARI Soundscape Listening Room project page, and the audio-source credit on a second line; QR block labelled with its actual target (`github.com/duane-edgington/perch-hoplite`) so a stale URL is visible at proof stage rather than invisible inside the code.
- **v4 — 20 Aug 2026.** Three figures built and embedded: FIG 2 (agile-modeling loop), FIG 3 (v0→v8 dated trajectory with labels-per-wave bars, the 6,489→304 callout, and v5–v8 marked as kept negative results), and the panel-9 per-class F1 inset. Generator committed as `make_poster_figures.py`; PNG (300 dpi) + PDF for each. Headline stat 2 changed to "~1,450 labels"; v2 count corrected to +50; panel 9 rebuilt around the chart with three non-duplicating bullets; FIG 1 placeholder now names its source URL. In FIG 3 the label bars stay teal even for v5–v8 — those labels were sound, it was the training configuration that failed; only the version ticks are marked red.
- **v5 — 20 Aug 2026.** FIG 1 (MARS observatory image, MBARI, cropped and credited), FIG 4 (April–May 2018 per-day calendar) and FIG 8 (threshold sweep, replacing panel 8's bullets) built and embedded; column 1 re-proportioned to give the MARS image a real slot. Apr 25 corrected 118 → 60 in panel 6. 21 April drawn as "pending review". FIG 5 is the only remaining placeholder.
- **20 Aug 2026 (no version bump).** Duane signed off on the 21 April "pending review" treatment. Added §0 pre-print checklist so the eight open items are visible at the top of this file rather than scattered through it. `retain_pct` discrepancy in the sweep CSV passed back to the project CLAUDE.
- **20 Aug 2026 — FIG 5 placed.** Panel 7 now carries the real by-day t-SNE (4-day, perplexity 30, presentation style) at its native 1.51 aspect; column 3 re-proportioned to fit it undistorted. Recorded in the checklist that this chat has no live repo or file-store access — every figure must be staged manually by Duane, which is worth telling Melissa at handoff so she doesn't expect the chat to retrieve files she names.
- **v6 — 20 Aug 2026.** Renumbered from the second v5 build (no content change at renumber). Versioning rule added: new number for every build that leaves the chat.
- **v7 — 20 Aug 2026.** Big structural revision. (1) v5–v8 removed from the trajectory figure and all poster text per hard instruction; FIG 3 rebuilt as a clean five-wave v0→v4 arc, with the ~1,450 campaign total reconciled in the subtitle rather than by an extra bar. (2) All-class t-SNE added as new panel 5 — the "linearly separable" premise was previously asserted and never shown. (3) Panel 3 restructured around a class-spectrogram row (slot and caption in place, PNG still needed). (4) MARS image reduced to 8.0 × 5.26 in to free the space. (5) Figure numbers dropped from captions; panel numbers 1–11 are the reference scheme.

**Recorded, not for the poster:** the first humpback exemplar tried (MARS_20201001_054822, 155–160 s) was rejected on expert review — all energy below 1000 Hz, no repeating phrase structure, reading as a low-frequency moan despite a confirmed `humpback_song` label. Blue whale is at least as plausible an alternate ID as gray whale, since October is out-of-season for gray whale in the bay but blue whale is dominant then. Unresolved; worth folding into the #13 review. The exemplar actually used is MARS_20201025_175314 @ 135–140 s, confirmed to show genuine humpback structure.
- **v8 — 20 Aug 2026.** Class-spectrogram row placed in panel 3 (16.05 × 4.10 in, ~301 dpi). Poster is figure-complete. Flagged that the delivered spectrogram PNG still carries the "Frequency (Hz, mel)" axis label the handoff notes described as corrected.
- **v9 — 20 Aug 2026.** No new content: panel 3 tightened (the spectrogram row sat ~1.4 in below its bullets, leaving dead space) and the freed height returned to the MARS image, now 9.4 × 6.18 in. Column 1 panels: 7.4 / 13.8 / 10.8.
- **v10 — 21 Aug 2026.** Three changes. (1) **April 21 2018 confirmed** (all 25 clips, D. Edgington, 21 Aug): bar recoloured, calendar retitled "eight days of orca", headline stat 7 → 8, panel 7 rewritten to 100 confirmed clips across three April days, and May 7 given its own amber key as "listened to, too faint to call". A `STATUS_OVERRIDE` in `make_poster_figures.py` carries the correction until the CSV is regenerated. (2) **Stat band reordered and de-jargoned** — the finding leads ("8 days"), the method win follows as "95% fewer" with the raw counts kept as supporting evidence, and version numbers were dropped from the cards. (3) **Jargon pass for an engineering audience** across every panel and figure; see §0c.
- **v11 — 21 Aug 2026.** Review round from the project CLAUDE, all four items actioned. (1) Loop diagram output box rewritten from "orca logit ≥ +1.16 operating threshold" to "Scan months of audio / keep scores above +1.16 (software default: 0.0)" — the last "logit" on the poster, and inside a graphic where the §0c pass had missed it. (2) Corrected spectrogram panel pulled from repo commit `d307d54`; y-axis now reads plain "Frequency (Hz)". (3) Reference verification added to the checklist as a human task, with the *Marine Mammal Science* 42(1)/2025 volume-year mismatch as the first lead — Claude cannot verify citations, since arxiv.org and doi.org are outside the container's allow-list. (4) Panel 5 caption gained "as labeled at the July 2026 snapshot", which explains the stale n=219 legend without regenerating a figure that honestly depicts that moment.

**Decision recorded — do not swap the 5-day by-day t-SNE into panel 8.** A 5-day version including April 21 exists in the repo and is deliberately not used. April 21 is genuinely confirmed orca (25/25 listened, no ambiguity), but the recording-spread check that validated April 25's separation comes back differently for it: 2 recordings inside a ~10-minute window (23:19–23:29) versus April 25's 10 recordings across 3.7 hours. A brief single-pass event is as well explained by animals passing close to the hydrophone — a range effect on the embedding — as by a distinct encounter, so its separation does not carry the same evidentiary weight. Panel 8 stays at 4 days, n=473. Newer is not the same as right for the panel.
- **v12 — 21 Aug 2026.** Terminology correction on J. Ryan's judgement (see §0d). Gray whales removed from the poster entirely. Panel 10's explanation of humpback's weak score changed from "some of its labels are really gray whale" to the class being broad by nature — song easy, non-song calls varied — which is both correct and a better argument. Panel 11's staged gray-whale-moan work replaced with splitting song from non-song calls as the next classifier question. F1 chart now uses plain display names ("humpback vocalization", "orca call", "dolphin call", "ship noise") rather than database class names. Panel 5's caption notes that its legend's `humpback_song` covers vocalization broadly.
- **v14 — 21 Aug 2026** (v13 skipped, as hotels skip the thirteenth floor). Panel 10 rebuilt on **orca_v10**: same three-season recipe as v4 (April 2018 + October 2020 + April 2026), retrained on 1,076 annotations against v4's 803, `--num-steps 512`. Eval set 296 → 459 held-out windows. Scores at each class's own cutoff: orca call 0.94 (n=61, +2.31), ship noise 0.80 (n=10, +1.81), dolphin call 0.69 (n=44, +1.93), humpback vocalization 0.62 (n=67, +0.79), other 0.59 (n=22, +1.31). Rows re-ordered by score; **ship noise is no longer hatched** — at n=10 it is a real result rather than v4's n=3 artifact, with a softened caveat rather than the old "artifact, not skill". A muted line under the panel header states that panels 1–9 present v4 and these scores come from the newer retrain, so the two never read as one model. Panel 11's stale bullet replaced and moved to lead: the retrain is **done** and was deliberately **three-season**, with May 2018 held out because the four-season mix is what previously inflated ship-noise false alarms. Two-bar design retained deliberately — Danelle and John are to see the busier version and weigh in.

**Three questions resolved with project CLAUDE, 21 Aug 2026 (still v14):**
1. **Headline card stays at 0.95.** It sits above panels 1–9 and makes a claim about the poster's main model (v4); moving it to `orca_v10`'s 0.94 to match one panel would put it out of step with the nine panels beneath it. The 0.95/0.94 difference is within run-to-run noise on a changed eval set anyway.
2. **The +1.16 / +2.31 ambiguity is now stated outright**, not left to the identity note. Panel 10's first bullet ends: *"orca_v10's +2.31 is its own optimum, not a correction to the +1.16 used with v4 elsewhere on this poster: different model, different evaluation set."* A Perch-literate attendee would otherwise read two cutoffs for one class as sloppiness.
3. **"other" now has a mechanism, not just a size change.** v4 measured `other` almost entirely on April 2018 clips — one season, one recording context. orca_v10 pools April 2018 + October 2020 + April 2026 catch-all clips in a held-out set for the first time. The chart note reads: *"'Other' fell furthest — it now pools catch-all clips from three seasons at once, and may hide several sound types itself."* Deliberately no counts: the 52→124 figure is a label count and would clash with the n=22 held-out support shown on the same row.

- **v15 — 21 Aug 2026.** Panel 11's first bullet removed. It framed May 2018 as something to fold back into training "once we understand why it inflated ship-noise false alarms" — but **May 2018 being absent from training is a feature, not a gap** (Duane, 21 Aug). No model presented on this poster has ever seen May 2018 recordings, so the 12/13/14/16 May confirmations are genuine performance on unseen months. Do not reinstate the bullet, and do not write anything implying May 2018 ought to be added to training.

  **Applied at v16.** Panel 7 now states the claim directly, immediately after the May bullet: *"Neither v4 nor orca_v10 has trained on May 2018 recordings — these are detections in a month the presented models have never seen."* (wording tightened at v17) This is the poster's only explicit cross-month generalization claim, and it depends on May 2018 staying out of training — another reason not to reinstate the removed panel 11 bullet.
- **v16 — 21 Aug 2026.** Cross-month generalization line added to panel 7 after the May bullet. Panel 7 now runs six bullets; the calendar figure and caption were unaffected, and column 3 still balances at 33.2 in.
- **v17 — 21 Aug 2026.** Panel 7's generalization line reworded to name the two presented models rather than claim "no model here". **May 2018 is a deliberate hold-out and stays one.** The v6/v7/v8 experiments did train on May 2018 and are being discarded; v4 and orca_v10 are clean, so May 2018 remains pure test data for both. The earlier phrasing was loose in a way that mattered: an experimental model *did* see May 2018, and a reviewer who knows the v6–v8 history could have read "no model here" as an overclaim. Naming v4 and orca_v10 makes the statement exactly true and survives that scrutiny.
- **v18 — 21 Aug 2026.** Reference block rebuilt from a verified BibTeX file supplied by Duane. **orcAI corrected 2025 → 2026** (Mar. Mamm. Sci. 42(1) e70083) — the volume-year mismatch flagged during review was a real error, not an artifact of early-view numbering. Both arXiv IDs verified correct. Added: Burns et al. year, article number for orcAI, fuller titles, and DOIs for the three formal publications (orcAI, Ryan 2016 OCEANS, McInnes 2023 NOAA). Type size dropped 17 → 16 pt to absorb the extra text; the block still sits inside its card. `references_verified.bib` is archived in the figure delivery folder — it holds five entries beyond the poster's seven, which were deliberately not added.
- **21 Aug 2026 (no version bump).** Duane approved the removal of figure numbers in favour of panel numbers 1–11 as the reference scheme. Subject only to Danelle's and John's view next week. Remaining judgment call still open: the 873-vs-~1,450 framing on the trajectory figure.
- **v19 — 21 Aug 2026.** Label count resolved with an exact figure. `SELECT COUNT(*) FROM annotations` across the four hoplite DBs gives **1,336** (April 2018 685, May 2018 260, October 2020 317, April 2026 74); the previous "~1,450" was an unsourced approximation from an earlier draft. Stat card now reads **1,336 labels / across every month and class today; 873 of them built the presented model** — the two numbers coexist with their meanings stated rather than contradicting each other. Trajectory subtitle carries the same distinction. **Note:** the "~8–10 h of one expert's time" claim came off the card in the process; it was measured against the older label set and predates the 20–21 Aug annotation work, so it needs re-estimating before it goes back on. Flagged in the checklist.
- **22 Aug 2026 (no poster change).** 300 dpi t-SNE renders arrived: `tsne_orca_by_day_5days_px30_pres_dpi300.png` and `tsne_orca_by_day_april2018_px30_pres_dpi300.png`, both 3567 × 2365 (≈310 dpi at print size), with provenance sidecars. Archived in the figure delivery set. **Neither replaces what is on the poster:** panel 8 uses the *4-day* set (Apr 13/18/25 + May 12), and the 5-day version is the one deliberately excluded at v11; the April-only version drops May 12 and so loses both the cross-month element and the caption's "12 May holds its own region a month later". No 300 dpi version of the all-class map (panel 5) exists yet either. Panel 8 therefore stays on the 155 dpi image pending a 4-day 300 dpi render.

  Two things to raise with project CLAUDE: (1) the April-only sidecar caption says "for the OCEANS poster (panel 8)", which suggests a different assumption about what panel 8 should show — worth reconciling; (2) its notes record a real discrepancy — 318 orca windows loaded across the four confirmed April days against a DB total of 319 April `orca_call` labels, i.e. one stray label outside the confirmed-day set, flagged there as not yet investigated.
- **v20 — 22 Aug 2026.** Pre-v0 origin note added under the trajectory figure in panel 6, where the 584-label first bar previously appeared from nowhere: *"Before v0: about four hours of hand review across six sessions on April 2018 audio — two 50-clip rounds of simple orca-or-not calls, then three 25-clip waves that gave dolphin and humpback their own classes, every humpback clip checked by John Ryan. That is where v0's 584 labels came from."* Phrased in round numbers because the source describes itself as a memory-based reconstruction. Column 2 rebalanced to make room: panel 4 10.6 in (loop image 9.6 × 3.53), panel 5 10.0 in (class map 7.8 × 6.5), panel 6 11.4 in.

  Also caught in the same pass: the **loop diagram's own footer still read "Labels: 1,450 across the campaign"** — a hard-coded string inside a figure that the v19 count change had missed. Now "1,336 labels across the campaign". No occurrence of 1,450 remains in either build script.

  **Not used, and available if wanted:** the fuller narrative version, including the 30-second context view — the tool grew that feature because a 5-second window is too short to tell humpback song from anything else, and it is the same feature that later confirmed the panel 3 song exemplar and rejected the first candidate. A nice thread, but there is no space for it without cutting something else.
- **22 Aug 2026 (no poster change).** Both open figure questions answered by project CLAUDE.

  **Panel 8 holds as-is.** The two 300 dpi renders delivered earlier both include 21 April because the generating script's day list had been edited for a separate exploratory question, and that edit changed what the script produces for every subsequent run. The April-only sidecar's "for the OCEANS poster (panel 8)" caption was written in that same error and should be disregarded. The script now supports `--exclude-days`, so the original 4-day set (Apr 13/18/25 + May 12) can be re-rendered at 300 dpi without permanently re-editing the day list; that file is queued and will be named `tsne_orca_by_day_4days_..._dpi300.png`. **Panel 8 stays on the 155 dpi 4-day image until it arrives — do not substitute either delivered file.** Panel 5's all-class map is a separate follow-up.

  **318 vs 319 resolved, no poster impact.** The missing label is a single pre-existing `orca_call` annotation on **23 April** (`MARS_20180423_230912`, 515–520 s) — a leftover from an earlier untracked labeling pass, the same character as the stray found on 21 April. April 23/24 remain in a confirmed cluster but pending individual review, so their status is unchanged. Nothing on the poster is affected: panel 8's 4-day figure never included 23 April, and panel 7's "eight days of orca" already excludes 23 and 24 April. Recorded so that "318, not 319" in that figure's console output is not re-investigated later.
- **v21 — 22 Aug 2026.** Panel 5's all-class map replaced with `tsne_apr2018_oct2020_apr2026_norm_dpi300.png` (2538 × 2116, ≈325 dpi at printed size). **Not merely a resolution bump — the data changed:** 1,076 labeled windows against the old 823, orca_call n=319 against 219, reflecting every label confirmed through the most recent work. Legend verified against the image itself: dolphin 206 + humpback 321 + negative 54 + orca 319 + other 124 + ship 52 = 1,076. Caption rewritten with the new counts; the old figure's "25 April 2026 humpback hard-negative labels land inside the humpback cluster" sentence dropped as instructed, since it was tied to the 823-window snapshot. The "as labeled at the July 2026 snapshot" clause added at v11 is also gone — the figure is now current, so the excuse for a stale legend is no longer needed.

  **Two consistency wins worth noting.** 1,076 + May 2018's 260 = **1,336**, the exact DB total on the stat card — the class map covers precisely the three seasons the models trained on, and the difference is the held-out month. The caption now says so: *"3 seasons — the three the models trained on; May 2018 is held out and does not appear."* That ties panel 5 to panel 7's cross-month claim. **Duane confirmed 22 Aug: stating the hold-out in both places is intended — keep it.** Panel 5 establishes that May is absent from the training data; panel 7 collects the payoff. Do not "de-duplicate" these.

  Cross-references adapted to our scheme: the supplied caption cited "Fig 3" and "panel 10"; figure numbers were retired at v10, so it now reads "the spectrogram row" and "panel 10".

  **Panel 8's by-day map is now the only sub-300 dpi figure on the poster**, pending the correctly-scoped 4-day render.
- **v22 — 22 Aug 2026.** Effort claim restored. Duane confirmed **8–10 h is total effort to date**: inclusive of the ~4 h of pre-v0 sessions, with the 20–21 Aug annotation work adding 30 minutes at most. The figure therefore covers all 1,336 labels, and the stat card reads **"1,336 labels / in ~8–10 h of one expert's time — all of it; 873 of them built the presented model"**. The "all of it" is doing deliberate work: it forecloses the reading that the hours cover only the model-building subset.

  **Double-counting closed.** Panel 6's pre-v0 note said "about four hours of hand review" while panel 4's loop diagram says "~8–10 h of one expert's time" — a reader could have added them to 12–14 h. The note now reads "four of those hours — nearly half the project's total", which subordinates it to the same budget instead of adding to it. Trimmed to 16 pt to fit the card after rewording.

  **Standing rule:** the hours figure appears in three places — stat card, loop diagram, and panel 6's note. If it is ever revised, all three move together. The loop diagram's copy is hard-coded inside `fig2_loop()` in `make_poster_figures.py`, which is exactly where the stale "1,450" hid at v20; grep the figure scripts, not just `build_poster.js`.
- **v23 — 22 Aug 2026.** Panel 8 swapped to `tsne_orca_by_day_4days_px30_pres_dpi300.png` (3567 × 2365, ≈310 dpi at printed size, commit `cce5da0`). Verified before swapping rather than on assurance: legend reads n=473 with 214 + 28 + 50 + 181 across Apr 13/18/25 + May 12, no 21 April, aspect 1.508 against the old 1.509 — a true drop-in. Caption and layout untouched, as instructed.

  **Every figure on the poster is now 300 dpi or vector, except the MARS observatory image (~145 dpi).** The two 155 dpi t-SNE files and the 823-window class map are superseded; the manifest lists them as do-not-use, alongside the two exploratory Apr-21-inclusive renders.
- **v24 — 22 Aug 2026.** US spelling sweep. Five instances fixed: "coloured" and "one colour scale" in the two t-SNE captions, "analysed" in panel 10, "nearest neighbours" in the loop diagram, and "grey"/"labelled" in the calendar's code comment. Three of the five were inside figure scripts rather than poster text. Rule recorded as §0e.
- **v25 — 22 Aug 2026.** Panel 6's bare "v4 (+8)" replaced with "Eight more of the same at v4 give the model used throughout this poster." The +8 meant eight labels, but the unit was last stated back at "v0 (584 labels)" and a long clause about 17 hard negatives and 6,489 → 304 sat in between, so by that point "(+8)" had no visible referent. It also failed to say those eight were themselves hard negatives, which the history records. Now both are explicit.

  **General note for future edits:** bare "(+N)" constructions in the version narrative depend on a unit stated earlier in the same bullet. If a bullet is split or reordered, re-state the unit.
- **v26 — 22 Aug 2026.** Reworded again: **"Eight more corrective labels produce v4, the model used throughout this poster."** The v25 phrasing ("Eight more of the same at v4 give the model...") was awkward — "of the same" leaned on the previous clause and "give the model" had no clear object. "Corrective labels" is the fix: it names what the eight are, and it echoes the second stat card's "from 17 corrective labels", so the two mentions of hard negatives now use one phrase across the poster.
- **v27 — 22 Aug 2026.** Pre-v0 note corrected on two counts. (1) **"three 25-clip waves" was factually wrong** — the source describes three *sessions* worked in 25-clip chunks at roughly 150 clips each, about 450 in total; my phrasing implied 75. Now "three longer sessions worked 25 clips at a time". (2) **"across six sessions" invited arithmetic that failed** — the sentence then described two rounds plus three sessions, which is five. The source says "~6 sessions" loosely; the total is now simply dropped, and the note describes the two phases without claiming a count. Note box grown to 1.05 in to hold the slightly longer text.

  Final wording: *"Before v0: four of those hours — nearly half the project's total — went on hand review of April 2018 audio. Two 50-clip rounds settled orca-or-not, then three longer sessions worked 25 clips at a time to give dolphin and humpback their own classes, every humpback clip checked by John Ryan. Hence v0's 584 labels."*
- **v28 — 23 Aug 2026.** Saturday's listening sessions folded in. **23 and 24 April 2018 are now confirmed orca days** — all 58 detections above the cutoff reviewed in 18 minutes: 55 orca, 2 humpback, 1 unlabeled. **13 May is no longer a single-clip day** — full review down to the 0.0 floor gives 8 confirmed orca clips (1 above the cutoff, 7 below) plus 2 dolphin. Calendar retitled "ten days of orca"; both stale caveats removed from the caption; the "detected, not individually listened to" legend entry deleted, since no day on the poster now carries that status. `STATUS_OVERRIDE` extended to cover 23 and 24 April until the CSV is regenerated.

  **Headline count changed 8 → 10, deliberately.** The instruction was to hold the headline, but that concerns the *framing* ("N days" versus "sustained presence"), not the count. Leaving "8 days" beside a calendar showing ten confirmed days would have been an internal contradiction. Reverting is a one-line change if Duane disagrees.

  **Framing question flagged, not acted on** — see the checklist. One caution recorded there: confirmed April days are 13, 18, 21, 23, 24, 25, which is six of the thirteen days from 13 to 25 April, with gaps at 14–17, 19, 20 and 22. "Sustained presence" holds; "continuous run" does not match what the calendar plots.

  Calendar subtitle changed from "Spring 2018 is episodic, not continuous" — which now reads as arguing against the data — to the plainer "Six of thirteen days from 13 to 25 April carry confirmed orca."

  **Not used:** `figures/gradio_apr2324_2018_orca_{1,2,3}.png`, three expert-confirmed example clips from the 23 April session. Available if a panel ever wants a second spectrogram.
- **23 Aug 2026 — April 23 framegrabs assessed, not placed (no poster change).** `figures/gradio_apr2324_2018_orca_{1,2,3}.png` fetched and archived in the figure delivery set. They are **Gradio review screenshots**, not figures: label buttons, waveform, audio transport and the 30 s context view alongside the 5 s clip. A usable crop exists — clip 1, `MARS_20180423_225912` at 310–315 s, score 1.437, shows clear orca harmonic structure between 500 Hz and 4 kHz. A sample crop is staged as `apr23_orca_spectrogram_CROP_sample.png`.

  **Recommended against placing it, for three reasons.** (1) **No room.** Column 3 is exactly full at 33.2 in; the crop needs ~3.7 in at a legible width, which means shrinking the calendar and the by-day t-SNE by roughly 15% each — two figures that are carrying real argument, in exchange for one that repeats a point. (2) **Panel 3 already shows an orca call spectrogram**, matched against three other classes under identical FFT parameters and one color scale; a second, unmatched orca spectrogram is weaker evidence than the first. (3) **The y-axis reads "Hz (mel)"** — the exact label the team corrected in the panel 3 figure, on the grounds that the binning is mel-scale while the axis renders linear. Placing this crop would reintroduce it.

  If Duane wants it anyway, the cleanest version is not a spectrogram at all: the *full* screenshot, captioned as the review tool, would show the human-in-the-loop step concretely — but that duplicates the loop diagram's message and would need the same space.
- **23 Aug 2026 (no poster change).** Review-panel images held pending a Duane/John discussion; Duane's leaning is to keep the MARS picture, which means the screenshot most likely stays off. All candidates archived in the figure delivery set with `_UNUSED` suffixes, and the space arithmetic written into the manifest so the "why not just shrink the other figures?" question is answered without re-deriving it. **Correction recorded against my own earlier advice:** the "shrink both neighbors by ~15%" estimate I gave was wrong — 15% of a figure's width buys only 15% of its height, yielding 2.25 in against the ~5.0 in needed.
- **v29 — 23 Aug 2026.** Stat card 3 reworded: **"1,336 labels / every one of them in ~8–10 h of one expert's time; 873 built the presented model."** The v22 phrasing ended "— all of it", which had no antecedent: labels are plural, so it would need "all of them", and the nearest noun was the time, making the phrase read as nonsense. "Every one of them" does the intended job — it forecloses the reading that the hours cover only the 873 — and does it grammatically. "873 of them built" also lost its now-redundant "of them".
- **v30 — 23 Aug 2026.** "one expert's time" → **"researchers' time"** on the stat card and in the loop diagram's header. The 8–10 h is collected time across Duane and John (mainly Duane, but the poster does not need to apportion it), and "researchers' time" is the extended abstract's own phrasing, so the poster and abstract now agree. Both instances changed together per the standing rule; the loop diagram's copy is inside `fig2_loop()`, not `build_poster.js`. Panel 6's pre-v0 note needed no change — "four of those hours" is neutral as to who spent them, and it already credits John by name for the humpback checks.
- **v31 — 23 Aug 2026. Major restructure: five columns, twelve panels, and the v10 result.** The sheet is 72 x 48 in, and at five columns each is 13.68 in wide with a 12.48 in content width — comfortable for 22–24 pt body text at roughly 65 characters a line. This added ~33 in of column length, which paid for the new panel *and* let several figures grow rather than shrink: the by-day t-SNE is now full column width at 12.48 x 8.27 in, the class map 9.6 x 8.0.

  **New panel 10, "Running the loop again"** — the only place v10 appears as a result. Carries a new comparison figure (`oceans2026_fig10_v10_holdout`) showing mean score on confirmed orca (2.61 v10 against 1.65 v4) and recall at the cutoff (80% against 61%), plus the four findings: 192 of 195 shared windows scored higher, 14 of 14 above-threshold detections confirmed orca by ear, four new days (2, 3, 7, 29 May), and the explicit note that May's four confirmed days become eight and panel 7's ten become fourteen. The precision caveat is stated as the source asked: "no false alarms among those reviewed", with "May is not exhaustively labeled, so this is not a month-wide false-alarm rate."

  **Fifth stat card added: "+19 points"** — more of a held-out month's orca found when the loop was run again.

  **The Gradio review screenshot went into panel 4**, under the loop diagram, at 12.48 x 5.45 in — the space five columns freed. Captioned with the six label names, since the buttons cannot be made readable at print size.

  **Panels 1–9 untouched on v4 numbers.** Panel 6's "the model used throughout this poster" softened to "the model built in this walkthrough". Panel 7's calendar caption now points forward: "Four more May days, found later by v10, are not shown here; see panel 10."

  **Open:** column 4 runs about 6 in shorter than its neighbors — the by-day t-SNE and sweep are both width-limited, so they cannot grow to fill it. Not visually serious, but a v10-validation framegrab in panel 10 would bring column 5 to full height, and column 4 could take a second figure if one exists.
- **v32 — 23 Aug 2026.** Panel 4's review capture swapped to the recommended v4 pick, `gradio_apr2324_2018_orca_3.png` (23 April 2018, MARS_20180423_225912, score 2.03) — a cleaner, stronger stepped-harmonic call than the earlier `_1`, and an April clip, which keeps it inside the v4 walkthrough. Caption now names the model role explicitly ("surfaced by v4"), gives the file and score, and states that reviewers also see a 30 s context view that the crop omits. Crop boundary detected programmatically off the card edge rather than eyeballed, so the two captures share a consistent framing.

  **Only one capture fits.** Working through every arrangement: a capture cropped to the clip card is 12.48 x 5.09 in at column width, and column 5 has 0.8 in of slack. Every rearrangement that frees ~6.5 in for a second capture ends up placing a v4-based panel *after* the v10 panel — which is precisely the muddle §0f exists to prevent. Recorded so the option is not re-explored from scratch.

  `gradio_may2018_v10_orca_may16_1.png` is fetched and archived in the delivery set, ready if a second capture ever earns space.
- **v33 — 23 Aug 2026.** Panel 12's second bullet was wrong: it listed "a separate score cutoff per class" as future work, but per-class cutoffs are already in use and panel 11 displays five of them. Corrected to the real future item — once humpback song and non-song calls become separate classes, each of the two will need its own cutoff — and the ecotype extension split into its own bullet. Panel 12 now has three bullets.

  **Naming point worth a decision (not applied):** Duane's phrasing for the split was "humpback_song and humpback_vocalization". Under §0d, *humpback vocalization* is the umbrella term covering song and non-song alike, so using it as the name of the non-song class would collide with the term used everywhere else on the poster. The bullet is currently written to avoid naming the two classes at all ("each of the two"). **Duane confirmed 23 Aug: the class names have not been discussed or decided, so the non-committal wording is correct and should stay.** If names are settled later and the non-song class ends up called `humpback_vocalization`, §0d's poster terminology needs revisiting at the same time — the two decisions are coupled.
- **v34 — 23 Aug 2026.** Two changes, both closing stale numbers.

  **Headline is now "14 days"**, with the attribution built into the card: *"of confirmed orca in April–May 2018: ten from v4, four more when the loop was re-run."* This takes the option that turns the two-model split into the argument rather than a discrepancy — an audience reading "14" against v4-only panels would otherwise ask how v4 found days its own recall could not reach. Confirmed days: April 13, 18, 21, 23, 24, 25 and May 2, 3, 7, 12, 13, 14, 16, 29. Panel 7's bullet now says "ten... from v4 alone", and its caption spells out the arithmetic: ten under v4, plus 2, 3, 7 and 29 May from v10, fourteen in all.

  **May 7's amber marker removed.** It read "listened to, too faint to call", which was true under v4 and is now false — v10 confirmed orca there. Since the calendar is a v4 figure, the day drops back to plain background rather than being colored as confirmed, and the four v10 days are named in the caption and panel 10 instead. The `too_faint_unlabeled` style and its legend entry are gone; no day on the poster now uses them. The calendar's own subtitle changed to "Ten confirmed days under v4; v10 later added four more in May."

  *(Superseded at v35: the calendar now shows all fourteen, with the four v10 days as markers rather than bars.)*
- **v35 — 23 Aug 2026.** The four v10-found days are now **on** the calendar, as hollow mint triangles below the axis for 2, 3, 7 and 29 May, with the legend key "also orca — found later by v10". Title reads "fourteen days of orca", matching the headline.

  **Why markers and not a third bar color.** The bars plot detections at v4's cutoff, and v4 had essentially none on those four days — that is why it missed them. A bar would have to plot v10's counts, putting two models' score distributions on one y-axis where equal heights mean different things; and v10's per-day counts are not in hand anyway. A marker asserts presence without implying magnitude, which is exactly the claim being made. The caption says so explicitly: "marked rather than drawn as bars, because v4 had almost no detections there to plot."

  Hollow mint rather than a new hue, so the shape reads as "the same finding, reached a different way" instead of a fourth category. Markers sit below the day labels, clear of the tick row. The calendar and the headline now agree at fourteen, and the v4/v10 distinction survives inside the figure rather than only in prose.
- **v36 — 23 Aug 2026.** Logo block built out. The single reserved rectangle is now five labeled slots in the arrangement Duane specified: top row left to right IEEE OES, OCEANS 2026, GENERAL POSTER; bottom row a QR slot with the MBARI logo at far right. Each is a dashed box captioned with what belongs in it, sized in final poster inches (see checklist item 6 for the table), so Melissa can drop files in without measuring. A line beneath notes that the conference logos and poster-type marker keep the positions and proportions supplied in the template.

  **No logo files are in hand.** They cannot be fetched here — no logos exist in the repo, and ieee.org, the conference site and mbari.org are all outside the container's allow-list. The conference marks are already inside `POSTER_TEMPLATE_72x48.pptx` at the mandated positions, so Melissa can lift them from there; MBARI's is the one that genuinely needs supplying, and a horizontal wordmark on transparent PNG will fit the 5.4 × 2.6 in slot far better than a stacked variant.

  **Flagged: two QR slots now exist** — this new one and the footer QR at bottom right. Recommend keeping only the footer one, since it has room for its URL printed beside it, which matters more than position (see the v11 note on why the printed URL travels with the code).
- **v37 — 23 Aug 2026.** Three conference logos placed from `poster/logos/`: IEEE OES, OCEANS 2026 Monterey, and the General Poster marker, right-aligned along the top row in the order Duane specified and each at its native aspect (2.15, 1.00 and 1.28 respectively), vertically centred on a common line so they read as a set despite differing shapes.

  **Note on fetching:** the GitHub contents API was rate-limited (403), so the directory listing came from scraping the repo's HTML tree page for `poster/logos/...` paths — worth remembering as the fallback when the API refuses. The raw file endpoint kept working throughout.

  **MBARI logo is not in that folder** — its slot stays dashed. QR slot also still open, and the two-QR question is unresolved.
- **v38 — 23 Aug 2026.** Review items 2–8 applied as decisions. Item 1 (logo block) was already done at v37; item 9 (QR) remains an asset.

  **2 — orca F1 now reads 0.95 in both places.** True value is 0.945; the panel 11 bar and its side note both moved from 0.94 to 0.95 to match the headline card, so one quantity shows one digit. The side note was a near-miss: changing the bar label alone would have left "Orca stays strong at 0.94" sitting beside a bar reading 0.95.

  **3 — the 181/111 puzzle is now explained in the prose**: "on 12 May, all 181 detections were reviewed and every one was orca (111 of them above our cutoff — the bar in the calendar)". Verified by SQL on the project side: 181 detections at the 0.0 floor, all reviewed and confirmed, 111 clearing +1.16. All three numbers were always right; only the wording made them look contradictory.

  **4** — panel 7's ten-days bullet now points forward to panel 10. **5** — the duplicated "192 of 195" removed from the v10 figure subtitle, kept in the bullet. **6** — cutoff-tuning months (October 2020, April 2026) now distinguished from May 2018 as the untouched final test. **8** — *Orcinus orca* added once in panel 1, matching the *Lagenorhynchus* binomial in panel 3.

  **7 — reconciled rather than applied verbatim.** The suggestion was "~8–10 h of expert labeling, total", but Duane specifically chose "researchers' time" at v30 because it is the extended abstract's own phrase, and the hours cover two people. Kept that and took the substance of the suggestion — the ambiguity fix — giving "all of them in ~8–10 h of researchers' time, total". The loop diagram's copy is unchanged and still agrees.

  **Left alone, flagged:** the trajectory figure annotates v4 as "orca F1 0.947 @ +1.16", a third rendering of the same quantity. It sits beside ROC-AUC 0.959 and cmap 0.830, where three decimals is the figure's own convention, and 0.947 against 0.95 reads as precision rather than disagreement — unlike 0.94 against 0.95, which was the actual problem. Say the word if it should round too.
- **v39 — 23 Aug 2026.** MBARI logo placed; logo block complete.

  **EPS handling worth recording.** `MBARI_logo+type.eps` is a 2014 Illustrator file. `convert` alone produced a 331 x 123 image — it had silently fallen back to the EPS's embedded low-resolution preview because ghostscript was absent. Installing ghostscript (available via the Ubuntu archives, which are on the container allow-list) and rendering with `gs -dEPSCrop -sDEVICE=pngalpha -r1200` gave 5502 x 2038 with transparency, roughly 860 dpi at the printed 6.4 in width. The trap is that the bad conversion succeeds quietly and looks plausible until it is enlarged.

  **QR: footer only.** The logo-block slot is gone, per Duane. One code, with its URL printed beside it — which is what makes a stale link visible at proof stage.

  **0.947 stays** in the trajectory figure, confirmed by Duane: three decimals is that figure's own convention, and it reads as precision rather than disagreement with the 0.95 shown elsewhere.
- **v40 — 23 Aug 2026.** MBARI moved to the footer, immediately left of the QR code, as the conference template requires. It is 4.60 x 1.70 in there — smaller than the 6.4 in it had in the header, but still ~1200 dpi from the 5502 px master. The three conference marks now sit alone in the top-right block and were re-centred vertically so the block doesn't read as top-heavy with a gap beneath. The footer contact block was narrowed to 54.5 in to clear the logo, and the QR and its printed URL shifted right to keep the sequence **MBARI → URL → QR** reading left to right.

  **Rule to keep:** MBARI's position is mandated, not a design choice. Do not move it back up to the logo block.
- **v41 — 24 Aug 2026. MBARI moved to the top-left corner** (template's mandated position), replacing v40's footer placement. **v40 is preserved** — `build_poster_v40_backup.js` regenerates it exactly if the footer position turns out to be the right reading of the template.

  The move cost more than a coordinate change: the navy title band had to shift right from x 0.6 to x 9.0 and narrow from 56.0 to 47.6 in to clear a white zone for the logo, since MBARI's navy-and-grey wordmark is invisible on a navy field. Title type dropped 62 → 54 pt, byline 30 → 28, strap 24 → 22, all to keep the title on one line in the narrower band. MBARI sits at 7.40 x 2.74 in, vertically centred on the band. The footer's MBARI is gone and the contact block widened back to 59.5 in; the QR and its URL are unchanged.

  **Print deliverables added for Melissa:**
  - `OCEANS2026_orca_poster_v41_FULLSIZE_72x48_RGB.pdf` — true 72 x 48 in (5184 x 3456 pt), vector text and shapes, images at ~300 dpi at final size. This is the file to open in Illustrator.
  - `OCEANS2026_orca_poster_v41_FULLSIZE_72x48_CMYK.pdf` — same, converted to DeviceCMYK by ghostscript. **Unprofiled**: no ICC target, so it is a mathematical conversion, not a press-matched one. See the note to Melissa in the checklist.
- **v42 — 24 Aug 2026. Two fixes from the first naive-reader test.**

  **The five cards read as column headers.** A reader tried to map each card to the column beneath it and could not be talked out of it — which is a layout failure, not a reader failure: five equal cards directly above five equal columns is an alignment cue strong enough to override an explanation. Rebuilt as **one continuous strip** with (1) a single background instead of five cards, (2) an explicit **"AT A GLANCE"** label at the left so the band announces its own job, and (3) **deliberately unequal item widths** (14.0 / 12.4 / 14.2 / 11.8 / 11.6 in) so no divider lands near a column gap — gaps are at 14.28 / 28.56 / 42.84 / 57.12 in, dividers at 19.6 / 32.5 / 47.2 / 59.5. Dividers are hairlines rather than card edges. **Do not restore equal widths**: the inequality is the fix.

  **"Foundation model" was opaque to the reader.** The title is unchanged — it is the accepted title in the proceedings and the abstract, and a mismatch between poster and record would be worse than an unfamiliar term. Instead the conference/track/date strap under the byline (which told a viewer standing at OCEANS nothing they did not know) was replaced by a plain-language gloss: *"A foundation model is a large model pre-trained on masses of data, reusable for new tasks with very little new labeling."* The term itself is correct as printed — "foundation model" is the standard term of art from Stanford's CRFM, 2021; "foundational model" is a common variant but not the accepted name.

  Full-size print file regenerated: `OCEANS2026_orca_poster_v44_FULLSIZE_72x48_RGB.pdf`.
- **24 Aug 2026 — CMYK is the print deliverable, not a fallback.** MBARI's plotters accept CMYK only; Melissa printed the v41 CMYK full size on plain matt without trouble. Earlier advice in this file (hand over RGB, convert in Illustrator with the shop's profile) is **superseded** for this workflow: generate CMYK here. A FOGRA39-profiled conversion was tested against the generic one and produced byte-identical output, so the unprofiled path costs nothing measurable — shift on the navy field and mint numerals is 8–11 levels out of 255. Both RGB and CMYK full-size files ship each version; the RGB one remains the editable Illustrator source.
- **v43 — 24 Aug 2026. John's review, both items.**

  **Panel 2** now credits the archive properly: "the Amazon Web Services (AWS) Pacific Ocean Sound recordings", with `registry.opendata.aws/pacific-sound` in the same bullet.

  **Panel 6 rebuilt around John's confusions**, each of which was a real design fault rather than a gap in his knowledge:
  - *ROC-AUC label overlapped the +1.16 text.* Both series labels moved to the left end of the lines, where nothing competes for the space.
  - *The green curved arrow from the callout to the highlighted point read as connecting "orca" to "cmap".* Replaced with a short straight leader, and the callout is right-aligned so it no longer overruns the axes.
  - *"orca F1 0.947 @ +1.16" in the callout.* Removed. F1 is not a plotted series here and is defined in panel 11; three metrics in one callout was the root of the confusion. The callout now reads only "v4 — the model built in this walkthrough / ROC-AUC 0.959 · cmap 0.830".
  - *Nobody knows what the metrics are.* A plain-language paragraph now sits under the pre-v0 note: ROC-AUC as "how well the classifier ranks a true orca window above a non-orca one — 1.0 perfect, 0.5 a coin flip"; cmap as "the same of each class separately, then averaged, so a rare class counts as much as a common one". It closes on why the numbers matter: **"v4's 0.959 and 0.830 are strong — and the point of this panel is that it reached them on 873 labels, not tens of thousands."** Panel 6 grew 12.8 → 14.2 in to hold it.

  **cmap confirmed by Duane, 24 Aug**, and the wording tightened at v44 to his definition: per-class average precision — how well the model ranks each class's true calls above the rest, across all thresholds — averaged with equal weight across classes on held-out data, 0–1, higher better.

  Neither metric appears anywhere else on the poster, so this paragraph is the only place they need explaining.
- **v44 — 24 Aug 2026.** cmap gloss replaced with Duane's confirmed definition, including "across all thresholds" and "averaged with equal weight", plus his optional clause explaining the gap to ROC-AUC: *"Because every class counts equally, weak classes pull it down — which is why it sits below ROC-AUC."* That last clause is worth its space: a curious reader who notices cmap sitting a tenth below ROC-AUC on every wave now has the answer in the same paragraph, instead of reading it as a defect.

  Fitted without growing the panel — the paragraph box already had headroom, so six lines occupy what four did before. Panel 6 stays at 14.2 in and column 3 at 33.0.
