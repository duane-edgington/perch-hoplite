# Poster v42 review — accuracy & clarity (follow-up to v35 review)

Reviewed rendered v42 (pptx → PDF). Focus: did the v35 must-fixes land, and is anything new
wrong? Short version: **the three v35 must-fixes are resolved. v42 is in good shape.** A couple
of v35 "should-fix" prose items were addressed via the figure/caption rather than the bullet
text — fine, not blocking. No new errors found.

## v35 must-fixes — STATUS

1. **Header placeholder → FIXED.** The "RESERVED — TEMPLATE BLOCK" is gone. v42 now has the
   MBARI logo (top-left), IEEE Oceanic Engineering Society + OCEANS logos and a "General Poster"
   marker (top-right). This was the most visible v35 problem; fully resolved.

2. **Stat card 0.95 vs panel 11's 0.94 → FIXED.** Panel 11's orca bar now reads **0.95**
   (side text: "Orca stays strong at 0.95… at the software default it drops to 0.78"). The stat
   card is 0.95. The two now agree. Consistent.

3. **May 12 "181/181" vs bar "111" → ADEQUATELY ADDRESSED.** The prose bullet still says
   "12 May confirmed 181/181," but the figure caption now clarifies: "12 May rests on 181
   reviewed clips… 111 [above cutoff]." Since all three numbers are individually correct
   (verified via SQL: 181 detections = 181 reviewed = all orca; 111 above +1.16), and the
   caption now carries the distinction, this is no longer confusing. Optional polish: reword the
   bullet to "all 181 reviewed, all orca (111 above cutoff)" for belt-and-suspenders, but not
   required.

## v35 should-fixes — STATUS

4. **10→14 forward-pointer → PRESENT (via figure).** Panel 7's bullet still reads "Ten confirmed
   orca days… from v4 alone," but the figure title now says "fourteen days of orca" and the
   caption ends "Fourteen days in all; see panel 10." So the 10→14 reconciliation is on the
   poster; it lives in the figure rather than the bullet. Good enough.

5. **"foundation model" defined → ADDED.** New header subtitle: "A foundation model is a large
   model pre-trained on masses of data, reusable for new tasks with very little new labeling."
   Nice touch for the mixed OCEANS audience (matches the "define term" commit).

## Not re-checked in detail (were fine in v35, no reason to change)
- Panel 10 (Running the loop again) — the strongest panel; v10 numbers correct.
- Panels 1-6, 8, 9, 12 — unchanged worked-example/method content.
- Orcinus orca binomial (v35 nice-to-have) — not verified in v42; low priority.

## New issues in v42
- None found. Header, stat band, panels 7/10/11 all check out.

## Bottom line
v42 resolves the v35 must-fixes and is in solid, presentable shape. The remaining items are
optional polish (the May-12 bullet wording; adding *Orcinus orca* once). John has v42 for
review; his feedback + Danelle's is the next input. No accuracy blockers for OCEANS 2026.

Reviewed from the pptx rendered to PDF (no v42 PDF in the repo — only the pptx). Color/font
specifics approximate; this is a content/accuracy pass.
