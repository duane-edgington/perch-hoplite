# Multi-Annotator Review in Gradio — Analysis & Recommendation

**For:** D. Edgington / J. Ryan discussion, week of Aug 31 2026
**Prepared:** Aug 28 2026
**Status:** recommendation, not yet implemented (one-line DB fix drafted, see §6)

---

## 0. TL;DR

1. **The current tool is DESTRUCTIVE across annotators.** `phase2_classify.py review` deletes any
   existing annotation for a window before inserting, *without regard to who made it*. If John
   reviews a window Duane already labeled, **Duane's label is gone** — not superseded, deleted.
2. **The review UI is already blind.** The radio is hardcoded `value="unlabeled"` and never reads
   the DB, so a second annotator cannot see the first's calls. Blind review is the accidental
   default. This makes (1) worse: John would silently overwrite labels he never saw.
3. **Recommendation: one DB, one annotation row per (window, annotator), never delete another
   annotator's row.** Scope the `DELETE` by `provenance`. No schema migration needed.
4. **Reconciliation is a derived artifact, not a mutation** — write a third `consensus:` row.
5. **Training selects a view under an explicitly recorded policy**, not by mutating labels.
6. **DO NOT let John relabel July 2015 until the DELETE is scoped**, or the 6 existing labels are lost.

---

## 1. What the code actually does today

From `phase2_classify.py`, the Gradio autosave path:

```python
prov = f"gradio_gui:{annotator_id}"
# DELETE existing annotation for this window first (any label), then INSERT fresh.
con.execute("""
    DELETE FROM annotations
    WHERE recording_id=? AND offsets=?
""", (rec_id, off_enc))
con.execute("""
    INSERT INTO annotations
        (recording_id, offsets, label, label_type, provenance)
    VALUES (?, ?, ?, ?, ?)
""", (rec_id, off_enc, store_label, int(lt), prov))
```

Facts established by reading the source:

| Question | Answer |
|---|---|
| Does `provenance` record the annotator? | **Yes** — `gradio_gui:<annotator_id>` |
| Can the DB hold two annotators' labels for one window? | **No** — the `DELETE` is unscoped |
| Does the UI prefill the radio from existing DB labels? | **No** — hardcoded `value="unlabeled"` |
| Is there an independent record of each session? | **Yes** — JSON under `provenance/labels/labels_<ts>_<annotator>.json` |
| Default `--annotator-id` | `analyst` (a generic bucket — always set it explicitly) |

**Consequence:** the DB currently represents "the most recent opinion of the most recent
annotator," while the per-session JSON files retain the full history. Recovery from an overwrite
would mean replaying JSON — possible, but not a design anyone should rely on.

---

## 2. Recommendation: one DB, additive rows

**One row per (window, annotator). Never delete another annotator's row.**

Rationale:
- `provenance` already carries the annotator, so this needs **no schema change**.
- Keeps existing behavior where it is wanted: changing *your own* mind still replaces *your own*
  label, because the `DELETE` still matches your own provenance.
- Agreement between annotators becomes computable, which is the main scientific payoff (§5).

### Why NOT separate DBs per annotator
- **Storage:** each month's DB is ~1.6 GB `usearch.index` + ~40 MB sqlite (finding #28).
  Duplicating per annotator multiplies the archive by the number of annotators for zero benefit —
  the embeddings are identical.
- **Analysis:** inter-annotator agreement would require joining across DB files, which is exactly
  the query you most want to be trivial.
- **Provenance:** two DBs drift. One DB with tagged rows cannot.

---

## 3. Reconciliation: a third row, not an edit

When Duane and John disagree, **do not modify either original row.** Write a new annotation with
provenance like:

```
consensus:duane+john
```

Then the record shows what each person heard *and* what they jointly concluded. Editing one
annotator's row to match the other destroys the disagreement, which is data.

Suggested provenance vocabulary:

| Provenance | Meaning |
|---|---|
| `gradio_gui:duane` | Duane's own call in a review session |
| `gradio_gui:john` | John's own call |
| `consensus:duane+john` | Jointly agreed after discussion |
| `csv_import:<who>` | Bulk import (already supported at line ~1177) |

---

## 4. Training consumes a VIEW under a recorded policy

Training must never resolve disagreements by mutating labels. Instead, select rows by an explicit,
documented policy — e.g.:

> *"Where a `consensus:` row exists, use it. Otherwise prefer `gradio_gui:john` (species-expert
> ground truth) over `gradio_gui:duane` (working labels). Ignore `analyst`-tagged rows."*

Record the policy string in the model card / training provenance so a given model's training set is
reconstructible. `_save_training_provenance` already dumps the full annotation list per run, so this
is mostly a documented convention rather than new code.

This matters because of the standing asymmetry Duane has already stated: **his humpback and `other`
labels are working labels; John's are authoritative.** The policy is where that asymmetry gets
encoded once, instead of being remembered ad hoc.

---

## 5. The payoff: inter-annotator agreement becomes computable

With both annotators' labels on the same windows in one DB, you can compute **Cohen's kappa** on
the overlap set, plus a per-class confusion matrix (where does Duane say `other` and John say
`humpback_song`?).

This is worth doing deliberately, because:
- Any expert-labeled bioacoustics paper gets asked for inter-annotator agreement by reviewers.
- It quantifies the "working label vs authoritative label" gap rather than asserting it.
- It identifies **which classes** need John's ear most — likely `other` and the humpback
  song/vocalization split, both already open threads.

Cheap way to seed it: have John review a modest overlap set (the July 2015 six is a start) rather
than a whole month, purely to measure agreement.

---

## 6. Proposed change — scope the DELETE by provenance

The whole fix:

```sql
-- BEFORE (destructive across annotators)
DELETE FROM annotations WHERE recording_id=? AND offsets=?

-- AFTER (destructive only to your own prior label)
DELETE FROM annotations WHERE recording_id=? AND offsets=? AND provenance=?
```

Passing `prov` as the third parameter. Behavior change:

| Scenario | Before | After |
|---|---|---|
| Duane changes his own mind | replaced | replaced (unchanged) |
| John labels a window Duane labeled | **Duane's label DELETED** | both rows coexist |
| Duane relabels after John | **John's label DELETED** | both rows coexist |

### Blind vs reconciliation mode

Blind review is already the behavior (`value="unlabeled"`, no DB read). So:
- **Blind mode = today's behavior.** No flag needed.
- **Reconciliation mode = the missing feature.** A `--show-existing-labels` flag would read
  existing annotations for each window and display them (ideally as text next to the radio, *not*
  as a prefilled selection, so a reviewer is never nudged into agreement by a pre-clicked button).

Recommend implementing the scoped `DELETE` **now** (small, safe, unblocks John) and treating
`--show-existing-labels` as a follow-up once you and John have decided whether reconciliation
happens in the tool or in conversation.

---

## 7. Immediate action items

| # | Action | Owner | When |
|---|---|---|---|
| 1 | **Ask John to listen to the July 2015 Gradio set** — especially the 4 `other` clips Duane could not identify | Duane | Monday |
| 2 | **Do not let John relabel `MARS_20150728_20150731_32kHz_norm`** until the scoped DELETE lands — or have him review read-only and dictate calls | Duane | before John's session |
| 3 | Apply the scoped `DELETE` patch | Duane | before any 2-annotator session |
| 4 | Decide reconciliation venue: in-tool (`--show-existing-labels`) or in conversation | Duane + John | at the meeting |
| 5 | Write the training label-selection policy string into the model card | Duane | before v11 training |
| 6 | Decide whether to seed an agreement set (kappa) and how big | Duane + John | at the meeting |
| 7 | **Always pass `--annotator-id`** — default is the generic `analyst`; the Aug 24 Apr 2026 session lacked it | Duane | ongoing |

---

## 8. The four July 2015 clips for John

DB `MARS_20150728_20150731_32kHz_norm`; review CSV `results/review_jul2015_orca_ge050.csv`.
Scores shown are **orca_v4** (the review set was built from the v4 detections CSV).

| Window | Offset | v4 | v10 | Duane's call | Note for John |
|---|---|---|---|---|---|
| `MARS_20150731_222345` | 335–340 s | 1.548 | 2.002 | `other` | **Both models' top hit, and concordant.** Duane: not orca. If he agrees, it is a clean calibration point for a high-scoring non-orca in 2015-era audio. If not, finding #29 changes. |
| `MARS_20150730_095345` | 315–320 s | 0.589 | 0.537 | `other` | unidentified |
| `MARS_20150731_064345` | 310–315 s | 0.572 | 0.874 | `other` | unidentified |
| `MARS_20150731_221345` | 255–260 s | 0.587 | <0.5 | `other` | unidentified; v10 dismissed it |
| `MARS_20150730_031345` | 450–455 s | 0.603 | 0.895 | `dolphin_call` | figure `gradio_jul30_2015_dolphin_450s_wid4531.png` |
| `MARS_20150731_232345` | 425–430 s | 0.711 | <0.5 | `dolphin_call` | figure `gradio_jul31_2015_dolphin_425s_wid20726.png`; **v10 scored it below 0.5** yet it is a real dolphin call — a v10 `dolphin_call` recall observation |

Context worth giving John: **July 2015 returned zero orca**, which is expected — late July sits
outside the spring window where every confirmed Monterey Bay Bigg's event has landed. The record
ends at midnight 7/31, so anything starting that evening is truncated; **August 2015 continues the
same deployment and Aug 1 is complete (144 files)**, so a boundary-spanning event would show there.
