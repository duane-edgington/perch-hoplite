# September 2015 — acoustic detections vs. Monterey Bay Whale Watch sightings

**D. Edgington, 2026-08-31.** First correlation of the full-archive campaign against an
independent sighting record.

---

## ⚠️ COPYRIGHT — READ BEFORE COMMITTING ANYTHING

The source states plainly:

> *"These sightings are copyright protected and may not be used without permission from Nancy
> Black. All rights reserved. Unauthorized use of this information is a violation of federal
> copyright law..."*

**`perch-hoplite` is a PUBLIC repository.** Do **not** commit the sightings table, a transcription
of it, or a derived CSV of it until written permission is obtained. **This document deliberately
contains only (a) our own detection times, which are ours, and (b) the minimum reference to
sighting dates needed to state the result.**

The permission conversation is now much easier: there is a concrete, favourable result to show.
**Ask J. Ryan to make the introduction.**

Source: Monterey Bay Whale Watch, *Marine Mammal Sightings List September 2015*,
`https://www.montereybaywhalewatch.com/sightings/slst1509/` (retrieved 2026-08-31).
Photos © 2015 Daniel Bianchetta.

---

## 1. The headline

**Our first confirmed orca call of September 2015 falls 17 minutes after a whale-watch trip
documented 14 killer whales preying on common dolphins in Monterey Bay.**

| Local time (PDT) | Event |
|---|---|
| 9/16 4:30 p.m. | MBWW trip logs **14 Killer Whales — predation on Common Dolphins**, with ~2,000 Long-beaked Common Dolphins present |
| **9/16 16:47:34** | **Our first confirmed orca call** (`MARS_20150916_234019` @435 s) |

Killer whales were logged on **every MBWW trip that day** — 7 animals at 8 a.m., 9 a.m., 1 p.m. and
2 p.m., rising to 14 by 4:30 p.m.

---

## 2. TIME BASE — the thing that made this legible

**Our timestamps are UTC; the sightings list is local (PDT = UTC−7 in September).** Converting is
not cosmetic — it moves an entire episode across a date boundary:

| | UTC framing | Local (PDT) framing |
|---|---|---|
| Episode A | spans 09-16 23:47 → 09-17 06:45 | **falls entirely within Sept 16** |
| Episode B | spans 09-28 05:05 → 07:54 | **spans the night of Sept 27→28** |

Under the UTC framing, Episode A looked like "one call, then a cluster seven hours later the next
morning." In local time it is **one call in late afternoon and a cluster before midnight on the
same day** — which is what actually aligns with the sighting record.

**Rule for all future correlation work: convert to local time before comparing with any sighting
source.** PDT (UTC−7) during daylight saving, PST (UTC−8) otherwise.

---

## 3. Episode A — Sept 16, all local

| Local (PDT) | Recording | Offset |
|---|---|---|
| 16:47:34 | `MARS_20150916_234019` | 435 s |
| 23:25:40 | `MARS_20150917_062020` | 320 s |
| 23:26:30 | `MARS_20150917_062020` | 370 s |
| 23:40:30 | `MARS_20150917_064020` | 10 s |
| 23:41:25 | `MARS_20150917_064020` | 65 s |
| **23:41:40** | `MARS_20150917_064020` | **80 s (v10 = 3.128)** |
| 23:41:55 | `MARS_20150917_064020` | 95 s |
| 23:42:00 | `MARS_20150917_064020` | 100 s |
| 23:45:15 | `MARS_20150917_064020` | 295 s |

Note the filenames still carry 09-17 because MARS names files in UTC.

**The dolphin call at 23:26:40 PDT — 10 seconds after an orca call in the same recording — now has
context.** The 4:30 p.m. trip recorded active predation on common dolphins with ~2,000 present.
Whether the acoustic co-occurrence is predation or simply co-presence is not settled by this, but
it is no longer an unmotivated coincidence.

---

## 4. Episode B — night of Sept 27→28, and NO sighting

| Local (PDT) | Recording | Offset |
|---|---|---|
| 9/27 22:05:24 | `MARS_20150928_050349` | 95 s |
| 9/27 22:06:39 | `MARS_20150928_050349` | 170 s |
| 9/27 23:10:09 | `MARS_20150928_060349` | 380 s |
| 9/28 00:18:39 | `MARS_20150928_071349` | 290 s |
| 9/28 00:20:29 | `MARS_20150928_071349` | 400 s |
| 9/28 00:21:04 | `MARS_20150928_071349` | 435 s |
| 9/28 00:23:09 | `MARS_20150928_071349` | 560 s |
| 9/28 00:26:14 | `MARS_20150928_072349` | 145 s |
| 9/28 00:54:14 | `MARS_20150928_075349` | 25 s |

**Every call is between 22:05 and 00:54 local — after dark.** The sightings list records no killer
whales on 9/27 or 9/28, but whale-watch trips run in daylight. **Absence of sighting at night is
not evidence of absence**, and this is precisely the gap that passive acoustics fills.

The log does show **2,500 Long-beaked Common Dolphins on both 9/28 trips** — the largest counts of
the month — consistent with our 13 dolphin calls that night.

**This is the strongest argument for the method: an apparent orca encounter that the visual record
could not have caught.**

---

## 5. The false negatives matter as much as the hit

| Local date | Killer whales sighted | Our confirmed calls |
|---|---|---|
| 9/1 | yes (p.m.) | **0** |
| 9/11 | yes (a.m. and p.m.) | **0** |
| 9/14 | yes (late p.m.) | **0** |
| **9/16** | **yes, all day** | **9** ✅ |
| 9/27 | none logged | 3 (night) |
| 9/28 | none logged | 6 (night) |

**Killer whales were sighted on four days; we detected them acoustically on one.** Coverage was
essentially complete on all four (9/1, 9/11 and 9/14 all ≈24 h recorded), so these are genuine
non-detections, not gaps.

Three explanations, not mutually exclusive, and this document does not choose between them:

1. **Range.** MARS sits ~25 km offshore in Monterey Canyon at ~890 m. The sightings list covers the
   whole Monterey Bay region, and whale-watch trips often work much closer inshore. **Animals can
   be in "the bay" and far outside hydrophone detection range.**
2. **Silence.** Bigg's killer whales are acoustically cryptic — they hunt marine mammals with acute
   hearing and are often silent for long periods. Being present is not being audible.
3. **Detector recall.** We already measured ~17% recall at v10's operating threshold on September
   (§ finding #34), and more than half the confirmed calls came only from a low-threshold second
   pass. **Days with few, faint calls could plausibly fall below even the pass-2 floor.**

**⚠️ UPDATE (Aug 31 2026) — RANGE IS NOW DEMONSTRATED FOR AT LEAST ONE MISS.** A public X post
from @MBayWhaleWatch (Oct 12 2015) locates the **10/11/2015** predation event *"just a few miles
from Monterey Harbor."* **Monterey Harbor → MARS node is 29.2 km (18.2 statute miles)**, so those
animals were **~24-28 km from the hydrophone** — actively hunting, vocal enough to film, and far
outside plausible detection range. See finding #44.
This does not prove all misses are range-limited, but it makes explanation 1 the leading candidate
and **changes the question** from *"why did we miss them?"* to **"what is the effective detection
radius, and which sightings fall inside it?"** The highest-value ask is therefore **sighting
LOCATIONS, not just dates.**

**The pass-2 test is still worth doing**, now scoped as a test of explanations 2 vs 3 on days where
the animals were plausibly in range: run the zoom-in protocol on 9/1, 9/11 and 9/14 specifically. If low-threshold review turns up calls on those days, explanation 3 dominates
and the recall estimate is worse than we thought. If it turns up nothing, explanations 1 and 2
dominate and the pipeline is behaving correctly.

---

## 6. What this establishes, and what it does not

**Establishes:**
- The pipeline detected a killer whale encounter that an independent, contemporaneous, expert
  visual record also documented — on the correct day, within 17 minutes of a logged predation event.
- The pipeline detected two further nights of activity that **the visual record could not have
  seen**, because the boats were not out.

**Does not establish:**
- **Pod identity.** Nothing here identifies CA95A1, CA26, CA180, or any matriline. The MBWW list
  gives counts, not IDs. **Earlier claims about specific pods remain unverified** (finding #39) and
  should not be repeated. Photo-ID from the California Killer Whale Project would be needed.
- **That the animals we heard are the animals they saw.** Same day and same region is suggestive,
  not conclusive.
- **Ecotype.** The 4:30 p.m. predation-on-dolphins entry is consistent with Bigg's, which is what
  `orca_v10` is trained on — but the sightings list does not state ecotype.

---

## 7. Asks

1. **Permission from Nancy Black** to use the sightings data. → J. Ryan introduction.
2. **The sightings record as a dataset**, not per-month PDFs. MBWW appears to publish monthly lists
   at `/sightings/slst<YYMM>/`. **~130 monthly pages is a scrapable, joinable series** — this is the
   join key for the whole campaign. Permission first.
3. **Photo-ID for 9/16 2015** from CKWP, to settle pod identity.
4. **Are ecotypes recorded** anywhere in the MBWW or CKWP records? It matters: a fish-eating
   Southern Resident vocalises very differently from Bigg's, and our classifier is trained on Bigg's.

---

## 8. Method note for the campaign

Add to the standard per-month loop, after review:

- Convert all confirmed detections **to local time** (PDT/PST) before any comparison.
- Compare against the sighting record for that month, and record **both directions**: sighted-and-
  detected, and **sighted-but-not-detected**. The false negatives quantify effective detection
  range and recall in a way nothing internal to the pipeline can.
- **Night-time detections with no possible sighting are a result in their own right**, not a gap.
