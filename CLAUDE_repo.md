# CLAUDE_repo.md — Handoff: build the public perch-hoplite release repo

**Purpose of this doc:** a self-contained brief so a FRESH chat can build the public
`perch-hoplite` release repo (github.com/duane-edgington) without needing the full context of the
main project conversation. Read this, then the two source-of-truth docs it points to, then build.

**One-line goal:** produce a clean, long-lived PUBLIC GitHub repo that supports the OCEANS 2026
poster, follow-on public datasets, and publications — a curated copy of the *important, durable*
stuff, NOT a mirror of the messy working repo.

---

## Read these first (they are the real spec)

Both are in the working repo (github.com/duane-edgington/perch-hoplite) and should be uploaded to
the fresh chat, or fetched via raw.githubusercontent.com:

1. **`CLAUDE_release_plan.md`** — THE blueprint. Full FAIR mapping, concrete repo file layout,
   reproducibility-bundle spec, model-card outline, licensing guidance, release sequence. Follow
   it. This handoff does not repeat its detail — it points at it and adds live context.
2. **`CLAUDE_perch_hoplite.md`** — the main project context/findings doc. Source of truth for the
   science story, model lineage, what's confirmed vs. tentative. Especially: the "Classifier
   Versioning" section (what `_clean` meant, why v0/v4/v10 are the canonical models) and findings
   #18 (May held-out), #20-#21 (v10 validation), #24-#26 (Sept 2024, John's strategic redirect).

Also useful in the working repo: `CLAUDE_embed.md` (resample+embed pipeline, for the
reproducibility bundle), `docs/agile_modeling_history.md` (the v0→v10 narrative).

---

## Live context the source docs don't fully capture (carry these into the build)

- **Canonical models:** `orca_v4.pt` (prior production, the poster's "worked example") and
  `orca_v10.pt` (current best, the poster's result). Both live at
  `/mnt/PAM_Analysis/perch-hoplite/models/` on spark. They are TINY (~708K total for the models
  dir) — the crown jewel, trivially shareable. **There is NO `orca_v4_clean.pt`** — `_clean` was
  the retired pre-normalization bootstrap era (v1_clean–v8_clean); do not include those.
- **The irreplaceable artifacts are <10 MB total:** models (708K), labels (52K), json_labels
  (636K), provenance (3.5M), example_clips (3.2M). Everything else (723 GB resamples, 45 GB DBs)
  is regenerable. So the public repo's core payload is small.
- **May 2018 is the permanent held-out test month** (finding #18) — never trained on. This is
  central to the validation story; make sure the release preserves/labels it as held-out.
- **John Ryan is co-author** on the OCEANS poster and must be credited. He's fine with public
  release (finding #26 pt 7). His view: a **notebook that runs the SoX resampling on the public
  AWS raw audio may itself suffice as "releasing the data"** (reproducibility-not-dataset) — he's
  ALSO open to releasing resampled data in an appropriate format/venue. Either path is acceptable;
  the notebook-on-raw approach is the lighter, John-endorsed default.
- **Raw audio is ALREADY public** (Pacific Ocean Sound / AWS Open Data). Do NOT re-host it. The
  release publishes *reproducibility* (script + manifest + checksums + models + labels), not a
  raw dataset.
- **Strategic direction (finding #26):** project focus is now full-archive seasonal/interannual
  ORCA analysis (Bigg's/transient), sightings-correlated. External ecotypes and external datasets
  (e.g. Palmer 2025) are TABLED. So the public repo should center the orca detector + agile-
  modeling method, not multi-ecotype ambitions.
- **cmap** on the poster/README = class mean average precision (macro mean of per-class AP,
  `perch_hoplite.agile.metrics.cmap`, sample_threshold=1). If the repo README reports it, describe
  it accurately: "class mean average precision — per-class AP averaged with equal weight across
  classes on held-out data."
- **Storage discipline (finding #26 pt 8):** thalassa is at 96%. Don't design the repo to require
  hosting bulk resampled audio. Curated small artifacts + reproducibility bundle only.

---

## What to INCLUDE (durable, important)

Per CLAUDE_release_plan.md's layout. Summary:
- Curated scripts: resampling (`new_32k_resample_sox.sh`), embedding (`phase1_embed_torch.py`),
  inference/review (`phase2_classify.py`), the held-out eval (`compare_may_holdout.py`), figure
  tools. Clean, documented, no dead-ends.
- Trained models `orca_v4.pt`, `orca_v10.pt` + `orca_v10.metrics.json` + a MODEL_CARD.md.
- Labels / annotation tables (confirmed labels per month) + a LABELS_README (schema, provenance,
  what "confirmed" means, annotator IDs).
- Confirmed-clip SUBSET (MB, not TB) — the windows referenced in the poster/paper + a few
  exemplars per class. NOT full months.
- Reproducibility bundle: SoX script + SOURCE_MANIFEST.csv (which public raw files → month) +
  VERSIONS.md (pinned deps incl. SoX version) + CHECKSUMS.md (sha256 of a sample of resampled
  outputs). This is what makes NOT hosting the bulk audio scientifically sound.
- Docs: README (overview, quickstart, reproduce steps, DOI + poster links), REPRODUCE.md,
  agile_modeling_history.md, the FAIR statement.
- Licensing: LICENSE (code, MIT or Apache-2.0), LICENSE-DATA (labels/clips, CC-BY-4.0),
  CITATION.cff. **Check Perch V2 upstream terms before redistributing derived model weights.**

## What to KEEP PRIVATE / EXCLUDE
- The v5–v8 detour and v1_clean–v8_clean retired models (present a clean v0→v4→v10 lineage).
- Un-reviewed / ambiguous labels presented as confirmed — e.g. the **April 2026 candidates are
  AMBIGUOUS pending John's blind review (finding #23)**; include only with that caveat, or omit.
- Full embedding DBs (45 GB) and full-month resampled WAVs (723 GB) — regenerable; don't host.
- Internal scratch, working notes, dead-end experiments, raw session logs.

---

## Suggested first steps for the fresh chat

1. Confirm scope with Duane: (a) new standalone repo `perch-hoplite` (or `perch-hoplite-orca`)
   under duane-edgington, vs. curating within the existing repo; (b) which release path for data
   — John's notebook-on-raw (lighter) vs. also a Zenodo snapshot of models+labels+clips (gets a
   DOI, good for the paper). Recommend: standalone clean repo + light Zenodo DOI; skip AWS.
2. Draft the top-level structure per CLAUDE_release_plan.md; stub README, MODEL_CARD, LICENSE(s),
   CITATION.cff.
3. Assemble the reproducibility bundle: SOURCE_MANIFEST.csv, VERSIONS.md (get exact SoX version:
   `sox --version` on spark), CHECKSUMS.md (sha256 a handful of resampled files).
4. Copy in the crown-jewel artifacts (<10 MB) + the confirmed-clip subset.
5. Write REPRODUCE.md: public raw → SoX resample → embed → infer → results, end to end.
6. Cross-link: repo README ↔ Zenodo DOI ↔ poster QR.

## Working-method notes (same discipline as the main project)
- Duane pushes via the **EM1 clone** (download → cp → grep-verify the new content → commit →
  push); Claude stages files to outputs, can't push directly. After any push, grep the pushed
  file on GitHub for the specific new text before believing it landed.
- Claude can fetch repo files via `git clone --depth 1` / raw.githubusercontent.com, and render
  pptx/pdf, but has no live spark or GitHub write access.
- Verify before acting: `rsync --dry-run`, grep-after-push, one working copy not stale clones.

---

## TODO / open items this build will touch or depend on
- Pin SoX version for VERSIONS.md/CHECKSUMS (also a standing CLAUDE_embed.md TODO).
- Confirm Perch V2 upstream license terms re: redistributing `orca_v*.pt` derived weights.
- Decide Zenodo vs. notebook-only for the data-release path (with Duane/John).
- The exact nohup resampling command (CLAUDE_embed.md TODO) belongs in REPRODUCE.md too.
