#!/usr/bin/env python3
"""
make_sperm_v11_input.py

Maps Danelle's sperm-whale selection table to the inputs perch-hoplite needs for a v11 seed set:

  1. a review CSV (same columns as the pass-1/pass-2 review files), one row per 5 s window,
     with filename = the 32 kHz resampled WAV name and window_start/window_end in seconds;
  2. a folder of symbolic links to those resampled WAVs, so one --audio-dir serves all of them;
  3. a manifest CSV recording where each resampled file was found (or that it is missing).

Input table: clicks-top-50_selections.txt (tab-separated, Raven-style). Columns used:
  Source Path        raw archive file, /mnt/PAM_Archive/YYYY/MM/MARS_YYYYMMDD_HHMMSS.wav
  Source Begin (s)   window start, seconds from the start of that file
  Source End (s)     window end
  rank, score        ranking and similarity score from the search (not a v10 logit)
  query_path         reference clip used for the search

The 32 kHz file name is the raw name with "_resampled_32kHz.wav". The resampled files are searched
under the roots listed in ROOTS (thalassa and the local drives), in order, by YYYY/MM.

Nothing is copied or moved. Links are created only when --link-dir is given.

Usage (from the repo, on spark-ae0e or spark-0626):
  python3 make_sperm_v11_input.py --selections clicks-top-50_selections.txt \
      --out-review sperm_v11_seed_review.csv --out-manifest sperm_v11_seed_manifest.csv \
      --link-dir /mnt/PAM_Analysis/perch-hoplite/sperm_v11_seed/audio
"""

import argparse
import csv
import os
import re
import sys

ROOTS = [
    "/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz",
    "/mnt/PAM_20TiB_32k",
    "/mnt/PAM_32k_Archive/resampled_32kHz",
    "/mnt/SL500/resampled_32kHz",
    "/mnt/SL500",
]
RAW_RE = re.compile(r"/(\d{4})/(\d{2})/(MARS_\d{8}_\d{6})\.wav$")
REVIEW_FIELDS = ["idx", "project", "filename", "window_start", "window_end", "label", "logits"]
LABEL = "sperm_whale_candidate"


def find_resampled(year, month, stem):
    name = f"{stem}_resampled_32kHz.wav"
    for root in ROOTS:
        for path in (os.path.join(root, year, month, name),):
            if os.path.isfile(path):
                return path
    return None


def main():
    ap = argparse.ArgumentParser(description="Map the sperm-whale selection table to perch-hoplite inputs.")
    ap.add_argument("--selections", required=True, help="tab-separated selection table")
    ap.add_argument("--out-review", required=True, help="review CSV to write")
    ap.add_argument("--out-manifest", required=True, help="manifest CSV to write")
    ap.add_argument("--link-dir", help="folder to fill with symlinks to the found resampled WAVs")
    args = ap.parse_args()

    with open(args.selections, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit("no rows in the selection table")

    review, manifest, missing = [], [], []
    for i, r in enumerate(rows, 1):
        m = RAW_RE.search(r["Source Path"])
        if not m:
            sys.exit(f"row {r['rank']}: Source Path does not match /YYYY/MM/MARS_YYYYMMDD_HHMMSS.wav: {r['Source Path']}")
        year, month, stem = m.group(1), m.group(2), m.group(3)
        resampled = find_resampled(year, month, stem)
        name32 = f"{stem}_resampled_32kHz.wav"
        start = float(r["Source Begin (s)"])
        end = float(r["Source End (s)"])
        review.append({"idx": r["rank"], "project": "MARS_20260901_20261031_32kHz", "filename": name32,
                       "window_start": f"{start:.1f}", "window_end": f"{end:.1f}", "label": LABEL,
                       "logits": r["score"]})
        manifest.append({"rank": r["rank"], "raw_path": r["Source Path"], "resampled_name": name32,
                         "found_path": resampled or "", "start_s": f"{start:.1f}", "end_s": f"{end:.1f}",
                         "score": r["score"], "query_path": r["query_path"]})
        if resampled is None:
            missing.append(name32)

    with open(args.out_review, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(review)
    with open(args.out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    files = sorted({x["resampled_name"] for x in manifest})
    found = [x for x in manifest if x["found_path"]]
    print(f"windows: {len(review)}   distinct 32 kHz files: {len(files)}   found: {len({x['resampled_name'] for x in found})}")
    print(f"wrote {args.out_review} and {args.out_manifest}")

    if args.link_dir:
        os.makedirs(args.link_dir, exist_ok=True)
        made = 0
        for name in sorted({x["resampled_name"] for x in found}):
            src = next(x["found_path"] for x in found if x["resampled_name"] == name)
            dst = os.path.join(args.link_dir, name)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
                made += 1
        print(f"links created in {args.link_dir}: {made} (existing links left as they are)")

    if missing:
        print(f"MISSING {len(set(missing))} resampled file(s); see the manifest found_path column:")
        for name in sorted(set(missing)):
            print("   ", name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
