#!/usr/bin/env python3
"""
make_sperm_clips.py

Cuts each sperm-whale selection out of its 32 kHz resampled file as a 5-second clip, so that each
review window is the whole clip and matches the selection exactly. Needed because Danelle's
selections start on half-second offsets (e.g. 342.5 s), while the database embed uses a 5-second
grid (0, 5, 10 ... s); the half-second windows never appear in the database.

Inputs:
  --selections   the tab-separated selection table (Source Path, Source Begin (s), rank, score, query_path)
  --link-dir     the folder of 32 kHz links made by make_sperm_v11_input.py
  --clip-dir     where the 5-second clips are written
  --out-review   review CSV pointing at the clips (window_start 0.0, window_end 5.0)

Clip names: MARS_YYYYMMDD_HHMMSS_off<start>s_resampled_32kHz.wav, so the source and offset are in the name.
Existing clips are not overwritten. Nothing in the archive is changed.

Usage (on spark-ae0e, from ~/perch-hoplite):
  python3 make_sperm_clips.py --selections clicks-top-50-selections.txt \
      --link-dir /mnt/PAM_Analysis/perch-hoplite/sperm_v11_seed/audio \
      --clip-dir /mnt/PAM_Analysis/perch-hoplite/sperm_v11_seed/clips \
      --out-review /mnt/PAM_Analysis/perch-hoplite/results/sperm_v11_clips_review.csv
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

RAW_RE = re.compile(r"(MARS_\d{8}_\d{6})\.wav$")
REVIEW_FIELDS = ["idx", "project", "filename", "window_start", "window_end", "label", "logits"]
LABEL = "sperm_whale_candidate"


def main():
    ap = argparse.ArgumentParser(description="Cut sperm-whale selections into 5-second clips and write a review CSV.")
    ap.add_argument("--selections", required=True)
    ap.add_argument("--link-dir", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out-review", required=True)
    args = ap.parse_args()

    if shutil.which("sox") is None:
        sys.exit("sox is not on PATH")
    os.makedirs(args.clip_dir, exist_ok=True)

    with open(args.selections, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    out, made, skipped, missing = [], 0, 0, []
    for r in rows:
        m = RAW_RE.search(r["Source Path"])
        if not m:
            sys.exit(f"rank {r['rank']}: cannot read the file stem from {r['Source Path']}")
        stem = m.group(1)
        start = float(r["Source Begin (s)"])
        end = float(r["Source End (s)"])
        dur = end - start
        src32 = os.path.join(args.link_dir, f"{stem}_resampled_32kHz.wav")
        if not os.path.exists(src32):
            missing.append(src32)
            continue
        clip_name = f"{stem}_off{start:.1f}s_resampled_32kHz.wav"
        clip_path = os.path.join(args.clip_dir, clip_name)
        if os.path.exists(clip_path):
            skipped += 1
        else:
            subprocess.run(["sox", src32, clip_path, "trim", f"{start:.3f}", f"{dur:.3f}"], check=True)
            made += 1
        out.append({"idx": r["rank"], "project": "SPERM_V11_CLIPS_32kHz", "filename": clip_name,
                    "window_start": "0.0", "window_end": f"{dur:.1f}", "label": LABEL,
                    "logits": r["score"]})

    with open(args.out_review, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(out)

    print(f"selections: {len(rows)}   clips written: {made}   already present: {skipped}   review rows: {len(out)}")
    print(f"wrote {args.out_review}")
    if missing:
        print(f"MISSING {len(missing)} linked source file(s):")
        for p in sorted(set(missing)):
            print("   ", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
