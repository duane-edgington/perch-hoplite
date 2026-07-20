#!/usr/bin/env python3
"""
labels_json_to_review_csv.py — turn an export_labels.py labels JSON into a
`phase2_classify.py review --detections-csv` input, so existing labeled windows can be
re-reviewed in Gradio (e.g. re-check humpback_song labels for gray-whale contamination,
issue #13).

`tools/export_labels.py` writes per-(month, species) JSON to
/mnt/PAM_Analysis/perch-hoplite/json_labels/labels_{month}_{species}.json, shaped:

    {"month": "2018_04", "species": "humpback_song", "annotations": [
        {"recording_32khz": "/…/MARS_20180430_061912_resampled_32kHz.wav",
         "annotation_offset_s": 190.0, "label_type": "positive", ...}, ...]}

The review tool locates each clip by `filename` + `window_start` + `--audio-dir` (the CSV
`idx` column is cosmetic — verified in phase2_classify.py), so we emit the same 7-column
format the working detection CSVs use:

    idx,project,filename,window_start,window_end,label,logits

Then feed it to review with `--target-label {species}` and a `--classes` list that includes
the new label option, e.g. `--classes humpback_song,gray_whale_moan,other,unlabeled`.

Examples
    python3 tools/labels_json_to_review_csv.py --selftest
    # April 2018 humpback labels -> review CSV (41 clips):
    python3 tools/labels_json_to_review_csv.py --month 2018_04 --species humpback_song
    # explicit paths:
    python3 tools/labels_json_to_review_csv.py \
        --json /mnt/PAM_Analysis/perch-hoplite/json_labels/labels_2026_04_humpback_song.json \
        --out  /mnt/PAM_Analysis/perch-hoplite/results/review_2026_04_humpback_graywhale.csv
"""
import argparse
import csv
import json
import os
import sys

JSON_DIR = "/mnt/PAM_Analysis/perch-hoplite/json_labels"
RESULTS_DIR = "/mnt/PAM_Analysis/perch-hoplite/results"
CSV_HEADER = ["idx", "project", "filename", "window_start", "window_end", "label", "logits"]


def json_to_rows(data, window_len=5.0, include_negatives=False):
    """Return list of (filename, window_start, window_end, label) from a labels JSON dict."""
    species = data.get("species", "unknown")
    rows = []
    for a in data.get("annotations", []):
        if not include_negatives and a.get("label_type") != "positive":
            continue
        rec = a.get("recording_32khz") or a.get("filename") or ""
        fn = os.path.basename(rec)
        if not fn:
            continue
        ws = float(a["annotation_offset_s"])
        rows.append((fn, ws, ws + window_len, a.get("species", species)))
    return rows


def write_csv(rows, out_path, project):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i, (fn, ws, we, label) in enumerate(rows):
            w.writerow([i, project, fn, ws, we, label, 0.0])
    return out_path


def selftest():
    import tempfile
    synthetic = {
        "month": "2018_04", "species": "humpback_song",
        "annotations": [
            {"recording_32khz": "/x/MARS_20180430_061912_resampled_32kHz.wav",
             "annotation_offset_s": 190.0, "label_type": "positive", "species": "humpback_song"},
            {"recording_32khz": "/x/MARS_20180430_121912_resampled_32kHz.wav",
             "annotation_offset_s": 10.0, "label_type": "positive", "species": "humpback_song"},
            {"recording_32khz": "/x/MARS_20180430_130000_resampled_32kHz.wav",
             "annotation_offset_s": 5.0, "label_type": "negative", "species": "humpback_song"},
        ],
    }
    ok = True
    rows = json_to_rows(synthetic)
    if len(rows) != 2:
        print(f"FAIL: expected 2 positive rows, got {len(rows)}"); ok = False
    if rows and (rows[0][0] != "MARS_20180430_061912_resampled_32kHz.wav"
                 or rows[0][1] != 190.0 or rows[0][2] != 195.0):
        print(f"FAIL: row 0 wrong: {rows[0]}"); ok = False
    rows_neg = json_to_rows(synthetic, include_negatives=True)
    if len(rows_neg) != 3:
        print(f"FAIL: expected 3 rows with negatives, got {len(rows_neg)}"); ok = False

    tmp = os.path.join(tempfile.mkdtemp(), "out.csv")
    write_csv(rows, tmp, "test_project")
    with open(tmp) as f:
        r = list(csv.DictReader(f))
    if list(r[0].keys()) != CSV_HEADER:
        print(f"FAIL: header {list(r[0].keys())}"); ok = False
    if r[0]["filename"] != "MARS_20180430_061912_resampled_32kHz.wav" or r[0]["label"] != "humpback_song":
        print(f"FAIL: csv row 0 {r[0]}"); ok = False
    if r[0]["window_end"] != "195.0":
        print(f"FAIL: window_end {r[0]['window_end']}"); ok = False
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="explicit labels JSON path (overrides --month/--species)")
    ap.add_argument("--month", help="e.g. 2018_04 (with --species, builds the JSON path)")
    ap.add_argument("--species", help="e.g. humpback_song (with --month)")
    ap.add_argument("--json-dir", default=JSON_DIR)
    ap.add_argument("--out", help="output review CSV path (default results/review_{month}_{species}.csv)")
    ap.add_argument("--project", help="CSV project field (cosmetic; default from month)")
    ap.add_argument("--window-len", type=float, default=5.0)
    ap.add_argument("--include-negatives", action="store_true",
                    help="also include label_type != positive (default positive-only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.json:
        json_path = args.json
    elif args.month and args.species:
        json_path = os.path.join(args.json_dir, f"labels_{args.month}_{args.species}.json")
    else:
        ap.error("provide --json, or both --month and --species")

    if not os.path.exists(json_path):
        ap.error(f"labels JSON not found: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    month = data.get("month", args.month or "unknown")
    species = data.get("species", args.species or "unknown")
    out = args.out or os.path.join(RESULTS_DIR, f"review_{month}_{species}.csv")
    project = args.project or f"MARS_{month}"

    rows = json_to_rows(data, window_len=args.window_len,
                        include_negatives=args.include_negatives)
    if not rows:
        print(f"No rows to write (positive-only={not args.include_negatives}).")
        sys.exit(1)
    write_csv(rows, out, project)
    print(f"Wrote {len(rows)} rows -> {out}")
    print(f"  species={species}  month={month}  project={project}")
    print("\nFeed to review, e.g.:")
    print(f"  phase2_classify.py review --target-label {species} \\")
    print(f"      --detections-csv {out} --num-results {max(len(rows), 25)} \\")
    print(f"      --classes {species},gray_whale_moan,other,unlabeled --serve --port 78XX")


if __name__ == "__main__":
    main()
