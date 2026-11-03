#!/usr/bin/env python3
"""
audit_stats_vs_db.py   (READ-ONLY)

Compares campaign_pass_statistics.csv with the per-month hoplite databases and the v10
inference CSVs, and reports every disagreement.

Definitions
  * Annotation counts use label_type 1 (positive) and unique windows (recording + offset).
  * Labels are counted separately: orca_call, humpback_song, dolphin_call, ship_noise, ROV_noise,
    other ("something I cannot identify yet"; never merged with ship or ROV).
  * Pass-1 / pass-2 split by v10 logit: a labeled window whose v10 logit is >= the threshold
    (default 2.31) is pass-1; below the threshold (including windows with no logit because they
    scored below the 0.0 inference cutoff) is pass-2.
  * Pass-2 evidence: labeled windows below the threshold. For orca this is orca_call windows
    with a v10 logit below 2.31.
  * v10_above_threshold in the CSV is compared with the number of v10 inference rows at or above
    the threshold.
  * PASS1_GAP: v10 windows at or above the threshold that have no annotation of any type, beyond
    the number of skipped clips recorded in pass1_unlabeled (windows never reviewed in pass-1).

Writes a detail CSV and prints a compact list of disagreements. Changes nothing.

Usage (from ~/perch-hoplite on spark-ae0e):
    python3 tools/audit_stats_vs_db.py --out-csv ~/audit_stats_vs_db.csv
Default range is 2015-01 through 2023-12 (--since / --until change it).
"""

import argparse
import csv
import glob
import os
import re
import sqlite3
import struct
from collections import defaultdict
from pathlib import Path

DB_RE = re.compile(r"^MARS_(\d{4})(\d{2})01_\d{8}_32kHz(_norm)?$")
STEM_RE = re.compile(r"(MARS_(\d{8})_\d{6})")
# short column name -> DB label
COLS = {"orca": "orca_call", "humpback": "humpback_song", "dolphin": "dolphin_call",
        "ship": "ship_noise", "rov": "ROV_noise", "other": "other"}


def first_offset(blob):
    if not blob:
        return 0.0
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", bytes(blob[: n * 8]))[0] if n else 0.0


def month_dbs(root):
    chosen = {}
    for d in sorted(Path(root).iterdir()):
        m = DB_RE.match(d.name)
        if m and (d / "hoplite.sqlite").exists():
            key = f"{m.group(1)}-{m.group(2)}"
            if key not in chosen or (m.group(3) and not chosen[key][1]):
                chosen[key] = (d, bool(m.group(3)))
    return {k: v[0] for k, v in sorted(chosen.items())}


def read_db(db_dir):
    """{window_key: {(label, label_type), ...}}"""
    con = sqlite3.connect(f"file:{db_dir / 'hoplite.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT r.filename, a.offsets, a.label, a.label_type "
                           "FROM annotations a JOIN recordings r ON r.id = a.recording_id").fetchall()
    finally:
        con.close()
    out = defaultdict(set)
    for fn, blob, label, ltype in rows:
        m = STEM_RE.search(fn)
        if m:
            out[(m.group(1), round(first_offset(blob), 1))].add((label, ltype))
    return out


def read_logits(results_dir, ym):
    """{window_key: v10 logit} from MARS_YYYYMM01_*_v10_orcaval.csv, or None if the file is missing."""
    y, m = ym.split("-")
    files = glob.glob(os.path.join(os.path.expanduser(results_dir), f"MARS_{y}{m}01_*_v10_orcaval.csv"))
    if not files:
        return None
    out = {}
    with open(files[0], newline="") as f:
        for r in csv.DictReader(f):
            if r.get("label", "orca_call") != "orca_call":
                continue
            mm = STEM_RE.search(r["filename"])
            if mm:
                out[(mm.group(1), round(float(r["window_start"]), 1))] = float(r["logits"])
    return out


def read_review_sets(results_dir):
    """Windows listed in review_*pass1*/pass2* CSVs (information only)."""
    sets = defaultdict(set)
    for path in glob.glob(os.path.join(os.path.expanduser(results_dir), "review_*.csv")):
        name = os.path.basename(path)
        if "pass1" not in name and "pass2" not in name:
            continue
        try:
            with open(path, newline="") as f:
                rd = csv.DictReader(f)
                if "filename" not in (rd.fieldnames or []) or "window_start" not in rd.fieldnames:
                    continue
                for r in rd:
                    mm = STEM_RE.search(r["filename"])
                    if mm:
                        sets[f"{mm.group(2)[:4]}-{mm.group(2)[4:6]}"].add(
                            (mm.group(1), round(float(r["window_start"]), 1)))
        except Exception:
            continue
    return sets


def ival(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def main():
    ap = argparse.ArgumentParser(description="Read-only audit of the stats CSV against the DBs and v10 logits.")
    ap.add_argument("--db-root", default="/mnt/PAM_Analysis/perch-hoplite/db")
    ap.add_argument("--results-dir", default="/mnt/PAM_Analysis/perch-hoplite/results")
    ap.add_argument("--stats-csv", default="~/perch-hoplite-orca_science/campaign_pass_statistics.csv")
    ap.add_argument("--out-csv", default="~/audit_stats_vs_db.csv")
    ap.add_argument("--since", default="2015-01")
    ap.add_argument("--until", default="2023-12")
    ap.add_argument("--threshold", type=float, default=2.31, help="pass-1 logit threshold (default 2.31)")
    args = ap.parse_args()
    T = args.threshold

    stats = {r["month"]: r for r in csv.DictReader(open(os.path.expanduser(args.stats_csv), newline=""))}
    review = read_review_sets(args.results_dir)
    out_rows, flags_by_month = [], {}

    for ym, db_dir in month_dbs(args.db_root).items():
        if not (args.since <= ym <= args.until):
            continue
        ann = read_db(db_dir)
        pos = {k: {l for l, t in v if t == 1} for k, v in ann.items()}
        logits = read_logits(args.results_dir, ym)
        db_unc = sum(1 for v in ann.values() if any(t == 2 for _, t in v))
        db_neg = sum(1 for v in ann.values() if any(t == 0 for _, t in v))
        db_tot = {s: 0 for s in COLS}
        p1 = {s: 0 for s in COLS}
        p2 = {s: 0 for s in COLS}
        nolog = {s: 0 for s in COLS}
        outside = {s: 0 for s in COLS}
        rv = review.get(ym, set())
        for k, labs in pos.items():
            lg = None if logits is None else logits.get(k)
            for s, l in COLS.items():
                if l not in labs:
                    continue
                db_tot[s] += 1
                if lg is not None and lg >= T:
                    p1[s] += 1
                else:
                    p2[s] += 1
                    if lg is None:
                        nolog[s] += 1
                if k not in rv:
                    outside[s] += 1
        p2_any = sum(p2.values())
        calc_v10_above = None if logits is None else sum(1 for lg in logits.values() if lg >= T)
        # above-threshold windows that carry no annotation of any type (skipped or never reviewed in pass-1)
        above_unlabeled = None if logits is None else sum(1 for k, lg in logits.items() if lg >= T and k not in ann)
        row = {"month": ym, "v10_csv_found": logits is not None, "calc_v10_above": calc_v10_above,
               "calc_above_unlabeled": above_unlabeled, "db_uncertain": db_unc, "db_negative": db_neg,
               "p2_any": p2_any, "p2_orca": p2["orca"], "nologit_any": sum(nolog.values())}
        for s in COLS:
            row[f"db_{s}"] = db_tot[s]
            row[f"calc_p1_{s}"] = p1[s]
            row[f"calc_p2_{s}"] = p2[s]
            row[f"nologit_{s}"] = nolog[s]
            row[f"outside_review_{s}"] = outside[s]
        flags = []
        r = stats.get(ym)
        if logits is None:
            flags.append("NO_V10_CSV (pass split cannot be computed)")
        if not r:
            flags.append("NO_CSV_ROW")
        else:
            reviewed = r["pass1_orca"] != ""
            done = r.get("pass2_done", "")
            row["csv_pass2_done"] = done
            if not reviewed and sum(db_tot.values()) > 0:
                flags.append("DB_HAS_LABELS_BUT_CSV_NOT_REVIEWED")
            if reviewed and above_unlabeled is not None and "pass1_unlabeled" in r:
                recorded = ival(r["pass1_unlabeled"])
                if above_unlabeled > recorded:
                    flags.append(f"PASS1_GAP above-threshold windows with no annotation={above_unlabeled}, "
                                 f"recorded skipped={recorded}")
            if calc_v10_above is not None and "v10_above_threshold" in r:
                v = (r["v10_above_threshold"] or "").strip()
                if not re.fullmatch(r"\d+", v):
                    flags.append(f"V10_ABOVE_NOT_NUMERIC '{v}' calc={calc_v10_above}")
                elif int(v) != calc_v10_above:
                    flags.append(f"V10_ABOVE csv={v} calc={calc_v10_above}")
            if reviewed:
                for s in COLS:
                    if f"pass1_{s}" not in r:
                        if db_tot[s]:
                            flags.append(f"{s.upper()}_COLUMN_MISSING db={db_tot[s]}")
                        continue
                    c1, c2 = ival(r.get(f"pass1_{s}")), ival(r.get(f"pass2_{s}"))
                    row[f"csv_p1_{s}"], row[f"csv_p2_{s}"] = c1, c2
                    if c1 + c2 != db_tot[s]:
                        flags.append(f"{s.upper()}_TOTAL csv={c1 + c2} db={db_tot[s]}")
                    elif logits is not None and (c1 != p1[s] or c2 != p2[s]):
                        flags.append(f"{s.upper()}_SPLIT csv={c1}/{c2} logit={p1[s]}/{p2[s]}")
                if logits is not None:
                    if done == "1" and p2_any == 0:
                        flags.append("PASS2_DONE_BUT_NO_SUBTHRESHOLD_LABELS")
                    if done not in ("1", "NP") and db_tot["orca"] > 0 and p2_any > 0:
                        flags.append(f"SUBTHRESHOLD_LABELS_EXIST_BUT_PASS2_DONE='{done}' "
                                     f"(sub-threshold labels={p2_any}, orca={p2['orca']})")
                    if done == "NP" and p2_any > 0:
                        flags.append(f"INFO_NOT_PLANNED_BUT_SUBTHRESHOLD_LABELS_EXIST "
                                     f"(labels={p2_any}, orca={p2['orca']})")
            if row["nologit_any"] > 0 and logits is not None:
                flags.append(f"INFO_NO_LOGIT labeled windows without a v10 score: {row['nologit_any']}")
        row["flags"] = "; ".join(flags)
        flags_by_month[ym] = flags
        out_rows.append(row)

    db_months = {x["month"] for x in out_rows}
    for ym in stats:
        if ym not in db_months and args.since <= ym <= args.until:
            out_rows.append({"month": ym, "flags": "CSV_ROW_BUT_NO_DB"})
            flags_by_month[ym] = ["CSV_ROW_BUT_NO_DB"]

    out_rows.sort(key=lambda x: x["month"])
    fields = []
    for x in out_rows:
        for k in x:
            if k not in fields:
                fields.append(k)
    out_path = os.path.expanduser(args.out_csv)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    errs = [(m, f) for m, f in sorted(flags_by_month.items()) if any(not x.startswith("INFO_") for x in f)]
    info = [m for m, f in sorted(flags_by_month.items()) if f and all(x.startswith("INFO_") for x in f)]
    print(f"months audited: {len(out_rows)}   with at least one error flag: {len(errs)}   (detail: {out_path})\n")
    for m, f in errs:
        print(f"{m}  " + " | ".join(f))
    kinds = defaultdict(int)
    for _, f in errs:
        for x in f:
            if not x.startswith("INFO_"):
                kinds[re.split(r"[ (=]", x)[0]] += 1
    print("\nflag counts:", dict(sorted(kinds.items())))
    print(f"informational only: {len(info)} months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
