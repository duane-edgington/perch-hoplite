#!/usr/bin/env python3
"""tools/export_slim_npz.py  --  perch-hoplite

Read embeddings back out of the Hoplite DB (no GPU), apply the orca_v4 linear
classifier, and write one SLIM .npz per recording into a perch/YYYY/MM/ tree that
mirrors the multispecies logits/ tree. The slim file carries the 5 class-logit
tracks + per-frame UTC epoch (what the temporal HMM needs); embeddings are omitted
by default (--with-embeddings to include the [T,1536] block for a future sequence
head). Pure CPU, minutes not the 37-min GPU embed.

Usage:
    python3 tools/export_slim_npz.py DB_PATH OUT_DIR \
        [--classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v4.pt] \
        [--month 2018/04] [--with-embeddings] [--force]

Notes / deliberate choices:
  * Frame order: window rows are sorted by the UNPACKED start_s in Python, NOT by
    `ORDER BY offsets` -- the offsets blob is little-endian float64, whose raw byte
    order is not numeric order, so a SQL blob sort can silently scramble frames.
  * epoch_seconds is UTC (filename YYYYMMDD_HHMMSS + start_s), matching the
    multispecies consolidator so the two join by absolute time. `.timestamp()` on a
    naive datetime would use local time and misalign -- we force UTC.
"""
import argparse, os, re, sqlite3, struct, sys, datetime as dt
import numpy as np

FALLBACK_CLASSES = ['dolphin_call', 'humpback_song', 'orca_call', 'other', 'ship_noise']
_TS = re.compile(r'MARS_(\d{8})_(\d{6})')


def rec_start(filename):
    """(utc_epoch_seconds, 'YYYYMMDD') from a MARS_YYYYMMDD_HHMMSS_... filename."""
    m = _TS.search(filename)
    if not m:
        raise ValueError(f'no MARS_YYYYMMDD_HHMMSS in {filename!r}')
    t = dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
    return t.replace(tzinfo=dt.timezone.utc).timestamp(), m.group(1)


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def load_classifier(path):
    import torch
    clf = torch.load(path, map_location='cpu', weights_only=False)
    beta = to_numpy(clf.beta).astype(np.float64)        # [1536, 5]
    bias = to_numpy(clf.beta_bias).astype(np.float64)   # [5]
    classes = [c.decode() if isinstance(c, bytes) else str(c)
               for c in list(getattr(clf, 'classes', FALLBACK_CLASSES))]
    assert beta.ndim == 2 and beta.shape[1] == len(classes) == bias.shape[0], \
        f'shape mismatch: beta {beta.shape}, bias {bias.shape}, classes {len(classes)}'
    return beta, bias, classes


def frames_for_recording(con, recording_id):
    """[(window_id, start_s)] sorted by start_s (numeric, not blob byte order)."""
    rows = con.execute(
        "SELECT id, offsets FROM windows WHERE recording_id = ?", (recording_id,)
    ).fetchall()
    parsed = [(wid, struct.unpack('<dd', off)[0]) for wid, off in rows]
    parsed.sort(key=lambda p: p[1])
    return parsed


def out_path(out_dir, filename, ymd, nested=True):
    stem = os.path.splitext(os.path.basename(filename))[0]
    sub = os.path.join(out_dir, ymd[:4], ymd[4:6]) if nested else out_dir
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, stem + '.npz')


def build_npz(logits, classes, epoch, with_embeddings, emb):
    out = {f'perch_{c}': logits[:, j].astype(np.float32) for j, c in enumerate(classes)}
    out['epoch_seconds'] = epoch.astype(np.float64)
    out['hop_sec'] = np.float64(5.0)
    out['classes'] = np.array(classes)
    if with_embeddings:
        out['embeddings'] = emb.astype(np.float32)
    return out


def export(db_path, out_dir, clf_path, with_embeddings=False, force=False, month=None):
    from perch_hoplite.db import sqlite_usearch_impl
    beta, bias, classes = load_classifier(clf_path)
    print(f'classifier: {beta.shape[0]}-dim -> {classes}', file=sys.stderr, flush=True)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_path, readonly=True)  # existing DB, no mutation
    con = sqlite3.connect(f'{db_path}/hoplite.sqlite')
    recs = con.execute("SELECT id, filename FROM recordings ORDER BY filename").fetchall()
    print(f'{len(recs)} recordings in DB', file=sys.stderr, flush=True)

    n_written, n_gap = 0, 0
    for i, (rid, filename) in enumerate(recs, 1):
        try:
            t0, ymd = rec_start(filename)
        except ValueError as e:
            print(f'  skip {filename}: {e}', file=sys.stderr); continue
        if month and f'{ymd[:4]}/{ymd[4:6]}' != month:
            continue
        dest = out_path(out_dir, filename, ymd)
        if os.path.exists(dest) and not force:
            continue
        frames = frames_for_recording(con, rid)
        if not frames:
            continue
        window_ids = [f[0] for f in frames]
        starts = np.array([f[1] for f in frames], dtype=np.float64)
        # contiguity sanity: 5 s non-overlapping
        if len(starts) > 1 and not np.allclose(np.diff(starts), 5.0):
            n_gap += 1
        emb = np.asarray(db.get_embeddings_batch(window_ids), dtype=np.float64)  # [T,1536], one call
        logits = emb @ beta + bias                          # [T,5]
        np.savez(dest, **build_npz(logits, classes, t0 + starts, with_embeddings, emb))
        n_written += 1
        if i % 200 == 0:
            print(f'  {i}/{len(recs)} recordings', file=sys.stderr, flush=True)

    print(f'\nwrote {n_written} .npz under {out_dir}', file=sys.stderr)
    if n_gap:
        print(f'NOTE: {n_gap} recording(s) had non-5s-contiguous frames (gaps/overlap)',
              file=sys.stderr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('db_path')
    ap.add_argument('out_dir')
    ap.add_argument('--classifier', default='/mnt/PAM_Analysis/perch-hoplite/models/orca_v4.pt')
    ap.add_argument('--month', default=None, help='filter to YYYY/MM')
    ap.add_argument('--with-embeddings', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    export(a.db_path, a.out_dir, a.classifier, a.with_embeddings, a.force, a.month)
