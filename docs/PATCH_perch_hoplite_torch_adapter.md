# PATCH — per-window peak normalization for the perch-hoplite PyTorch pipeline

Fixes the low-amplitude divergence found on 2026-07-09 (see
`FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md`). Peak-normalizes every 5-second window
to 0.25 before the model, restoring cos 1.0 vs live TF on low-amplitude MARS audio.

Apply this in **`perch_hoplite_torch_adapter.py`** (the single point both embedding and
re-embedding detection paths flow through). Two edits.

---

## Edit 1 — add the helper (near the top of the file, after the imports)

```python
PEAK_TARGET = 0.25   # Perch's internal peak-norm target; pre-normalizing to it keeps the
                     # log-mel frontend numerically stable at low input amplitude.

def peak_normalize_windows(frames: np.ndarray, target_peak: float = PEAK_TARGET) -> np.ndarray:
    """Per-window peak normalization to `target_peak`, computed in float64.

    Each row of `frames` (shape [num_windows, samples]) is scaled so its peak
    magnitude equals `target_peak`. This is idempotent with Perch's internal
    peak_norm(0.25): the resulting embedding is the identical canonical, amplitude-
    invariant Perch embedding, but the frontend arithmetic stays in a numerically
    stable range so the port matches the reference TF model (cos 1.0) even on very
    low-amplitude input. Silent windows (peak ~ 0) are left unscaled.
    """
    f = np.atleast_2d(frames).astype(np.float64)
    peak = np.abs(f).max(axis=-1, keepdims=True)
    scale = np.where(peak > 1e-12, target_peak / np.maximum(peak, 1e-12), 1.0)
    return (f * scale).astype(np.float32)
```

## Edit 2 — call it inside `embed()`, right after framing

Find the framing lines in `embed()`:

```python
        framed = self.frame_audio(audio_array, self.window_size_s, self.hop_size_s)
        if framed.ndim == 1:
            framed = framed[None, :]
        x = torch.from_numpy(np.ascontiguousarray(framed)).float().to(self.device)
```

and insert the normalization between the reshape and the tensor conversion:

```python
        framed = self.frame_audio(audio_array, self.window_size_s, self.hop_size_s)
        if framed.ndim == 1:
            framed = framed[None, :]
        framed = peak_normalize_windows(framed)          # <-- FIX: per-window peak-norm to 0.25
        x = torch.from_numpy(np.ascontiguousarray(framed)).float().to(self.device)
```

That's the whole fix for the embedding path. Because `build_db()` (phase-1 embedding) and
any detection code that re-embeds audio both call this `embed()`, this one change covers
both. Nothing else in the adapter changes.

---

## If a detection/inference path does NOT go through the adapter

If some code turns a raw 5-second clip into an embedding without calling the adapter's
`embed()` (e.g. a standalone inference helper), apply the same normalization to that clip
before the model. For a single 1-D clip:

```python
from perch_hoplite_torch_adapter import peak_normalize_windows
clip = peak_normalize_windows(clip_1d[None, :])[0]   # -> peak 0.25, then embed as usual
```

The rule: **any audio→embedding path must peak-normalize each 5 s window to 0.25 first.**
Keep that invariant in one helper so it can't drift between the embed and detect paths.

---

## Re-embed / retrain checklist (existing DBs carry the pre-fix divergence)

1. Apply the patch above.
2. Re-run **phase-1 embedding** to rebuild the DB(s) with normalized embeddings.
3. Re-attach your existing **annotations** by (filename, offset) — labels reference
   windows, not embedding values, so they carry over. Keep your annotated files as-is.
4. **Retrain** the linear classifier on the regenerated embeddings.
5. **Re-run detections**, and re-validate a known result (e.g. April 13 2018 orca count)
   to confirm the headline numbers hold — a few boundary detections may shift.

## Optional: drop `vol 3` from the resampling script

Per-window normalization erases any constant gain, so `vol 3` in
`new_32k_resample_sox.sh` no longer affects embeddings and is a mild clipping risk on loud
events. Safe to remove (keep the resample + 10 Hz high-pass). Low priority.

---

### Verification reference (what "fixed" looks like)

Real MARS windows from `MARS_20180413_065913`, port vs live TF:

| window raw peak | cos RAW | cos after norm→0.25 |
|---|---|---|
| 0.0028 | 0.936 | 1.00000 |
| 0.0018 | 0.756 | 1.00000 |
| 0.0015 | 0.771 | 1.00000 |

(Full sweep confirmed cos 1.00000 at peak 0.25 for all clips; the normalized embedding is
amplitude-invariant — scaling a clip 1000× or 0.01× before normalizing gives cos 0.9999999
to the original, confirming the fix does not distort the embedding.)
