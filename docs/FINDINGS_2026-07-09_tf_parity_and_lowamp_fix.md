# Perch 2.0 PyTorch port — TF-parity verification & low-amplitude fix
### Findings, 2026-07-09 (verified on Colab A100 against the live Kaggle TF Perch 2.0 SavedModel)

## Summary

We verified the native PyTorch port directly against Google's **live** TensorFlow
Perch 2.0 model (`google/bird-vocalization-classifier/tensorFlow2/perch_v2`, loaded in
Colab), rather than only against archived reference arrays. Two conclusions:

1. **The port is faithful to TF — at healthy signal amplitude.** On a normal-amplitude
   clip (peak ≈ 0.32) the full raw-audio→embedding pipeline matches the live TF model at
   **cosine 0.9998**. The reverse-engineered architecture and the ONNX-extracted weights
   are correct (a wrong weight could not produce end-to-end parity).

2. **The port diverged from TF at low absolute amplitude — now diagnosed and fixed.**
   On real MARS hydrophone windows (peak ≈ 0.0015–0.003) the port disagreed with live TF
   at **cosine 0.76–0.94**. The cause is numerical, in the log-mel frontend, and it is
   **fully resolved by peak-normalizing each 5 s window to 0.25 before the model**, which
   restores **cosine 1.0000** on every window tested.

We also **confirmed, directly from the live TF model's spectrogram output, that the
frontend is log-mel and not PCEN**: three of four test clips bottom out at exactly
`0.1·ln(1e-5) = −1.151293`, the hard floor of `0.1·log(max(mel, 1e-5))`. PCEN has no such
constant floor. (The one clip that didn't reach the floor is simply loud enough that no
mel bin clamps — expected, not a discrepancy.)

## Root cause

Perch's frontend internally peak-normalizes each 5 s window to a fixed peak of 0.25 before
computing the log-mel spectrogram. When the raw input window is very low-amplitude
(MARS data sits near peak 0.002), that internal normalization multiplies the signal by a
large factor (~125×). At that amplification the port's frontend arithmetic and the live TF
frontend's arithmetic diverge — the two implementations amplify the low-order (float32)
detail of the input differently, and the difference lands in the many mel bins sitting near
the `1e-5` log floor. The effect scales with how much amplification is needed, so it is
severe for quiet windows and absent for loud ones:

| clip / window peak | cos(port, live TF) |
|---|---|
| 0.32   (loud) | 0.9998 |
| 0.0024 | 0.9472 |
| 0.0006–0.0007 (MARS-typical) | 0.55 / 0.43 |

A per-window amplitude audit of a representative 10-minute MARS file
(`MARS_20180413_065913`) found **all 120 windows** below peak 0.005 — i.e. the entire
file sits in the divergence zone. So this is not an edge case for MBARI data; it is the
normal operating regime, and it must be fixed for the port to be trustworthy on real
recordings.

## The fix

**Peak-normalize each 5-second window to 0.25 before embedding.** Verified on real MARS
orca-call windows: every tested window went from cos 0.76–0.94 (raw) to **cos 1.00000**
(normalized).

Why this is correct and non-distorting: Perch's internal peak-norm is idempotent —
`peak_norm(peak_norm(x, 0.25), 0.25) = peak_norm(x, 0.25)`. Pre-normalizing each window to
0.25 therefore yields the **identical canonical Perch embedding** (Perch is amplitude-
invariant per window by design), while keeping the frontend arithmetic in a numerically
stable range so the port and TF agree. It changes nothing about the science; it removes a
numerical artifact.

**Scope: normalize per 5-second window, not per file.** Perch normalizes each window
independently, so file-level normalization can leave quiet passages under-normalized. The
robust fix mirrors the model: normalize every window to peak 0.25 at the point of
embedding (see `PATCH_perch_hoplite_torch_adapter.md`). This single change covers both the
bulk embedding path (`build_db` / phase-1) and any detection path that re-embeds audio,
because both go through the adapter's `embed()`.

## Consequences for existing results

- **All embeddings produced before this fix carry the divergence** (they were computed
  from un-normalized, low-amplitude MARS audio). They are not the canonical Perch
  embeddings and should be regenerated.
- **Existing detections are likely internally consistent but not canonical.** The
  classifier was trained on divergent embeddings, so it may separate classes fine
  (consistent with the good ROC-AUC / t-SNE) — but the embeddings themselves differ from
  standard Perch. Re-embed with normalization, then retrain the classifier on the
  regenerated embeddings.
- **Annotations are preserved.** Human labels reference audio windows by
  file/offset, not by embedding value, so they survive re-embedding. Rebuild the DB,
  re-attach annotations by (filename, offset), retrain, and re-run detections.
- **Re-validate a known result after re-embedding** (e.g. the April 13 2018 orca count):
  a few boundary detections may shift. Confirm the headline numbers hold rather than
  assuming they carry over unchanged.

## Note on the `vol 3` SoX step

The resampling script's `vol 3` gain is now irrelevant to the embeddings — per-window
normalization erases any constant gain. It is also a mild clipping risk on loud events
(peak × 3 can exceed 1.0). Recommendation: **drop `vol 3`** from the resample step (keep
the resample and 10 Hz high-pass); normalization at embed time handles amplitude. Low
priority given the fix, but it removes a latent clipping hazard.

## Longer-term (optional) hardening

The most robust fix would make the port's frontend numerically stable at any amplitude
without relying on pre-normalization — i.e. ensure the internal peak-norm, mel matmul, and
log all run in float64 (the FFT already does). Then the port matches TF at any input level
with no preprocessing dependency. The per-window normalization above is the proven,
immediate fix; the float64 frontend is the "no one can forget to normalize" version, worth
doing eventually in the `perch-pytorch` repo's `perch_frontend_torch.py`.
