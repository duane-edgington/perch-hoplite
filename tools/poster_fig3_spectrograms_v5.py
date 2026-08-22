#!/usr/bin/env python3
"""
poster_fig3_class_spectrograms.py — matched 4-panel spectrogram row for the poster
(one confirmed exemplar clip per class: orca_call, dolphin_call, humpback_song, ship_noise).

Requested by poster chat: "same time and frequency axes on all four, same colormap, same
window length, exported at 300 dpi" — panel 3's biggest gap was that the poster was entirely
abstract (diagrams/curves/bars) with no one ever seeing the actual sound.

Mel-spectrogram recipe matches phase2_classify.py's review display EXACTLY (n_fft=512,
hop=128, n_mels=128, fmin=10, fmax=16000, power=2.0, power_to_db ref=max) so this is
visually consistent with every other spectrogram already in this project's figures/.

Exemplars (all already expert-confirmed labels, no new review needed; offsets decoded from
the hoplite DB's `annotations.offsets` blob, which is 2 little-endian doubles = (start, end)):

    orca_call      MARS_20180413_075913_resampled_32kHz.wav   370.0-375.0 s
    dolphin_call   MARS_20180420_200913_resampled_32kHz.wav   205.0-210.0 s
    humpback_song  MARS_20201025_175314_resampled_32kHz.wav   135.0-140.0 s
    ship_noise     MARS_20180430_195912_resampled_32kHz.wav   130.0-135.0 s

NOTE on humpback_song exemplar (Aug 20 2026): the first candidate tried
(MARS_20201001_054822, 155-160s) was rejected on expert review (D. Edgington) — even in
30s context it showed all energy below 1000 Hz with no repeating phrase structure, i.e. it
read as a low-frequency moan (gray whale/blue whale candidate, species undetermined), NOT
humpback song, despite being a confirmed humpback_song label. Surfaced as a possible new
mislabeling case worth flagging for the #13 gray-whale review (though Oct 2020 gray whale
presence is considered out-of-season per local naturalist sources; blue whale, which IS
dominant in Monterey Bay in October and shares the low-frequency simple-call profile, is at
least as plausible an alternate ID here). Current exemplar (Oct 25, 175314, 135-140s) was
chosen from a batch of 8 candidates after visual/audio confirmation of higher-frequency
content and repeating structure consistent with genuine humpback song.

Usage:
    python3 tools/poster_fig3_class_spectrograms.py \
        --out /mnt/PAM_Analysis/perch-hoplite/results/poster_fig3_class_spectrograms.png
    python3 tools/poster_fig3_class_spectrograms.py --selftest
"""
import argparse
import os
import sys

import numpy as np

AUDIO_ROOT = "/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz"

# (label, display_name, subdir under AUDIO_ROOT, filename, start_s, end_s)
EXEMPLARS = [
    ("orca_call", "Orca call", "2018/04", "MARS_20180413_075913_resampled_32kHz.wav", 370.0, 375.0),
    ("dolphin_call", "Pacific white-sided dolphin", "2018/04", "MARS_20180420_200913_resampled_32kHz.wav", 205.0, 210.0),
    ("humpback_song", "Humpback song", "2020/10", "MARS_20201025_175314_resampled_32kHz.wav", 135.0, 140.0),
    ("ship_noise", "Ship noise", "2018/04", "MARS_20180430_195912_resampled_32kHz.wav", 130.0, 135.0),
]

# Exact match to phase2_classify.py's "mel" spectrogram-type block.
# n_fft=2048 (up from phase2_classify.py's default of 512): at sr=32000, 512 gives only
# 62.5 Hz/bin frequency resolution, which leaves some of the 128 mel filters near the low
# end (fmin=10 Hz) with NO FFT bin inside them at all -- librosa's "Empty filters detected"
# warning, and the literal cause of black/zero bars around 500-800 Hz that clip out
# humpback's low-frequency content. 2048 gives 15.625 Hz/bin -- fine enough that every mel
# filter gets real energy. Trade-off: coarser time resolution (larger hop), acceptable for
# humpback (slow-varying low-freq moans) but worth knowing if this is reused for fast
# transients (e.g. orca clicks) where the phase2_classify.py 512 default may still be right.
N_FFT = 2048
HOP = N_FFT // 4
N_MELS = 128
F_MIN = 10.0
F_MAX = 16000.0
COLORMAP = "viridis"  # project convention for registered figures (phase2_classify.py
                       # defaults to inferno; viridis is what's used in every Apr/May
                       # registered figure to date — keep consistent with those, not the
                       # phase2_classify.py internal default)


def compute_mel_db(y, sr, n_fft=N_FFT, hop=None):
    import librosa
    if hop is None:
        hop = n_fft // 4
    S = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=sr,
        n_fft=n_fft, hop_length=hop,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    f = librosa.mel_frequencies(n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    t = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)
    return S_db, f, t


def load_window(path, start_s, end_s):
    import soundfile as sf
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(start_s * sr))
        n = int(round((end_s - start_s) * sr))
        y = f.read(n, dtype="float32")
    return y, sr


def diagnose_empty_bins(S_db, f, label):
    """Check for near-zero-energy mel bins (symptom of FFT/mel-filter resolution mismatch
    at n_fft=512 — flagged by librosa's 'Empty filters detected' warning). A bin whose max
    value across all time frames is far below the rest is a real data gap, not a display
    issue, and no colormap/axis change will fix it."""
    col_max = S_db.max(axis=1)  # best case per mel bin, across time
    floor = col_max.min()
    suspects = [(f[i], col_max[i]) for i in range(len(f)) if col_max[i] <= floor + 1.0]
    if suspects:
        print(f"  [{label}] SUSPECT EMPTY BINS (near-floor across all time): "
              + ", ".join(f"{hz:.0f}Hz(max={v:.1f}dB)" for hz, v in suspects))
    else:
        print(f"  [{label}] no obviously-empty bins detected")
    return suspects


def build_figure(rows, out_path, dpi=300, window_len=5.0, colormap=COLORMAP, log_freq=False):
    """rows: list of (label, display_name, S_db, f, t) — same window_len for all."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared color scale across all four panels so brightness is comparable, not just axes.
    vmax = max(S_db.max() for _, _, S_db, _, _ in rows)
    vmin = max(vmax - 80, min(S_db.min() for _, _, S_db, _, _ in rows))

    for ax, (label, name, S_db, f, t) in zip(axes, rows):
        im = ax.pcolormesh(t, f, S_db, cmap=colormap, vmin=vmin, vmax=vmax, shading="gouraud")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, window_len)
        if log_freq:
            ax.set_yscale("log")
            ax.set_ylim(max(F_MIN, 20), F_MAX)
        else:
            ax.set_ylim(F_MIN, F_MAX)
        if ax is axes[0]:
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("Power (dB)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_context(path, center_start, center_end, out_path, n_fft=N_FFT, colormap=COLORMAP,
                 context_s=30.0, dpi=200):
    """30-second context view around a focal window, matching the project's established
    diagnostic (see gradio_30s_context_*.png): humpback's repeating phrase structure is often
    invisible in a 5s clip alone and only shows up at 30s. Highlights the focal window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    win_len = center_end - center_start
    pad = (context_s - win_len) / 2.0
    ctx_start = max(0.0, center_start - pad)

    with sf.SoundFile(path) as f:
        sr = f.samplerate
        total_s = len(f) / sr
        ctx_end = min(total_s, ctx_start + context_s)
        f.seek(int(ctx_start * sr))
        n = int(round((ctx_end - ctx_start) * sr))
        y = f.read(n, dtype="float32")

    S_db, mel_f, t = compute_mel_db(y, sr, n_fft=n_fft)
    t = t + ctx_start  # shift to absolute time for the highlight box to line up

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, mel_f, S_db, cmap=colormap, shading="gouraud")
    ax.axvspan(center_start, center_end, edgecolor="yellow", facecolor="none", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{os.path.basename(path)} — {context_s:.0f}s context "
                 f"(focal window {center_start:.0f}-{center_end:.0f}s highlighted)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/mnt/PAM_Analysis/perch-hoplite/results/poster_fig3_class_spectrograms.png")
    ap.add_argument("--audio-root", default=AUDIO_ROOT)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--colormap", default=COLORMAP, help="matplotlib colormap (default viridis)")
    ap.add_argument("--log-freq", action="store_true", help="log-scale frequency axis")
    ap.add_argument("--n-fft", type=int, default=N_FFT,
                    help=f"FFT window size (default {N_FFT}; phase2_classify.py's 'mel' mode "
                         "uses 512, which leaves empty low-freq mel filters -- see comment "
                         "at top of file)")
    ap.add_argument("--diagnose", action="store_true",
                    help="print suspected empty/zero-energy mel bins per class before plotting")
    ap.add_argument("--context-check", metavar="LABEL",
                     help="instead of building the 4-panel figure, plot a 30s-context view "
                          "around that class's exemplar window (e.g. --context-check humpback_song) "
                          "to check whether it shows humpback's repeating-phrase structure. "
                          "Combine with --wav/--start/--end to check an arbitrary candidate "
                          "clip instead of the hardcoded EXEMPLARS entry.")
    ap.add_argument("--wav", help="override: specific filename to check (with --context-check)")
    ap.add_argument("--subdir", default="2020/10", help="override: subdir under --audio-root for --wav")
    ap.add_argument("--start", type=float, help="override: window start seconds (with --wav)")
    ap.add_argument("--end", type=float, help="override: window end seconds (with --wav)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.context_check:
        if args.wav:
            label = args.context_check
            path = os.path.join(args.audio_root, args.subdir, args.wav)
            start_s, end_s = args.start, args.end
            if start_s is None or end_s is None:
                print("--wav requires --start and --end too.")
                sys.exit(1)
        else:
            match = next((e for e in EXEMPLARS if e[0] == args.context_check), None)
            if not match:
                print(f"No exemplar for label '{args.context_check}'. Options: "
                      + ", ".join(e[0] for e in EXEMPLARS))
                sys.exit(1)
            label, name, subdir, fname, start_s, end_s = match
            path = os.path.join(args.audio_root, subdir, fname)
        out = args.out.replace(".png", f"_context_{label}.png") if args.out.endswith(".png") \
              else args.out + f"_context_{label}.png"
        out_path = plot_context(path, start_s, end_s, out, n_fft=args.n_fft, colormap=args.colormap)
        print(f"Wrote 30s context view: {out_path}  (source: {os.path.basename(path)} @ "
              f"{start_s}-{end_s}s)")
        sys.exit(0)

    rows = []
    for label, name, subdir, fname, start_s, end_s in EXEMPLARS:
        path = os.path.join(args.audio_root, subdir, fname)
        if not os.path.exists(path):
            print(f"ERROR: audio file not found: {path}")
            sys.exit(1)
        y, sr = load_window(path, start_s, end_s)
        S_db, f, t = compute_mel_db(y, sr, n_fft=args.n_fft)
        rows.append((label, name, S_db, f, t))
        print(f"  {label}: {fname} @ {start_s}-{end_s}s  (sr={sr}, {len(y)} samples, n_fft={args.n_fft})")
        if args.diagnose:
            diagnose_empty_bins(S_db, f, label)

    out = build_figure(rows, args.out, dpi=args.dpi, colormap=args.colormap, log_freq=args.log_freq)
    print(f"\nWrote {out}")
    print("Register with tools/register_figure.py (--type matplotlib_plot) before committing.")


def selftest():
    """Synthetic 4-panel test — no real audio needed. Validates the plotting path only."""
    import tempfile
    sr = 32000
    rng = np.random.default_rng(0)
    rows = []
    names = ["Orca call (synthetic)", "Dolphin (synthetic)", "Humpback (synthetic)", "Ship noise (synthetic)"]
    for i, name in enumerate(names):
        y = (rng.standard_normal(5 * sr) * 0.01).astype(np.float32)
        # inject a tone so each panel has visible structure
        tt = np.arange(len(y)) / sr
        y += 0.1 * np.sin(2 * np.pi * (500 + i * 800) * tt).astype(np.float32)
        S_db, f, t = compute_mel_db(y, sr)
        rows.append((f"class_{i}", name, S_db, f, t))

    out = os.path.join(tempfile.mkdtemp(), "selftest.png")
    ok = True
    try:
        build_figure(rows, out)
    except Exception as e:
        print(f"FAIL: build_figure raised: {e}")
        ok = False
    if ok and not (os.path.exists(out) and os.path.getsize(out) > 5000):
        print("FAIL: output file missing or too small")
        ok = False
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
