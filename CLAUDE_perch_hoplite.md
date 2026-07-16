# CLAUDE.md — Perch-Hoplite Project Context

This file provides context for Claude (AI assistant) when working on this
repository. It captures the project state, conventions, and key decisions
so that Claude can pick up where we left off without re-explaining everything.

---

## Project Summary

Applying Google Perch V2 bioacoustics embeddings + perch-hoplite agile
modeling to detect and classify marine mammal vocalizations in MARS
hydrophone data at MBARI. The pipeline runs entirely in PyTorch on a local
NVIDIA GB10 DGX — no TensorFlow, no Colab.

**Author:** Duane R. Edgington — MBARI (Monterey Bay Aquarium Research Institute)
**GitHub:** https://github.com/duane-edgington/perch-hoplite

---

## System Architecture

| Machine | Role | Notes |
|---|---|---|
| **ICEFISH** (Mac M1 Max) | Dev workstation, scp gateway | 134.89.114.25 / VPN 134.89.74.134 |
| **spark-ae0e** (134.89.11.107) | Primary compute — NVIDIA GB10 DGX | Working dir `~/perch-hoplite/` |
| **spark-0626** (134.89.11.174) | Spare DGX | |
| **thalassa** | NFS server | thalassa.shore.mbari.org |

---

## Key Paths

```
# Permanent pipeline data (NFS)
/mnt/PAM_Analysis/perch-hoplite/
    db/          — embedding databases (normalized, suffix _norm)
    models/      — trained classifiers (.pt + .metrics.json)
    results/     — inference CSVs and plots
    logs/        — embedding and inference logs
    provenance/  — label and training JSON records
    example_clips/ — 10 peak-normalized 5s example clips + manifest.json

/mnt/PAM_Analysis/perch_weights/    — Perch V2 weights (ONNX-extracted)
    weights.npz
    graph_manifest.json

# Audio (32kHz resampled WAV)
/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/
    2018/04/    — April 2018 (4,320 files, 518,400 windows)
    2018/05/    — May 2018   (4,464 files, 535,680 windows)
    2020/10/    — October 2020 (4,504 files, 535,278 windows)

# Old location (read-only fallback — DO NOT write here)
/mnt/PAM_Analysis/duane_scratch/perch_hoplite/

# PyTorch Perch V2 port
~/perch-pytorch/
    perch_hoplite_torch_adapter.py  ← MUST be up to date (normalization fix)
    perch_embedder_torch.py
    perch_frontend_torch.py
    perch_weights/
    const__pad1_output_0.npy
```

---

## Critical: Low-Amplitude Normalization (July 9 2026)

**All embeddings must use per-window peak normalization to 0.25.**

MARS hydrophone audio has typical peak amplitude 0.0015–0.003. Without
normalization, the PyTorch port diverges from TF Perch V2 at cosine 0.43–0.94
on real MARS data (verified against live TF model on Colab A100).

The fix is in `perch_hoplite_torch_adapter.py` — `peak_normalize_windows()`
is called inside `embed()` before the model. This is transparent to callers.

**All DBs with `_norm` suffix were embedded with this fix (July 9 2026+).**
Old DBs without `_norm` suffix are pre-fix and should not be used for new work.

See: `docs/FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md`

---

## Venv

```bash
# Always activate before running anything
source ~/perch-hoplite/venv/bin/activate

# Key packages: torch 2.12.1+cu130, gradio==6.15.1 (pinned), librosa,
# soundfile, perch-hoplite (core, no TF extras)
```

**Gradio version must be 6.15.1** — other versions have audio playback issues.
**Browser for Gradio: Chrome (incognito)** — Safari has audio playback issues
with data: URIs.

---

## Classifier Versioning (new era — July 9 2026)

Old classifiers (v1_clean through v8_clean) were trained on un-normalized
embeddings and are retired. New versioning starts at v0:

| Version | ROC-AUC | top1_acc | cmap | Training DB | Notes |
|---|---|---|---|---|---|
| v0 | 0.9773 | 0.9405 | 0.8810 | April 2018 norm | Baseline normalized |
| v1 | 0.9533 | 0.9559 | 0.7999 | April + October 2020 norm | Cross-season |
| v2 | 0.9654 | 0.9438 | 0.8930 | April 2018 norm (expanded) | More dolphin/other labels |
| v3 | 0.9467 | 0.9481 | 0.7370 | April 2018 + Oct 2020 + April 2026 norm | 3-season, 17 Apr2026 humpback |
| v4 | 0.9590 | 0.9650 | 0.8297 | April 2018 + Oct 2020 + April 2026 norm | Best cross-season, 25 Apr2026 humpback |
| v5 | 0.9303 | 0.9301 | 0.5945 | 3-season context DB (30s Gaussian avg) | Context embedding experiment — WORSE than v4 |

**Best for October 2020 analysis:** v1
**Best for April/May 2018 analysis:** v2
**Best for April 2026 / cross-season:** v4

---

## Annotation State (July 9 2026)

| DB | Annotations |
|---|---|
| MARS_20180401_20180430_32kHz_norm | 219 orca + 195 dolphin + 41 humpback + 24 ship + 51 other + 54 neg |
| MARS_20201001_20201031_32kHz_norm | 209 humpback + 5 dolphin |
| MARS_20260401_20260430_32kHz_norm | 25 humpback (all high-scoring orca FPs — hard negatives) |

---

## Standard Commands

```bash
# Embed a full month
nohup python3 phase1_embed_torch.py \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --device cuda --compile \
    > /mnt/PAM_Analysis/perch-hoplite/logs/embed_april2018.log 2>&1 &

# Train classifier
time python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier-out /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --num-steps 256 --train-ratio 0.8

# Run inference
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20180401_20180430_v2_detections.csv \
    --logit-threshold 0.0

# Review annotations (Chrome incognito, viridis mel is preferred display)
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v2.pt \
    --target-label orca_call \
    --num-results 25 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7861 \
    > /mnt/PAM_Analysis/perch-hoplite/logs/review_7861.log 2>&1 &

# Kill Gradio (by port to avoid killing inference)
pkill -f "port 7861"

# Monthly plot
python3 tools/plot_monthly.py \
    --input /mnt/PAM_Analysis/perch-hoplite/results/MARS_20180401_20180430_v2_detections.csv \
    --output-dir /mnt/PAM_Analysis/perch-hoplite/results \
    --title "April 2018 MARS Hydrophone v2"

# t-SNE
python3 tools/plot_tsne.py \
    --db-dir \
        /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
        /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
    --output /mnt/PAM_Analysis/perch-hoplite/results/tsne_combined.png \
    --title "Perch V2 Embeddings — April 2018 + October 2020"
```

---

## Spectrogram Preferences

**Preferred display:** `--spectrogram-type mel --colormap viridis`

| Mode | Flag | Best for |
|---|---|---|
| Linear STFT | `--spectrogram-type linear` | Orca, dolphin (default) |
| Mel | `--spectrogram-type mel --colormap viridis` | Humpback, general review |
| Perch frontend | `--spectrogram-type perch` | Model inspection |
| Gray mel | `--spectrogram-type mel --colormap gray` | Publication figures |

---

## Known Issues / Pending Work

1. **Mel spectrogram banding** — minor horizontal artifacts in mel/pcen/perch modes (partially fixed)
2. **plot_monthly dedup** — should dedup on `(idx, label)` not just `idx`
3. **Inference /tmp write speed** — 266K row CSV takes ~33 min to write
4. **May 2018 expert review** — May 12 (190 orca) and May 14 (45) need confirmation
5. **Repo reorganization** — `reorganize_repo.sh` ready to run
6. **April 2026 orca FPs** — Apr 14, 16 still show elevated humpback FPs; need 2026 orca examples to resolve
7. **May 2026 embedding** — not yet done

---

## Scientific Results Summary

| Month | Key finding |
|---|---|
| April 13 2018 | 289 orca detections — confirmed Bigg's orca hunting event ✅ |
| May 12 2018 | 190 orca detections — probable event, expert review pending |
| May 14 2018 | 45 orca detections — secondary event, expert review pending |
| October 2020 | Oct 5-12 cluster confirmed — zero orca vocalizations (Bigg's orca silent during hunts) ✅ |
| April 2026 | Apr 21 dominant (101 v4 detections) — all reviewed clips are humpback FP; consistent with Bigg's orca acoustic silence. Apr 17-24 CA51A/CA50B event window shows 129 detections but no confirmed orca vocalizations. |

**Context embedding experiments (July 15 2026):**
- 30s Gaussian-weighted t-SNE: orca completely separated from humpback (zero overlap) — colleague-suggested method ✅
- Context DB (v5 classifier): ROC-AUC dropped 0.959→0.930, cmap 0.830→0.595 — context averaging hurts training
- Context post-processing filter (orca/humpback ratio): suppressed April 13 2018 orca — filter fails because Bigg's orca calls are brief discrete bursts, not sustained bouts
- **Conclusion:** Raw embeddings + v4 classifier remain best. Temporal sequence modeling is the right path for disambiguation.

---

## Repo Structure

```
perch-hoplite/
├── src/                    — modular Python library
│   ├── spectrogram.py      — 4-mode spectrogram generation
│   ├── audio.py            — audio encoding + 30s context
│   ├── torch_model.py      — TF mock + model loading
│   ├── train.py            — PyTorch classifier training
│   ├── infer.py            — inference + NFS-safe CSV writing
│   ├── review.py           — Gradio labeling GUI
│   └── paths.py            — canonical path definitions
├── tools/                  — standalone scripts
│   ├── merge_annotations.py
│   ├── merge_dbs.py
│   ├── plot_monthly.py
│   ├── plot_tsne.py
│   ├── extract_example_clips.py
│   └── review_example_clips.py
├── docs/                   — documentation and analysis
│   ├── pytorch_port_summary.md   — PyTorch Conference 2026 poster
│   ├── october_2020_analysis.md
│   ├── FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md
│   └── PROGRESS_2026-07-09.md
├── figures/                — plots and screenshots
├── phase2_classify.py      — main CLI
├── phase1_embed_torch.py   — embedding pipeline
└── README.md
```
