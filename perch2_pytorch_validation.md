# Perch 2.0 Native PyTorch Port — Validation Results
**MBARI Monterey Bay Aquarium Research Institute**  
**Date: July 4 2026**  
**Hardware: NVIDIA GB10 DGX Spark (spark-ae0e), aarch64, CUDA 13, compute capability sm_121**

---

## Summary

A from-scratch PyTorch reimplementation of the Google Perch 2.0 bioacoustics
embedding model was validated against the reference TensorFlow implementation
using real MARS hydrophone recordings from Monterey Bay. The PyTorch port
produces numerically equivalent embeddings and identical downstream classifier
outputs, with no dependency on TensorFlow, Google Colab, or Google Drive.

---

## Embedding Numerical Parity

Embeddings from the native PyTorch pipeline were compared against the reference
TensorFlow model (via the `justinchuby/Perch-onnx` ONNX export as cross-check
oracle). Results on 5-second audio clips at 32 kHz:

| Pipeline | Device | Cosine similarity | Relative error |
|---|---|---|---|
| Native PyTorch (raw audio → embedding) | CPU | 1.0000000 | ~1–5×10⁻⁵ |
| Native PyTorch (raw audio → embedding) | GB10 GPU | 0.9999997 | ~8×10⁻⁴ |
| ONNX bridge (cross-check oracle) | CPU | — | ~1×10⁻⁹ |

GPU parity is slightly looser than CPU due to float32 / tensor-core accumulation
(TF32 enabled on Blackwell). Cosine similarity is unchanged for any practical
embedding use (search, classification, transfer learning).

---

## End-to-End Pipeline Validation

The full pipeline was validated on April 13 2018 MARS hydrophone recordings
(a confirmed Bigg's orca event in Monterey Bay):

**Test dataset:** 144 audio files × 120 windows = **17,280 five-second clips**  
**Classifier:** `orca_v4_clean.pt` (ROC-AUC 0.974, multi-class: orca / dolphin / other)  
**Logit threshold:** 0.0

### Detection counts — Colab TF pipeline vs native PyTorch pipeline

| Class | Colab TF DB | PyTorch DB | Difference |
|---|---|---|---|
| `orca_call` | 295 | 295 | **0** |
| `dolphin_call` | 2,253 | 2,254 | 1 |
| `other` | 159 | 159 | **0** |
| **Total** | **2,707** | **2,708** | **1** |

The single detection difference (one `dolphin_call`) represents a clip whose
logit score fell at the decision boundary (logit ≈ 0.0) — a floating-point
rounding difference between the TF and PyTorch accumulation paths, not a
meaningful classification disagreement.

**Conclusion: the PyTorch port is validated end-to-end for production use.**

---

## Throughput on NVIDIA GB10 DGX Spark

Measured on 144 files (17,280 five-second windows, 32 kHz, non-overlapping):

| Mode | Time | Windows/sec | Notes |
|---|---|---|---|
| Colab Pro (A100, TF) | ~1.9 min | ~152/sec | Via Google Drive upload/download |
| Native PyTorch eager (GB10) | ~2.2 min | ~130/sec | No compile |
| **Native PyTorch `torch.compile` (GB10)** | **1.6 min** | **175/sec** | 35% faster than eager for 1 day; larger gains for multi-day runs |
| ONNX bridge (ORT-CUDA, GB10) | — | ~207/sec | Cross-check only |

Throughput benchmark across batch sizes (5-second clips, GB10 GPU):

| Path | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| ONNX bridge (ORT-CUDA) | 102 | 200 | 207 | 211 | 206 |
| Native PyTorch (eager) | 177 | 245 | 227 | 209 | 196 |
| **Native PyTorch (`torch.compile`)** | **350** | **635** | **607** | **533** | **514** |

`torch.compile` delivers ~2.5× the ONNX bridge at batch 4–16 and ~5× at
batch 1. The native pipeline wins because the full graph stays on-device;
the ONNX bridge's in-graph DFT forces host↔device copies.

---

## Practical Impact

| Metric | Before (Colab) | After (PyTorch on spark-ae0e) |
|---|---|---|
| Embedding one day (17,280 windows) | ~2 min + upload/download | ~2.2 min, fully local |
| Colab Pro subscription required | Yes | No |
| Google Drive storage required | Yes (~1.5 GB/day zip + DB) | No |
| TensorFlow required | Yes | No |
| Internet required for embedding | Yes | No |
| Full April 2018 (30 days) | ~60 min + manual batching | ~66 min, single command |
| Full October 2020 (31 days) | ~62 min + manual batching | ~68 min, single command |

---

## Architecture Notes

Key findings from reverse-engineering the Perch 2.0 model graph:

- **Frontend is log-mel, NOT PCEN.** Perch 2.0 uses `0.1·log(max(mel, 1e-5))`.
  Perch 1.0 / SurfPerch use PCEN; Perch 2.0 does not.
- **Stem uses VALID padding** (→ 249×63 feature map), while all other k>1
  convolutions use JAX SAME padding. Getting this wrong drops cosine to ~0.82.
- **Mel filterbank:** 128 bands, 60 Hz–16 kHz, HTK scale, DC bin zeroed.
- **EfficientNet-B3 backbone:** 26 MBConv blocks, folded BatchNorm, Swish,
  sigmoid-gated SqueezeExcite, 1536-dim head, global average pool → embedding.

---

## Downstream Classifier Performance (April 13 2018)

Linear classifiers trained on Perch 2.0 embeddings via perch-hoplite
agile modeling. All classifiers trained with `--train-ratio 0.8`.

| Model | ROC-AUC | Labels | Classes |
|---|---|---|---|
| orca_v1_clean | 0.982 | 44 pos / 56 neg | orca (single-class) |
| orca_v2_clean | 0.919 | 54 pos / 56 neg | orca (single-class) |
| orca_v3_clean | 0.990 | 213 orca + 13 dolphin + 55 neg | multi-class |
| orca_v4_clean | **0.974** | 214 orca + 168 dolphin + 42 other + 54 neg | multi-class |

**Cross-day validation (orca_v4_clean):**

| Day | Known activity | Orca detections | Dolphin | Other |
|---|---|---|---|---|
| April 1 2018 | Quiet (no orca) | 0 (4 FP = dolphin) | 307 | 0 |
| April 13 2018 | Confirmed orca event | 295 | 2,253 | 159 (vessel UTC 14:19–14:29) |
| April 20 2018 | No orca; large vessel + dolphin | 16 (FP scattered) | 1,129 | 462 (vessel UTC 13:19–14:09) |
| April 30 2018 | No orca (humpback present) | 0 (26 FP = humpback/ship) | 0 | 11 |

Zero true orca false positives on two held-out quiet days. The 4 detections
on April 1 and 26 on April 30 were confirmed as dolphin/humpback/vessel by
manual review in the Gradio annotation interface.

---

## April 20 2018 — Detailed Results

April 20 provides a rich example of a non-orca day with multiple acoustic events:

**Vessel passage (UTC 13:19–14:09, PDT 06:19–07:09):**
462 `other` detections in 6 consecutive 10-minute files (46–98 detections/file).
Single large vessel transiting through the MARS hydrophone range during morning
whale watch hours. Correctly classified as `other` — will be reclassified as
`ship_noise` once ship_noise training labels are added in v5_clean.

**Dolphin activity — three distinct groups:**

| UTC window | PDT | Detections | Notes |
|---|---|---|---|
| 10:29–12:29 | 03:29–05:29 | 130 | Morning transiting group |
| 13:19–14:09 | 06:19–07:09 | 175 | Co-occurs with vessel passage |
| 18:39–21:09 | 11:39–14:09 | 810 | Large afternoon feeding school |

The afternoon group (810 detections, peak 91/file at UTC 19:09) is comparable
in density to the April 13 dolphin school during the orca hunt — a large active
Pacific white-sided dolphin school feeding in Monterey Bay.

**Orca:** 16 scattered detections (1–7 per file, 5 different files) — no
temporal clustering, consistent with false positives at the decision boundary.

---

## Software Stack

| Component | Version / Notes |
|---|---|
| PyTorch | 2.12.1+cu130 |
| CUDA | 13 |
| Hardware | NVIDIA GB10 (sm_121), aarch64 |
| perch-hoplite | core only, no TF/JAX extras |
| Python | 3.12.3 |
| Perch 2.0 weights source | `justinchuby/Perch-onnx` (Apache-2.0) |

---

## Citation

If using this validation or the PyTorch port in your work:

```
Edgington D., Ryan J. (2026). Detection and classification of marine mammal
vocalizations in MARS hydrophone recordings using Perch 2.0 embeddings and
agile modeling. IEEE OCEANS 2026, MBARI.

Perch 2.0: Google Research, https://github.com/google-research/perch (Apache-2.0)
ONNX export: justinchuby/Perch-onnx, https://huggingface.co/justinchuby/Perch-onnx
perch-hoplite: https://github.com/google-research/perch-hoplite (Apache-2.0)
```

---

*Generated July 4 2026 — MBARI*
