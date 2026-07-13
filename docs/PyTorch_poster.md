# PyTorch Conference 2026 — Poster Abstract

**Title:** Perch 2.0 and Perch-Hoplite in Pure PyTorch: An End-to-End TensorFlow-Free Bioacoustics Pipeline

**Submitted:** July 12 2026
**Update deadline:** July 26 2026

---

Perch 2.0 (Google) classifies ~15,000 species and produces general-purpose audio embeddings used across conservation. It ships as TensorFlow — a problem on the NVIDIA GB10 (Grace Blackwell, sm_121, CUDA 13), where accelerated TF is unavailable but PyTorch runs natively. Rather than wrap the TF model, I reimplemented Perch 2.0's embedding model: a log mel-spectrogram frontend and EfficientNet-B3 embedder as an idiomatic torch.nn.Module, weights from TF model graph. It reproduces TF embeddings at cosine ~1.0 on the GB10. This meant reverse-engineering details — the frontend uses log scaling (not PCEN), and the stem uses VALID padding. With torch.compile it runs ~2.5× faster than ONNX (~635 clips/s), fully on-device. I replaced perch-hoplite's TensorFlow model loading with the native PyTorch port, so the embed→search→classify loop runs 100% TensorFlow-free. Monterey Accelerated Research System hydrophone audio is quiet, so per-window amplitude normalization was essential for parity. On 1.56M MARS embeddings (2018, 2020, 2026), a linear classifier (ROC-AUC 0.959) distinguishes Bigg's orca, Pacific white-sided dolphin, humpback, and ship noise; orca validated for 2018. Code to follow.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| July 12 2026 | Initial submission | — |
| July 12 2026 | "unnecessary TensorFlow calls" → "replaced TF model loading with PyTorch port" | Accuracy — TF calls were necessary in original design; we replaced them |

---

*Character count: 1,199 / 1,200*
*Update deadline: July 26 2026 — update if May 2018 orca confirmed by John Ryan*
