# Perch Hoplite Marine Bioacoustics Pipeline
## MBARI — NVIDIA DGX SPARC (spark-ae0e, spark-0626)

---
## Overview

Built on Google perch-hoplite
https://github.com/google-research/perch-hoplite

Hoplite is a system for storing large volumes of embeddings from machine
perception models. We focus on combining vector search with active learning
workflows, aka [agile modeling](https://arxiv.org/abs/2505.03071).

In brief, agile modeling is a process for rapidly developing classifiers using
embeddings from a pre-trained 'foundation' model. For bioacoustics work, we
find that new classifiers can often be developed for new signals in under
an hour.

**How does it work?**

We first use a bioacoustics model to convert the unlabeled audio data into
embeddings - these are like semantic 'fingerprints' of 5-second audio clips.
Then, you can *search* the embeddings of your data by providing an example of
what you're looking for. You then give feedback on the results - which examples
are and are not what you're looking for. From this feedback, we can quickly
train a classifier. You can then improve on the classifier with
*active learning*: Examine the classifier outputs, provide more feedback, and
re-train the classifier.

A key feature of this workflow is that we pre-compute the embeddings. This
may take a while if you have a large amount of data, but the subsequent search
and classifier training is very efficient.
## System Overview

| Machine | Role | IP / Access |
|---|---|---|
| **ICEFISH** (Mac) | Developer workstation | Local — scp gateway to spark |
| **spark-ae0e** | Active learning, inference, Gradio server | 134.89.11.107 |
| **spark-0626** | Spare / parallel runs | 134.89.11.174 |
| **Google Colab** (A100) | Phase 1 embedding only | colab.research.google.com |

### Why the split?

The spark servers have NVIDIA GB10 (Blackwell, compute capability 12.1) GPUs.
TensorFlow 2.17 does not support XLA on compute capability 12.x, so the
Perch V2 model cannot run inference on spark for embedding. The Colab A100
(compute capability 8.0) works fine. Once the embedding database is built on
Colab and transferred to NFS, all subsequent steps (training, review, inference)
run on spark without needing the GPU for the Perch model.

| | spark-ae0e | spark-0626 |
|---|---|---|
| IP address | 134.89.11.107 | 134.89.11.174 |
| PAM_Analysis | /mnt/PAM_Analysis (NFS4, rw) | /mnt/PAM_Analysis (NFS4, rw) |
| PAM_Archive | /mnt/PAM_Archive (NFS4, rw) | /mnt/PAM_Archive (NFS4, rw) |
| NFS server | thalassa.shore.mbari.org | thalassa.shore.mbari.org |
| Python venv | ~/gmwd/new3-12_whale_detection/gmwd/venv | same |
| perch-hoplite | 1.0.1 | 1.0.1 |

Both spark machines share the same NFS volumes. Databases, models, results,
and labels written on one are immediately visible on the other.

---

## Directory Structure

```
Google Colab (temporary, per-session)
    /tmp/mbari_audio/<dataset>/     ← audio uploaded from Google Drive
    /content/drive/My Drive/
        MBARI_perch/
            audio/                  ← zipped WAV files from ICEFISH
            db/                     ← completed Hoplite DBs → scp to spark

ICEFISH (Mac, ~/Desktop/colab_staging/)
    ← staging area for zipping audio before Google Drive upload
    ← also used as scp relay: Colab DB → ICEFISH → spark NFS

spark-ae0e / spark-0626 (NFS shared)
/mnt/PAM_Analysis/duane_scratch/perch_hoplite/
    db/                             ← Hoplite embedding databases
        MARS_20180413_20180413_32kHz/   ← April 13 2018 (144 files, 17,280 embeddings)
    models/                         ← trained classifiers (.pt + .metrics.json)
        orca_v1.pt                  ← first classifier, ROC-AUC 0.754
    results/                        ← inference CSVs, score histogram PNGs
    labels/                         ← annotation CSVs (bootstrap + active learning)
    queries/
        cetaceans/                  ← orca, dolphin, whale reference clips
        anthropogenic/              ← boat, ROV, sonar reference clips
    logs/                           ← persistent logs

/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/
    resampled_32kHz/                ← source audio resampled to 32 kHz
        2018/04/                    ← MARS recordings April 2018

/mnt/PAM_Archive/                   ← RAW AUDIO (read-only, 273 TB)
    2015/ 2016/ ... 2026/
```

---

## Installation Status (spark-ae0e as of June 2026)

Already installed in venv — do not reinstall:

```
perch-hoplite        1.0.1
gradio               6.15.1
soundfile            0.13.1
librosa              0.11.0
scipy                (for spectrogram rendering in GUI)
matplotlib           (for spectrogram rendering in GUI)
```

Known harmless warnings at startup (not errors):
- `Unable to register cuFFT/cuDNN/cuBLAS factory` — two TF builds registering CUDA plugins; GPU still works.
- `MessageFactory has no attribute GetPrototype` — protobuf version mismatch, cosmetic only.
- `NUMA node read from SysFS had negative value` — BIOS topology quirk, TF defaults correctly to node 0.
- `NodeDef mentions attribute use_shardy_partitioner` — model saved with newer JAX/XLA; attribute is ignored.
- `Failed to load class list ... duplicate entries` — cosmetic, does not affect embeddings.

---

## Programs

| File | Runs on | Purpose |
|---|---|---|
| `MBARI_perch_phase1_embed.py` | **Google Colab** | Build Hoplite embedding DB from audio |
| `prepare_audio_for_colab.sh` | **ICEFISH** (Mac) | Zip resampled audio and stage for Google Drive |
| `phase2_classify.py` | **spark-ae0e** | Active learning: search, label, train, review, infer |
| `convert_scores_to_labels.py` | **spark-ae0e** | Convert Google model score CSVs to Hoplite label CSVs |

---

## Complete Workflow

### Phase 1 — Build the Embedding Database (Colab)

The embedding step requires a GPU compatible with TF 2.17 + XLA. Use Google
Colab (A100 runtime) because the spark GB10 GPUs are not compatible.

#### Step 1a — Prepare audio on ICEFISH

```bash
# On ICEFISH Mac — zip a date range of resampled 32 kHz WAV files
# and copy to Google Drive for Colab to access
chmod +x prepare_audio_for_colab.sh
./prepare_audio_for_colab.sh
# Follow prompts — output goes to ~/Google Drive/My Drive/MBARI_perch/audio/
```

Edit `prepare_audio_for_colab.sh` to set `DATE_START`, `DATE_END`, and `MAX_FILES`
before running. The script creates a zip archive and a Colab config snippet.

#### Step 1b — Run embedding on Colab

1. Open `MBARI_perch_phase1_embed.py` in Google Colab (A100 GPU runtime)
2. Mount Google Drive when prompted
3. Set `GDRIVE_AUDIO_ZIP` to match the zip created in Step 1a
4. Run all cells — embedding takes ~15 min per 75-second shard on A100
5. When complete, the DB is saved to Google Drive

#### Step 1c — Transfer DB from Colab to spark

```bash
# On ICEFISH — download DB from Google Drive, then scp to spark NFS
scp -r ~/Downloads/<db_folder> duane@134.89.11.107:/mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/
```

The DB path stored in hoplite_metadata will reference the Colab path
`/tmp/mbari_audio/...`. This is overridden at runtime with `--audio-dir`
(see Step 4 below).

---

### Phase 2 — Active Learning Loop (spark-ae0e)

All Phase 2 steps run on spark-ae0e. The Gradio labeling GUI is accessed
from any browser on the MBARI network — including ICEFISH.

#### Step 2 — Import bootstrap labels (optional)

If you have existing annotations from Raven Pro, PAMGuard, or the Google
species model score CSVs, import them first:

```bash
# From Google model score CSVs
python3 convert_scores_to_labels.py \
    --scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/ \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/bootstrap_orca.csv \
    --target-class orca_call \
    --threshold 0.7

# Import into DB
python3 phase2_classify.py label \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --labels-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/bootstrap_orca.csv \
    --annotator-id duane
```

#### Step 3 — Train initial classifier

```bash
python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier-out /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --num-steps 256
```

Metrics are printed after training and saved to `orca_v1.metrics.json`.
Target ROC-AUC > 0.90 before running full inference.

#### Step 4 — Review and label (active learning)

Launch the Gradio labeling GUI on spark, open it on ICEFISH browser:

```bash
# On spark-ae0e
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --target-label orca_call \
    --num-results 50 \
    --sample-size 5000 \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --serve --port 7860 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_7860.log 2>&1 &
```

**`--audio-dir` is required** because the DB was built on Colab where audio
was at `/tmp/mbari_audio/...`. This flag overrides that stored path with the
actual location on spark NFS.

Then on ICEFISH, open: **http://134.89.11.107:7860**

The GUI shows the top-50 highest-scoring candidates with:
- Spectrogram (0–16 kHz, log-power, inferno colormap)
- Waveform
- Custom audio player with high-contrast progress bar
- Positive / Negative / Unlabeled radio buttons

Audio is automatically normalized to −3 dBFS for comfortable listening
(the resampled MARS files have very low amplitude at native levels).

**Labeling strategy:**
- `positive` = orca call clearly present
- `negative` = background noise, dolphin, ship, silence — anything that is NOT orca
- `unlabeled` = genuinely ambiguous — skip, do not save
- Focus on adding negatives early: the bootstrap has ~464 positives and only ~3 negatives

Click **💾 Save Labels to DB** when done. The status box confirms save counts.

To monitor the server:
```bash
tail -f /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_7860.log
```

To stop the server:
```bash
pkill -f "phase2_classify.py review"
```

#### Step 5 — Retrain with new labels

```bash
nohup python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier-out /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v2.pt \
    --num-steps 256 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/train_orca_v2.log 2>&1 &
tail -f /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/train_orca_v2.log
```

Repeat Steps 4–5 until ROC-AUC is satisfactory. Increment the version number
(`orca_v2.pt`, `orca_v3.pt`, ...) to preserve each iteration.

For margin sampling (finding hard negatives near the decision boundary):
```bash
nohup python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v2.pt \
    --target-label orca_call \
    --num-results 50 \
    --sample-size 17280 \
    --margin-target-score 0.0 \
    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \
    --serve --port 7860 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_7860.log 2>&1 &
```

#### Step 6 — Check label counts at any time

```bash
python3 phase2_classify.py stats \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz
```

#### Step 7 — Full inference → detections CSV

```bash
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v2.pt \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_detections.csv \
    --logit-threshold 0.0 \
    --plot-distribution /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_logit_dist.png
```

---

## Current Status (as of June 4 2026)

| Item | Status |
|---|---|
| DB: MARS April 13 2018 | ✅ 144 files, 17,280 embeddings |
| Bootstrap labels | ✅ 464 positive orca_call, 3 negative (from Google model scores) |
| orca_v1.pt | ✅ ROC-AUC 0.754 — too few negatives |
| Active learning round 1 | ✅ +58 positives, +39 negatives via Gradio GUI |
| orca_v2.pt | ✅ ROC-AUC 0.690 — class imbalance 522 pos / 42 neg |
| Active learning round 2 | 🔄 in progress — margin sampling near score=0 |
| orca_v3.pt | 🔲 pending round 2 labels |
| Full April 2018 DB | 🔲 planned |
| Multi-class labels | 🔲 planned — dolphin_whistle, dolphin_click, boat_motor, background |

**Note on ROC-AUC trajectory:** v2 dropped from 0.754 to 0.690 due to 12:1
class imbalance. Round 2 uses `--margin-target-score 0.0` to surface clips
near the decision boundary. Target: ROC-AUC > 0.90 by v4–v5.

---

## Multi-class Classification

Run review/label once per sound class with a distinct `--target-label`.
All labels accumulate in the same DB. Train once — the classifier is
automatically multi-class.

Suggested label names:
```
orca_call           # Bigg's / resident orca vocalizations
dolphin_whistle     # common/bottlenose dolphin tonal whistles
dolphin_click       # odontocete echolocation clicks
humpback_song       # humpback whale song units
blue_whale_call     # blue whale 20 Hz calls
fin_whale_call      # fin whale 20 Hz doublets
sperm_whale_click   # sperm whale codas / clicks
boat_motor          # vessel engine noise
rov_thruster        # ROV/AUV thruster noise
background          # featureless background / flow noise
```

---

## Gradio Labeling GUI — Details

The GUI runs on spark and is accessed from any browser on the MBARI network.
It does not require any software installation on the client machine.

Per-clip display:
- **Header**: filename, time offset, classifier score
- **Spectrogram**: 0–16 kHz, 60 dB dynamic range, inferno colormap
- **Waveform**: full 5-second window
- **Player**: HTML5 audio player with normalization to −3 dBFS
- **Radio**: positive / negative / unlabeled

Labels are written to the Hoplite SQLite DB on click of **💾 Save Labels to DB**.
Only one analyst should label a given DB at a time to avoid SQLite write conflicts.

For concurrent multi-analyst campaigns, use Label Studio:
```bash
docker run -d -p 8080:8080 \
    -v /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labelstudio:/label-studio/data \
    heartexlabs/label-studio:latest
```
Access at http://134.89.11.107:8080

---

## Model Selection Guide

| Model | Best for | Notes |
|---|---|---|
| `perch_v2` | Broadest coverage, recommended start | Requires A100/V100 for embedding |
| `multispecies_whale` | Baleen whale calls, biotwangs | Pre-trained on cetaceans |
| `humpback` | Humpback song specifically | Narrow but accurate |
| `surfperch` | Coral reef soundscapes | Not suitable for deep-water MBARI sites |
| `perch_8` | Bird sounds only | Not useful for marine work |

Each model requires its own separate DB — embeddings from different models
cannot be mixed in a single database.

---

## Disk Usage

```bash
df -h /mnt/PAM_Analysis /mnt/PAM_Archive
du -sh /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/* 2>/dev/null | sort -h
```

PAM_Analysis: ~94% full as of June 2026 (3.4 TB free). Monitor before large embedding runs.
Rough estimate: ~9 MB per hour of audio at Perch V2 defaults (5-second windows, 1536-dim float16).
