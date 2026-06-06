# Perch Hoplite Marine Bioacoustics Pipeline
## MBARI — NVIDIA DGX SPARC (spark-ae0e, spark-0626)

---
## Overview

Built on Google perch-hoplite
https://github.com/google-research/perch-hoplite

https://github.com/google-research/perch/blob/main/chirp/projects/whale_demo/agile_modeling_noaa_demo.ipynb

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

https://research.google/blog/how-ai-trained-on-birds-is-surfacing-underwater-mysteries/ 

https://arxiv.org/abs/2512.03219 


## System Overview

| Machine | Role | IP / Access |
|---|---|---|
| **ICEFISH** (Mac) | Developer workstation | Local — scp gateway to spark |
| **spark-ae0e** | Active learning, inference, Gradio server | 134.89.11.107 |
| **spark-0626** | Spare / parallel runs | 134.89.11.174 |
| **Google Colab** (A100) | Phase 1 embedding only | colab.research.google.com |

### Why the split?

The spark servers have NVIDIA GB10 (Blackwell, compute capability 12.1) GPUs.
TensorFlow 2.17 is the highest version ported to GB10 ARM/Nvidia chip, which does not support XLA on compute capability 12.x, so the
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

The embedding step requires a GPU compatible with TF 2.20rc or higher + XLA. Use Google
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

⚠️ **Always kill the Gradio review server before training.** The review server
holds several GB of GPU memory. Training will fail with `cudaSetDevice() out of memory`
if both run simultaneously.

```bash
pkill -f "phase2_classify.py review"
```

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

## Current Status (as of June 5 2026)

| Item | Status |
|---|---|
| DB: MARS April 13 2018 | ✅ 144 files, 17,280 embeddings — **RESET, 0 annotations** |
| DB: MARS April 1 2018 | ✅ 144 files, 17,280 embeddings (separate, 0 annotations) |
| DB: MARS_combined | ✅ 34,560 embeddings (Apr 1 + Apr 13 merged, experimental) |
| orca_v1_clean.pt | ✅ **ROC-AUC 0.9821** — 44 pos / 56 neg, clean labels, train_ratio=0.8 |
| orca_v2_clean.pt | ✅ **ROC-AUC 0.9191** — 54 pos / 56 neg, 110 clean labels, train_ratio=0.8 |
| Multi-class labels | 🔲 planned — dolphin_whistle, dolphin_click, boat_motor |
| Full April 2018 DB | 🔲 planned via Colab Pro batches |

### Orca event timing on April 13 2018 (UTC)

From annotation analysis, orca calls were detected during:
- **00:19–00:29 UTC** — brief overnight activity
- **06:49–09:49 UTC** — morning feeding event
- **12:59 UTC** — midday
- **14:19–18:29 UTC** — main afternoon feeding event (densest)
- **20:49 UTC** — late activity

In **Pacific Daylight Time (UTC−7):** main event was 07:19–11:29 local —
consistent with morning gray whale calf migration through the bay.

**Best windows for NEGATIVE labels (quiet hours on April 13 UTC):**
- 03:00–06:00 UTC (20:00–23:00 PDT previous day)
- 10:00–12:00 UTC (03:00–05:00 PDT)
- 21:00–23:59 UTC (14:00–16:59 PDT)

### Key Lessons from Development Phase

1. **Dolphin false positives** — Perch V2 embeddings place dolphin whistles
   near orca calls. High classifier scores do NOT always mean orca. Always
   listen and check the spectrogram. Dolphin whistles show high-frequency
   (3–14 kHz) upsweeping/downsweeping tonal arcs vs orca's 1–6 kHz banded
   harmonic bursts.

2. **Cross-day DB merging degrades performance** — merging embeddings from
   different days hurt ROC-AUC. Stay within one deployment day for training.
   The 17,280 windows on April 13 contain natural negatives in the quiet hours.

3. **Kill Gradio before training** — review server holds GPU memory; training
   will fail with OOM if both run simultaneously.

4. **Eval set size matters** — with train_ratio=0.9 and ~550 labels, only
   ~55 eval examples → high ROC-AUC variance. Use --train-ratio 0.8 for
   reliable metrics.

5. **Label deliberately** — target positives from known active UTC hours,
   negatives from known quiet UTC hours. Random sampling wastes labeling effort.

### Clean labeling strategy (current)

- **Positive labels:** target files from UTC 06:49–20:49 (orca active hours)
- **Negative labels:** target files from UTC 21:00–06:00 (quiet hours)
- **Goal:** 150 positives + 100 negatives before first clean training run
- **Train ratio:** 0.8 for reliable eval signal
- **Target ROC-AUC:** > 0.90 before full inference

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

### Species acoustic signatures for MARS hydrophone

Understanding where each species' energy falls in the spectrogram is
critical for correct labeling, especially since the Gradio spectrogram
displays 0–16 kHz. Note that blue whale and fin whale calls are
**infrasonic** — their fundamental energy is below the 0 Hz axis of
the display. They appear as very low-frequency energy near the bottom
of the spectrogram, and only their harmonics may be visible.

#### Orca (Killer Whale) — `orca_call`
- **Frequency range:** 0.5–25 kHz, most energy 1–6 kHz
- **Call types:** discrete pulsed calls, whistles, echolocation clicks
- **Spectrogram signature:** structured banded harmonic bursts, 1–6 kHz,
  clearly distinct from background. Duration 0.5–2 seconds per call.
- **Monterey Bay seasonality:** peak mid-March through mid-May (Bigg's orca
  hunting gray whale calves). Other ecotypes present year-round.

#### Common/Bottlenose Dolphin — `dolphin_whistle`, `dolphin_click`
- **Whistles:** 3–14 kHz, upsweeping/downsweeping tonal arcs
- **Clicks:** broadband, >10 kHz
- **Important:** Perch V2 embeddings place dolphin and orca calls near
  each other — dolphin whistles are the most common **false positive**
  for the orca classifier. Score alone cannot distinguish them; always
  check the spectrogram.

#### Blue Whale — `blue_whale_call`
- **Frequency range:** 10–40 Hz (infrasonic — below display range)
- **Peak energy:** "B call" harmonics centered at 15–16 Hz and 30–43 Hz
- **Intensity:** up to 189 dB re 1 μPa — among the loudest animal sounds
- **Propagation:** hundreds to thousands of miles across ocean basins
- **Spectrogram signature:** very low-frequency energy at bottom of display,
  may appear as bright band near 0 Hz. Harmonics occasionally visible up
  to ~80 Hz.
- **References:** [Thompson et al. 2017](https://www.nature.com/articles/s41598-017-09423-7),
  [DOSITS](https://dosits.org/galleries/audio-gallery/marine-mammals/baleen-whales/blue-whale/)

#### Fin Whale — `fin_whale_call`
- **Frequency range:** 13–40 Hz (infrasonic), peak at ~20 Hz
- **Call structure:** "20-Hz pulse" — short downsweep ~1 second, 40→13 Hz
- **Secondary component:** higher-frequency component (HFC) at 85–140 Hz
  occasionally visible in spectrogram
- **Peak energy:** tightly centered at 20 Hz
- **Spectrogram signature:** very similar to blue whale — near-bottom energy.
  The HFC at 85–140 Hz may be the only visible feature in the 0–16 kHz display.
- **References:** [DOSITS](https://dosits.org/galleries/audio-gallery/marine-mammals/baleen-whales/fin-whale/),
  [Širović et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0967064519300736)

#### Humpback Whale — `humpback_song`
- **Frequency range:** 20 Hz–10 kHz, most energy 100 Hz–4 kHz
- **Spectrogram signature:** complex song units with rich harmonic structure,
  visible across a wide frequency range. Clearly audible and visually
  distinctive.

#### Gray Whale — `gray_whale_call`
- **Frequency range:** 40 Hz–1.6 kHz, peak energy below 100 Hz
- **Call types:**
  - **M3 migratory moan** — most common during migration; dominant energy
    20–200 Hz, narrow bandwidth, low-frequency moan
  - **S1/M1 knocks** — bongo- or metallic-drum-like broadband pulses;
    peak energy 100 Hz–1.6 kHz
- **Spectrogram signature:** M3 moans appear as faint low-frequency energy
  near the bottom of the display. S1/M1 knocks may be visible up to ~1.6 kHz
  as broadband transient bursts.
- **Ecological note:** Gray whale calves migrate northward through Monterey
  Bay mid-March through mid-May — the same window as peak Bigg's orca activity.
  Orca prey on these calves. Gray whale vocalizations and orca attack calls
  may co-occur in the same recordings.
- **References:** [DOSITS](https://dosits.org/galleries/audio-gallery/marine-mammals/baleen-whales/gray-whale/),
  [Tyack & Clark 2000](https://pdfs.semanticscholar.org/bc3c/b56a934cc176b7afc3847257e8730df06747.pdf)

#### Note on infrasonic species at 32 kHz sample rate
The MARS files are resampled to 32 kHz, giving a Nyquist of 16 kHz.
Blue whale and fin whale fundamentals (10–40 Hz) are well within this
range and fully captured. However, the Gradio spectrogram uses a linear
frequency axis — the bottom ~2% of the display covers 0–320 Hz where
all baleen whale energy concentrates. Consider using a log-frequency
spectrogram for baleen whale work (future enhancement).

---

## Spectrogram Examples — What to Look For

These examples show the three main patterns you will encounter in the
Gradio labeling interface. The spectrogram shows frequency (Hz) on the
Y axis and time (0–5 seconds) on the X axis. Color intensity indicates
sound energy (bright = loud).

### Orca call — label POSITIVE
**Score: 3.629 | File: MARS_20180413_083913 | 495–500s**

![Orca call spectrogram](orca.png)

Characteristic orca call signature: discrete bright energy bursts with
harmonic stacking visible between 1–6 kHz, appearing as horizontal
banded structures. The call is clearly structured and distinct from
background noise. The waveform shows clear amplitude peaks corresponding
to the calls. UTC 08:39 = PDT 01:39 — overnight orca feeding activity.

---

### Ocean background — label NEGATIVE
**Score: −2.815 | File: MARS_20180413_235913 | 130–135s**

![Background noise spectrogram](background.png)

Featureless broadband noise across all frequencies. Energy is uniformly
distributed with no structured features. This is typical deep-water
ambient noise — flow noise, distant shipping, and biological background.
The waveform is stationary with no transients. UTC 23:59 = PDT 16:59 —
mid-afternoon, well outside the orca event window. Label confidently negative.

---

### Dolphin — label NEGATIVE (for orca classifier)
**Score: 3.709 | File: MARS_20180413_163913 | 310–315s**

![Dolphin spectrogram](dolphin.png)

High-frequency tonal whistles with distinctive upsweeping and downsweeping
frequency modulation, extending from ~3 kHz up to 14+ kHz. These are
common/bottlenose dolphin whistles — entirely different structure from
orca calls. Score is high (3.709) because the Perch V2 embedding places
dolphin and orca calls near each other in embedding space. This is a
classic **false positive** case — mark NEGATIVE for the orca classifier.
When building a multi-class classifier, these would be labeled
`dolphin_whistle` as a separate positive class.

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

## Provenance and Reproducibility

Every labeling session and training run writes a JSON audit record so that
any classifier can be fully reproduced from scratch given the original audio.

### Directory structure

```
/mnt/PAM_Analysis/duane_scratch/perch_hoplite/provenance/
    labels/
        labels_20260604_132500_analyst.json   ← one file per labeling session
        labels_20260605_110000_analyst.json
    training/
        train_20260604_150000_orca_v1.json    ← one file per training run
        train_20260605_125234_orca_v5.json
```

### Labeling session record

Written automatically when **💾 Save Labels to DB** is clicked.

```json
{
  "session_id":        "20260605_132500_analyst",
  "timestamp":         "2026-06-05T13:25:00",
  "db_dir":            "/mnt/.../MARS_20180413_20180413_32kHz",
  "classifier":        "/mnt/.../models/orca_v4.pt",
  "annotator_id":      "analyst",
  "query_label":       "orca_call",
  "annotation_count":  50,
  "positive_count":    6,
  "negative_count":    44,
  "annotations": [
    {
      "window_id": 2398,
      "filename":  "MARS_20180413_143913_resampled_32kHz.wav",
      "offset_s":  95.0,
      "end_s":     100.0,
      "label":     "negative",
      "label_type": 2,
      "score":     3.241
    }
  ]
}
```

### Training run record

Written automatically after each `train` command completes.

```json
{
  "session_id":    "20260605_135000_orca_v5",
  "timestamp":     "2026-06-05T13:50:15",
  "db_dir":        "/mnt/.../MARS_20180413_20180413_32kHz",
  "classifier_out": "/mnt/.../models/orca_v5.pt",
  "elapsed_s":     530.1,
  "eval_scores":   {"roc_auc": 0.787, "top1_acc": 1.0, "cmap": 0.703},
  "annotation_counts": {"orca_call_type1": 782, "orca_call_type2": 83},
  "train_args":    {"num_steps": 256, "learning_rate": 0.001, ...}
}
```

### Reproducing a classifier from scratch

Given a provenance record and the original audio:

1. Re-embed the audio with `phase1_embed.py` using the same `--model` and `--shard-len`
2. Import the annotations using the window filenames and offsets from the label records:
   ```bash
   python3 phase2_classify.py label        --db-dir <new_db>        --labels-csv <reconstructed_from_provenance.csv>        --annotator-id duane
   ```
3. Train with the same parameters from the training record

## Utilities

| File | Purpose |
|---|---|
| `merge_annotations.py` | Copy annotations between DBs (annotations only — use with care) |
| `merge_dbs.py` | Full DB merge: SQLite + USearch index (correct way to combine DBs) |

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
