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

https://arxiv.org/abs/2512.03219 https://arxiv.org/abs/2508.04665 


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
| `phase2_classify_logmel.py` | **spark-ae0e** | same as above, with log mel spectrogram display |
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

```bash
nohup python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
    --classifier-out /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v3_clean.pt \
    --num-steps 256 \
    --train-ratio 0.8 \
    > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/train_orca_v3_clean.log 2>&1 &
sleep 5 && tail -f /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/train_orca_v3_clean.log
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

### Step 8 - Review detections produced by perch-hoplite system

```bash
nohup python3 phase2_classify.py review     --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz     --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1_clean.pt     --target-label orca_call     --detections-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_detections.csv     --num-results 25     --detections-offset 1     --classes orca_call,dolphin_call,other,unlabeled     --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04     --serve --port 7860     > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_7860.log 2>&1 &
```

### optional Step 8 -- Review detections, logmel spectrogram display, optional --grayscale spectrogram display

```bash
nohup python3 phase2_classify_logmel.py review     --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz     --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1_clean.pt     --target-label orca_call     --detections-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_detections.csv     --num-results 25     --detections-offset 200     --classes orca_call,dolphin_call,other,unlabeled     --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04  --grayscale  --serve --port 7860     > /mnt/PAM_Analysis/duane_scratch/perch_hoplite/logs/review_7860.log 2>&1 &
```

--detections-offset is updated manually from 0 to n to select which annotations to review

0 (1 to --num-results here 25)

25 (26 to 50)

...

200 (201 to 225)

...

---

## Current Status (as of June 25 2026)

| Item | Status |
|---|---|
| DB: MARS April 13 2018 | ✅ 144 files, 17,280 embeddings |
| DB: MARS April 1 2018 | ✅ 144 files, 17,280 embeddings (separate) |
| DB: MARS_combined | ✅ 34,560 embeddings (Apr 1 + Apr 13 merged, experimental) |
| orca_v1_clean.pt | ✅ **ROC-AUC 0.9821** — 44 pos / 56 neg, clean labels, train_ratio=0.8 |
| orca_v2_clean.pt | ✅ **ROC-AUC 0.9191** — 54 pos / 56 neg, 110 clean labels, train_ratio=0.8 |
| orca_v3_clean.pt | ✅ **ROC-AUC 0.9900** — multi-class: 213 orca + 13 dolphin + 1 other + 55 neg |
| Inference v1_clean | ✅ 227 detections — 213 orca confirmed, 13 dolphin (false pos), 1 other |
| Inference v3_clean | ✅ 321 orca + 205 dolphin + 1 other (527 total) — multi-class |
| Multi-class labels | ✅ orca_call / dolphin_call / other — all 227 v1 detections reviewed |
| Full April 2018 DB | 🔲 planned via Colab Pro batches |
| humpback_song class | 🔲 planned — May 2 2018 embedding next |
| fin_whale_call class | 🔲 planned — May 2 2018 (Google model scores available) |
| ship_noise class | 🔲 planned — replaces generic `other` for vessel noise |

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

1. **Dolphin false positives** — Perch V2 embeddings place Pacific white-sided
   dolphin pulsed calls near orca calls (Henderson et al. 2011, JASA). High
   classifier scores do NOT always mean orca. Always listen and check the
   spectrogram. Use label `dolphin_call` (not `dolphin_whistle`) since it is
   the pulsed calls — not tonal whistles — that cause confusion with orca.
   **Resolved in v3_clean** by adding dolphin_call as a separate class.

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
- **Multi-class:** use `--classes orca_call,dolphin_call,other,unlabeled` when
  reviewing inference detections to separate species in one pass
- **Train ratio:** 0.8 for reliable eval signal
- **Target ROC-AUC:** > 0.90 before full inference ✅ achieved (v3_clean: 0.990)

**Guidelines**
Use **unlabeled** (skip it) when:

- You genuinely cannot tell what it is after listening and looking at the spectrogram
- The signal is so faint you can't make out any structure
- You're uncertain between two classes and don't want to bias the classifier

Use **other** when:

- You can clearly hear something but it's definitely not orca or dolphin — e.g. a boat, a loud transient, flow noise artifact, or an unknown biological sound
- The spectrogram shows clear structure but it doesn't match either target class

Give your best guess (**orca_call** or **dolphin_call**) when:

You can see or hear partial call structure that looks like one class more than the other, even if faint
The UTC timestamp is in the known orca active window (14:19–18:29 UTC) — a faint signal in that window is more likely orca than not
You're 60%+ confident — a soft label is still better than no label for a classifier

The key principle: unlabeled is the right choice when you're truly 50/50. But if you're leaning even slightly toward one class, label it — the classifier can handle a few soft labels. What hurts the classifier most is confidently wrong labels (labeling a dolphin as orca), not uncertain-but-correct labels.
For faint signals specifically: if you see the characteristic banded harmonic structure of orca in the spectrogram even if quiet, trust the spectrogram over the audio and label it orca_call. The spectrogram is often more informative than your ears for faint calls.

### Inference results summary (April 13 2018)

| Model | ROC-AUC | Orca detections | Dolphin detections | Notes |
|---|---|---|---|---|
| orca_v1_clean | 0.982 | 227 | — | Single class |
| orca_v2_clean | 0.919 | 239 | — | Single class |
| orca_v3_clean | **0.990** | **321** | **205** | Multi-class — separates species |

All detections cluster within known orca active hours (UTC 06:49–20:49) with
zero false positives in quiet periods — confirming strong temporal precision.
v3_clean finds 94 more orca detections than v1_clean because the dolphin class
absorbs the false positives that previously suppressed borderline orca calls.

---

## Multi-class Classification

Run review/label once per sound class with a distinct `--target-label`.
All labels accumulate in the same DB. Train once — the classifier is
automatically multi-class.

Suggested label names:
```
orca_call           # Bigg's / resident orca vocalizations
humpback_song       # humpback whale song units
fin_whale_call      # fin whale 20 Hz pulse
dolphin_call        # Pacific white-sided dolphin burst pulse calls
ship_noise          # vessel engine noise (regular low-freq pulsing)
other               # catch-all: ROV thruster, unknown bio, unclassified
unlabeled           # skip — always last, always gray
```

For multi-class Gradio review sessions, pass `--classes` with comma-separated
labels in the order you want them displayed. Up to 6 named classes supported
(plus unlabeled = 7 buttons total). Button colors by position:

| Position | Color | Suggested use |
|---|---|---|
| 1 | Green | orca_call |
| 2 | Amber | humpback_song |
| 3 | Blue | fin_whale_call |
| 4 | Purple | dolphin_call |
| 5 | Teal | ship_noise |
| 6 | Orange | other |
| last | Gray | unlabeled |

Example for May 2 2018 review session:
```bash
--classes orca_call,humpback_song,fin_whale_call,dolphin_call,ship_noise,other,unlabeled
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

#### Pacific White-Sided Dolphin — `dolphin_call`
- **Whistles:** 3–14 kHz, upsweeping/downsweeping tonal arcs
- **Clicks:** broadband, >10 kHz
- **Calls:** 
- **Important:** Perch V2 embeddings place dolphin and orca calls near
  each other — dolphin calls are the most common **false positive**
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
spectrogram for baleen whale work. Use logmel spectrogram display, optional --grayscale spectrogram display.

---

## Spectrogram Examples — What to Look For

These examples show the three main patterns you will encounter in the
Gradio labeling interface. The spectrogram shows frequency (Hz) on the
Y axis and time (0–5 seconds) on the X axis. Color intensity indicates
sound energy (bright = loud).

### Orca call — label orca_call
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

### Dolphin call — label NEGATIVE (for orca classifier)
**Score: 3.709 | File: MARS_20180413_163913 | 310–315s**

![Dolphin spectrogram](dolphin.png)

High-frequency tonal whistles with distinctive upsweeping and downsweeping
frequency modulation, extending from ~3 kHz up to 14+ kHz. These are
Pacific white-sided dolphin calls — entirely different structure from
orca calls. Score is high (3.709) because the Perch V2 embedding places
dolphin and orca calls near each other in embedding space. This is a
classic **false positive** case — mark NEGATIVE for the orca classifier.
When building a multi-class classifier, these would be labeled
`dolphin_call` as a separate positive class.

---

### Dolphin burst pulse call — label DOLPHIN_CALL (multi-class classifier)
**Score: 0.020 | File: MARS_20180413_172913 | 380–385s**

![Dolphin burst pulse call spectrogram](dolphin_call.png)

Dense vertical striping across 2–14 kHz — the characteristic signature of
Pacific white-sided dolphin burst pulse calls (Henderson et al. 2011, JASA).
Unlike the tonal upsweeping whistles in the previous example, burst pulses
appear as rapid broadband click trains producing continuous vertical streaks
across a wide frequency range. The score is very low (0.020) — the v3_clean
multi-class classifier correctly assigns this to `dolphin_call` rather than
`orca_call`. UTC 17:29 = PDT 10:29 — within the known afternoon dolphin
activity window. In the multi-class Gradio interface, label these as
`dolphin_call` (amber button).

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
