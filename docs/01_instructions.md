# Instructions

## 1. Competition Overview

BirdCLEF+ 2026 is a passive acoustic monitoring competition for the Brazilian Pantanal. The goal is to identify calling wildlife species from hidden 1-minute soundscape recordings. Each soundscape is evaluated as **12 contiguous 5-second windows**, and each row in `submission.csv` contains probabilities for the target species columns.

The task is multi-taxon rather than bird-only. The target set includes birds, amphibians, mammals, reptiles, and insects, which makes the problem closer to ecosystem soundscape recognition than ordinary single-species bird-call classification.

Key repository facts:

| Item | Value |
|---|---:|
| Submission target columns | **234** |
| Training recordings | **35,549** |
| Training primary labels | **206** |
| Deduplicated labeled soundscape windows | **739** |
| Current EfficientNet-B0 public score | **0.646** |
| Current Perch v2 public score | **0.770** |

## 2. What The Notebooks Need To Answer

The competition reduces to five practical questions:

1. **Which species are present in each 5-second window?**
   The output is probability-based, so the model needs calibrated confidence rather than only top-1 accuracy.

2. **How do we bridge clean clips and noisy soundscapes?**
   Training audio is mostly short focal recordings, while scoring audio is long passive soundscape audio with overlapping species, background noise, and sparse calls.

3. **How do we handle severe label imbalance?**
   Class counts range from **1** to **499** recordings, and the top **30** labels account for **40.3%** of the training set.

4. **How do we use soundscape structure?**
   Deduplicated soundscape windows are strongly multi-label, with a median of **4** labels and a maximum of **10**. Site, hour, and co-occurrence signals are useful for calibration and post-processing.

5. **How do we finish scoring on Kaggle CPU?**
   The strongest model is not useful unless inference finishes inside the competition scoring limit. Submission paths focus on loading artifacts, batching soundscape windows, and writing `submission.csv`.

## 3. Tasks In This Repository

| Task | Notebook | Purpose |
|---|---|---|
| Explore the dataset | `1_bc2026_eda.ipynb` | Audit labels, soundscape annotations, metadata shift, and acoustic examples |
| Build a simple baseline | `2_bc2026_effnet_b0.ipynb` | Train and score a 5-second mel-spectrogram EfficientNet-B0 model |
| Build the lead model | `3_bc2026_perch_v2.ipynb` | Train and score a shallow classifier on frozen Google Perch v2 embeddings |

The notebooks are intentionally ordered from understanding to baseline to stronger transfer model. EDA explains why validation, calibration, and runtime choices matter; EfficientNet validates the end-to-end Kaggle pipeline; Perch v2 gives the best current score.

## 4. Current Solution Approach

### 4.1 EDA-Driven Modeling

The EDA notebook establishes the constraints that drive modeling:

- **Class imbalance** calls for per-class metrics, class-aware sampling, rare-class augmentation, or calibration by label frequency.
- **Secondary labels** reveal co-occurrence patterns that can later support soft targets or post-processing.
- **Soundscape labels** are closer to hidden-test audio than clean clips, so they are the best source for calibration and domain diagnostics.
- **Metadata shift** appears through rating, collection source, geography, site, and hour concentration.
- **Representative spectrograms** confirm that 5-second crops need to handle sparse calls and background energy.

### 4.2 EfficientNet-B0 Baseline

EfficientNet-B0 converts each 5-second clip into a normalized mel-spectrogram and trains a compact CNN classifier. Its role is reliability:

- Pure PyTorch inference.
- Small model size.
- Clear fallback when TensorFlow or Perch runtime changes.
- Public score: **0.646**.

This baseline is not the current leader, but it is useful for sanity checks, runtime comparison, and future ensembles.

### 4.3 Perch v2 Lead Model

Perch v2 uses a pretrained bioacoustic representation and trains a shallow PyTorch probe over frozen **1,536-dimensional embeddings**. This model currently leads the repo:

- Validation accuracy: **0.8392**.
- Public score: **0.770**.
- CPU submission path uses the Perch CPU export.
- Full-file batching reads each 60-second soundscape once and reshapes it into **12** windows.

The result shows that foundation bioacoustic features transfer better than the small CNN baseline for this dataset.

## 5. Next Questions

The next experiments are ordered by likely gain:

1. **Can Perch predictions improve with soundscape priors?**
   Test hour, site, and co-occurrence logit offsets from labeled soundscape windows.

2. **Which labels are weak despite strong validation accuracy?**
   Use Perch validation predictions and per-class metrics to inspect rare labels and non-bird taxa.

3. **Can EfficientNet add complementary signal?**
   Compare EfficientNet and Perch errors before adding any ensemble cost.

4. **Can Perch be distilled into a faster PyTorch model?**
   Use Perch outputs as soft targets if TensorFlow inference becomes a scoring bottleneck.

5. **Can calibration improve leaderboard score without retraining?**
   Tune class-level thresholds or logit scaling against soundscape-like validation examples.

## 6. Source Links

- Kaggle competition overview: https://www.kaggle.com/competitions/birdclef-2026/overview
- Kaggle competition page: https://www.kaggle.com/competitions/birdclef-2026
- Kaggle dataset mirror used for public file descriptions: https://www.kaggle.com/datasets/llkh0a/birdclef-2026-repack
