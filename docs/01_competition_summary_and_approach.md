# Competition Summary And Approach

## 1. Storyline

BirdCLEF+ 2026 turns passive acoustic monitoring into a multi-label soundscape problem. The hidden test set contains Brazilian Pantanal soundscapes, and each 1-minute recording is evaluated as **12 contiguous 5-second windows**. A useful solution therefore needs two qualities at the same time: strong wildlife representations and CPU inference that finishes inside Kaggle scoring.

The project now has that split clearly mapped. **EfficientNet-B0** proves the pipeline is stable and gives a simple PyTorch fallback. **Perch v2** provides stronger bioacoustic features and, after switching to the CPU export with full-file batching, becomes the current lead submission.

## 2. Task Shape

The submission contract contains **234 target columns** across birds, amphibians, mammals, reptiles, and insects. The local training metadata used in this repo has **35,549 short recordings** across **206 primary labels**, plus expert-labeled soundscape windows that better match the hidden-test domain.

This gap between short clean clips and long soundscapes is the central modeling challenge. Clean clips are useful for learning species signatures, while soundscape labels are better for calibration, co-occurrence, site/hour priors, and domain shift checks.

## 3. Evidence So Far

| Notebook | Representation | Public score | Main lesson |
|---|---|---:|---|
| `2_bc2026_effnet_b0.ipynb` | 5-second mel-spectrogram CNN | **0.646** | Stable CPU-safe baseline |
| `3_bc2026_perch_v2.ipynb` | Frozen Perch embeddings + probe | **0.770** | Foundation features transfer strongly |

The public leaderboard result matches the validation story. EfficientNet-B0 reaches **0.5464** validation accuracy and establishes a working submission path. Perch v2 reaches **0.8392** validation accuracy and improves the public score by **+0.124**, which is large enough to make it the lead model.

## 4. Dataset Signals

- **Class imbalance** is severe: class counts range from **1** to **499** recordings.
- The top **30** labels account for **40.3%** of recordings, so head classes can dominate aggregate metrics.
- Secondary labels appear in **4,372** recordings with **7,431** total mentions, making co-occurrence a meaningful signal.
- Deduplicated soundscape windows are strongly multi-label, with a median of **4** labels and a maximum of **10**.
- Soundscape site and hour structure matter because hidden-test audio is evaluated as long contextual recordings, not isolated clips.
- Non-bird taxa are strategically important because they are less naturally covered by bird-specialized pretrained models.

## 5. Current Approach

The lead path is **Perch v2 submission mode**. It combines a Google Perch CPU SavedModel with the uploaded PyTorch probe artifact. The speed improvement comes from reading each 60-second soundscape once, reshaping it into 12 windows, and batching multiple files per TensorFlow call.

EfficientNet-B0 stays in the project as the fallback and comparison model. It is less accurate publicly, but it is pure PyTorch, smaller, and easier to reason about when runtime or TensorFlow compatibility becomes unstable.

## 6. Next Experiments

The next runs are ordered by expected gain and risk:

1. **Notebook 3 submission rerun after each Perch change**: this is the current leaderboard model, so every scoring-facing change gets validated there first.
2. **Notebook 3 train mode for calibration diagnostics**: use validation predictions and per-class metrics to find weak labels, especially rare taxa and non-bird classes.
3. **Notebook 1 EDA only when adding new priors**: rerun EDA if site/hour, co-occurrence, or soundscape coverage features change.
4. **Notebook 2 submission as fallback**: rerun only after artifact-path, inference, or ensemble changes.
5. **Notebook 2 train mode only for a new checkpoint**: useful for distillation, pretrained-weight experiments, or class-aware training.

## 7. Refinement Roadmap

The strongest next modeling ideas are:

1. Add site/hour priors from train soundscape labels to Perch predictions.
2. Calibrate class logits using soundscape validation windows rather than only clip-level validation.
3. Improve non-bird taxa with proxy classes, taxonomy-aware priors, or targeted calibration.
4. Compare EffNet and Perch errors to find ensemble candidates that improve public score without much CPU cost.
5. Distill Perch outputs into a faster PyTorch student if the Perch path becomes too heavy for future submissions.

## 8. Source Links

- Kaggle competition overview: https://www.kaggle.com/competitions/birdclef-2026/overview
- Kaggle competition page: https://www.kaggle.com/competitions/birdclef-2026
- Kaggle dataset mirror used for public file descriptions: https://www.kaggle.com/datasets/llkh0a/birdclef-2026-repack
