# Perch v2 Probe Results

## 1. Role

The Perch v2 experiment evaluates Google Perch embeddings as frozen bioacoustic features. A shallow PyTorch probe is trained on top of 1,536-dimensional embeddings for the same 206 primary labels used by the EfficientNet-B0 baseline.

Perch v2 is best treated as an offline feature generator, teacher model, or distillation source unless hidden-test inference has cached embeddings or a proven Kaggle-compatible runtime.

The Perch workflow is now split into two notebooks. The training notebook uses cached Perch embeddings and does not load TensorFlow or the Perch SavedModel. The submission notebook is the only place that loads the CPU Perch export for hidden-test soundscape scoring.

## 2. Training Setup

| Item | Value |
|---|---|
| Feature model | Google Perch v2 |
| Embedding shape | 35,549 x 1,536 |
| Probe | LayerNorm, Linear, ReLU, Dropout, Linear |
| Classes | 206 |
| Epochs | 20 max with early stopping |
| Latest observed run | Early stopped after 7 epochs |
| Diagnostic outputs | Validation predictions, per-class recall, weak-label diagnostics, calibration JSON, soundscape priors |
| Training runtime requirement | Cached `train_embeddings.npz` |
| Submission runtime requirement | TensorFlow 2.20+ and CPU Perch export |
| Uploaded artifact path | `/kaggle/input/models/tuannm3812/birdclef-perch-v2-artifacts/pytorch/default/3/perch_v2` for latest probe/calibration artifacts |
| Submission mode | Loads `best_perch_probe.pt` and `labels.json`; skips train embeddings, probe training, diagnostics, and artifact zip |
| Submission speedup | Prefers the CPU Perch export and batches full 60-second soundscape files into 12 windows per file |
| CPU public score | **0.770** |
| Training notebook | `notebooks/03_perch_v2_train.ipynb` |
| Submission notebook | `notebooks/04_perch_v2_submit.ipynb` |

## 3. Validation History

| Epoch | Train loss | Valid accuracy |
|---:|---:|---:|
| 1 | 1.3073 | 0.8357 |
| 2 | 0.5471 | 0.8333 |
| 3 | 0.3382 | 0.8392 |
| 4 | 0.2109 | 0.8374 |
| 5 | 0.1429 | 0.8336 |
| 6 | 0.1105 | 0.8332 |
| 7 | 0.0913 | 0.8367 |

Best validation accuracy from the latest Kaggle run: **0.8392**.

## 4. Kaggle Runtime Notes

An earlier submission failure came from using the CUDA-only Perch export on CPU. The submission notebook must use the CPU Perch export, while the training notebook avoids this issue by training from cached embeddings only.

The Kaggle model has two useful versions attached. Version **3** contains the
latest probe, calibration, and diagnostics. Version **1** still contains
`train_embeddings.npz`, but the training notebook now prefers the dedicated
`perch-meta` cached embedding dataset when it is attached.

The current Perch submission stack uses:

| Component | Path |
|---|---|
| Competition data | `/kaggle/input/competitions/birdclef-2026` |
| TensorFlow 2.20 wheels | `/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0` |
| Perch artifact | `/kaggle/input/models/tuannm3812/birdclef-perch-v2-artifacts/pytorch/default/3/perch_v2` |
| Preferred cached train embeddings | `/kaggle/input/datasets/jaejohn/perch-meta/full_perch_arrays.npz` with `/kaggle/input/datasets/jaejohn/perch-meta/full_perch_meta.parquet` |
| Legacy cached train embeddings | `/kaggle/input/models/tuannm3812/birdclef-perch-v2-artifacts/pytorch/default/1/perch_v2/train_embeddings.npz` |
| Probe checkpoint | `best_perch_probe.pt` |
| Label map | `labels.json` |

## 5. Interpretation

The probe reaches strong validation accuracy almost immediately, which indicates that Perch embeddings already encode most of the useful acoustic structure. The best epoch in the latest run is epoch 3; early stopping ends the run at epoch 7 after validation stops improving while train loss continues to fall.

Compared with the offline-safe EfficientNet-B0 run, Perch v2 improves validation accuracy by about **31.2 percentage points**. The CPU submission also improves the public score from **0.646** to **0.770**, a **+0.124** gain over EffNet-B0. The result is important because it shows foundation-model representations are highly valuable for this competition, provided the CPU inference path stays optimized.

The fast starter notebook runs well on CPU because it uses the **`perch_v2_cpu`** SavedModel, reads each 60-second soundscape once with `soundfile`, reshapes it into **12 contiguous 5-second windows**, and batches multiple files per TensorFlow call. The dedicated Perch submission notebook now mirrors that strategy while keeping the training notebook focused on artifacts and diagnostics.

The most practical next step is to strengthen Perch without losing CPU viability:

1. Add soundscape priors from site/hour metadata and validate the effect on train soundscapes.
2. Calibrate logits per class, especially rare and non-bird taxa.
3. Compare EfficientNet failures against Perch successes to identify ensemble opportunities.
4. Distill Perch predictions or embeddings into a faster PyTorch student.
5. Keep early stopping; the latest run suggests extra probe epochs are unlikely to help much after the first few epochs.

## 6. Added Diagnostics

The notebook now saves lightweight diagnostic artifacts:

1. `validation_predictions.csv`: row-level validation target, top-1 prediction, top-1 confidence, top-5 labels, and correctness flags.
2. `per_class_validation_metrics.csv`: class support, top-1 recall, top-5 recall, and mean top-1 confidence.
3. `weak_label_diagnostics.csv`: labels ranked by low recall, high-confidence mistakes, and most common wrong prediction.
4. `temperature_grid.csv` and `calibration.json`: validation log-loss sweep for global temperature scaling.
5. `soundscape_priors.json`: deduplicated soundscape priors by hour, site, and co-occurrence.
6. `summary.json`: best validation accuracy, top-5 validation accuracy, log loss, selected temperature, best epoch, epochs run, and embedding shape.

These files make the Perch result more actionable. Low top-1 recall with high top-5 recall points to class-confusion problems that may benefit from calibration or soft targets, while low top-5 recall suggests missing acoustic coverage or hard domain shift.

## 7. New Experiment Hooks

The Perch training and submission notebooks now include the next experimental
layer without changing the default score path:

1. **Temperature calibration.** The training notebook collects validation logits, sweeps a global temperature, and saves the best setting in `calibration.json`.
2. **Weak-label inspection.** The training notebook writes `weak_label_diagnostics.csv` so rare labels, non-bird taxa, and high-confidence confusions can be reviewed directly.
3. **Soundscape priors.** The training notebook deduplicates `train_soundscapes_labels.csv` and builds hour, site, and co-occurrence logit-offset tables in `soundscape_priors.json`.

`CFG.use_soundscape_priors` is still `False` by default. This keeps the current
**0.770** Perch result comparable until a fresh Kaggle run validates that the
prior offsets improve scoring. After uploading a newly trained artifact, turn
`CFG.use_soundscape_priors = True` for a controlled leaderboard test.

Recommended test order:

1. Run the training notebook once and inspect `calibration.json`,
   `weak_label_diagnostics.csv`, and `soundscape_prior_summary.csv`.
2. Submit with calibration only, leaving `CFG.use_soundscape_priors = False`.
3. Submit with `CFG.use_soundscape_priors = True`.
4. If priors help, tune `hour_prior_weight`, `site_prior_weight`, and
   `cooccurrence_prior_weight` conservatively.
