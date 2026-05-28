# Perch v2 Probe Results

## 1. Role

The Perch v2 experiment evaluates Google Perch embeddings as frozen bioacoustic features. A shallow PyTorch probe is trained on top of 1,536-dimensional embeddings for the same 206 primary labels used by the EfficientNet-B0 baseline.

Perch v2 is best treated as an offline feature generator, teacher model, or distillation source unless hidden-test inference has cached embeddings or a proven Kaggle-compatible runtime.

The Perch notebook now has two modes: **`CFG.mode = "train"`** for the full experiment and **`CFG.mode = "submission"`** for loading the probe artifact and writing `submission.csv`. This is useful for score experiments, but it is still operationally heavier than EfficientNet because hidden-test audio is embedded through the Perch SavedModel during scoring.

## 2. Training Setup

| Item | Value |
|---|---|
| Feature model | Google Perch v2 |
| Embedding shape | 35,549 x 1,536 |
| Probe | LayerNorm, Linear, ReLU, Dropout, Linear |
| Classes | 206 |
| Epochs | 20 max with early stopping |
| Latest observed run | Early stopped after 7 epochs |
| Diagnostic outputs | Validation predictions, per-class recall, summary JSON |
| Runtime requirement | TensorFlow 2.20+ for the Perch export |
| Uploaded artifact path | `/kaggle/input/models/tuannm3812/birdclef-perch-v2-artifacts/pytorch/default/1/perch_v2` |
| Submission mode | Loads `best_perch_probe.pt` and `labels.json`; skips train embeddings, probe training, diagnostics, and artifact zip |
| Submission speedup | Prefers the CPU Perch export and batches full 60-second soundscape files into 12 windows per file |
| CPU public score | **0.770** |
| Primary notebook | `notebooks/03_perch_v2.ipynb` |

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

An earlier notebook 3 failure came from trying to install `tensorflow==2.20.0` from PyPI during an offline Kaggle run. The current notebook uses the local TensorFlow 2.20 wheel input and keeps Perch execution local.

The current Perch submission stack uses:

| Component | Path |
|---|---|
| TensorFlow 2.20 wheels | `/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0` |
| Perch artifact | `/kaggle/input/models/tuannm3812/birdclef-perch-v2-artifacts/pytorch/default/1/perch_v2` |
| Probe checkpoint | `best_perch_probe.pt` |
| Label map | `labels.json` |

## 5. Interpretation

The probe reaches strong validation accuracy almost immediately, which indicates that Perch embeddings already encode most of the useful acoustic structure. The best epoch in the latest run is epoch 3; early stopping ends the run at epoch 7 after validation stops improving while train loss continues to fall.

Compared with the offline-safe EfficientNet-B0 run, Perch v2 improves validation accuracy by about **31.2 percentage points**. The CPU submission also improves the public score from **0.646** to **0.770**, a **+0.124** gain over EffNet-B0. The result is important because it shows foundation-model representations are highly valuable for this competition, provided the CPU inference path stays optimized.

The fast starter notebook runs well on CPU because it uses the **`perch_v2_cpu`** SavedModel, reads each 60-second soundscape once with `soundfile`, reshapes it into **12 contiguous 5-second windows**, and batches multiple files per TensorFlow call. Notebook 3 now mirrors that submission strategy while keeping the PyTorch probe artifact path.

The most practical next step is to strengthen Perch without losing CPU viability:

1. Add soundscape priors from site/hour metadata and validate the effect on train soundscapes.
2. Calibrate logits per class, especially rare and non-bird taxa.
3. Compare EfficientNet failures against Perch successes to identify ensemble opportunities.
4. Distill Perch predictions or embeddings into a faster PyTorch student.
5. Keep early stopping; the latest run suggests extra probe epochs are unlikely to help much after the first few epochs.

## 6. Added Diagnostics

The notebook now saves three lightweight diagnostic artifacts:

1. `validation_predictions.csv`: row-level validation target, top-1 prediction, top-1 confidence, top-5 labels, and correctness flags.
2. `per_class_validation_metrics.csv`: class support, top-1 recall, top-5 recall, and mean top-1 confidence.
3. `summary.json`: best validation accuracy, top-5 validation accuracy, best epoch, epochs run, and embedding shape.

These files make the Perch result more actionable. Low top-1 recall with high top-5 recall points to class-confusion problems that may benefit from calibration or soft targets, while low top-5 recall suggests missing acoustic coverage or hard domain shift.
