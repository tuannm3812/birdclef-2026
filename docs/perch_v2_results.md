# Perch v2 Probe Results

## 1. Role

The Perch v2 experiment evaluates Google Perch embeddings as frozen bioacoustic features. A shallow PyTorch probe is trained on top of 1,536-dimensional embeddings for the same 206 primary labels used by the EfficientNet-B0 baseline.

Perch v2 is best treated as an offline feature generator, teacher model, or distillation source unless hidden-test inference has cached embeddings or a proven Kaggle-compatible runtime.

## 2. Training Setup

| Item | Value |
|---|---|
| Feature model | Google Perch v2 |
| Embedding shape | 35,549 x 1,536 |
| Probe | LayerNorm, Linear, ReLU, Dropout, Linear |
| Classes | 206 |
| Epochs | 20 max with early stopping |
| Latest observed run | Early stopped after 7 epochs |
| Runtime requirement | TensorFlow 2.20+ for the attached Perch export |
| Primary notebook | `notebooks/3_bc2026_perch_v2.ipynb` |

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

The observed notebook 3 failure came from trying to install `tensorflow==2.20.0` from PyPI while Kaggle internet was disabled. The notebook now uses the attached `/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0` wheel input directly and keeps Perch execution local.

Recommended Perch workflow:

1. Attach `/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0` for TensorFlow 2.20 wheels.
2. Attach `/kaggle/input/datasets/jaejohn/perch-meta` or another Perch SavedModel input for local embedding extraction.
3. Keep the final Perch experiment reproducible by using attached wheels and model inputs rather than runtime downloads.
4. Restart the Kaggle session after any TensorFlow upgrade if TensorFlow was imported earlier.

## 5. Interpretation

The probe reaches strong validation accuracy almost immediately, which indicates that Perch embeddings already encode most of the useful acoustic structure. The best epoch in the latest run is epoch 3; early stopping ends the run at epoch 7 after validation stops improving while train loss continues to fall.

Compared with the offline-safe EfficientNet-B0 run, Perch v2 improves validation accuracy by about **31.2 percentage points**. The result is important because it shows foundation-model representations are highly valuable for this competition, but the submission path is operationally heavier than the PyTorch baseline.

The most practical next step is to use Perch as a teacher:

1. Distill Perch predictions or embeddings into a faster PyTorch student.
2. Keep early stopping; the latest run suggests extra probe epochs are unlikely to help much after the first few epochs.
3. Use Perch features for error analysis and class-similarity diagnostics.
4. Compare EfficientNet failures against Perch successes to identify rare-class or domain-shift patterns.
5. Train a PyTorch student on Perch-informed soft targets so hidden inference can stay lightweight.
