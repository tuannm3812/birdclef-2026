# EfficientNet-B0 Results

## 1. Role

EfficientNet-B0 is the primary competition-safe baseline. It uses 5-second mono mel-spectrogram crops and a PyTorch EfficientNet-B0 classifier over the 206 training labels.

This model is the preferred Kaggle submission path because it is fast, self-contained, and avoids TensorFlow/Perch runtime constraints during hidden reruns. The notebook disables external pretrained weights by default so it can run with Kaggle internet turned off.

The notebook now follows a single-run flow: train the checkpoint, load that checkpoint directly, then write `submission.csv`. This keeps submission logic simple and avoids broad artifact search code.

## 2. Training Setup

| Item | Value |
|---|---|
| Backbone | EfficientNet-B0 |
| Input | 5-second mono mel-spectrogram |
| Classes | 206 |
| Epochs | 5 |
| Pretrained weights | Disabled by default for internet-off Kaggle reruns |
| Loss | Cross entropy with label smoothing |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Primary notebook | `notebooks/2_bc2026_effnet_b0.ipynb` |

## 3. Validation History

| Epoch | Train loss | Valid loss | Valid accuracy | LR |
|---:|---:|---:|---:|---:|
| 1 | 3.0352 | 2.0567 | 0.6134 | 0.000271 |
| 2 | 1.7437 | 1.7819 | 0.6968 | 0.000196 |
| 3 | 1.2337 | 1.7117 | 0.7153 | 0.000104 |
| 4 | 0.8811 | 1.6919 | 0.7256 | 0.000029 |
| 5 | 0.7143 | 1.6846 | 0.7318 | 0.000000 |

Best validation accuracy from the saved reference run: **0.7318**.

## 4. Interpretation

The baseline learns steadily across all 5 epochs, with validation accuracy improving from **0.6134** to **0.7318**. Validation loss flattens by the final epochs, which suggests the current setup is a solid baseline but should benefit more from data and inference improvements than simply training longer.

The next most useful improvements are:

1. Multi-crop inference for soundscape windows.
2. Class-aware sampling or rare-class augmentation.
3. Per-class validation metrics and confusion analysis.
4. Secondary-label soft targets once the single-label baseline is stable.
5. Threshold or calibration work informed by soundscape label overlap.

The current inference path intentionally raises a clear error if hidden-test audio is missing. The only exception is the tiny public dry-run case, where Kaggle may expose a 3-row sample submission without test audio.
