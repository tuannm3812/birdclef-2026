# EfficientNet-B0 Results

## 1. Role

EfficientNet-B0 is the primary competition-safe baseline. It uses 5-second mono mel-spectrogram crops and a PyTorch EfficientNet-B0 classifier over the 206 training labels.

This model is the preferred Kaggle submission path because it is fast, self-contained, and avoids TensorFlow/Perch runtime constraints during hidden reruns. The notebook disables external pretrained downloads and forces offline Hugging Face mode so hidden reruns do not fail on network calls.

The notebook supports two modes: **`CFG.mode = "train"`** for producing the checkpoint and **`CFG.mode = "submission"`** for scored reruns that load an attached checkpoint artifact and write `submission.csv` quickly.

## 2. Training Setup

| Item | Value |
|---|---|
| Backbone | EfficientNet-B0 |
| Input | 5-second mono mel-spectrogram |
| Classes | 206 |
| Epochs | 15 max with early stopping |
| Latest observed run | 15 completed epochs |
| Pretrained weights | External downloads disabled; optional local Kaggle input checkpoint supported |
| Loss | Cross entropy with label smoothing |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Submission mode | Loads `best_effnet_b0.pt` and `labels.json` from an attached artifact dataset |
| Primary notebook | `notebooks/2_bc2026_effnet_b0.ipynb` |

## 3. Validation History

| Epoch | Train loss | Valid loss | Valid accuracy | LR |
|---:|---:|---:|---:|---:|
| 1 | 4.3488 | 3.5253 | 0.2451 | 0.000297 |
| 2 | 3.1525 | 2.8786 | 0.4076 | 0.000287 |
| 3 | 2.5361 | 2.5778 | 0.4765 | 0.000271 |
| 4 | 2.0606 | 2.4620 | 0.5198 | 0.000250 |
| 5 | 1.5823 | 2.5089 | 0.5298 | 0.000225 |
| 6 | 1.1074 | 2.6487 | 0.5149 | 0.000196 |
| 7 | 0.7937 | 2.7033 | 0.5276 | 0.000166 |
| 8 | 0.6623 | 2.6473 | 0.5339 | 0.000134 |
| 9 | 0.6045 | 2.6386 | 0.5295 | 0.000104 |
| 10 | 0.5737 | 2.5857 | 0.5361 | 0.000075 |
| 11 | 0.5551 | 2.5792 | 0.5418 | 0.000050 |
| 12 | 0.5435 | 2.5465 | 0.5425 | 0.000029 |
| 13 | 0.5336 | 2.5445 | 0.5453 | 0.000013 |
| 14 | 0.5279 | 2.5369 | 0.5464 | 0.000003 |
| 15 | 0.5244 | 2.5462 | 0.5426 | 0.000000 |

Best validation accuracy from the latest Kaggle run: **0.5464** at epoch **14**.

## 4. Pretrained Weight Policy

Do not leave Kaggle internet enabled for the final competition notebook. If pretrained EfficientNet initialization is needed, create a separate Kaggle dataset that contains the timm/Hugging Face weight file, usually `model.safetensors` or a `.pt` checkpoint, and point `CFG.pretrained_weight_path` to that file. The notebook always calls `timm.create_model(..., pretrained=False)` and then loads compatible local tensors, adapting the RGB stem to mono audio and skipping incompatible classifier weights.

## 5. Interpretation

The offline-safe baseline now trains from random initialization unless a local pretrained checkpoint is attached. That explains the lower validation accuracy compared with earlier pretrained-style runs, but the full 15-epoch run still improves meaningfully: validation accuracy rises from **0.2451** to **0.5464**.

The curve starts to overfit after the middle epochs: training loss keeps falling sharply, while validation loss bottoms out around epoch **14** and accuracy only improves slowly after epoch **8**. The saved best checkpoint should therefore come from epoch **14**, not the final epoch. If a local EfficientNet pretrained checkpoint is attached, expect a stronger starting point and faster convergence.

The next most useful improvements are:

1. Run the 15-epoch early-stopping configuration and compare the best checkpoint.
2. Attach a local EfficientNet-B0 pretrained weight file if allowed by the final Kaggle setup.
3. Add multi-crop inference for soundscape windows.
4. Add class-aware sampling or rare-class augmentation.
5. Use secondary-label soft targets once the single-label baseline is stable.

The current inference path intentionally raises a clear error if hidden-test audio is missing. The only exception is the tiny public dry-run case, where Kaggle may expose a 3-row sample submission without test audio.

## 6. Submission Runtime

Do not submit a notebook that trains the model during scoring. Kaggle scoring has a tight runtime budget, so the final run should use **`CFG.mode = "submission"`**:

1. Run training once with `CFG.mode = "train"`.
2. Save the generated `best_effnet_b0.pt` and `labels.json` as a Kaggle dataset.
3. Attach that dataset to the submission notebook.
4. Set `CFG.mode = "submission"` and, if needed, `CFG.submission_artifact_dir`.
5. Submit the notebook so it only loads the checkpoint, scores test windows, and writes `submission.csv`.
