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
| Latest observed run | 5 completed epochs |
| Pretrained weights | External downloads disabled; optional local Kaggle input checkpoint supported |
| Loss | Cross entropy with label smoothing |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Submission mode | Loads `best_effnet_b0.pt` and `labels.json` from an attached artifact dataset |
| Primary notebook | `notebooks/2_bc2026_effnet_b0.ipynb` |

## 3. Validation History

| Epoch | Train loss | Valid loss | Valid accuracy | LR |
|---:|---:|---:|---:|---:|
| 1 | 4.3343 | 3.5052 | 0.2463 | 0.000271 |
| 2 | 3.1107 | 2.8757 | 0.4025 | 0.000196 |
| 3 | 2.4525 | 2.5757 | 0.4813 | 0.000104 |
| 4 | 1.8975 | 2.4568 | 0.5225 | 0.000029 |
| 5 | 1.4742 | 2.4470 | 0.5273 | 0.000000 |

Best validation accuracy from the latest Kaggle run: **0.5273**.

## 4. Pretrained Weight Policy

Do not leave Kaggle internet enabled for the final competition notebook. If pretrained EfficientNet initialization is needed, create a separate Kaggle dataset that contains the timm/Hugging Face weight file, usually `model.safetensors` or a `.pt` checkpoint, and point `CFG.pretrained_weight_path` to that file. The notebook always calls `timm.create_model(..., pretrained=False)` and then loads compatible local tensors, adapting the RGB stem to mono audio and skipping incompatible classifier weights.

## 5. Interpretation

The offline-safe baseline now trains from random initialization unless a local pretrained checkpoint is attached. That explains the lower validation accuracy compared with earlier pretrained-style runs, but the learning curve is still improving at epoch 5: validation accuracy rises from **0.2463** to **0.5273**, and validation loss continues to drop.

The notebook now allows up to **15** epochs with early stopping. This is worth running because the current 5-epoch curve has not saturated. If a local EfficientNet pretrained checkpoint is attached, expect a much stronger starting point and faster convergence.

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
