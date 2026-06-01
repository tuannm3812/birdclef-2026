# Notebook Structure

Keep this folder small. Notebooks should be promoted here only when they are
project-owned, reproducible, and tied to a documented experiment.

## Canonical Notebooks

| Notebook | Role |
|---|---|
| `01_eda.ipynb` | Dataset audit and figures |
| `02_effnet_b0.ipynb` | Protected EfficientNet-B0 fallback and submission path |
| `03_perch_v2_train.ipynb` | Perch probe training, diagnostics, and artifact packaging |
| `04_perch_v2_submit.ipynb` | Protected Perch v2 submission reference |
| `05_onnx_sed_submit.ipynb` | Protected distilled SED ONNX champion submission |
| `06_onnx_perch_speed_test.ipynb` | ONNX Perch runtime experiment |
| `07_onnx_perch_sed_blend.ipynb` | Protected ONNX Perch + SED champion submission |
| `08_onnx_perch_sed_blend_w025.ipynb` | Blend-weight variant with Perch weight 0.25 |
| `09_onnx_perch_sed_blend_proxy6.ipynb` | Protected narrow proxy-mapping champion submission |
| `10_onnx_perch_sed_soundscape_calibrated.ipynb` | Protected soundscape-calibrated champion submission |
| `11_onnx_perch_sed_calibrated_min10_ap001.ipynb` | Support-thresholded calibration variant |
| `12_onnx_perch_sed_calibrated_shrink050.ipynb` | Shrunk-calibration variant |
| `13_onnx_perch_sed_temporal_residual.ipynb` | Active temporal residual blend experiment |

## Active Notebook Lane

Use **one** active notebook slot at a time. The current lane is:

| Notebook | Purpose |
|---|---|
| `13_onnx_perch_sed_temporal_residual.ipynb` | Train a lightweight temporal residual model on soundscape labels, then blend it with the 0.893 calibrated champion |

Do not add separate notebooks for every public reference. Review external
notebooks in `docs/`, then promote only the cleaned project-owned version.

## Perch v2 Policy

Keep `03_perch_v2_train.ipynb` and `04_perch_v2_submit.ipynb`. They preserve
the current champion path and artifact history. Do not keep iterating on direct
TensorFlow Perch CPU submission unless we need to reproduce version 14; new
leaderboard work should use ONNX-based notebooks.

## Promotion Rules

1. A notebook must have one clear mode: EDA, training, or submission.
2. A submission notebook must avoid training and heavy diagnostics.
3. Public-reference notebooks stay in `docs/*_review.md`, not in this folder.
4. Protected baselines should not be overwritten by experiments.
5. New experiments should update `docs/06_next_steps.md` before another notebook
   is added.
