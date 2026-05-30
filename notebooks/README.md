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
| `06_onnx_perch_speed_test.ipynb` | Active ONNX Perch runtime experiment |

## Active Notebook Lane

Use **one** active notebook slot at a time. The current lane is:

| Notebook | Purpose |
|---|---|
| `06_onnx_perch_speed_test.ipynb` | Measure ONNX Perch inference speed without blending or sequence modeling |

If this succeeds on Kaggle, add the next notebook only:

| Reserved notebook | Purpose |
|---|---|
| `07_onnx_perch_sed_blend.ipynb` | Blend ONNX Perch and SED predictions only after both paths finish reliably |

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
