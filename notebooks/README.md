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

## Next Notebook Slot

Use **one** active notebook slot at a time:

| Planned notebook | Purpose |
|---|---|
| `05_onnx_sed_submit.ipynb` | Reproduce a fast ONNX distilled SED submission baseline |

Do not add separate notebooks for every public reference. Review external
notebooks in `docs/`, then promote only the cleaned project-owned version.

## Promotion Rules

1. A notebook must have one clear mode: EDA, training, or submission.
2. A submission notebook must avoid training and heavy diagnostics.
3. Public-reference notebooks stay in `docs/*_review.md`, not in this folder.
4. Protected baselines should not be overwritten by experiments.
5. New experiments should update `docs/06_next_steps.md` before another notebook
   is added.

