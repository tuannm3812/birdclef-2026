# Coding Standards

## 1. Repository Scope

This repository is intentionally notebook-first. Kaggle notebooks are the executable source of truth, while `docs/` captures analysis, model results, and project decisions.

Keep the root small:

- `notebooks/` for Kaggle notebooks.
- `docs/` for reports, results, and supporting EDA artifacts.
- `README.md` for the high-level project overview.

Avoid adding local-only folders such as `data/`, `models/`, `outputs/`, `configs/`, or `scripts/` unless the project direction changes back to local training.

## 2. Notebook Naming

Use numbered, stable notebook names:

1. `01_eda.ipynb`
2. `02_effnet_b0.ipynb`
3. `03_perch_v2_train.ipynb`
4. `04_perch_v2_submit.ipynb`
5. `05_onnx_sed_submit.ipynb`

Notebook names should describe the actual Kaggle workflow. Keep training and submission together for simple baselines, but split them when artifact management, runtime constraints, or Kaggle scoring safety make the responsibilities meaningfully different.

Reserve new numbers for promoted project-owned workflows only. The current
planned order is `06_onnx_perch_speed_test.ipynb` after ONNX SED succeeds, then
`07_onnx_perch_sed_blend.ipynb` only after both standalone paths run
successfully.

Use numbered documentation filenames so the reading order is obvious:

1. `01_project_overview.md`
2. `02_coding_standards.md`
3. `03_eda_insights.md`
4. `04_effnet_b0_results.md`
5. `05_perch_v2_results.md`
6. `06_next_steps.md`

## 3. Code Style

Follow **PEP 8** for Python code:

- Use **4 spaces** for indentation.
- Keep lines to **79 characters or fewer** where practical.
- Prefer **concise, optimized syntax** such as list comprehensions, f-strings, and small utility functions when they improve readability.
- Add **type hints** for functions and class methods when the type is clear.
- Group imports in this order:
  1. Standard library
  2. Third-party libraries
  3. Local modules
- Separate import groups with a blank line.

Use **Google-style docstrings** for reusable functions and classes:

```python
def func(x: int) -> int:
    """One-line summary.

    Args:
        x (int): Description.

    Returns:
        int: Description.
    """
```

Add short inline comments only when they explain **why** a decision was made. Avoid comments that restate what the code already says.

## 4. Notebook Style

Each notebook should include:

- A short purpose statement at the top.
- A clear `CFG` section for tunable values.
- Explicit mode flags when runtime behavior differs between **training** and **submission**.
- Kaggle path auto-detection where practical.
- Markdown insight cells after important plots or metrics.
- Artifact-writing cells for reusable outputs such as `submission.csv`, histories, or plots.

Prefer readable, self-contained notebook code over imports from local project modules. Kaggle should be able to run the notebook after attaching only the required competition datasets and model inputs.

When notebook code changes, clear all outputs before committing and rerun the notebook on Kaggle to regenerate trusted outputs. Keep committed notebooks lightweight; Kaggle is the execution record.

Competition notebooks should not depend on internet access during final reruns. For pretrained models, attach local Kaggle input weights and load them explicitly; do not rely on runtime downloads from Hugging Face, timm, or other external hubs.

For runtime packages that differ from the Kaggle image, prefer attached wheelhouse datasets. If an exploratory notebook allows internet installation, gate it behind an explicit config flag and keep the default offline-safe.

Submission notebooks must be optimized for Kaggle scoring limits. Do not run EDA, model training, TensorFlow embedding extraction, or large artifact creation in a scored submission path. Attach trained checkpoints or cached features as Kaggle inputs and keep submission mode focused on loading, inference, and writing `submission.csv`.

## 5. Plot Style

Use the Viridis palette as the default visual language across notebooks:

- Use `sns.color_palette("viridis", ...)` for categorical or sequential accents.
- Use `"viridis"` as the default colormap for heatmaps and spectrogram-like plots.
- Change color palettes only when a specific chart needs clearer contrast, semantic coloring, or accessibility improvement.
- Keep chart titles short and analytical; avoid decorative styling.

## 6. Documentation Style

Documentation should be written for a competition reviewer or teammate who wants the reasoning quickly:

- Use numbered sections.
- Use numbered markdown filenames for stable reading order.
- Lead with findings and implications.
- Include exact metrics when available.
- Link notebooks and docs with relative paths.
- Keep model result pages separate by model.
- Keep broad narrative in the root `README.md`; keep detailed evidence in focused docs.
- Store documentation figures under `docs/figures/`.

## 7. Git Hygiene

Do not commit:

- Raw Kaggle audio.
- Local checkpoints.
- Kaggle working directories.
- Large embedding arrays.
- Python caches or notebook checkpoints.
- Ad hoc experiment dumps.

Commit lightweight artifacts only when they directly support the written analysis, such as figures used by the EDA markdown and model result pages.
