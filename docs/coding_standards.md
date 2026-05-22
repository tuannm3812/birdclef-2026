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

1. `1_bc2026_eda.ipynb`
2. `2_bc2026_effnet_b0.ipynb`
3. `3_bc2026_perch_v2.ipynb`

Notebook names should describe the actual Kaggle workflow. Do not split training and submission into separate notebooks when the competition flow is meant to run end-to-end.

## 3. Notebook Style

Each notebook should include:

- A short purpose statement at the top.
- A clear `CFG` section for tunable values.
- Kaggle path auto-detection where practical.
- Markdown insight cells after important plots or metrics.
- Artifact-writing cells for reusable outputs such as `submission.csv`, histories, or plots.

Prefer readable, self-contained notebook code over imports from local project modules. Kaggle should be able to run the notebook after attaching only the required competition datasets and model inputs.

When notebook code changes, clear all outputs before committing and rerun the notebook on Kaggle to regenerate trusted outputs. Keep committed notebooks lightweight; Kaggle is the execution record.

Competition notebooks should not depend on internet access during final reruns. For pretrained models, attach local Kaggle input weights and load them explicitly; do not rely on runtime downloads from Hugging Face, timm, or other external hubs.

## 4. Plot Style

Use the Viridis palette as the default visual language across notebooks:

- Use `sns.color_palette("viridis", ...)` for categorical or sequential accents.
- Use `"viridis"` as the default colormap for heatmaps and spectrogram-like plots.
- Change color palettes only when a specific chart needs clearer contrast, semantic coloring, or accessibility improvement.
- Keep chart titles short and analytical; avoid decorative styling.

## 5. Documentation Style

Documentation should be written for a competition reviewer or teammate who wants the reasoning quickly:

- Use numbered sections.
- Lead with findings and implications.
- Include exact metrics when available.
- Link notebooks and docs with relative paths.
- Keep model result pages separate by model.
- Keep broad narrative in the root `README.md`; keep detailed evidence in focused docs.

## 6. Git Hygiene

Do not commit:

- Raw Kaggle audio.
- Local checkpoints.
- Kaggle working directories.
- Large embedding arrays.
- Python caches or notebook checkpoints.
- Ad hoc experiment dumps.

Commit lightweight artifacts only when they directly support the written analysis, such as figures used by the EDA markdown and model result pages.
