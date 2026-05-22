# BirdCLEF+ 2026

<p align="center">
  <img src="https://www.birds.cornell.edu/home/wp-content/uploads/2018/11/aab.jpg" alt="Cornell Lab bird soundscape" width="100%">
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/birdclef-2026">
    <img src="https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle competition">
  </a>
  <img src="https://img.shields.io/badge/Notebook-EDA%20%7C%20EffNet%20%7C%20Perch-F37626?logo=jupyter" alt="Notebook workflow">
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch" alt="PyTorch EfficientNet-B0">
  <img src="https://img.shields.io/badge/TensorFlow-Perch%20v2-FF6F00?logo=tensorflow" alt="TensorFlow Perch v2">
</p>

BirdCLEF+ 2026 bioacoustic classification workspace with three Kaggle notebooks: EDA, EfficientNet-B0 submission baseline, and Google Perch v2 probe. The project focuses on dataset insight, reproducible notebook execution, and clear model result reporting.

## 1. Notebooks

| Notebook | Purpose |
|---|---|
| [1_bc2026_eda.ipynb](notebooks/1_bc2026_eda.ipynb) | Dataset audit, class imbalance, secondary labels, metadata bias, soundscape domain analysis, and spectrogram inspection |
| [2_bc2026_effnet_b0.ipynb](notebooks/2_bc2026_effnet_b0.ipynb) | EfficientNet-B0 training plus fast checkpoint-based submission mode |
| [3_bc2026_perch_v2.ipynb](notebooks/3_bc2026_perch_v2.ipynb) | Perch v2 embedding extraction, diagnostics, and teacher-signal analysis |

EfficientNet-B0 is the main submission path because it is fast, pure PyTorch, and configured for Kaggle runs without internet access. Perch v2 gives stronger validation features and is most useful as an offline teacher or feature source.

## 2. Key EDA Findings

- **35,549** recordings across **206** primary labels.
- Complete taxonomy coverage for train labels: **206/206**.
- No duplicate rows in `train.csv`, `taxonomy.csv`, or `sample_submission.csv`.
- `train_soundscapes_labels.csv` deduplicates from **1,478** rows to **739** unique soundscape segments.
- Median class size is **125** recordings; the range is **1-499**.
- Top **30** labels account for **40.3%** of training recordings.
- Secondary labels include **161** distinct labels and **7,431** mentions.
- Only **847** recordings (**2.38%**) fall inside the rough Pantanal geography box used in EDA.
- Deduplicated soundscape segments are strongly multi-label, with a median of **4** labels and a maximum of **10**.
- The EDA notebook includes deeper soundscape coverage checks: site/hour concentration, partially labeled files, soundscape-only species, taxonomic activity by hour, and label co-occurrence.

Full analysis: [docs/eda_full_insights.md](docs/eda_full_insights.md).

## 3. Model Results

| Model | Representation | Epochs | Best validation accuracy | Role |
|---|---|---:|---:|---|
| EfficientNet-B0 | 5-second mel-spectrogram | 5 observed / 15 max | **0.5273** | Reliable submission baseline |
| Perch v2 probe | Frozen 1,536-d embeddings | 7 observed / 20 max | **0.8392** | Stronger feature experiment and teacher candidate |

Result notes:

- [EfficientNet-B0 results](docs/effnet_b0_results.md)
- [Perch v2 results](docs/perch_v2_results.md)

Perch v2 requires a compatible TensorFlow runtime. Notebook 3 expects TensorFlow 2.20 wheels from `/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0` and a local Perch SavedModel input such as `/kaggle/input/datasets/jaejohn/perch-meta` when internet is off.

## 4. Submission

Use [2_bc2026_effnet_b0.ipynb](notebooks/2_bc2026_effnet_b0.ipynb) for competition submission. For experiments, keep `CFG.mode = "train"` to produce `best_effnet_b0.pt` and `labels.json`. For scored reruns, attach that artifact dataset, set `CFG.mode = "submission"`, and run only checkpoint loading plus inference.

Perch v2 is not the scored submission path because TensorFlow setup and embedding extraction exceed the competition runtime budget. Use it as a teacher, diagnostic model, or distillation source. The public sample submission has only **3** rows, so a successful public run confirms notebook mechanics but does not prove hidden-test runtime.

## 5. Repository Layout

```text
notebooks/
  1_bc2026_eda.ipynb
  2_bc2026_effnet_b0.ipynb
  3_bc2026_perch_v2.ipynb

docs/
  coding_standards.md
  eda_full_insights.md
  effnet_b0_results.md
  perch_v2_results.md
  eda_artifacts/
```

Standards: [docs/coding_standards.md](docs/coding_standards.md).
