# BirdCLEF+ 2026

<p align="center">
  <img src="https://www.birds.cornell.edu/home/wp-content/uploads/2018/11/aab.jpg" alt="Cornell Lab bird soundscape" width="100%">
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/birdclef-2026">
    <img src="https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle competition">
  </a>
  <img src="https://img.shields.io/badge/Notebook-EDA%20%7C%20EffNet%20%7C%20Perch-F37626?logo=jupyter" alt="Notebook workflow">
  <img src="https://img.shields.io/badge/Public-EffNet%200.646%20%7C%20Perch%200.770-2EA44F" alt="Public scores">
</p>

BirdCLEF+ 2026 bioacoustic classification workspace with three Kaggle notebooks: EDA, EfficientNet-B0 baseline, and Google Perch v2 submission. The project focuses on dataset insight, CPU-safe inference, reproducible notebook execution, and clear model result reporting.

## 1. Competition Snapshot

The competition asks participants to identify wildlife species in Brazilian Pantanal soundscapes. During scoring, hidden `test_soundscapes/` are mounted and each 1-minute file is scored as **12 contiguous 5-second windows** with probability columns for the target species.

Current CPU submission results:

| Notebook | Public score | Role |
|---|---:|---|
| [2_bc2026_effnet_b0.ipynb](notebooks/2_bc2026_effnet_b0.ipynb) | **0.646** | Reliable PyTorch fallback |
| [3_bc2026_perch_v2.ipynb](notebooks/3_bc2026_perch_v2.ipynb) | **0.770** | Current lead submission |

Competition instructions and approach: [docs/1_instructions.md](docs/1_instructions.md).

## 2. Notebooks

| Notebook | Purpose |
|---|---|
| [1_bc2026_eda.ipynb](notebooks/1_bc2026_eda.ipynb) | Dataset audit, class imbalance, secondary labels, metadata bias, soundscape domain analysis, and spectrogram inspection |
| [2_bc2026_effnet_b0.ipynb](notebooks/2_bc2026_effnet_b0.ipynb) | EfficientNet-B0 training plus fast checkpoint-based submission mode |
| [3_bc2026_perch_v2.ipynb](notebooks/3_bc2026_perch_v2.ipynb) | Perch v2 probe training plus CPU-optimized submission mode |

Perch v2 is now the lead submission path after the successful CPU run. EfficientNet-B0 remains important as a simpler fallback and possible ensemble component.

## 3. Key EDA Findings

- **35,549** recordings across **206** primary labels.
- Complete taxonomy coverage for train labels: **206/206**.
- No duplicate rows in `train.csv`, `taxonomy.csv`, or `sample_submission.csv`.
- `train_soundscapes_labels.csv` deduplicates from **1,478** rows to **739** unique soundscape segments.
- Median class size is **125** recordings; the range is **1-499**.
- Top **30** labels account for **40.3%** of training recordings.
- Secondary labels include **161** distinct labels and **7,431** mentions.
- Deduplicated soundscape segments are strongly multi-label, with a median of **4** labels and a maximum of **10**.

Full analysis: [docs/3_eda_full_insights.md](docs/3_eda_full_insights.md).

## 4. Model Results

| Model | Representation | Best validation accuracy | Public score | Role |
|---|---|---:|---:|---|
| EfficientNet-B0 | 5-second mel-spectrogram | **0.5464** | **0.646** | Reliable fallback |
| Perch v2 probe | Frozen 1,536-d embeddings | **0.8392** | **0.770** | Lead submission |

Result notes:

- [EfficientNet-B0 results](docs/4_effnet_b0_results.md)
- [Perch v2 results](docs/5_perch_v2_results.md)

## 5. Submission

Use [3_bc2026_perch_v2.ipynb](notebooks/3_bc2026_perch_v2.ipynb) as the current primary submission:

```python
CFG.mode = "submission"
```

It loads the uploaded Perch probe artifact, uses the CPU Perch export, batches full 60-second soundscapes into 12 windows per file, and writes `/kaggle/working/submission.csv`.

Use [2_bc2026_effnet_b0.ipynb](notebooks/2_bc2026_effnet_b0.ipynb) as the fallback submission:

```python
CFG.mode = "submission"
```

Confirmed EffNet artifact directory:

```text
/kaggle/input/models/tuannm3812/irdclef-efficientnet-b0-artifacts/pytorch/default/1/effnet_b0
```

## 6. Repository Layout

```text
notebooks/
  1_bc2026_eda.ipynb
  2_bc2026_effnet_b0.ipynb
  3_bc2026_perch_v2.ipynb

docs/
  1_instructions.md
  2_coding_standards.md
  3_eda_full_insights.md
  4_effnet_b0_results.md
  5_perch_v2_results.md
  eda_artifacts/
```

Standards: [docs/2_coding_standards.md](docs/2_coding_standards.md).
