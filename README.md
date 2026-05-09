# BirdCLEF+ 2026

<p align="center">
  <img src="https://www.birds.cornell.edu/home/wp-content/uploads/2018/11/aab.jpg" alt="Cornell Lab bird soundscape" width="100%">
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/birdclef-2026">
    <img src="https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle competition">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch EfficientNet-B0">
  <img src="https://img.shields.io/badge/TensorFlow-Perch%20v2-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow Perch v2">
  <img src="https://img.shields.io/badge/Notebooks-3%20Kaggle--ready-F37626?logo=jupyter&logoColor=white" alt="Three Kaggle-ready notebooks">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Project status">
</p>

Professional Kaggle workspace for the BirdCLEF+ 2026 bioacoustic classification competition. The repo separates exploratory notebooks, Kaggle-ready training notebooks, reusable Python modules, and command-line scripts for repeatable experiments.

## Project Structure

```text
configs/                 Experiment configuration files
data/                    Local data mount, ignored by Git
models/                  Local checkpoints, ignored by Git
notebooks/               Curated Kaggle notebooks
notebooks/archive/       Original reference notebooks
outputs/                 Local experiment outputs, ignored by Git
reports/eda/             Kaggle EDA artifacts and written insights
scripts/                 CLI utilities and notebook generator
src/birdclef2026/        Reusable Python package
```

The canonical Kaggle notebooks are:

- `notebooks/01_data_eda.ipynb`
- `notebooks/02_effnet_b0_baseline.ipynb`
- `notebooks/03_perch_v2_probe.ipynb`

EDA artifacts and summarized findings are available in [reports/eda](reports/eda).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,perch]"
```

If you only want the EfficientNet baseline, `pip install -e ".[dev]"` is enough.

## Data Layout

Put the Kaggle competition files under `data/raw/birdclef-2026/`:

```text
data/raw/birdclef-2026/
  train.csv
  taxonomy.csv
  train_soundscapes_labels.csv
  sample_submission.csv
  train_audio/
  train_soundscapes/
  test_soundscapes/
```

On Kaggle, the notebooks auto-detect common dataset mount paths. For scripts, pass the Kaggle input directory explicitly:

```bash
python scripts/train_effnet.py --config configs/effnet_b0.yaml --data-root /kaggle/input/birdclef-2026 --output-dir /kaggle/working/outputs/effnet_b0
```

## Workflow

Create cross-validation folds:

```bash
python scripts/prepare_folds.py --config configs/effnet_b0.yaml
```

Run quick EDA tables:

```bash
python scripts/run_eda.py --config configs/effnet_b0.yaml
```

Train the EfficientNet-B0 mel-spectrogram baseline:

```bash
python scripts/train_effnet.py --config configs/effnet_b0.yaml
```

Extract Perch embeddings when you have a local or Kaggle SavedModel path:

```bash
python scripts/extract_perch_embeddings.py --config configs/perch_probe.yaml --perch-model-dir /kaggle/input/perch-v2/saved_model
```

Train the shallow probe on extracted embeddings:

```bash
python scripts/train_perch_probe.py --config configs/perch_probe.yaml
```

## Notes

- `data/`, `models/`, and `outputs/` are intentionally ignored by Git.
- Config values live in `configs/*.yaml`; avoid hard-coding Kaggle paths inside scripts.
- The EfficientNet baseline trains on `primary_label` as a clean starting point.
- The Perch v2 path extracts frozen embeddings and trains a shallow classification probe.
