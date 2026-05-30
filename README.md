# BirdCLEF+ 2026

<p align="center">
  <img src="https://www.birds.cornell.edu/home/wp-content/uploads/2018/11/aab.jpg" alt="Cornell Lab bird soundscape" width="100%">
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/birdclef-2026">
    <img src="https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle competition">
  </a>
  <img src="https://img.shields.io/badge/Notebook-EDA%20%7C%20EffNet%20%7C%20Perch%20%7C%20ONNX%20SED-F37626?logo=jupyter" alt="Notebook workflow">
  <img src="https://img.shields.io/badge/Public-EffNet%200.646%20%7C%20Perch%200.770%20%7C%20ONNX%20SED%200.822-2EA44F" alt="Public scores">
</p>

BirdCLEF+ 2026 bioacoustic classification workspace for identifying wildlife
species in Brazilian Pantanal soundscapes. The project documents the path from
dataset understanding to a reliable EfficientNet-B0 baseline, a stronger Google
Perch v2 transfer-learning model, and the next ONNX-based submission lane.

## 1. Competition Snapshot

The competition asks participants to identify wildlife species in Brazilian Pantanal soundscapes. During scoring, hidden `test_soundscapes/` are mounted and each 1-minute file is scored as **12 contiguous 5-second windows** with probability columns for the target species.

The task is multi-taxon rather than bird-only. The target set includes birds,
amphibians, mammals, reptiles, and insects, so the problem is closer to
ecosystem soundscape recognition than ordinary single-species bird-call
classification.

## 2. Key Highlights

| Area | Result |
|---|---|
| Dataset scale | **35,549** training recordings across **206** primary labels |
| Output contract | **234** species probability columns |
| Soundscape labels | **739** deduplicated labeled 5-second windows |
| EfficientNet-B0 | **0.5464** best validation accuracy, **0.646** public score |
| Perch v2 probe | **0.8392** best validation accuracy, **0.770** public score |
| ONNX distilled SED | **0.822** public score |
| Best gain | ONNX distilled SED improved public score by **+0.176** over EfficientNet-B0 |

Project overview and approach: [docs/01_project_overview.md](docs/01_project_overview.md).

## 3. What We Achieved

- Built a structured EDA notebook that audits label quality, class imbalance,
  metadata shift, soundscape overlap, and representative spectrograms.
- Trained a compact EfficientNet-B0 mel-spectrogram baseline that remains a
  dependable PyTorch fallback.
- Trained a shallow classifier on frozen **1,536-dimensional** Google Perch v2
  embeddings, producing the first strong transfer-learning result.
- Promoted a distilled SED ONNX submission path that is now the best current
  public score in the repo.
- Optimized the Perch scoring path for CPU by reading full 60-second
  soundscapes once and reshaping them into **12** contiguous 5-second windows.
- Added lightweight result documentation so EDA findings, model behavior, and
  next experiments are easy to review without opening every notebook.

## 4. Lessons Learned

- Foundation bioacoustic features transfer much better than a small CNN trained
  from scratch for this dataset.
- Clean-clip validation is useful but incomplete; labeled soundscape windows are
  closer to hidden scoring and should guide calibration.
- Class imbalance is severe enough that aggregate accuracy can hide weak rare
  labels and non-bird taxa.
- Secondary labels and co-occurrence patterns are valuable, but they should be
  introduced after the single-label pipeline is stable.
- CPU-safe inference matters as much as model quality because Kaggle scoring
  depends on finishing hidden soundscape inference inside the runtime limit.

## 5. Notebooks

| Notebook | Purpose |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Dataset audit, class imbalance, secondary labels, metadata bias, soundscape domain analysis, and spectrogram inspection |
| [02_effnet_b0.ipynb](notebooks/02_effnet_b0.ipynb) | EfficientNet-B0 training plus fast checkpoint-based submission mode |
| [03_perch_v2_train.ipynb](notebooks/03_perch_v2_train.ipynb) | Perch v2 probe training, diagnostics, calibration, and artifact packaging |
| [04_perch_v2_submit.ipynb](notebooks/04_perch_v2_submit.ipynb) | Lean Perch v2 scoring notebook for CPU Kaggle submission |
| [05_onnx_sed_submit.ipynb](notebooks/05_onnx_sed_submit.ipynb) | Protected distilled SED ONNX champion submission |
| [06_onnx_perch_speed_test.ipynb](notebooks/06_onnx_perch_speed_test.ipynb) | Active ONNX Perch runtime experiment |

ONNX distilled SED is now the protected champion. Perch v2 version 14 remains a
valuable protected baseline, and EfficientNet-B0 remains the simpler fallback.
New leaderboard work should continue through the ONNX lane: ONNX Perch speed
testing next, then blending only if both standalone paths complete reliably.

## 6. Key EDA Findings

- **35,549** recordings across **206** primary labels.
- Complete taxonomy coverage for train labels: **206/206**.
- No duplicate rows in `train.csv`, `taxonomy.csv`, or `sample_submission.csv`.
- `train_soundscapes_labels.csv` deduplicates from **1,478** rows to **739** unique soundscape segments.
- Median class size is **125** recordings; the range is **1-499**.
- Top **30** labels account for **40.3%** of training recordings.
- Secondary labels include **161** distinct labels and **7,431** mentions.
- Deduplicated soundscape segments are strongly multi-label, with a median of **4** labels and a maximum of **10**.

Full analysis: [docs/03_eda_insights.md](docs/03_eda_insights.md).

## 7. Model Results

| Model | Representation | Best validation accuracy | Public score | Role |
|---|---|---:|---:|---|
| EfficientNet-B0 | 5-second mel-spectrogram | **0.5464** | **0.646** | Reliable fallback |
| Perch v2 probe | Frozen 1,536-d embeddings | **0.8392** | **0.770** | Protected baseline |
| ONNX distilled SED | 5-fold SED ONNX student | N/A | **0.822** | Current champion |

Successful Kaggle submissions to preserve:

| Submission | Public score | Current role | Next action |
|---|---:|---|---|
| EfficientNet-B0 version 9 | **0.646** | CPU-safe fallback | Keep exact notebook and artifact inputs unchanged |
| Perch v2 version 14 | **0.770** | Protected baseline | Keep as reference while moving new work to ONNX |
| ONNX distilled SED version 1 | **0.822** | Current champion | Preserve and test ONNX Perch speed next |

Result notes:

- [EfficientNet-B0 results](docs/04_effnet_b0_results.md)
- [Perch v2 results](docs/05_perch_v2_results.md)
- [Distilled SED review](docs/07_distilled_sed_review.md)
- [ProtoSSM review](docs/08_protossm_review.md)
- [ONNX SED results](docs/09_onnx_sed_results.md)
- [ONNX Perch speed results](docs/10_onnx_perch_speed_results.md)
- [Next steps](docs/06_next_steps.md)

## 8. Repository Layout

```text
notebooks/
  README.md
  01_eda.ipynb
  02_effnet_b0.ipynb
  03_perch_v2_train.ipynb
  04_perch_v2_submit.ipynb
  05_onnx_sed_submit.ipynb
  06_onnx_perch_speed_test.ipynb

docs/
  01_project_overview.md
  02_coding_standards.md
  03_eda_insights.md
  04_effnet_b0_results.md
  05_perch_v2_results.md
  06_next_steps.md
  07_distilled_sed_review.md
  08_protossm_review.md
  09_onnx_sed_results.md
  10_onnx_perch_speed_results.md
  figures/eda/
```

Standards: [docs/02_coding_standards.md](docs/02_coding_standards.md).
Notebook promotion rules: [notebooks/README.md](notebooks/README.md).
