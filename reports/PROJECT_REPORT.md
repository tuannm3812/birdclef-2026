# BirdCLEF+ 2026 Project Report

## Abstract

This project develops a reproducible Kaggle workflow for the BirdCLEF+ 2026 bioacoustic classification competition. The repository organizes exploratory data analysis, baseline modeling, Perch v2 transfer-learning experiments, and submission notebooks into a single professional workspace. The analysis shows that the dataset contains 35,549 training recordings across 206 primary classes, with substantial class imbalance, uneven metadata quality, source concentration, and a distinct soundscape domain. Two modeling directions were implemented: a competition-safe EfficientNet-B0 mel-spectrogram baseline and a higher-performing Google Perch v2 embedding probe. The EfficientNet-B0 model reached 0.7318 validation accuracy and was used as the dependable submission path. The Perch v2 probe reached 0.8403 validation accuracy on frozen embeddings, indicating strong representational value, but direct hidden-test inference is constrained by Kaggle runtime and TensorFlow/CUDA compatibility.

## 1. Project Objectives

The project has four primary objectives:

1. Establish a clean and repeatable repository structure for BirdCLEF+ 2026 experimentation.
2. Produce an exploratory analysis that identifies data quality issues, class imbalance, domain shift, and modeling implications.
3. Implement a fast baseline model that can train and submit reliably on Kaggle.
4. Evaluate Perch v2 embeddings as a stronger transfer-learning representation while documenting operational limitations.

The resulting workflow is intended to support both competition submission and later research iteration.

## 2. Repository Design

The repository separates exploratory notebooks, reusable source code, model artifacts, and written reports:

- `notebooks/01_data_eda.ipynb`: Kaggle-ready exploratory data analysis.
- `notebooks/02_effnet_b0_baseline.ipynb`: EfficientNet-B0 training notebook.
- `notebooks/03_perch_v2_probe.ipynb`: Perch v2 embedding and probe notebook.
- `notebooks/04_effnet_b0_submission.ipynb`: EfficientNet-B0 submission notebook.
- `notebooks/05_perch_v2_submission.ipynb`: Perch v2 probe submission experiment.
- `src/birdclef2026/`: reusable Python package for audio, data, models, and configuration.
- `scripts/`: command-line utilities and notebook generation script.
- `reports/eda/`: lightweight EDA outputs and written insight summary.
- `artifacts/`: committed model metadata and lightweight trained artifacts.

This design keeps Kaggle notebooks usable while avoiding notebook-only experimentation. The notebook generator, `scripts/create_kaggle_notebooks.py`, is the source of truth for notebook content.

## 3. Dataset Summary

The EDA run identified the following dataset characteristics:

- Training metadata contains 35,549 recordings across 206 primary labels.
- All 35,549 training audio paths were present during the Kaggle run.
- The taxonomy file contains 234 labels, including all 206 training labels and 28 labels not present in train.
- The public sample submission contains 3 rows, indicating that public notebook execution is mainly a smoke test for the hidden evaluation pipeline.
- `train.csv`, `taxonomy.csv`, and `sample_submission.csv` contain no duplicate rows.
- `train_soundscapes_labels.csv` contains 1,478 rows but deduplicates to 739 unique soundscape segments.

The soundscape duplication result is operationally important. Clean clip metadata appears structurally safe, but soundscape prevalence and overlap analysis should be performed on deduplicated rows.

## 4. Exploratory Findings

### 4.1 Class Imbalance

The primary-label distribution is long-tailed:

- Median recordings per class: 125.
- Minimum recordings per class: 1.
- Maximum recordings per class: 499.
- Top 10 labels account for 13.9% of recordings.
- Top 30 labels account for 40.3% of recordings.
- Four classes are singletons.

The largest classes are near an apparent cap of approximately 500 recordings. Therefore, the main imbalance risk is not one dominant class but the separation between capped head classes and rare tail classes. This motivates per-class validation metrics, class-aware sampling, and rare-class augmentation.

### 4.2 Secondary Labels

Secondary labels provide a noisy but useful multi-label signal:

- 161 distinct secondary labels appear.
- There are 7,431 total secondary-label mentions.
- The most frequent secondary labels include `grekis`, `whtdov`, `undtin1`, `yecpar`, and `rufhor2`.

The baseline model treats `primary_label` as the training target for clarity and submission reliability. Secondary labels are best reserved for future soft-label training, mixup target construction, co-occurrence priors, or confusion analysis.

### 4.3 Metadata Quality And Source Bias

Recording quality is uneven. The EDA found 12,849 recordings with rating `0.0`, while ratings `4.0` and `5.0` together account for 14,863 recordings. Aves dominate the dataset with 34,799 rows and median rating 3.5, while Amphibia, Mammalia, Insecta, and Reptilia are substantially smaller and tend to have lower ratings.

Source fields also show concentration:

- Collection split: XC has 23,043 recordings and iNat has 12,506.
- The most common license, `by-nc-sa`, accounts for 22,843 recordings.
- The `type` field is empty for 12,975 rows.
- The top author contributes 2,874 recordings.

These fields may act as hidden confounders. Validation that ignores author, source, geography, or recording conditions can overstate generalization.

### 4.4 Geography And Soundscape Domain

All training rows include coordinates. Only 847 recordings, or 2.38%, fall inside the rough Pantanal box used in the EDA, though these rows cover 119 species. This suggests that geography should be treated as a possible domain-shift variable.

The deduplicated soundscape labels are strongly multi-label:

- 739 unique labeled segments.
- Mean labels per segment: 4.22.
- Median labels per segment: 4.
- 90th percentile: 7 labels.
- Maximum: 10 labels.

Labeled soundscape segments cluster most strongly around evening and night hours, especially 20:00-23:00. This supports hour-aware validation diagnostics and threshold calibration.

## 5. Modeling Methodology

### 5.1 EfficientNet-B0 Baseline

The baseline converts each recording into a 5-second mono mel-spectrogram and trains an EfficientNet-B0 classifier over 206 primary labels. The approach prioritizes:

- fast Kaggle execution,
- pure PyTorch deployment,
- straightforward artifact packaging,
- reproducible fold assignment,
- compatibility with competition submission constraints.

The committed artifact includes `best_effnet_b0.pt`, `labels.json`, `history.csv`, and fold metadata.

### 5.2 Perch v2 Probe

The Perch v2 workflow uses frozen Google Perch embeddings of dimension 1,536 and trains a shallow PyTorch probe. This separates representation learning from classifier fitting and tests whether a bioacoustic foundation model provides stronger features than a small supervised baseline.

The embedding matrix inspected during training has shape 35,549 x 1,536. It was not committed because the `.npz` artifact is approximately 202 MB. The committed probe artifact includes `best_perch_probe.pt`, `labels.json`, and training history.

## 6. Results

| Model | Representation | Epochs | Best validation accuracy | Notes |
|---|---|---:|---:|---|
| EfficientNet-B0 | 5-second mel-spectrogram | 5 | 0.7318 | Reliable Kaggle submission path |
| Perch v2 probe | Frozen 1,536-d Perch embeddings | 10 | 0.8403 | Stronger validation performance, but operationally constrained for hidden inference |

The Perch probe outperforms the EfficientNet-B0 baseline by approximately 10.9 percentage points in validation accuracy. However, its hidden-test submission path is less reliable because direct Perch SavedModel inference on Kaggle can be slow and sensitive to TensorFlow/XLA/CUDA compatibility. Consequently, the EfficientNet-B0 model remains the primary competition-safe artifact, while Perch v2 is most useful as an offline teacher, feature generator, or later distillation source.

## 7. Submission Considerations

EfficientNet-B0 is the preferred current submission path because it avoids TensorFlow installation issues and runs within practical competition limits. The Perch v2 submission notebook is retained as an experimental pathway, but it should be used cautiously unless the hidden-test runtime is proven acceptable or cached embeddings are available for the evaluated rows.

The public sample submission has only 3 rows, so successful local or public dry-run execution does not guarantee hidden-test feasibility. This is especially important for heavy feature extraction workflows.

## 8. Limitations

The current project has several limitations:

- Validation accuracy is not a complete proxy for competition score, particularly under soundscape domain shift.
- The EfficientNet baseline uses primary labels only and does not yet exploit secondary labels.
- Soundscape annotations are treated diagnostically, not as a fully integrated training objective.
- Perch v2 direct inference is operationally constrained on Kaggle.
- Metadata confounders such as author, collection, quality rating, and geography are analyzed but not yet fully incorporated into validation or training.

## 9. Recommended Next Steps

1. Add per-class validation metrics and confusion analysis.
2. Introduce class-aware sampling or rare-class augmentation.
3. Add multi-crop inference for EfficientNet-B0.
4. Explore secondary-label soft targets after the baseline remains stable.
5. Use soundscape hour and label-overlap patterns for threshold calibration.
6. Test knowledge distillation from Perch embeddings into a faster PyTorch model.
7. Create validation splits that account for source, geography, and recording groups.

## 10. Conclusion

This project establishes a professional, reproducible BirdCLEF+ 2026 workflow with clear separation between data analysis, baseline modeling, transfer-learning experiments, and submission packaging. The EDA identifies the main risks for model generalization: class imbalance, duplicated soundscape labels, metadata quality variation, source concentration, and soundscape domain shift. The EfficientNet-B0 baseline provides a dependable submission-ready model, while the Perch v2 probe demonstrates the value of bioacoustic foundation embeddings for future improvement. The most promising next phase is not simply scaling the backbone, but improving validation fidelity, calibration, rare-class handling, and efficient use of Perch-derived knowledge.
