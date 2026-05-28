# Next Steps

## 1. Current Position

Perch v2 is the lead path with a **0.770** public score, while
EfficientNet-B0 remains the reliable PyTorch fallback at **0.646**. The next
work should improve Perch calibration and soundscape behavior before spending
time on larger models.

The main risk is optimizing clean-clip validation accuracy while hidden scoring
happens on noisy 1-minute soundscapes split into 5-second windows. Every new
experiment should therefore report both normal validation metrics and at least
one soundscape-like diagnostic.

## 2. Priority Experiments

### 2.1 Add Perch Soundscape Priors

Status: implemented in `03_perch_v2.ipynb`; needs a fresh Kaggle train run and
leaderboard validation.

Goal: use labeled soundscape structure without overfitting it.

Work items:

1. Deduplicate `train_soundscapes_labels.csv` by soundscape file and window.
2. Build simple prior tables for species by hour, site, and co-occurrence.
3. Apply small logit offsets to Perch predictions.
4. Tune offsets on held-out soundscape-like windows, not on leaderboard alone.

Success signal:

- Public score improves or stays stable while per-window soundscape diagnostics
  improve.
- No large boost for species with tiny or unreliable support.

Deliverables:

- Updated `03_perch_v2.ipynb`.
- A small prior summary table saved during training mode.
- A note added to `05_perch_v2_results.md`.

### 2.2 Inspect Weak Labels

Status: implemented in `03_perch_v2.ipynb` via
`weak_label_diagnostics.csv`; needs review after the next training run.

Goal: identify where Perch is confidently wrong or missing rare/non-bird taxa.

Work items:

1. Use `per_class_validation_metrics.csv` from the Perch notebook.
2. Rank labels by low top-1 recall, low top-5 recall, and high confidence error.
3. Cross-check weak labels against class count, taxonomic group, rating, and
   secondary-label frequency.
4. Create a short table of the top action labels.

Success signal:

- Clear split between calibration problems, class-confusion problems, and data
  coverage problems.

Deliverables:

- Add a weak-label section to `05_perch_v2_results.md`.
- Optional figure under `docs/figures/perch/`.

### 2.3 Test Lightweight Calibration

Status: implemented in `03_perch_v2.ipynb` via `temperature_grid.csv` and
`calibration.json`; needs a controlled submission test.

Goal: improve probability ranking without retraining the embedding model.

Work items:

1. Add temperature scaling for Perch probe logits.
2. Test class-level bias terms based on validation prevalence.
3. Compare global calibration versus rare-class-specific calibration.
4. Keep submission output as probabilities for all target columns.

Success signal:

- Better validation log loss or ranking proxy.
- Public score improves without slower inference.

Deliverables:

- Calibration config in `03_perch_v2.ipynb`.
- Updated result table in `05_perch_v2_results.md`.

### 2.4 Compare Perch And EfficientNet Errors

Goal: decide whether an ensemble is worth the CPU cost.

Work items:

1. Save compatible validation predictions from both notebooks.
2. Join predictions by validation row.
3. Measure cases where EfficientNet is right and Perch is wrong.
4. Estimate ensemble gain before implementing submission blending.

Success signal:

- EfficientNet adds meaningful unique wins, especially on labels Perch misses.

Deliverables:

- Error-overlap table in either `04_effnet_b0_results.md` or
  `05_perch_v2_results.md`.
- If useful, a simple weighted-average submission path.

### 2.5 Distill Perch Into A Faster Student

Goal: reduce dependency on TensorFlow Perch inference if scoring runtime becomes
fragile.

Work items:

1. Generate Perch soft labels for clean clips and selected soundscape windows.
2. Train an EfficientNet or small audio CNN student on hard plus soft targets.
3. Compare runtime, validation score, and public score against both existing
   notebooks.

Success signal:

- Student closes part of the gap to Perch while keeping PyTorch-only inference.

Deliverables:

- New notebook only if the workflow becomes distinct enough, likely
  `04_perch_distillation.ipynb`.
- Result document only after a successful Kaggle run.

## 3. Recommended Order

1. Soundscape priors for Perch.
2. Weak-label inspection.
3. Lightweight calibration.
4. Perch/EfficientNet error overlap.
5. Distillation only if runtime or ensemble cost becomes a blocker.

## 4. Guardrails

- Do not optimize only against the public leaderboard.
- Do not add heavy test-time augmentation until CPU runtime is measured.
- Do not use raw soundscape label counts without deduplication.
- Do not split train and submission notebooks unless Kaggle execution becomes
  hard to manage.
- Keep attached model artifacts and wheelhouses outside git.
