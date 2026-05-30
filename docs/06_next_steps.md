# Next Steps

## 1. Current Position

ONNX Perch + SED is the protected champion with a **0.890** public score. ONNX
distilled SED remains the strongest simpler baseline at **0.822**, Perch v2 is
the strongest older baseline at **0.770**, and EfficientNet-B0 remains the
reliable PyTorch fallback at **0.646**.

The main risk is hidden-test runtime. Public notebook runs can pass with only
three sample rows and no test audio, while real submissions must score hidden
soundscapes. Every new experiment should therefore report validation quality and
submission runtime risk.

## 2. Priority Experiments

### 2.1 Freeze The Two Working Baselines

Status: urgent.

Goal: keep reproducible copies of the only two successful submissions.

Work items:

1. Preserve the exact EfficientNet-B0 version 9 notebook and artifact inputs.
2. Preserve the exact Perch v2 version 14 notebook and artifact inputs.
3. Record attached Kaggle inputs, artifact versions, public scores, and runtime.
4. Avoid overwriting those notebooks while testing new variants.

Success signal:

- We can rerun or restore both baselines without guessing paths or versions.

Deliverables:

- Add a baseline registry table to `README.md` or `05_perch_v2_results.md`.
- Keep experimental submission notebooks as separate versions, not replacements.

### 2.2 Preserve Distilled SED ONNX Champion

Status: succeeded with **0.822** public score.

Goal: keep the new best submission reproducible before adding Perch back into
the scoring path.

Work items:

1. Preserve `05_onnx_sed_submit.ipynb`.
2. Record attached Kaggle inputs and notebook version.
3. Avoid changing the champion notebook while testing ONNX Perch.
4. Document the result in `docs/09_onnx_sed_results.md`.

Success signal:

- We can restore the **0.822** path without guessing inputs or code changes.

Deliverables:

- `docs/09_onnx_sed_results.md`.

### 2.3 Test ONNX Perch Speed

Status: succeeded in `06_onnx_perch_speed_test.ipynb`.

Goal: measure whether ONNX Perch can replace the slower TensorFlow Perch
submission path.

Work items:

1. Run `06_onnx_perch_speed_test.ipynb` on Kaggle.
2. Attach the ONNX Perch no-DFT model input.
3. Score hidden-style 60-second files as 12 contiguous 5-second windows.
4. Report wall time per file and projected hidden-test runtime.
5. Avoid blending, priors, sequence modeling, and heavy post-processing.

Result:

- The public dry-run timing used 20 `train_soundscapes` files.
- Median runtime was **1.86 seconds per 60-second file**.
- ONNX outputs were `(48, 14795)` label logits and `(48, 1536)` embeddings per
  4-file batch.

Deliverables:

- `docs/10_onnx_perch_speed_results.md`.

### 2.4 Preserve ONNX Perch + SED Champion

Status: succeeded with **0.890** public score.

Goal: keep the new best submission reproducible before testing variants.

Work items:

1. Preserve `07_onnx_perch_sed_blend.ipynb`.
2. Record attached Kaggle inputs and notebook version.
3. Keep the exact-mapped conservative blend unchanged as the protected
   champion.
4. Document the result in `docs/11_onnx_perch_sed_blend_results.md`.

Success signal:

- We can restore the **0.890** path without guessing inputs or code changes.

Deliverables:

- `docs/11_onnx_perch_sed_blend_results.md`.

### 2.5 Tune Simple Blend Variants

Status: next.

Goal: test small changes around the champion without adding a fragile modeling
stack.

Work items:

1. Sweep Perch blend weight near the current conservative value.
2. Save each variant as a clearly named Kaggle notebook version, not a
   replacement for the champion.
3. Record exact Perch mapping count and unmapped target classes from the run
   output.
4. Consider one genus-proxy variant only after exact-mapped blend-weight tests.

Success signal:

- A controlled variant improves over **0.890** without increasing timeout risk.

Deliverables:

- A short result note comparing blend weights and mapping counts.

### 2.6 Add Perch Soundscape Priors

Status: implemented in `03_perch_v2_train.ipynb`; needs a fresh Kaggle train run and
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

- Updated `03_perch_v2_train.ipynb` and `04_perch_v2_submit.ipynb`.
- A small prior summary table saved by the training notebook.
- A note added to `05_perch_v2_results.md`.

### 2.7 Inspect Weak Labels

Status: implemented in `03_perch_v2_train.ipynb` via
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

### 2.8 Test Lightweight Calibration

Status: implemented in `03_perch_v2_train.ipynb` via `temperature_grid.csv` and
`calibration.json`; needs a controlled submission test through
`04_perch_v2_submit.ipynb`.

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

- Calibration config in `03_perch_v2_train.ipynb` and scoring use in
  `04_perch_v2_submit.ipynb`.
- Updated result table in `05_perch_v2_results.md`.

### 2.9 Compare Perch And EfficientNet Errors

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

### 2.10 Distill Perch Into A Faster Student

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

1. Preserve the **0.890** ONNX Perch + SED champion.
2. Select the **0.890** blend and **0.822** ONNX SED submissions for final-score
   tracking unless a stronger variant appears.
3. Run only small blend-weight variants next.
4. Try genus proxy mapping only as a separate variant after weight tuning.
5. Avoid full ProtoSSM-style sequence modeling until simple variants stop
   improving.

## 4. Guardrails

- Do not optimize only against the public leaderboard.
- Do not assume public dry runs measure hidden-test runtime; public runs can have
  three rows and zero audio files.
- Do not add heavy test-time augmentation until CPU runtime is measured.
- Do not use raw soundscape label counts without deduplication.
- Keep the Perch training and submission notebooks split because artifact
  management and CPU scoring now have different constraints.
- Keep direct TensorFlow Perch notebooks as protected references, not the active
  improvement lane.
- Keep attached model artifacts and wheelhouses outside git.
