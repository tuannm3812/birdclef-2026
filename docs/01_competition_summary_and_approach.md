# Competition Summary And Approach

## 1. Competition Objective

BirdCLEF+ 2026 is a CPU-scored Kaggle code competition for passive acoustic monitoring in the Brazilian Pantanal. The task is to predict which wildlife species are present in each **5-second window** of hidden **1-minute soundscape recordings**.

The target set contains **234 submission columns** spanning birds, amphibians, mammals, reptiles, and insects. The training set in this repository currently contains **35,549 short recordings** across **206 primary labels**, plus expert-labeled in-domain train soundscapes.

## 2. Evaluation And Runtime Constraints

The leaderboard score is based on multi-label ranking quality across species. The practical implementation target is a `submission.csv` with one row per soundscape window and one probability column per species.

Key constraints:

- Hidden `test_soundscapes/` are populated only during Kaggle scoring.
- Scored notebooks must run on **CPU** within the competition runtime budget.
- Internet must not be required during final scoring.
- Checkpoints, pretrained weights, wheel files, and model exports must be attached as Kaggle inputs.
- Submission mode should load artifacts, run inference, and write `submission.csv`; it should not train models or create large diagnostic bundles.

## 3. Current Public Scores

| Notebook | Mode | Public score | Runtime note | Role |
|---|---|---:|---|---|
| `2_bc2026_effnet_b0.ipynb` | CPU submission | **0.646** | Completed successfully | Fast PyTorch baseline |
| `3_bc2026_perch_v2.ipynb` | CPU submission | **0.770** | Completed successfully | Stronger foundation-feature submission |

The public leaderboard confirms the validation pattern: Perch features are materially stronger than the EfficientNet-B0 baseline. The public gap is **+0.124** in favor of Perch.

## 4. Dataset Insights That Drive Modeling

- Class imbalance is severe: training labels range from **1** to **499** recordings.
- The top **30** labels account for **40.3%** of recordings, so naive training favors common species.
- Secondary labels are common enough to matter: **4,372** recordings include secondary labels, with **7,431** total secondary-label mentions.
- In-domain train soundscape labels are especially valuable because they match the hidden-test recording style more closely than isolated training clips.
- Non-bird taxa are likely high-leverage because they are rarer and less covered by bird-specialized models.
- CPU inference is now a first-class modeling constraint; a slower model is not useful if it cannot finish scoring.

## 5. Recommended Approach

### 5.1 Primary Submission

Use **Notebook 3: Perch v2** as the current primary scored submission because it has the best public score:

1. Set `CFG.mode = "submission"`.
2. Attach TensorFlow 2.20 wheels.
3. Attach the Google Perch CPU SavedModel.
4. Attach the uploaded Perch artifact containing `best_perch_probe.pt` and `labels.json`.
5. Keep full-file batched inference enabled so each 60-second file is read once and reshaped into 12 windows.

### 5.2 Reliable Fallback

Keep **Notebook 2: EfficientNet-B0** as the fallback submission because it is simpler, pure PyTorch, and already successful on CPU:

1. Set `CFG.mode = "submission"`.
2. Attach the EfficientNet artifact directory containing `best_effnet_b0.pt` and `labels.json`.
3. Use it whenever TensorFlow/Perch runtime becomes unstable.

The confirmed EfficientNet artifact directory is:

```text
/kaggle/input/models/tuannm3812/irdclef-efficientnet-b0-artifacts/pytorch/default/1/effnet_b0
```

### 5.3 Next Refinements

The highest-value next steps are:

1. Add light post-processing to Perch predictions using soundscape priors from site/hour metadata.
2. Use train soundscape labels to calibrate species thresholds or logits.
3. Distill Perch predictions into a faster PyTorch model for a runtime-safe ensemble.
4. Improve non-bird taxa handling with proxy labels, class-specific priors, or targeted augmentation.
5. Compare Perch and EffNet predictions to identify classes where an ensemble improves public score without large runtime cost.

## 6. Source Links

- Kaggle competition overview: https://www.kaggle.com/competitions/birdclef-2026/overview
- Kaggle competition page: https://www.kaggle.com/competitions/birdclef-2026
- Kaggle dataset mirror used for public file descriptions: https://www.kaggle.com/datasets/llkh0a/birdclef-2026-repack

## 7. Decision Log

- **EffNet-B0 remains useful** because it is the simplest CPU-safe baseline and validates the submission pipeline.
- **Perch v2 is now the lead model** because it finishes CPU scoring and improves the public score from **0.646** to **0.770**.
- **Do not add training to submission mode** for either notebook. Training belongs in experiment mode; scoring should only load artifacts and infer.
- **Keep notebook outputs clear after code edits** and rerun on Kaggle to regenerate trusted execution outputs.
