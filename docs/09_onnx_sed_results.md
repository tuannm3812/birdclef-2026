# ONNX Distilled SED Results

## 1. Summary

The distilled SED ONNX submission is now the best public result in this
workspace.

| Submission | Notebook | Public score | Status |
|---|---|---:|---|
| ONNX distilled SED version 1 | `05_onnx_sed_submit.ipynb` | **0.822** | New champion |
| Perch v2 version 14 | `04_perch_v2_submit.ipynb` | **0.770** | Protected baseline |
| EfficientNet-B0 version 9 | `02_effnet_b0.ipynb` | **0.646** | CPU-safe fallback |

The ONNX SED path improves public score by **+0.052** over the previous Perch
v2 champion and by **+0.176** over EfficientNet-B0.

## 2. What Worked

The result confirms the main lesson from the reference notebook review:
submission-time ONNX student inference is a better direction than direct
TensorFlow Perch inference for this competition.

Important strengths:

1. It predicts all **234** submission columns.
2. It avoids TensorFlow Perch inference during scoring.
3. It uses compact ONNX folds with CPU execution.
4. It keeps scoring focused on 12 contiguous 5-second windows per soundscape.
5. It finished hidden-test scoring where later direct Perch attempts timed out.

## 3. Interpretation

This should become the new protected champion until another submission beats it.
The previous Perch v2 version 14 remains valuable, but mainly as a baseline and
source of modeling ideas. New leaderboard work should stay in the ONNX lane.

The score also suggests that full 234-column SED modeling plus temporal
smoothing is more competition-aligned than a 206-class clean-clip Perch probe.

## 4. Recommended Next Step

Create `06_onnx_perch_speed_test.ipynb` as a runtime experiment, not a full
modeling notebook.

The next notebook should:

1. Load the ONNX Perch no-DFT model.
2. Score 60-second files as 12 contiguous 5-second windows.
3. Report seconds per soundscape and projected hidden-test runtime.
4. Avoid blending, priors, sequence models, and post-processing.

ONNX Perch later passed the CPU speed check, and the historical blend notebook
is preserved at `archive/07_onnx_perch_sed_blend.ipynb`.
