# ONNX Perch + SED Blend Results

## 1. Summary

`10_onnx_perch_sed_soundscape_calibrated.ipynb` is now the best public
submission in this workspace.

| Submission | Notebook | Public score | Status |
|---|---|---:|---|
| ONNX Perch + SED soundscape-calibrated version 2 | `10_onnx_perch_sed_soundscape_calibrated.ipynb` | **0.893** | New champion |
| ONNX Perch + SED min10/AP0.01 version 1 | `archive/11_onnx_perch_sed_calibrated_min10_ap001.ipynb` | **0.892** | Did not improve |
| ONNX Perch + SED proxy6 version 1 | `archive/09_onnx_perch_sed_blend_proxy6.ipynb` | **0.892** | Protected baseline |
| ONNX Perch + SED blend version 2 | `archive/07_onnx_perch_sed_blend.ipynb` | **0.890** | Protected baseline |
| ONNX Perch + SED W0.25 version 2 | `archive/08_onnx_perch_sed_blend_w025.ipynb` | **0.890** | Tied champion |
| ONNX distilled SED version 2 | `05_onnx_sed_submit.ipynb` | **0.822** | Protected baseline |
| Perch v2 version 14 | `04_perch_v2_submit.ipynb` | **0.770** | Protected baseline |
| EfficientNet-B0 version 9 | `02_effnet_b0.ipynb` | **0.646** | CPU-safe fallback |

The soundscape-calibrated blend improves public score by **+0.071** over ONNX
SED alone and by **+0.247** over the original EfficientNet-B0 fallback.

The first blend-weight variant increased Perch weight from **0.15** to
**0.25** and tied the public score at **0.890**. That suggests the simple
exact-mapped blend is robust, but more Perch weight alone is not the next clear
source of improvement.

The narrow proxy variant improved public score from **0.890** to **0.892** by
leaving anonymous insect sonotypes unchanged and adding same-genus Perch proxies
only for named unmapped taxa.

The soundscape-calibrated variant improved public score from **0.892** to
**0.893** by learning lightweight per-class blend weights from labeled
train-soundscape windows.

The support-thresholded calibration variant dropped back to **0.892**. This
suggests the low-support calibration changes in notebook `10` may contribute
useful public-leaderboard signal, or at minimum should not be removed by a
simple positives/AP threshold.

## 2. What Worked

The result validates the staged ONNX workflow:

1. ONNX SED gave a strong fast baseline.
2. ONNX Perch passed the speed test at about **1.86 seconds per 60-second file**.
3. Exact scientific-name mapping let Perch contribute signal to the BirdCLEF
   234-column output contract.
4. A conservative exact blend was enough to produce a large gain.
5. A narrow proxy variant added a small extra gain without broad sonotype
   proxying.
6. Labeled train-soundscape windows provided a small additional calibration
   gain.

## 3. Interpretation

The main lesson is that Perch remains highly useful, but the successful
submission form is **ONNX Perch as a lightweight signal blended into ONNX SED**,
not direct TensorFlow Perch scoring.

The soundscape-calibrated notebook should now be protected. Future experiments
should branch from it as small variants, not overwrite it.

## 4. Recommended Next Experiments

Use the remaining submissions carefully. The next work should stay small and
controlled:

1. Preserve the **0.893** soundscape-calibrated submission.
2. Do not continue pruning calibration weights with hard support thresholds.
3. If testing another calibration variant, prefer a smoother/shrunk version
   rather than dropping low-support changes entirely.
4. Try light temporal smoothing or rank blending only after calibration variants
   stop improving.

Avoid jumping directly to the full ProtoSSM-style stack until simple blend
variants stop improving.
