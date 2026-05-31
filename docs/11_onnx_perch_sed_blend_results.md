# ONNX Perch + SED Blend Results

## 1. Summary

`09_onnx_perch_sed_blend_proxy6.ipynb` is now the best public submission in
this workspace.

| Submission | Notebook | Public score | Status |
|---|---|---:|---|
| ONNX Perch + SED proxy6 version 1 | `09_onnx_perch_sed_blend_proxy6.ipynb` | **0.892** | New champion |
| ONNX Perch + SED blend version 2 | `07_onnx_perch_sed_blend.ipynb` | **0.890** | Protected baseline |
| ONNX Perch + SED W0.25 version 2 | `08_onnx_perch_sed_blend_w025.ipynb` | **0.890** | Tied champion |
| ONNX distilled SED version 2 | `05_onnx_sed_submit.ipynb` | **0.822** | Protected baseline |
| Perch v2 version 14 | `04_perch_v2_submit.ipynb` | **0.770** | Protected baseline |
| EfficientNet-B0 version 9 | `02_effnet_b0.ipynb` | **0.646** | CPU-safe fallback |

The proxy6 blend improves public score by **+0.070** over ONNX SED alone and by
**+0.246** over the original EfficientNet-B0 fallback.

The first blend-weight variant increased Perch weight from **0.15** to
**0.25** and tied the public score at **0.890**. That suggests the simple
exact-mapped blend is robust, but more Perch weight alone is not the next clear
source of improvement.

The narrow proxy variant improved public score from **0.890** to **0.892** by
leaving anonymous insect sonotypes unchanged and adding same-genus Perch proxies
only for named unmapped taxa.

## 2. What Worked

The result validates the staged ONNX workflow:

1. ONNX SED gave a strong fast baseline.
2. ONNX Perch passed the speed test at about **1.86 seconds per 60-second file**.
3. Exact scientific-name mapping let Perch contribute signal to the BirdCLEF
   234-column output contract.
4. A conservative exact blend was enough to produce a large gain.
5. A narrow proxy variant added a small extra gain without broad sonotype
   proxying.

## 3. Interpretation

The main lesson is that Perch remains highly useful, but the successful
submission form is **ONNX Perch as a lightweight signal blended into ONNX SED**,
not direct TensorFlow Perch scoring.

The proxy6 notebook should now be protected. Future experiments should branch
from it as small variants, not overwrite it.

## 4. Recommended Next Experiments

Use the remaining submissions carefully. The next work should stay small and
controlled:

1. Preserve the **0.892** proxy6 submission.
2. If testing another variant today, tune proxy weight only, not the full Perch
   weight.
3. Try a small proxy-weight change such as **0.08** or **0.10** as a separate
   notebook version.
4. Try light temporal smoothing or rank blending only after proxy-weight
   variants stop improving.

Avoid jumping directly to the full ProtoSSM-style stack until simple blend
variants stop improving.
