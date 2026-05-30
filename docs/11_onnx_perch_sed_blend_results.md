# ONNX Perch + SED Blend Results

## 1. Summary

`07_onnx_perch_sed_blend.ipynb` is now the best public submission in this
workspace.

| Submission | Notebook | Public score | Status |
|---|---|---:|---|
| ONNX Perch + SED blend version 2 | `07_onnx_perch_sed_blend.ipynb` | **0.890** | New champion |
| ONNX Perch + SED W0.25 version 2 | `08_onnx_perch_sed_blend_w025.ipynb` | **0.890** | Tied champion |
| ONNX distilled SED version 2 | `05_onnx_sed_submit.ipynb` | **0.822** | Protected baseline |
| Perch v2 version 14 | `04_perch_v2_submit.ipynb` | **0.770** | Protected baseline |
| EfficientNet-B0 version 9 | `02_effnet_b0.ipynb` | **0.646** | CPU-safe fallback |

The blend improves public score by **+0.068** over ONNX SED alone and by
**+0.244** over the original EfficientNet-B0 fallback.

The first blend-weight variant increased Perch weight from **0.15** to
**0.25** and tied the public score at **0.890**. That suggests the simple
exact-mapped blend is robust, but more Perch weight alone is not the next clear
source of improvement.

## 2. What Worked

The result validates the staged ONNX workflow:

1. ONNX SED gave a strong fast baseline.
2. ONNX Perch passed the speed test at about **1.86 seconds per 60-second file**.
3. Exact scientific-name mapping let Perch contribute signal to the BirdCLEF
   234-column output contract.
4. A conservative blend was enough to produce a large gain without adding
   sequence models, priors, proxy labels, or post-processing gates.

## 3. Interpretation

The main lesson is that Perch remains highly useful, but the successful
submission form is **ONNX Perch as a lightweight signal blended into ONNX SED**,
not direct TensorFlow Perch scoring.

This notebook should now be protected. Future experiments should branch from it
as small variants, not overwrite it.

## 4. Recommended Next Experiments

Use the remaining submissions carefully. The next work should stay small and
controlled:

1. Inspect the exact mapping count and unmapped taxa from
   `perch_mapping_diagnostics.csv`.
2. Consider genus-level proxy mapping only for unmapped classes, and only as a
   separate variant.
3. If testing weights again, try **0.20** before going higher than **0.25**.
4. Try light temporal smoothing or rank blending only after mapping variants
   are understood.

Avoid jumping directly to the full ProtoSSM-style stack until simple blend
variants stop improving.
