# ONNX Perch Speed Results

## 1. Summary

`06_onnx_perch_speed_test.ipynb` successfully loaded and ran the ONNX Perch
no-DFT model on Kaggle CPU.

| Metric | Result |
|---|---:|
| Audio source | `train_soundscapes` public dry-run fallback |
| Files timed | **20** |
| Batch size | **4** soundscape files |
| Windows per batch | **48** |
| Median seconds per 60-second file | **1.86** |
| Output label shape | `(48, 14795)` |
| Output embedding shape | `(48, 1536)` |

This is fast enough to justify moving to the next controlled experiment:
`07_onnx_perch_sed_blend.ipynb`.

## 2. Timing Detail

| Batch | Files | Load seconds | Inference seconds | Seconds per file |
|---:|---:|---:|---:|---:|
| 1 | 4 | 0.30 | 7.94 | 2.06 |
| 2 | 4 | 0.30 | 7.77 | 2.02 |
| 3 | 4 | 0.31 | 7.11 | 1.86 |
| 4 | 4 | 0.32 | 7.11 | 1.86 |
| 5 | 4 | 0.31 | 7.06 | 1.84 |

The first batches are slightly slower, then the run stabilizes around
**1.84-1.86 seconds per file**.

## 3. Interpretation

The key result is that ONNX Perch inference itself is not the bottleneck that
caused the direct TensorFlow Perch submission to time out. The ONNX path is
therefore a valid candidate for blending with the current ONNX SED champion.

The public run did not expose hidden `test_soundscapes`, so this timing used
`train_soundscapes`. That is still useful for runtime measurement because the
files have the same 60-second soundscape structure.

## 4. Caveat

The speed test confirms runtime only. It does not yet solve the modeling step:
ONNX Perch returns **14,795** label logits, while the competition submission
needs **234** target columns.

Before blending, the next notebook must map Perch logits to the submission
columns using Perch label metadata and the BirdCLEF taxonomy.

## 5. Recommended Next Step

Create `07_onnx_perch_sed_blend.ipynb`.

The blend notebook should:

1. Reuse the working ONNX SED champion path.
2. Reuse ONNX Perch waveform inference from the speed test.
3. Map ONNX Perch logits to the 234 submission columns.
4. Start with a simple conservative blend, such as mostly SED plus a smaller
   Perch contribution.
5. Avoid sequence modeling, priors, and heavy post-processing until the simple
   blend has a successful submission.

