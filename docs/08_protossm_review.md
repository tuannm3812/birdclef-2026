# ProtoSSM Notebook Review

## 1. Summary

The downloaded `birdclef-2026-perch-v2-protossm-0-925.ipynb` is a strong
reference, but it is too complex to copy directly into our working notebooks.
It combines ONNX Perch inference, train-soundscape sequence modeling, distilled
SED ONNX folds, and several post-processing gates.

Its core lesson is valuable: **the successful high-scoring Perch path uses ONNX
Perch, not TensorFlow Perch**. That is the main reason it can be much faster
than our direct TensorFlow Perch submission.

## 2. What It Does Differently

| Area | Approach |
|---|---|
| Perch runtime | Prefers `perch_v2_no_dft.onnx` with ONNX Runtime CPU |
| Fallback | Can fall back to TensorFlow Perch CPU if ONNX is unavailable |
| Label coverage | Predicts all **234** submission columns |
| Training signal | Uses labeled full train-soundscape files, 708 windows from 59 files |
| Sequence modeling | Trains ProtoSSM-like temporal models inside the notebook |
| Additional model | Blends with public distilled SED ONNX folds |
| Postprocessing | Rank blend, site/hour priors, temporal smoothing, sonotype mirroring, rare-class dampening |
| Saved output | Final `submission.csv` after ProtoSSM + SED blend |

## 3. Why It May Beat Our Current Perch

Our current Perch submission runs the TensorFlow SavedModel over hidden audio.
That path timed out. The ProtoSSM notebook uses an ONNX Perch export, which is
designed for much faster CPU inference. It also augments raw Perch logits with
temporal sequence modeling and SED predictions, rather than relying on a shallow
probe over Perch embeddings.

The notebook also predicts all 234 output columns. Our Perch probe covers the
206 train primary labels, leaving 28 columns at zero.

## 4. Risks

1. **Very high complexity.**
   The notebook is only six cells, but the main cell is large and mixes path
   discovery, Perch inference, cache handling, sequence training, calibration,
   post-processing, and file writing.

2. **Hard to debug.**
   A failure in hidden scoring would be harder to isolate than in our split
   train/submit notebooks.

3. **Many external inputs.**
   It depends on public ONNX Perch, Perch metadata caches, distilled SED ONNX
   folds, ONNX Runtime wheels, TensorFlow wheels, and the Google Perch model.

4. **Saved output is a dry run.**
   The inspected output used 20 train soundscape files because no public hidden
   `test_soundscapes` were mounted. The claimed public score is promising, but
   our local inspection cannot verify hidden runtime.

5. **Post-processing overfit risk.**
   Site/hour priors, sonotype mirroring, adaptive thresholds, and residual
   correction may help public score but should be added incrementally in our own
   workflow.

## 5. What To Borrow

1. ONNX Perch inference path.
2. Full 234-column output contract.
3. Soundscape-window sequence modeling.
4. Distilled SED ONNX blend.
5. Window-level temporal smoothing.

## 6. What Not To Copy Directly

Do not drop the full notebook into `notebooks/` as-is. Instead, extract one
controlled experiment at a time:

1. First: reproduce the distilled SED ONNX submission path.
2. Second: test ONNX Perch inference speed alone.
3. Third: add simple rank blending between ONNX Perch and SED.
4. Last: consider sequence models and post-processing gates.

## 7. Recommendation

Do **not** continue improving the TensorFlow Perch submission path. Keep Perch
v2 version 14 as the protected champion, keep EfficientNet-B0 version 9 as the
protected fallback, and move new work toward ONNX-based inference:

1. Distilled SED ONNX baseline.
2. ONNX Perch speed test.
3. ONNX Perch + SED blend.
4. Project-owned sequence model only if the simpler ONNX paths succeed.

