# Distilled SED Notebook Review

## 1. Why It Matters

The downloaded `bc2026-distilled-sed.ipynb` points to the most promising next
direction: keep Perch as a teacher during training, but submit a fast PyTorch or
ONNX student model. This avoids the hidden-test timeout risk from running the
full TensorFlow Perch SavedModel during submission.

## 2. What The Notebook Does Differently

| Area | Approach |
|---|---|
| Submission runtime | ONNX SED fold models on CPU, not TensorFlow Perch |
| Teacher model | Perch v2 ONNX used during training for embedding distillation |
| Student model | `tf_efficientnet_b0.ns_jft_in1k` with SED attention head |
| Output classes | Full `sample_submission.csv` target set, **234** classes |
| Training data | Focal recordings plus labeled soundscape windows |
| Audio speedup | Pre-extracted waveform cache instead of repeated OGG decoding |
| Inference | 12 contiguous 5-second windows per 60-second soundscape |
| Postprocessing | Fold averaging, clip/frame blend, Gaussian smoothing over windows |

## 3. Key Lessons To Borrow

1. **Use Perch as a teacher, not as the submitted model.**
   Direct Perch CPU inference is strong but fragile. Distillation keeps part of
   the Perch signal while replacing the slow TensorFlow scoring path.

2. **Train all 234 submission columns.**
   Our current Perch probe covers 206 train primary labels, leaving 28 target
   columns at zero. The SED notebook trains against the `sample_submission.csv`
   label order, which is a cleaner competition-aligned output contract.

3. **Use labeled soundscape windows directly.**
   The notebook builds a 739-window soundscape label matrix and mixes those
   examples into training. This addresses the clean-clip versus hidden-soundscape
   domain gap better than clean-clip validation alone.

4. **Export to ONNX for CPU submission.**
   Five ONNX fold models are about 20 MB each and run with
   `CPUExecutionProvider`. This is far more submission-friendly than loading
   TensorFlow Perch.

5. **Smooth across the 12 windows.**
   Gaussian smoothing across adjacent 5-second windows is a cheap way to reduce
   jitter in soundscape predictions.

## 4. Important Caveat

The saved notebook output is still a public/debug run:

- It found no public `test_soundscapes`.
- It used 5 train soundscape files for debug.
- It wrote 60 debug rows.

So it does not prove hidden-test runtime by itself. However, its architecture is
much more likely to fit the Kaggle CPU limit than direct Perch inference because
the submitted model is a compact ONNX SED student.

## 5. Recommended Next Step

Create a project-owned distilled SED workflow:

1. Start from the notebook's inference path and reproduce a Kaggle submission
   using the attached public ONNX folds.
2. If it finishes, document the score and runtime as a new baseline.
3. Only then decide whether to train our own student folds.
4. Keep EfficientNet-B0 version 9 and Perch v2 version 14 unchanged as protected
   baselines.

