from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    path: str | Path,
    sample_rate: int,
    duration: float,
    offset: float | None = None,
) -> np.ndarray:
    target_len = int(sample_rate * duration)
    y, _ = librosa.load(path, sr=sample_rate, mono=True, offset=offset or 0.0, duration=duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


def random_offset(duration_available: float | None, clip_duration: float) -> float:
    if duration_available is None or duration_available <= clip_duration:
        return 0.0
    return float(np.random.uniform(0.0, duration_available - clip_duration))


def mel_spectrogram(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)
