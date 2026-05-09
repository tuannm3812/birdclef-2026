from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from birdclef2026.audio import load_audio, mel_spectrogram, random_offset


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 32000
    duration: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 16000


class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        audio_cfg: AudioConfig,
        train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.audio_cfg = audio_cfg
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        duration = float(row["duration"]) if "duration" in row and pd.notna(row["duration"]) else None
        offset = random_offset(duration, self.audio_cfg.duration) if self.train else 0.0
        y = load_audio(row["filepath"], self.audio_cfg.sample_rate, self.audio_cfg.duration, offset)

        if self.train and random.random() < 0.5:
            gain = random.uniform(0.75, 1.25)
            y = y * gain

        spec = mel_spectrogram(
            y=y,
            sample_rate=self.audio_cfg.sample_rate,
            n_fft=self.audio_cfg.n_fft,
            hop_length=self.audio_cfg.hop_length,
            n_mels=self.audio_cfg.n_mels,
            fmin=self.audio_cfg.fmin,
            fmax=self.audio_cfg.fmax,
        )
        x = torch.from_numpy(spec).unsqueeze(0)
        y_idx = torch.tensor(self.label_to_idx[row["primary_label"]], dtype=torch.long)
        return x, y_idx
