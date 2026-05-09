from __future__ import annotations

import timm
import torch
from torch import nn


class BirdClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PerchProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
