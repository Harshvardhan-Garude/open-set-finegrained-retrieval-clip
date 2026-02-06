"""Embedding backbones used in the CNN baseline.

The original project used EfficientNet-B0 with a projection head to produce
L2-normalized embeddings for metric learning / retrieval.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EffB0Embedding(nn.Module):
    """EfficientNet-B0 backbone with a projection head.

    Output: L2-normalized embedding vector of dimension `embed_dim`.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Remove classifier; keep features + pooling
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        in_dim = backbone.classifier[1].in_features  # type: ignore[index]
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)
