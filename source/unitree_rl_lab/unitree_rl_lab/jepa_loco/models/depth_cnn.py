"""프레임 단위 depth 인코더. 2채널 (정규화 depth, mask) [N, 2, 64, 112] → f_t [N, feat_dim].

GRU·Mamba 비교에서 동일하게 유지하는 visual encoder (CLAUDE.md §4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        in_hw: tuple[int, int] = (64, 112),
        channels: tuple[int, ...] = (16, 32, 64),
        kernels: tuple[int, ...] = (5, 3, 3),
        strides: tuple[int, ...] = (2, 2, 2),
        feat_dim: int = 128,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for co, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv2d(c, co, k, s), nn.ELU()]
            c = co
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, *in_hw)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, feat_dim), nn.ELU())
        self.in_channels, self.in_hw, self.feat_dim = in_channels, tuple(in_hw), feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))
