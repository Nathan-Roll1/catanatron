from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


def _make_activation(name: str) -> nn.Module:
    if name == "mish":
        return nn.Mish()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation {name!r}, expected 'mish' or 'relu'")


class ResidualBlock(nn.Module):
    """FC residual block: x -> FC -> BN -> Act -> FC -> BN -> (+x) -> Act

    All layers use the same *dim*; no projection shortcut is needed when
    input and output widths match (which is always the case inside the trunk).
    """

    def __init__(self, dim: int, activation: str = "mish") -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.act(out + identity)


class ResNetTrunk(nn.Module):
    """Deep residual trunk.

    1. Project ``input_dim`` -> ``trunk_channels`` (Linear + BN + Act)
    2. Pass through *num_blocks* :class:`ResidualBlock` layers

    Supports optional per-block gradient checkpointing to trade compute
    for GPU memory during training.

    Shape: ``(batch, input_dim)`` -> ``(batch, trunk_channels)``
    """

    def __init__(
        self,
        input_dim: int,
        trunk_channels: int = 256,
        num_blocks: int = 20,
        activation: str = "mish",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, trunk_channels),
            nn.BatchNorm1d(trunk_channels),
            _make_activation(activation),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(trunk_channels, activation) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = cp.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
