from __future__ import annotations

import torch
import torch.nn as nn

from .base import NetBase, register_net


class _ResidualBlock(nn.Module):
    """Simple residual MLP block with optional dropout."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin1(x)
        y = self.act1(y)
        y = self.dropout(y)
        y = self.lin2(y)
        y = self.act2(y)
        return x + y


@register_net(name="fcnet")
class FCNet(NetBase):
    """Feed-forward network with stacked residual layers."""

    def __init__(
        self,
        in_dim: int,
        width: int = 256,
        depth: int = 4,
        out_dim: int = 1,
        dropout: float = 0.0,
        out_unc: bool = False,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.proj = nn.Linear(in_dim, width)
        blocks = []
        for _ in range(depth):
            blocks.append(_ResidualBlock(width, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(width, out_dim)
        self.out_dim = out_dim
        self.out_unc = out_unc
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if out_unc:
            self.logvar = nn.Linear(width, 1)
        else:
            self.logvar = None
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        h = self.proj(x)
        h = self.final_act(h)
        h = self.blocks(h)
        h = self.dropout(h)
        y = self.head(h)
        if self.out_dim == 1:
            y = y.squeeze(-1)
        if self.out_unc and self.logvar is not None:
            return y, self.logvar(h).squeeze(-1)
        return y
