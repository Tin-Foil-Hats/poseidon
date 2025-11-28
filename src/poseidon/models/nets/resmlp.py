from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from .base import NetBase, register_net


class ResBlock(nn.Module):
    def __init__(self, width: int, *, dropout: float = 0.0, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin1(self.norm(x))
        h = self.act(h)
        h = self.dropout(h)
        h = self.lin2(h)
        return x + h


@register_net(name="resmlp")
class ResMLP(NetBase):
    def __init__(
        self,
        in_dim: int,
        width: int = 256,
        depth: int = 4,
        out_unc: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout=dropout) for _ in range(depth)])
        self.head = nn.Linear(width, 1)
        self.out_unc = out_unc
        self.logvar = nn.Linear(width, 1) if out_unc else None

    def forward(self, x: torch.Tensor):
        h = self.input(x)
        h = self.blocks(h)
        y = self.head(h).squeeze(-1)
        if self.out_unc and self.logvar is not None:
            return y, self.logvar(h).squeeze(-1)
        return y
