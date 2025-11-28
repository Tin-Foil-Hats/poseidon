from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .base import NetBase, register_net


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
}


@register_net(name="mlp")
class MLP(NetBase):
    """Simple feed-forward MLP with configurable depth and activation."""

    def __init__(
        self,
        in_dim: int,
        width: int = 256,
        depth: int = 4,
        act: str = "relu",
        out_dim: int = 1,
        out_unc: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if act not in _ACTIVATIONS:
            raise ValueError(f"unknown activation '{act}'")
        activation = _ACTIVATIONS[act]
        layers: list[nn.Module] = []
        prev_dim = in_dim

                     
        layers.append(nn.Linear(prev_dim, width))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

                       
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(width, width))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)
        self.out_dim = out_dim
        self.out_unc = out_unc
        self.logvar = nn.Linear(width, 1) if out_unc else None

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        y = self.head(h)
        if self.out_dim == 1:
            y = y.squeeze(-1)
        if self.out_unc and self.logvar is not None:
            return y, self.logvar(h).squeeze(-1)
        return y
