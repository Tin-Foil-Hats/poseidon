from __future__ import annotations

import torch
import torch.nn as nn

from .base import NetBase, register_net


@register_net(name="siren_stable")
class SIRENStable(NetBase):
    """Sine-like smooth implicit net using SiLU activations for stability."""

    def __init__(self, in_dim: int, width: int = 256, depth: int = 4, out_unc: bool = False) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d, width), nn.SiLU()])
            d = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.out_unc = out_unc
        self.logvar = nn.Linear(width, 1) if out_unc else None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        y = self.head(h).squeeze(-1)
        if self.out_unc and self.logvar is not None:
            return y, self.logvar(h).squeeze(-1)
        return y
