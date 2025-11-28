from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NetBase, register_net


class _Sine(nn.Module):
    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class _SirenLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dropout = dropout
        self.weight = nn.Parameter(torch.empty(dim_out, dim_in))
        self.bias: Optional[nn.Parameter]
        if use_bias:
            self.bias = nn.Parameter(torch.empty(dim_out))
        else:
            self.bias = None
        self.activation = _Sine(w0)
        self._init_weights(c=c, w0=w0)

    def _init_weights(self, c: float, w0: float) -> None:
        bound = 1.0 / self.dim_in if self.is_first else math.sqrt(c / self.dim_in) / w0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return self.activation(out)


@register_net(name="siren")
class SIREN(NetBase):
    """Sinusoidal representation network with optional Gaussian head."""

    def __init__(
        self,
        in_dim: int,
        width: int = 256,
        depth: int = 4,
        omega0: float = 30.0,
        omega0_hidden: float = 1.0,
        omega0_initial: Optional[float] = None,
        out_unc: bool = False,
        dropout: float = 0.0,
        final_activation: Optional[str] = None,
        learn_hidden_omegas: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("SIREN depth must be >= 1")
        if learn_hidden_omegas:
            import warnings

            warnings.warn(
                "learn_hidden_omegas is no longer supported and will be ignored",
                RuntimeWarning,
                stacklevel=2,
            )

        self.layers = nn.ModuleList()
        first_w0 = omega0_initial if omega0_initial is not None else omega0
        for idx in range(depth):
            is_first = idx == 0
            layer_in = in_dim if is_first else width
            w0_layer = first_w0 if is_first else omega0_hidden
            layer_dropout = 0.0 if is_first else dropout
            self.layers.append(
                _SirenLayer(
                    dim_in=layer_in,
                    dim_out=width,
                    w0=w0_layer,
                    is_first=is_first,
                    dropout=layer_dropout,
                )
            )

        self.final_linear = nn.Linear(width, width)
        bound = math.sqrt(6.0 / width) / max(omega0_hidden, 1e-6)
        nn.init.uniform_(self.final_linear.weight, -bound, bound)
        nn.init.uniform_(self.final_linear.bias, -bound, bound)

        self.final_activation: nn.Module
        if final_activation == "relu":
            self.final_activation = nn.ReLU(inplace=True)
        elif final_activation == "gelu":
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()

        self.head = nn.Linear(width, 1)
        nn.init.uniform_(self.head.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.head.bias)

        self.out_unc = out_unc
        if out_unc:
            self.logvar = nn.Linear(width, 1)
            nn.init.uniform_(self.logvar.weight, -1e-4, 1e-4)
            nn.init.zeros_(self.logvar.bias)
        else:
            self.logvar = None

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.final_linear(h)
        h = self.final_activation(h)
        y = self.head(h).squeeze(-1)
        if self.out_unc and self.logvar is not None:
            return y, self.logvar(h).squeeze(-1)
        return y
