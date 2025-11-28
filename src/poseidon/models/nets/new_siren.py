"""Siren network re-exported with a NetBase adapter."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .activations import Sine
from .base import NetBase, register_net
from .utils import cast_tuple, exists


class _SirenLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float,
        *,
        is_first: bool,
        c: float = 6.0,
        use_bias: bool = True,
        resnet: bool = False,
    ) -> None:
        super().__init__()
        self.resnet = bool(resnet)
        self.is_first = bool(is_first)
        self.dim_in = int(dim_in)
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self._init_weights(weight, bias, c=c, w0=w0)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.activation = Sine(w0)

    def _init_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor], *, c: float, w0: float) -> None:
        dim = float(self.dim_in)
        bound = (1.0 / dim) if self.is_first else (c / dim) ** 0.5 / w0
        weight.uniform_(-bound, bound)
        if bias is not None:
            bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        out = self.activation(out)
        if self.resnet:
            out = 0.5 * (x + out)
        return out


class _SirenNetCore(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int = 5,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        c: float = 6.0,
        use_bias: bool = True,
        final_activation: Optional[nn.Module] = None,
        resnet: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        for idx in range(num_layers):
            is_first = idx == 0
            layers.append(
                _SirenLayer(
                    dim_in=dim_in if is_first else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    is_first=is_first,
                    c=c,
                    use_bias=use_bias,
                    resnet=resnet and is_first,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_activation = final_activation if final_activation is not None else nn.Identity()
        self.last = _SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_first=False, c=c, use_bias=use_bias)

    def forward(self, x: torch.Tensor, mods: Optional[torch.Tensor] = None) -> torch.Tensor:
        mods_seq = cast_tuple(mods, len(self.layers))
        for layer, mod in zip(self.layers, mods_seq):
            x = layer(x)
            if exists(mod):
                x *= rearrange(mod, "d -> () d")
        x = self.final_activation(x)
        return self.last(x)


@register_net(name="new_siren", aliases=("siren_new",))
class NewSiren(NetBase):
    """Adapter around the custom SirenNet implementation."""

    def __init__(
        self,
        in_dim: int,
        width: int = 512,
        depth: int = 6,
        omega0: float = 30.0,
        omega0_hidden: float = 1.0,
        omega0_initial: Optional[float] = None,
        use_bias: bool = True,
        c: float = 6.0,
        out_activation: Optional[str] = None,
        resnet: bool = False,
    ) -> None:
        super().__init__()
        final_act: Optional[nn.Module]
        if out_activation == "relu":
            final_act = nn.ReLU(inplace=True)
        elif out_activation == "gelu":
            final_act = nn.GELU()
        else:
            final_act = None

        core = _SirenNetCore(
            dim_in=in_dim,
            dim_hidden=width,
            dim_out=1,
            num_layers=depth,
            w0=omega0_hidden,
            w0_initial=omega0_initial if omega0_initial is not None else omega0,
            c=c,
            use_bias=use_bias,
            final_activation=final_act,
            resnet=resnet,
        )
        self.model = core
        self.head = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.model(inputs)
        out = self.head(out)
        return out.squeeze(-1)

