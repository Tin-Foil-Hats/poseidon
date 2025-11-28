import torch
import torch.nn as nn

from .base import PEBase
from .registry import register_pe
from .utils import deg2rad, wrap_lon_pi, time_normalize


@register_pe(name="rff", aliases=["perff"])
class PERFF(PEBase):
    """Random Fourier Features; time input uses configurable normalization."""

    def __init__(self, dim_in=2, n_feat=256, sigma=10.0, use_time=True, time_norm: str = "zscore"):
        super().__init__()
        self.use_time = bool(use_time)
        self.time_norm = str(time_norm).lower()
        d = int(dim_in) + (1 if self.use_time else 0)
        B = torch.randn(d, n_feat) / float(sigma)
        self.register_buffer("B", B)
        self._time_stats: dict | None = None

    def bind_context(self, ctx: dict):
        self._time_stats = ctx.get("time") if ctx is not None else None

    def feat_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, lat_deg, lon_deg, t_sec=None):
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))
        U = [lat, lon]
        if self.use_time and (t_sec is not None):
            U.append(time_normalize(t_sec, self.time_norm, self._time_stats))
        U = torch.stack(U, dim=-1)
        Z = U @ self.B
        return torch.cat([torch.sin(Z), torch.cos(Z)], dim=-1)
