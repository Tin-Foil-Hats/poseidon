import torch
import torch.nn as nn

from .base import PEBase
from .registry import register_pe
from .utils import deg2rad, wrap_lon_pi, time_normalize


@register_pe(
    name="rff_ard_bands",
    aliases=["perff_ard_bands", "rff_ard", "ard_rff"],
)
class PERFFARDBands(PEBase):
    """Random Fourier Features with multiple frequency bands and ARD scalings."""

    def __init__(
        self,
        dim_in=2,
        use_time=True,
        time_norm="zscore",
        sigmas=(0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0),
        n_feat_per_band=128,
        learn_ard=True,
    ):
        super().__init__()
        self.use_time = bool(use_time)
        self.time_norm = str(time_norm).lower()
        self.sigmas = tuple(float(s) for s in sigmas)
        self.n_feat_per_band = int(n_feat_per_band)
        d = int(dim_in) + (1 if self.use_time else 0)

        B0 = [torch.randn(d, self.n_feat_per_band) / s for s in self.sigmas]
        self.register_buffer("B0", torch.cat(B0, dim=1))
        self.m_total = self.B0.shape[1]

        self.learn_ard = bool(learn_ard)
        if self.learn_ard:
            self.theta = nn.Parameter(torch.zeros(d))
            self.eps = 1e-4
        else:
            self.register_buffer("alpha_fixed", torch.ones(d))

        self._time_stats: dict | None = None

    def bind_context(self, ctx: dict):
        self._time_stats = (ctx or {}).get("time")

    def feat_dim(self) -> int:
        return 2 * self.m_total

    def forward(self, lat_deg, lon_deg, t_sec=None):
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))
        U = [lat, lon]
        if self.use_time and (t_sec is not None):
            U.append(time_normalize(t_sec, self.time_norm, self._time_stats))
        U = torch.stack(U, dim=-1)

        if self.learn_ard:
            alpha = torch.nn.functional.softplus(self.theta) + self.eps
        else:
            alpha = self.alpha_fixed
        B = alpha.unsqueeze(1) * self.B0

        Z = U @ B
        return torch.cat([torch.sin(Z), torch.cos(Z)], dim=-1)
