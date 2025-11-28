import math
import torch
import torch.nn as nn

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, wrap_lon_pi, time_normalize


@register_pe(name="fourier", aliases=["pe_fourier"])
class PEFourier(PEBase):
    """Deterministic Fourier features; time input uses configurable normalization."""

    def __init__(self, n_lat=16, n_lon=16, n_t=8, use_time=True, add_xyz=True, time_norm: str = "zscore"):
        super().__init__()
        self.use_time = bool(use_time)
        self.add_xyz = bool(add_xyz)
        self.time_norm = str(time_norm).lower()
        self.Blat = nn.Parameter(2 * math.pi * torch.linspace(1, n_lat, n_lat), requires_grad=False)
        self.Blon = nn.Parameter(2 * math.pi * torch.linspace(1, n_lon, n_lon), requires_grad=False)
        self.Bt = nn.Parameter(2 * math.pi * torch.linspace(1, n_t, n_t), requires_grad=False)
        self._time_stats: dict | None = None
        self._layout_cache: dict | None = None

    def bind_context(self, ctx: dict):
        self._time_stats = ctx.get("time") if ctx is not None else None

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(self, lat_deg, lon_deg, t_sec=None):
        builder = FeatureBuilder()
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))

        for angle, bank in ((lat, self.Blat), (lon, self.Blon)):
            proj = angle[..., None] * bank
            builder.add_space(torch.sin(proj))
            builder.add_space(torch.cos(proj))

        if self.add_xyz:
            x = (torch.cos(lat) * torch.cos(lon))[..., None]
            y = (torch.cos(lat) * torch.sin(lon))[..., None]
            z = torch.sin(lat)[..., None]
            builder.add_space(x)
            builder.add_space(y)
            builder.add_space(z)

        if self.use_time and t_sec is not None:
            t_ch = time_normalize(t_sec, self.time_norm, self._time_stats)
            if t_ch is not None:
                proj_t = t_ch[..., None] * self.Bt
                builder.add_time(torch.sin(proj_t))
                builder.add_time(torch.cos(proj_t))

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is None:
            self._update_layout_cache()
        return dict(self._layout_cache) if self._layout_cache is not None else {"full": self.feat_dim()}

    def _update_layout_cache(self) -> None:
        device = self.Blat.device
        zeros = torch.zeros(1, dtype=torch.float32, device=device)
        features = self.forward(zeros, zeros, zeros if self.use_time else None)
        del features                                     
