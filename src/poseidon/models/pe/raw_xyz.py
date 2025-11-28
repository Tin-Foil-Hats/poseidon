import torch

from .base import PEBase
from .registry import register_pe
from .utils import deg2rad, wrap_lon_pi


@register_pe(name="rawxyz", aliases=["raw_xyz", "perawxyz"])
class PERawXYZ(PEBase):
    """Geocentric xyz plus optional fixed-frequency time harmonics."""

    def __init__(self, use_time: bool = True, omegas=None):
        super().__init__()
        self.use_time = bool(use_time)
        self.omegas = list(omegas or [])

    def feat_dim(self) -> int:
        return 3 + (2 * len(self.omegas) if self.use_time else 0)

    def forward(self, lat_deg, lon_deg, t_sec=None):
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        feats = [x, y, z]
        if self.use_time and (t_sec is not None) and self.omegas:
            for w in self.omegas:
                feats += [torch.sin(w * t_sec), torch.cos(w * t_sec)]
        return torch.stack(feats, dim=-1)
