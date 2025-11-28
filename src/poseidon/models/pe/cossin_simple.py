import math
import torch
import torch.nn as nn

from .base import PEBase
from .registry import register_pe
from .utils import deg2rad, wrap_lon_pi, normalize_lon_bounds


@register_pe(name="cossin_simple", aliases=["cossin"])
class PECosSinSimple(PEBase):
    """Cyclical lat/lon features with optional daily/annual harmonics."""

    def __init__(
        self,
        add_xy: bool = True,
        include_daily: bool = True,
        include_annual: bool = True,
        extra_periods_s=None,
    ):
        super().__init__()
        self.add_xy = bool(add_xy)
        self.include_daily = bool(include_daily)
        self.include_annual = bool(include_annual)
        self.extra_periods_s = list(extra_periods_s or [])

        self.register_buffer("lat_min", torch.tensor(0.0))
        self.register_buffer("lat_max", torch.tensor(1.0))
        self.register_buffer("lon_min", torch.tensor(-1.0))
        self.register_buffer("lon_max", torch.tensor(1.0))

        self.register_buffer("omega_day", torch.tensor(2 * math.pi / 86400.0))
        self.register_buffer("omega_year", torch.tensor(2 * math.pi / (365.2422 * 86400.0)))
        extra = [2 * math.pi / p for p in self.extra_periods_s]
        self.register_buffer("omega_extra", torch.tensor(extra, dtype=torch.float32))
        self._lon_crosses_seam = False

    def bind_context(self, ctx: dict):
        bbox = ctx["bbox"]
        self.lat_min.fill_(float(math.radians(bbox["lat_min"])))
        self.lat_max.fill_(float(math.radians(bbox["lat_max"])))
        lon_min, lon_max, crosses = normalize_lon_bounds(bbox["lon_min"], bbox["lon_max"])
        self.lon_min.fill_(float(lon_min))
        self.lon_max.fill_(float(lon_max))
        self._lon_crosses_seam = crosses

    def feat_dim(self) -> int:
        d = 4
        if self.add_xy:
            d += 2
        if self.include_daily:
            d += 2
        if self.include_annual:
            d += 2
        d += 2 * int(self.omega_extra.numel())
        return d

    def forward(self, lat_deg, lon_deg, t_sec):
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))
        if self._lon_crosses_seam:
            seam_threshold = self.lon_min - self.lon_min.new_tensor(math.pi)
            lon = torch.where(
                lon < seam_threshold,
                lon + self.lon_min.new_tensor(2 * math.pi),
                lon,
            )
        lat_p = 2 * (lat - self.lat_min) / torch.clamp(self.lat_max - self.lat_min, min=1e-9) - 1
        lon_p = 2 * (lon - self.lon_min) / torch.clamp(self.lon_max - self.lon_min, min=1e-9) - 1
        f = [
            torch.sin(lat_p)[..., None],
            torch.cos(lat_p)[..., None],
            torch.sin(lon_p)[..., None],
            torch.cos(lon_p)[..., None],
        ]
        if self.add_xy:
            f += [lat_p[..., None], lon_p[..., None]]
        if self.include_daily:
            a = self.omega_day * t_sec
            f += [torch.sin(a)[..., None], torch.cos(a)[..., None]]
        if self.include_annual:
            a = self.omega_year * t_sec
            f += [torch.sin(a)[..., None], torch.cos(a)[..., None]]
        if self.omega_extra.numel() > 0:
            A = t_sec[..., None] * self.omega_extra
            f += [torch.sin(A), torch.cos(A)]
        return torch.cat(f, dim=-1)
