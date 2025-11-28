from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .base import PEBase
from poseidon.constants import DEFAULT_TIDAL_PERIODS_S
from .registry import register_pe
from .utils import FeatureBuilder


def _associated_legendre_polynomial(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt(torch.clamp(1 - x, min=0) * torch.clamp(1 + x, min=0))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _sh_renorm(l: int, m: int) -> float:
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4.0 * math.pi * math.factorial(l + m)))


def _real_spherical_harmonic(m: int, l: int, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if m == 0:
        return _sh_renorm(l, 0) * _associated_legendre_polynomial(l, 0, torch.cos(theta))
    if m > 0:
        return math.sqrt(2.0) * _sh_renorm(l, m) * torch.cos(m * phi) * _associated_legendre_polynomial(l, m, torch.cos(theta))
    return math.sqrt(2.0) * _sh_renorm(l, -m) * torch.sin(-m * phi) * _associated_legendre_polynomial(l, -m, torch.cos(theta))


@register_pe(name="satclip_sh", aliases=["spherical_harmonics_satclip"])
class SatclipSphericalHarmonics(PEBase):
    """Spherical harmonic positional encoder inspired by SatCLIP. currently calculates on the fly, need to hardcode elements later."""

    def __init__(
        self,
        max_degree: int = 6,
        normalize: bool = False,
        epsilon: float = 1e-6,
        include_time: bool = False,
        time_periods_s: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        if max_degree < 1:
            raise ValueError("max_degree must be >= 1")
        self.max_degree = int(max_degree)
        self.normalize = bool(normalize)
        self.epsilon = float(epsilon)
        self.include_time = bool(include_time)
        if time_periods_s is None and self.include_time:
            time_periods_s = list(DEFAULT_TIDAL_PERIODS_S)
        if time_periods_s is not None:
            uniq = []
            seen = set()
            for val in time_periods_s:
                val = float(val)
                if val <= 0:
                    continue
                if val in seen:
                    continue
                uniq.append(val)
                seen.add(val)
            time_periods_s = uniq
        self.time_periods_s = time_periods_s
        self._layout: Optional[dict] = None

    def feat_dim(self) -> int:
        base = self.max_degree * self.max_degree
        if self.include_time and self.time_periods_s:
            base += 2 * len(self.time_periods_s)
        return base

    def feature_layout(self) -> dict[str, int]:
        if self._layout is None:
            layout = {"space": self.max_degree * self.max_degree}
            if self.include_time and self.time_periods_s:
                layout["time"] = 2 * len(self.time_periods_s)
            layout["full"] = self.feat_dim()
            return layout
        return dict(self._layout)

    def forward(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor, t_sec: Optional[torch.Tensor] = None):
        lat_flat = lat_deg.reshape(-1)
        lon_flat = lon_deg.reshape(-1)
        if lat_flat.numel() == 0:
            raise ValueError("SatclipSphericalHarmonics received empty input")

        phi = torch.deg2rad(lon_flat + 180.0)
                                                                         
        theta = torch.deg2rad(90.0 - lat_flat)

        comps = []
        for l in range(self.max_degree):
            for m in range(-l, l + 1):
                y = _real_spherical_harmonic(m, l, phi, theta)
                comps.append(y)
        features = torch.stack(comps, dim=1)

        if self.normalize:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            std = torch.clamp(std, min=self.epsilon)
            features = (features - mean) / std

        builder = FeatureBuilder()
        builder.add_space(features)

        if self.include_time and self.time_periods_s:
            if t_sec is None:
                raise ValueError("SatclipSphericalHarmonics include_time=True requires time input")
            t_flat = t_sec.reshape(-1).float()
            time_feats = []
            for period in self.time_periods_s:
                omega = 2.0 * math.pi / period
                time_feats.append(torch.sin(omega * t_flat))
                time_feats.append(torch.cos(omega * t_flat))
            time_matrix = torch.stack(time_feats, dim=1)
            if self.normalize:
                mean_t = time_matrix.mean(dim=0, keepdim=True)
                std_t = time_matrix.std(dim=0, keepdim=True)
                std_t = torch.clamp(std_t, min=self.epsilon)
                time_matrix = (time_matrix - mean_t) / std_t
            builder.add_time(time_matrix)

        out = builder.build()
        self._layout = builder.layout()
        return out