from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, time_normalize, wrap_lon_pi


def _minmax_scale(values: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    rng = max(hi - lo, 1e-6)
    return (values - lo) / rng * 2.0 - 1.0


@register_pe(name="xyzminmax", aliases=("pe_xyzminmax",))
class PEXyzMinMax(PEBase):
    """Positional encoding with xyz lift plus min-max normalization."""

    def __init__(
        self,
        radius: float = 1.0,
        include_time: bool = True,
        time_norm: str = "minmax",
        spatial_steps: int = 128,
        time_cycles_per_year: tuple[float, ...] | None = None,
        time_cycle_period_days: float = 365.2425,
    ):
        super().__init__()
        self.radius = float(radius)
        self.include_time = bool(include_time)
        self.time_norm = str(time_norm).lower()
        self.spatial_steps = max(int(spatial_steps), 8)
        if not self.include_time:
            time_cycles_per_year = ()
        cycles = tuple(float(freq) for freq in (time_cycles_per_year or ()))
        self.time_cycles: tuple[float, ...] = tuple(freq for freq in cycles if abs(freq) > 0.0)
        self.time_cycle_period_sec = max(float(time_cycle_period_days) * 86400.0, 1.0)
        self._time_stats: dict | None = None
        self._layout_cache: dict | None = None
        self._spatial_bounds: Dict[str, Tuple[float, float]] | None = None

    def bind_context(self, ctx: dict | None) -> None:
        self._time_stats = ctx.get("time") if ctx is not None else None
        bbox = (ctx or {}).get("bbox") if ctx is not None else None
        if bbox is None:
            self._spatial_bounds = None
            return

        lat_min = math.radians(float(bbox["lat_min"]))
        lat_max = math.radians(float(bbox["lat_max"]))
        lon_min = math.radians(float(bbox["lon_min"]))
        lon_max = math.radians(float(bbox["lon_max"]))

        lat_lin = torch.linspace(lat_min, lat_max, steps=self.spatial_steps)
        lon_lin = torch.linspace(lon_min, lon_max, steps=self.spatial_steps)
        lat_grid, lon_grid = torch.meshgrid(lat_lin, lon_lin, indexing="ij")
        cos_lat = torch.cos(lat_grid)
        x = cos_lat * torch.cos(lon_grid)
        y = cos_lat * torch.sin(lon_grid)
        z = torch.sin(lat_grid)
        self._spatial_bounds = {
            "x": (float(x.min()), float(x.max())),
            "y": (float(y.min()), float(y.max())),
            "z": (float(z.min()), float(z.max())),
        }

    def feat_dim(self) -> int:
        if not self.include_time:
            return 3
        return 3 + 1 + 2 * len(self.time_cycles)

    def forward(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        t_sec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lat = deg2rad(lat_deg.float())
        lon = wrap_lon_pi(deg2rad(lon_deg.float()))
        cos_lat = torch.cos(lat)
        x = cos_lat * torch.cos(lon)
        y = cos_lat * torch.sin(lon)
        z = torch.sin(lat)
        coords = torch.stack((x, y, z), dim=-1) * self.radius

        if self._spatial_bounds is not None:
            bounds_x = self._spatial_bounds["x"]
            bounds_y = self._spatial_bounds["y"]
            bounds_z = self._spatial_bounds["z"]
            coords_x = _minmax_scale(coords[..., 0], bounds_x)
            coords_y = _minmax_scale(coords[..., 1], bounds_y)
            coords_z = _minmax_scale(coords[..., 2], bounds_z)
            coords = torch.stack((coords_x, coords_y, coords_z), dim=-1)

        builder = FeatureBuilder()
        builder.add_space(coords)

        if self.include_time:
            if t_sec is None:
                t_sec = torch.zeros_like(lat, dtype=torch.float32)
            t_scaled = time_normalize(t_sec.float(), self.time_norm, self._time_stats)
            if t_scaled is None:
                t_scaled = t_sec.float()
            if t_scaled.dim() == coords.dim() - 1:
                t_scaled = t_scaled.unsqueeze(-1)
            builder.add_time(t_scaled)

            if self.time_cycles:
                if self._time_stats is not None and "mean" in self._time_stats:
                    center = float(self._time_stats.get("mean", 0.0))
                else:
                    center = 0.0
                t_centered = t_sec.float() - center
                base = t_centered / self.time_cycle_period_sec
                two_pi = 2.0 * math.pi
                cycle_terms = []
                for freq in self.time_cycles:
                    phase = base * (two_pi * freq)
                    cycle_terms.append(torch.sin(phase))
                    cycle_terms.append(torch.cos(phase))
                cycle_tensor = torch.stack(cycle_terms, dim=-1)
                builder.add_time(cycle_tensor)

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int | list[str]]:
        if self._layout_cache is None:
            layout: dict[str, int | list[str]] = {"space": 3, "order": ["space"], "full": self.feat_dim()}
            if self.include_time:
                layout["time"] = 1 + 2 * len(self.time_cycles)
                layout["order"].append("time")
            return layout
        return dict(self._layout_cache)
