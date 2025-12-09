from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, time_normalize, wrap_lon_pi

_EARTH_RADIUS_KM = 6371.0088


def _coerce_scales(values: Iterable[float] | None, *, min_scale_km: float = 1.0) -> tuple[float, ...]:
    if values is None:
        return ()
    parsed = []
    for value in values:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        if scale < min_scale_km:
            scale = min_scale_km
        parsed.append(scale)
    return tuple(sorted(set(parsed)))


def _minmax_scale_tensor(values: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    rng = max(hi - lo, 1e-6)
    return (values - lo) / rng * 2.0 - 1.0


@register_pe(name="fourier_physical", aliases=("pe_fourier_physical",))
class PEFourierPhysical(PEBase):
    """Fourier features parameterized by physical spatial/temporal scales.

    The encoder projects latitude/longitude coordinates (in degrees) onto a
    bank of sinusoidal features whose wavelengths correspond to the provided
    ``spatial_scales_km``. Temporal inputs are expanded similarly according to
    ``time_scales_hours``. Frequencies are derived using the domain's bounding
    box (if available) to ensure longitude distances respect the mean latitude.
    """

    def __init__(
        self,
        spatial_scales_km: Sequence[float] = (20.0, 50.0, 100.0, 200.0),
        time_scales_hours: Sequence[float] | None = (6.0, 24.0, 168.0),
        include_xyz: bool = True,
        add_time_channel: bool = True,
        time_norm: str = "zscore",
        spatial_steps: int = 256,
    ) -> None:
        super().__init__()
        spatial = _coerce_scales(spatial_scales_km, min_scale_km=1.0)
        if not spatial:
            raise ValueError("fourier_physical requires at least one spatial scale")
        self.spatial_scales_km = spatial
        self.include_xyz = bool(include_xyz)
        self.add_time_channel = bool(add_time_channel)
        self.time_norm = str(time_norm)
        self.time_scales_hours = _coerce_scales(time_scales_hours, min_scale_km=1.0)
        self.spatial_steps = max(int(spatial_steps), 8)

        self._freq_lat = torch.empty(len(self.spatial_scales_km), dtype=torch.float32)
        self._freq_lon = torch.empty(len(self.spatial_scales_km), dtype=torch.float32)
        self._freq_time = torch.empty(len(self.time_scales_hours), dtype=torch.float32)

        self._time_stats: dict | None = None
        self._layout_cache: dict | None = None
        self._spatial_bounds: dict[str, tuple[float, float]] | None = None

    def bind_context(self, ctx: dict | None) -> None:
        ctx = ctx or {}
        self._time_stats = ctx.get("time")
        bbox = ctx.get("bbox")
        if bbox is not None:
            lat_min = math.radians(float(bbox.get("lat_min", 0.0)))
            lat_max = math.radians(float(bbox.get("lat_max", 0.0)))
            lon_min = math.radians(float(bbox.get("lon_min", 0.0)))
            lon_max = math.radians(float(bbox.get("lon_max", 0.0)))
        else:
            lon_min = -math.pi
            lon_max = math.pi
            lat_min = -math.pi / 2.0
            lat_max = math.pi / 2.0

        spatial_wavenumbers = [2.0 * math.pi / scale for scale in self.spatial_scales_km]
        self._freq_lat = torch.tensor(spatial_wavenumbers, dtype=torch.float32)
        self._freq_lon = torch.tensor(spatial_wavenumbers, dtype=torch.float32)

        if self.time_scales_hours:
            time_freqs = [2.0 * math.pi / (scale * 3600.0) for scale in self.time_scales_hours]
            self._freq_time = torch.tensor(time_freqs, dtype=torch.float32)
        else:
            self._freq_time = torch.empty(0, dtype=torch.float32)

        lat_lin = torch.linspace(lat_min, lat_max, steps=self.spatial_steps)
        lon_lin = torch.linspace(lon_min, lon_max, steps=self.spatial_steps)
        lat_grid, lon_grid = torch.meshgrid(lat_lin, lon_lin, indexing="ij")
        cos_lat_grid = torch.cos(lat_grid)
        x = cos_lat_grid * torch.cos(lon_grid)
        y = cos_lat_grid * torch.sin(lon_grid)
        z = torch.sin(lat_grid)
        self._spatial_bounds = {
            "x": (float(x.min()), float(x.max())),
            "y": (float(y.min()), float(y.max())),
            "z": (float(z.min()), float(z.max())),
        }

        self._layout_cache = None

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        t_sec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lat = deg2rad(lat_deg.float())
        lon = wrap_lon_pi(deg2rad(lon_deg.float()))
        builder = FeatureBuilder()

        freq_lat = self._freq_lat.to(lat.device)
        freq_lon = self._freq_lon.to(lon.device)
        if freq_lat.numel() == 0 or freq_lon.numel() == 0:
            raise RuntimeError("fourier_physical encoder requires precomputed spatial frequencies")

        lat_km = lat.unsqueeze(-1) * _EARTH_RADIUS_KM
        cos_lat = torch.cos(lat).clamp(min=1e-3)
        lon_km = lon.unsqueeze(-1) * (_EARTH_RADIUS_KM * cos_lat.unsqueeze(-1))

        lat_proj = lat_km * freq_lat
        lon_proj = lon_km * freq_lon
        builder.add_space(torch.sin(lat_proj))
        builder.add_space(torch.cos(lat_proj))
        builder.add_space(torch.sin(lon_proj))
        builder.add_space(torch.cos(lon_proj))

        if self.include_xyz:
            cos_lat = torch.cos(lat)
            x = (cos_lat * torch.cos(lon)).unsqueeze(-1)
            y = (cos_lat * torch.sin(lon)).unsqueeze(-1)
            z = torch.sin(lat).unsqueeze(-1)
            if self._spatial_bounds is not None:
                x = _minmax_scale_tensor(x, self._spatial_bounds["x"])
                y = _minmax_scale_tensor(y, self._spatial_bounds["y"])
                z = _minmax_scale_tensor(z, self._spatial_bounds["z"])
            builder.add_space(x)
            builder.add_space(y)
            builder.add_space(z)

        if self.add_time_channel and t_sec is not None:
            t_norm = time_normalize(t_sec.float(), self.time_norm.lower(), self._time_stats)
            if t_norm is not None:
                builder.add_time(t_norm.unsqueeze(-1))

        if self.time_scales_hours and t_sec is not None and self._freq_time.numel() > 0:
            freq_time = self._freq_time.to(t_sec.device)
            proj_t = t_sec.float().unsqueeze(-1) * freq_time
            builder.add_time(torch.sin(proj_t))
            builder.add_time(torch.cos(proj_t))

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int | list[str]]:
        if self._layout_cache is not None:
            return dict(self._layout_cache)
        layout = {
            "space": (4 * len(self.spatial_scales_km)) + (3 if self.include_xyz else 0),
            "time": 0,
            "order": ["space"],
        }
        if self.add_time_channel:
            layout["time"] += 1
        if self.time_scales_hours:
            layout["time"] += 2 * len(self.time_scales_hours)
        if layout["time"] > 0:
            layout["order"].append("time")
        layout["full"] = layout["space"] + layout["time"]
        return layout
