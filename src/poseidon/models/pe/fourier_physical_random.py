from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, time_normalize, wrap_lon_pi

_EARTH_RADIUS_KM = 6371.0088


def _validate_range(bounds: Tuple[float, float], *, name: str) -> Tuple[float, float]:
    lo, hi = bounds
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"{name} bounds must be finite")
    if lo <= 0 or hi <= 0:
        raise ValueError(f"{name} bounds must be > 0")
    if hi <= lo:
        raise ValueError(f"{name} bounds must satisfy hi > lo")
    return float(lo), float(hi)


def _sample_lengths(
    count: int,
    bounds: Tuple[float, float],
    *,
    log_uniform: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0)
    lo, hi = bounds
    if log_uniform:
        lo_log = math.log(lo)
        hi_log = math.log(hi)
        samples = torch.rand(count, generator=generator).mul_(hi_log - lo_log).add_(lo_log).exp_()
    else:
        samples = torch.rand(count, generator=generator).mul_(hi - lo).add_(lo)
    return samples


def _sample_spatial_matrix(
    count: int,
    bounds: Tuple[float, float],
    *,
    log_uniform: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = _sample_lengths(count, bounds, log_uniform=log_uniform, generator=generator)
    if lengths.numel() == 0:
        return torch.empty(2, 0), lengths
    dirs = torch.randn(2, count, generator=generator)
    dirs = dirs / torch.clamp(dirs.norm(dim=0, keepdim=True), min=1e-6)
    omegas = (2.0 * math.pi) / lengths
    matrix = dirs * omegas
    return matrix, lengths


def _sample_temporal_freqs(
    count: int,
    bounds_hours: Tuple[float, float],
    *,
    log_uniform: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    lengths = _sample_lengths(count, bounds_hours, log_uniform=log_uniform, generator=generator)
    if lengths.numel() == 0:
        return torch.empty(0)
    length_seconds = lengths * 3600.0
    return (2.0 * math.pi) / length_seconds


@register_pe(name="fourier_physical_random", aliases=("pe_fourier_physical_random", "rff_physical"))
class PERandomFourierPhysical(PEBase):
    """Random Fourier features drawn over physical spatial and temporal scales."""

    def __init__(
        self,
        *,
        spatial_features: int = 256,
        temporal_features: int = 128,
        spatial_scale_bounds_km: Tuple[float, float] = (20.0, 500.0),
        temporal_scale_bounds_hr: Tuple[float, float] = (6.0, 240.0),
        spatial_log_uniform: bool = True,
        temporal_log_uniform: bool = True,
        include_time_channel: bool = False,
        time_norm: str = "zscore",
        seed: Optional[int] = None,
        normalize_features: bool = False,
        include_annual: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_features = max(int(spatial_features), 0)
        self.temporal_features = max(int(temporal_features), 0)
        self.include_time_channel = bool(include_time_channel)
        self.time_norm = str(time_norm).lower()
        self.normalize_features = bool(normalize_features)
        self.include_annual = bool(include_annual)
        self.spatial_scale_bounds_km = _validate_range(spatial_scale_bounds_km, name="spatial_scale_bounds_km")
        self.temporal_scale_bounds_hr = _validate_range(temporal_scale_bounds_hr, name="temporal_scale_bounds_hr")
        self.spatial_log_uniform = bool(spatial_log_uniform)
        self.temporal_log_uniform = bool(temporal_log_uniform)

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        space_matrix, space_lengths = _sample_spatial_matrix(
            self.spatial_features,
            self.spatial_scale_bounds_km,
            log_uniform=self.spatial_log_uniform,
            generator=gen,
        )
        time_freqs = _sample_temporal_freqs(
            self.temporal_features,
            self.temporal_scale_bounds_hr,
            log_uniform=self.temporal_log_uniform,
            generator=gen,
        )

        self.register_buffer("space_matrix", space_matrix, persistent=False)
        self.register_buffer("space_lengths_km", space_lengths, persistent=False)
        self.register_buffer("time_freqs", time_freqs, persistent=False)
        if self.include_annual:
            annual_freqs = torch.tensor(
                [
                    2.0 * math.pi / (365.2422 * 86400.0),
                    2.0 * math.pi / (182.6211 * 86400.0),
                ],
                dtype=torch.float32,
            )
        else:
            annual_freqs = torch.empty(0)
        self.register_buffer("annual_freqs", annual_freqs, persistent=False)

        self._time_stats: Optional[dict] = None
        self._layout_cache: Optional[dict] = None
        self._time_ref: float = 0.0

    def bind_context(self, ctx: Optional[dict]) -> None:
        self._time_stats = (ctx or {}).get("time")
        self._layout_cache = None
        stats = self._time_stats or {}
        mean_val = stats.get("mean") if isinstance(stats, dict) else None
        self._time_ref = float(mean_val) if mean_val is not None else 0.0

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        t_sec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        builder = FeatureBuilder()

        lat = deg2rad(lat_deg.float())
        lon = wrap_lon_pi(deg2rad(lon_deg.float()))
        cos_lat = torch.cos(lat)
        lat_km = lat * _EARTH_RADIUS_KM
        lon_km = lon * (_EARTH_RADIUS_KM * cos_lat.clamp(min=1e-3))
        coords = torch.stack((lat_km, lon_km), dim=-1)

        if self.space_matrix.numel() == 0:
            raise RuntimeError("PERandomFourierPhysical was configured with zero spatial features.")
        space_mat = self.space_matrix.to(device=coords.device, dtype=coords.dtype)
        proj_space = coords @ space_mat
        builder.add_space(torch.sin(proj_space))
        builder.add_space(torch.cos(proj_space))

        if self.temporal_features > 0:
            if t_sec is None:
                raise ValueError("PERandomFourierPhysical requires t_sec when temporal features are enabled.")
            t_float = t_sec.float()
            centers = time_normalize(t_float, self.time_norm, self._time_stats)
            time_base = centers if centers is not None else t_float
            if self.include_time_channel:
                builder.add_time(time_base[..., None])
            time_freqs = self.time_freqs.to(device=t_float.device, dtype=t_float.dtype)
            proj_time = t_float.unsqueeze(-1) * time_freqs
            builder.add_time(torch.sin(proj_time))
            builder.add_time(torch.cos(proj_time))
        elif self.include_time_channel and t_sec is not None:
            t_float = t_sec.float()
            centers = time_normalize(t_float, self.time_norm, self._time_stats)
            time_base = centers if centers is not None else t_float
            builder.add_time(time_base[..., None])

        if self.include_annual:
            if t_sec is None:
                raise ValueError("PERandomFourierPhysical requires t_sec when include_annual is true.")
            t_float = t_sec.float()
            t_rel = t_float - self._time_ref
            annual_freqs = self.annual_freqs.to(device=t_float.device, dtype=t_float.dtype)
            if annual_freqs.numel() > 0:
                proj_annual = t_rel.unsqueeze(-1) * annual_freqs
                annual_feats = torch.cat([torch.sin(proj_annual), torch.cos(proj_annual)], dim=-1)
                builder.add_time(annual_feats)

        features = builder.build()
        if self.normalize_features:
            features = torch.clamp(features, min=-1.0, max=1.0)
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is None:
            layout = {"space": 2 * self.spatial_features, "time": 0, "order": ["space"]}
            time_dim = 0
            if self.temporal_features > 0:
                time_dim = 2 * self.temporal_features + int(self.include_time_channel)
            elif self.include_time_channel:
                time_dim = 1
            if self.include_annual and self.annual_freqs.numel() > 0:
                time_dim += 2 * int(self.annual_freqs.numel())
            layout["time"] = time_dim
            if time_dim > 0:
                layout["order"].append("time")
            layout["full"] = layout["space"] + layout.get("time", 0)
            return layout
        return dict(self._layout_cache)

    def extra_repr(self) -> str:
        return (
            f"spatial_features={self.spatial_features}, temporal_features={self.temporal_features}, "
            f"spatial_scale_bounds_km={self.spatial_scale_bounds_km}, "
            f"temporal_scale_bounds_hr={self.temporal_scale_bounds_hr}"
        )
