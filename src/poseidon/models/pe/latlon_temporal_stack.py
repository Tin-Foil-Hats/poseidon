from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, time_normalize


def _minmax(values: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    rng = max(hi - lo, 1e-6)
    return (values - lo) / rng * 2.0 - 1.0


def _to_float_seq(values: Sequence[float] | None) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in values]


@register_pe(name="latlon_temporal_stack")
class PELatLonTemporalStack(PEBase):
    """Hybrid encoder with min-max spatial scaling and rich temporal cycles.

    The encoder exposes bounded spatial coordinates, optional spherical
    sin/cos features, a normalized time scalar, and configurable seasonal
    sinusoids (annual/weekly/daily or custom periods). This helps SIREN models
    latch onto well-defined low-frequency structure before modeling residuals.
    """

    def __init__(
        self,
        *,
        include_space_minmax: bool = True,
        include_space_trig: bool = True,
        include_time_scalar: bool = True,
        time_norm: str = "minmax",
        time_fill: float = 0.0,
        center_time_for_cycles: bool = True,
        annual_period_days: float = 365.2422,
        annual_harmonics: Sequence[int] | None = (1,),
        include_weekly_cycle: bool = True,
        include_daily_cycle: bool = False,
        extra_cycle_period_hours: Sequence[float] | None = None,
        default_lat_bounds: Tuple[float, float] = (-90.0, 90.0),
        default_lon_bounds: Tuple[float, float] = (-180.0, 180.0),
        default_time_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.include_space_minmax = bool(include_space_minmax)
        self.include_space_trig = bool(include_space_trig)
        self.include_time_scalar = bool(include_time_scalar)
        self.time_norm = str(time_norm).lower()
        self.time_fill = float(time_fill)
        self.center_time_for_cycles = bool(center_time_for_cycles)

        self._lat_bounds = tuple(float(v) for v in default_lat_bounds)
        self._lon_bounds = tuple(float(v) for v in default_lon_bounds)
        self._time_bounds = tuple(float(v) for v in default_time_bounds)

        periods_sec: list[float] = []
        extra_periods = _to_float_seq(extra_cycle_period_hours)
        periods_sec.extend([h * 3600.0 for h in extra_periods if h > 0])
        if include_weekly_cycle:
            periods_sec.append(7.0 * 24.0 * 3600.0)
        if include_daily_cycle:
            periods_sec.append(24.0 * 3600.0)

        base_annual_period = max(float(annual_period_days), 1e-3) * 24.0 * 3600.0
        annual_freqs = []
        if annual_harmonics:
            base_omega = 2.0 * math.pi / base_annual_period
            for h in annual_harmonics:
                if h is None:
                    continue
                k = int(h)
                if k > 0:
                    annual_freqs.append(base_omega * k)

        seen_periods: set[float] = set()
        for period in periods_sec:
            if period > 0 and period not in seen_periods:
                seen_periods.add(period)

        cycle_omegas = [2.0 * math.pi / p for p in seen_periods]
        cycle_omegas.extend(annual_freqs)
        if cycle_omegas:
            omega_tensor = torch.tensor(sorted(cycle_omegas), dtype=torch.float32)
        else:
            omega_tensor = torch.empty(0, dtype=torch.float32)
        self.register_buffer("cycle_omegas", omega_tensor, persistent=False)

        self._time_stats: Optional[dict] = None
        self._layout_cache: Optional[dict] = None

    def bind_context(self, ctx: Optional[dict]) -> None:
        ctx = ctx or {}
        bbox = ctx.get("bbox") if isinstance(ctx, dict) else None
        if isinstance(bbox, dict):
            lat_min = float(bbox.get("lat_min", self._lat_bounds[0]))
            lat_max = float(bbox.get("lat_max", self._lat_bounds[1]))
            lon_min = float(bbox.get("lon_min", self._lon_bounds[0]))
            lon_max = float(bbox.get("lon_max", self._lon_bounds[1]))
            if lat_max <= lat_min:
                lat_max = lat_min + 1e-3
            if lon_max <= lon_min:
                lon_max = lon_min + 1e-3
            self._lat_bounds = (lat_min, lat_max)
            self._lon_bounds = (lon_min, lon_max)

        self._time_stats = ctx.get("time") if isinstance(ctx, dict) else None
        if isinstance(self._time_stats, dict):
            tmin = self._time_stats.get("min")
            tmax = self._time_stats.get("max")
            if tmin is not None and tmax is not None and tmax > tmin:
                self._time_bounds = (float(tmin), float(tmax))
        self._layout_cache = None

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        t_sec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lat = lat_deg.float()
        lon = lon_deg.float()
        builder = FeatureBuilder()

        if self.include_space_minmax:
            lat_scaled = _minmax(lat, self._lat_bounds)[..., None]
            lon_scaled = _minmax(lon, self._lon_bounds)[..., None]
            builder.add_space(lat_scaled)
            builder.add_space(lon_scaled)

        if self.include_space_trig:
            lat_rad = lat * (math.pi / 180.0)
            lon_rad = lon * (math.pi / 180.0)
            builder.add_space(torch.sin(lat_rad)[..., None])
            builder.add_space(torch.cos(lat_rad)[..., None])
            builder.add_space(torch.sin(lon_rad)[..., None])
            builder.add_space(torch.cos(lon_rad)[..., None])

        if t_sec is None:
            t_values = lat.new_full(lat.shape, self.time_fill)
        else:
            t_values = t_sec.float()

        t_normed = None
        if self.include_time_scalar:
            t_normed = time_normalize(t_values, self.time_norm, self._time_stats)
            if t_normed is None:
                if self.time_norm == "minmax":
                    t_normed = _minmax(t_values, self._time_bounds)
                else:
                    # Fall back to z-score-like scaling if stats unavailable.
                    mean = 0.5 * sum(self._time_bounds)
                    std = max((self._time_bounds[1] - self._time_bounds[0]) / 2.0, 1.0)
                    t_normed = (t_values - mean) / std
            builder.add_time(t_normed[..., None])

        if self.cycle_omegas.numel() > 0:
            if t_normed is None:
                pivot = t_values
            else:
                pivot = t_values
            if self.center_time_for_cycles:
                stats_mean = float((self._time_stats or {}).get("mean", 0.0))
                pivot = pivot - stats_mean
            phases = pivot[..., None] * self.cycle_omegas
            builder.add_time(torch.sin(phases))
            builder.add_time(torch.cos(phases))

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is not None:
            return dict(self._layout_cache)

        dummy = torch.zeros(1)
        feats = self.forward(dummy, dummy, dummy)
        return dict(self._layout_cache) if self._layout_cache is not None else {"full": feats.shape[-1]}
