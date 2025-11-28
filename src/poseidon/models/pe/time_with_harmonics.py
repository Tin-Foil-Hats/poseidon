from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

from poseidon.constants import (
    DEFAULT_DIURNAL_PERIODS_S,
    DEFAULT_SEMIDIURNAL_PERIODS_S,
    DEFAULT_TIDAL_PERIODS_S,
)

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, time_normalize


def _to_periods(seq: Iterable[float] | None) -> list[float]:
    return [float(p) for p in seq] if seq is not None else []


@register_pe(name="time_with_harmonics", aliases=["temporal_harmonics"])
class PETimeWithHarmonics(PEBase):
    """Raw lat/lon plus enriched temporal harmonics (Fourier bands + tidal cycles)."""

    def __init__(
        self,
        *,
        time_norm: str = "zscore",
        include_bias: bool = False,
        fourier_bands: int = 0,
        use_centered_time: bool = True,
        include_diurnal: bool = False,
        include_semidiurnal: bool = False,
        include_default_tides: bool = False,
        include_weekly: bool = False,
        include_annual: bool = False,
        extra_periods_s: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.time_norm = str(time_norm).lower()
        self.include_bias = bool(include_bias)
        self.use_centered_time = bool(use_centered_time)
        self.fourier_bands = max(int(fourier_bands), 0)

        periods: list[float] = []
        if include_default_tides:
            periods.extend(DEFAULT_TIDAL_PERIODS_S)
        else:
            if include_diurnal:
                periods.extend(DEFAULT_DIURNAL_PERIODS_S)
            if include_semidiurnal:
                periods.extend(DEFAULT_SEMIDIURNAL_PERIODS_S)
        if include_weekly:
            periods.append(7.0 * 24.0 * 3600.0)
        if include_annual:
            periods.append(365.2422 * 24.0 * 3600.0)
        periods.extend(_to_periods(extra_periods_s))

                                            
        seen: set[float] = set()
        dedup_periods: list[float] = []
        for p in periods:
            if p > 0 and p not in seen:
                dedup_periods.append(float(p))
                seen.add(float(p))
        self.periods = dedup_periods

        freqs = 2.0 * math.pi * torch.arange(1, self.fourier_bands + 1, dtype=torch.float32)
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        omegas = torch.tensor([
            2.0 * math.pi / p for p in self.periods
        ], dtype=torch.float32) if self.periods else torch.empty(0, dtype=torch.float32)
        self.register_buffer("period_omegas", omegas, persistent=False)

        self._time_stats: dict | None = None
        self._layout_cache: dict | None = None

    def bind_context(self, ctx: dict | None) -> None:
        self._time_stats = ctx.get("time") if ctx is not None else None
        self._layout_cache = None

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(
        self,
        lat_deg: torch.Tensor | None,
        lon_deg: torch.Tensor | None,
        t_sec: torch.Tensor | None,
    ) -> torch.Tensor:
        if lat_deg is None or lon_deg is None:
            raise ValueError("PETimeWithHarmonics requires both latitude and longitude inputs.")

        lat = lat_deg.float()
        lon = lon_deg.float()
        if t_sec is None:
            t_input = torch.zeros_like(lat, dtype=torch.float32)
        else:
            t_input = t_sec.float()

        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)

        builder = FeatureBuilder()
        builder.add_space(torch.sin(lat_rad)[..., None])
        builder.add_space(torch.cos(lat_rad)[..., None])
        builder.add_space(torch.sin(lon_rad)[..., None])
        builder.add_space(torch.cos(lon_rad)[..., None])

                                      
        t_normed = time_normalize(t_input, self.time_norm, self._time_stats)
        if t_normed is None:
            t_normed = t_input
        builder.add_time(t_normed[..., None])

                                                                    
        if self.use_centered_time:
            stats_mean = float((self._time_stats or {}).get("mean", 0.0))
            t_center = t_input - stats_mean
        else:
            t_center = t_input

        if self.fourier_freqs.numel() > 0:
            scaled = t_normed[..., None] * self.fourier_freqs
            builder.add_time(torch.sin(scaled))
            builder.add_time(torch.cos(scaled))

        if self.period_omegas.numel() > 0:
            phases = t_center[..., None] * self.period_omegas
            builder.add_time(torch.sin(phases))
            builder.add_time(torch.cos(phases))

        if self.include_bias:
            builder.add_other(torch.ones_like(t_normed)[..., None])

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is None:
            self._update_layout_cache()
        return dict(self._layout_cache) if self._layout_cache is not None else {"full": 0}

    def _update_layout_cache(self) -> None:
        sample = torch.zeros(1, dtype=torch.float32)
        builder = FeatureBuilder()
        sample_rad = sample * (math.pi / 180.0)
        builder.add_space(torch.sin(sample_rad)[..., None])
        builder.add_space(torch.cos(sample_rad)[..., None])
        builder.add_space(torch.sin(sample_rad)[..., None])
        builder.add_space(torch.cos(sample_rad)[..., None])
        t_normed = time_normalize(sample, self.time_norm, self._time_stats)
        if t_normed is None:
            t_normed = sample
        builder.add_time(t_normed[..., None])

        if self.fourier_freqs.numel() > 0:
            scaled = t_normed[..., None] * self.fourier_freqs
            builder.add_time(torch.sin(scaled))
            builder.add_time(torch.cos(scaled))

        stats_mean = float((self._time_stats or {}).get("mean", 0.0))
        if self.period_omegas.numel() > 0:
            t_center = sample - stats_mean if self.use_centered_time else sample
            phases = t_center[..., None] * self.period_omegas
            builder.add_time(torch.sin(phases))
            builder.add_time(torch.cos(phases))

        if self.include_bias:
            builder.add_other(torch.ones_like(sample)[..., None])

        builder.build()
        self._layout_cache = builder.layout()
