from __future__ import annotations

import math

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, time_normalize


@register_pe(name="time_with_raw", aliases=["time_raw", "raw_time"])
class PETimeWithRaw(PEBase):
    """Encode lat/lon via sin/cos of radians plus a normalized time channel."""

    def __init__(self, time_norm: str = "zscore", include_bias: bool = False) -> None:
        super().__init__()
        self.time_norm = str(time_norm).lower()
        self.include_bias = bool(include_bias)
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
            raise ValueError("PETimeWithRaw requires both latitude and longitude inputs.")

        lat = lat_deg.float()
        lon = lon_deg.float()
        t_input = t_sec.float() if t_sec is not None else torch.zeros_like(lat, dtype=torch.float32)

        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)

        builder = FeatureBuilder()
        builder.add_space(torch.sin(lat_rad)[..., None])
        builder.add_space(torch.cos(lat_rad)[..., None])
        builder.add_space(torch.sin(lon_rad)[..., None])
        builder.add_space(torch.cos(lon_rad)[..., None])

        t_enc = time_normalize(t_input, self.time_norm, self._time_stats)
        if t_enc is None:
            t_enc = t_input
        builder.add_time(t_enc[..., None])

        if self.include_bias:
            builder.add_other(torch.ones_like(t_enc)[..., None])

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
        builder.add_space(torch.sin(sample * (math.pi / 180.0))[..., None])
        builder.add_space(torch.cos(sample * (math.pi / 180.0))[..., None])
        builder.add_space(torch.sin(sample * (math.pi / 180.0))[..., None])
        builder.add_space(torch.cos(sample * (math.pi / 180.0))[..., None])
        t_enc = time_normalize(sample, self.time_norm, self._time_stats)
        if t_enc is None:
            t_enc = sample
        builder.add_time(t_enc[..., None])
        if self.include_bias:
            builder.add_other(torch.ones_like(sample)[..., None])
        builder.build()
        self._layout_cache = builder.layout()
