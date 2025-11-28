from __future__ import annotations

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, time_normalize


@register_pe(name="time_only", aliases=["temporal_only", "time"])
class PETimeOnly(PEBase):
    """Positional encoder that exposes only time-derived features."""

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
        builder = FeatureBuilder()

        if t_sec is None:
            ref = None
            for candidate in (lat_deg, lon_deg):
                if candidate is not None:
                    ref = candidate
                    break
            if ref is None:
                raise ValueError("PETimeOnly requires at least one tensor to infer batch shape.")
            t_sec = torch.zeros_like(ref, dtype=torch.float32)
        else:
            t_sec = t_sec.float()

        t_enc = time_normalize(t_sec, self.time_norm, self._time_stats)
        if t_enc is None:
            t_enc = t_sec
        builder.add_time(t_enc[..., None])

        if self.include_bias:
            bias = torch.ones_like(t_enc, dtype=t_enc.dtype)
            builder.add_other(bias[..., None])

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
        t_enc = time_normalize(sample, self.time_norm, self._time_stats)
        if t_enc is None:
            t_enc = sample
        builder.add_time(t_enc[..., None])
        if self.include_bias:
            builder.add_other(torch.ones_like(sample)[..., None])
        builder.build()
        self._layout_cache = builder.layout()
