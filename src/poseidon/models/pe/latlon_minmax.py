from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, time_normalize


def _minmax(values: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    rng = max(hi - lo, 1e-6)
    return (values - lo) / rng * 2.0 - 1.0


@register_pe(name="latlon_minmax", aliases=("pe_latlon_minmax", "location_minmax"))
class PELatLonMinMax(PEBase):
    """Minimal positional encoder with per-axis min/max scaling.

    The encoder maps latitude, longitude, and (optional) time to the [-1, 1]
    interval using dataset statistics supplied via :meth:`bind_context`. When
    context is missing, it falls back to configurable defaults so the network
    still receives bounded coordinates.
    """

    def __init__(
        self,
        *,
        include_time: bool = True,
        time_fill: float = 0.0,
        default_lat_bounds: Tuple[float, float] = (-90.0, 90.0),
        default_lon_bounds: Tuple[float, float] = (-180.0, 180.0),
        default_time_bounds: Tuple[float, float] = (0.0, 1.0),
        time_norm: str = "minmax",
    ) -> None:
        super().__init__()
        self.include_time = bool(include_time)
        self.time_fill = float(time_fill)
        self.time_norm = str(time_norm).lower()
        self._lat_bounds = tuple(float(v) for v in default_lat_bounds)
        self._lon_bounds = tuple(float(v) for v in default_lon_bounds)
        self._time_bounds = tuple(float(v) for v in default_time_bounds)
        self._time_stats: Optional[dict] = None
        self._layout_cache: Optional[dict] = None

    def bind_context(self, ctx: Optional[dict]) -> None:
        ctx = ctx or {}
        bbox = ctx.get("bbox")
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
        self._time_stats = ctx.get("time") if self.include_time else None
        if isinstance(self._time_stats, dict):
            tmin = self._time_stats.get("min")
            tmax = self._time_stats.get("max")
            if tmin is not None and tmax is not None and tmax > tmin:
                self._time_bounds = (float(tmin), float(tmax))
        self._layout_cache = None

    def feat_dim(self) -> int:
        return 2 + (1 if self.include_time else 0)

    def forward(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        t_sec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lat = lat_deg.float()
        lon = lon_deg.float()
        lat_scaled = _minmax(lat, self._lat_bounds)[..., None]
        lon_scaled = _minmax(lon, self._lon_bounds)[..., None]

        builder = FeatureBuilder()
        builder.add_space(lat_scaled)
        builder.add_space(lon_scaled)

        if self.include_time:
            if t_sec is None:
                t = lat.new_full(lat.shape, self.time_fill)
            else:
                t = t_sec.float()
            if self.time_norm == "minmax":
                t_scaled = time_normalize(t, "minmax", self._time_stats)
                if t_scaled is None:
                    t_scaled = _minmax(t, self._time_bounds)
            elif self.time_norm == "zscore":
                t_scaled = time_normalize(t, "zscore", self._time_stats)
                if t_scaled is None:
                    mean = 0.5 * sum(self._time_bounds)
                    std = max((self._time_bounds[1] - self._time_bounds[0]) / 2.0, 1.0)
                    t_scaled = (t - mean) / std
            else:
                t_scaled = time_normalize(t, self.time_norm, self._time_stats)
                if t_scaled is None:
                    t_scaled = _minmax(t, self._time_bounds)
            builder.add_time(t_scaled[..., None] if t_scaled.dim() == lat.dim() else t_scaled)

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is not None:
            return dict(self._layout_cache)
        layout = {"space": 2, "order": ["space"], "full": self.feat_dim()}
        if self.include_time:
            layout["time"] = 1
            layout["order"].append("time")
        return layout
