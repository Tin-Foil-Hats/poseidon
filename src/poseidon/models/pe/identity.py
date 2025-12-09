import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder


@register_pe(name="identity", aliases=["raw", "passthrough", "none"])
class PEIdentity(PEBase):
    """Pass-through encoder that exposes raw lat/lon (+ optional time)."""

    def __init__(
        self,
        include_time: bool = True,
        time_fill: float = 0.0,
        lat_scale: float = 1.0,
        lon_scale: float = 1.0,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.include_time = bool(include_time)
        self.time_fill = float(time_fill)
        self.lat_scale = float(lat_scale)
        self.lon_scale = float(lon_scale)
        self.time_scale = float(time_scale)

    def feat_dim(self) -> int:
        return 2 + (1 if self.include_time else 0)

    def forward(self, lat_deg, lon_deg, t_sec=None):
        lat = lat_deg.float() * self.lat_scale
        lon = lon_deg.float() * self.lon_scale
        builder = FeatureBuilder()
        builder.add_space(lat[..., None])
        builder.add_space(lon[..., None])

        if self.include_time:
            if t_sec is None:
                t = lat.new_full(lat.shape, self.time_fill)
            else:
                t = t_sec.float()
            t = t * self.time_scale
            builder.add_time(t[..., None])

        self._layout_cache = builder.layout()
        return builder.build()

    def feature_layout(self) -> dict[str, int]:
        order = ["space"]
        if self.include_time:
            order.append("time")
        return {
            "space": 2,
            "time": 1 if self.include_time else 0,
            "full": self.feat_dim(),
            "order": order,
        }
