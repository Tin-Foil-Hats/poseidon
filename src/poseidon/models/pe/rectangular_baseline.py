import math
import torch
import torch.nn as nn

from poseidon.constants import (
    DEFAULT_TIDAL_PERIODS_S,
    DEFAULT_SEMIDIURNAL_PERIODS_S,
    DEFAULT_DIURNAL_PERIODS_S,
)
from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, time_normalize, wrap_lon_pi, normalize_lon_bounds


@register_pe(name="rect_baseline", aliases=["rect", "perectangularbaseline"])
class PERectangularBaseline(PEBase):
    """Scale lat/lon to [-1, 1] using bbox and add tidal/annual harmonics."""

    def __init__(
        self,
        include_annual: bool = True,
        tidal_periods_s=None,
        time_norm: str = "zscore",
        include_semidiurnal: bool = False,
        include_diurnal: bool = False,
        include_default_tides: bool = False,
    ):
        super().__init__()
        self.include_annual = bool(include_annual)
        periods = list(tidal_periods_s or [])
        if include_default_tides:
            periods.extend(DEFAULT_TIDAL_PERIODS_S)
        else:
            if include_semidiurnal:
                periods.extend(DEFAULT_SEMIDIURNAL_PERIODS_S)
            if include_diurnal:
                periods.extend(DEFAULT_DIURNAL_PERIODS_S)
        if not periods:
            periods = [44714.16, 43200.0]
                                                  
        seen: set[float] = set()
        uniq_periods: list[float] = []
        for p in periods:
            if p in seen:
                continue
            seen.add(p)
            uniq_periods.append(p)
        self.tidal_periods_s = uniq_periods
        self.time_norm = str(time_norm).lower()

        self.register_buffer("lat_min", torch.tensor(0.0))
        self.register_buffer("lat_max", torch.tensor(1.0))
        self.register_buffer("lon_min", torch.tensor(-1.0))
        self.register_buffer("lon_max", torch.tensor(1.0))

        self.register_buffer("omega_year", torch.tensor(2 * math.pi / 365.2422))
        self.register_buffer(
            "omega_tide",
            torch.tensor([2 * math.pi / p for p in self.tidal_periods_s], dtype=torch.float32),
        )

        self._time_stats: dict | None = None
        self._layout_cache: dict | None = None
        self._lon_crosses_seam = False

    def bind_context(self, ctx: dict):
        bbox = ctx["bbox"]
        self.lat_min.fill_(float(math.radians(bbox["lat_min"])))
        self.lat_max.fill_(float(math.radians(bbox["lat_max"])))
        lon_min, lon_max, crosses = normalize_lon_bounds(bbox["lon_min"], bbox["lon_max"])
        self.lon_min.fill_(float(lon_min))
        self.lon_max.fill_(float(lon_max))
        self._lon_crosses_seam = crosses
        self._time_stats = ctx.get("time")
        self._update_layout_cache()

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def forward(self, lat_deg, lon_deg, t_sec):
        builder = FeatureBuilder()
        features = self._encode(lat_deg, lon_deg, t_sec, builder)
        self._layout_cache = builder.layout()
        return features

    def _encode(self, lat_deg, lon_deg, t_sec, builder: FeatureBuilder) -> torch.Tensor:
        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))
        if self._lon_crosses_seam:
            seam_threshold = self.lon_min - self.lon_min.new_tensor(math.pi)
            lon = torch.where(
                lon < seam_threshold,
                lon + self.lon_min.new_tensor(2 * math.pi),
                lon,
            )
        lat_p = 2 * (lat - self.lat_min) / torch.clamp(self.lat_max - self.lat_min, min=1e-9) - 1
        lon_p = 2 * (lon - self.lon_min) / torch.clamp(self.lon_max - self.lon_min, min=1e-9) - 1

        t_sec = t_sec if t_sec is not None else torch.zeros_like(lat_p)
        t_ch = time_normalize(t_sec, self.time_norm, self._time_stats)
        time_base = t_ch if t_ch is not None else t_sec

        builder.add_space(lat_p[..., None])
        builder.add_space(lon_p[..., None])
        builder.add_time(time_base[..., None])

        if self.include_annual:
            aY = self.omega_year * (t_sec / 86400.0)
            builder.add_time(torch.sin(aY)[..., None])
            builder.add_time(torch.cos(aY)[..., None])
        if self.omega_tide.numel() > 0:
            A = t_sec[..., None] * self.omega_tide
            builder.add_time(torch.sin(A))
            builder.add_time(torch.cos(A))

        return builder.build()

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is None:
            self._update_layout_cache()
        return dict(self._layout_cache) if self._layout_cache is not None else {"full": super().feat_dim()}

    def _update_layout_cache(self) -> None:
        device = self.lat_min.device
        sample_lat = torch.zeros(1, dtype=torch.float32, device=device)
        sample_lon = torch.zeros(1, dtype=torch.float32, device=device)
        sample_t = torch.zeros(1, dtype=torch.float32, device=device)
        builder = FeatureBuilder()
        self._encode(sample_lat, sample_lon, sample_t, builder)
        self._layout_cache = builder.layout()
