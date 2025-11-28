import math
import torch

from .base import PEBase
from .registry import register_pe
from .utils import FeatureBuilder, deg2rad, time_normalize, wrap_lon_pi

TIDE_PERIOD_HOURS = {
    "M2": 12.4206012,
    "S2": 12.0,
    "N2": 12.65834751,
    "K1": 23.93447213,
    "O1": 25.81933871,
    "M4": 6.2103006,
}


def _omega_rad_per_sec(hours: float) -> float:
    return 2.0 * math.pi / (hours * 3600.0)


@register_pe(name="rff_tides", aliases=["rff_with_tides", "perff_with_tides"])
class PERFFWithTides(PEBase):
    """RFF PE + explicit tidal harmonics."""

    def __init__(
        self,
        dim_in=2,
        n_feat=256,
        sigma=0.5,
        use_time=True,
        include_time_channel=True,
        time_norm="zscore",
        tide_names=("M2", "S2", "N2", "K1", "O1", "M4"),
        add_annual=False,
        include_time_of_day=False,
        include_time_of_week=False,
        include_time_of_month=False,
        include_time_of_year=False,
        include_time_of_span=False,
    ):
        super().__init__()
        self.use_time = bool(use_time)
        self.include_time_channel = bool(include_time_channel)
        self.time_norm = str(time_norm).lower()
        self.tide_names = tuple(tide_names)
        self.add_annual = bool(add_annual)
        self.include_time_of_day = bool(include_time_of_day)
        self.include_time_of_week = bool(include_time_of_week)
        self.include_time_of_month = bool(include_time_of_month)
        self.include_time_of_year = bool(include_time_of_year)
        self.include_time_of_span = bool(include_time_of_span)

        d = int(dim_in) + (1 if self.use_time else 0)
        B = torch.randn(d, n_feat) / float(sigma)
        self.register_buffer("B", B)

        omegas = [_omega_rad_per_sec(TIDE_PERIOD_HOURS[n]) for n in self.tide_names]
        if self.add_annual:
            omegas += [
                2.0 * math.pi / (365.2422 * 86400.0),
                2.0 * math.pi / (182.6211 * 86400.0),
            ]
        self.register_buffer("omegas", torch.tensor(omegas, dtype=torch.float32))

        self._time_stats = None
        self._t0_sec = None
        self._extra_periods: list[float] = []
        self._layout_cache: dict | None = None

    def bind_context(self, ctx: dict):
        self._time_stats = (ctx or {}).get("time")
        tstats = self._time_stats or {}
        mean = tstats.get("mean")
        self._t0_sec = float(mean) if mean is not None else 0.0

        extra_periods: list[float] = []
        if self.include_time_of_day:
            extra_periods.append(24.0 * 3600.0)
        if self.include_time_of_week:
            extra_periods.append(7.0 * 24.0 * 3600.0)
        if self.include_time_of_month:
            extra_periods.append(30.4375 * 24.0 * 3600.0)
        if self.include_time_of_year:
            extra_periods.append(365.2422 * 24.0 * 3600.0)
        if self.include_time_of_span and self._time_stats is not None:
            tmin = self._time_stats.get("min")
            tmax = self._time_stats.get("max")
            if tmin is not None and tmax is not None:
                span = float(tmax) - float(tmin)
                if span > 0:
                    extra_periods.append(span)
        self._extra_periods = extra_periods
        self._layout_cache = None

    def feat_dim(self) -> int:
        layout = self.feature_layout()
        return int(layout.get("full", 0))

    def _time_channel(self, t_sec):
        if not self.use_time or t_sec is None:
            return None
        if self.time_norm == "zscore":
            if self._time_stats is None:
                raise RuntimeError("PERFFWithTides: call bind_context(ctx) before forward.")
            mu_val = self._time_stats.get("mean")
            sd_val = self._time_stats.get("std")
            mu = float(mu_val) if mu_val is not None else 0.0
            sd = float(sd_val) if (sd_val is not None and sd_val) else 1.0
            return (t_sec - mu) / sd
        if self.time_norm in ("scale", "minmax"):
            raise NotImplementedError("Only 'zscore' time normalization is implemented.")
        return t_sec

    def forward(self, lat_deg, lon_deg, t_sec=None):
        builder = FeatureBuilder()

        lat = deg2rad(lat_deg)
        lon = wrap_lon_pi(deg2rad(lon_deg))

        U = [lat, lon]
        time_channel = self._time_channel(t_sec)
        if self.use_time:
            if time_channel is None:
                raise RuntimeError("PERFFWithTides: time channel required but unavailable.")
            U.append(time_channel)
        U = torch.stack(U, dim=-1)
        Z = U @ self.B
        rff = torch.cat([torch.sin(Z), torch.cos(Z)], dim=-1)
        builder.add_space(rff)

        if t_sec is None:
            raise RuntimeError("PERFFWithTides requires t_sec for temporal harmonics.")

        if self.include_time_channel and time_channel is not None:
            builder.add_time(time_channel[..., None])
        else:
            tn = time_normalize(t_sec, self.time_norm, self._time_stats)
            if tn is not None:
                builder.add_time(tn[..., None])

        t_rel = t_sec - (self._t0_sec if self._t0_sec is not None else 0.0)
        if self.omegas.numel() > 0:
            Zt = t_rel.unsqueeze(-1) * self.omegas.to(t_rel.dtype).to(t_rel.device)
            tides = torch.cat([torch.sin(Zt), torch.cos(Zt)], dim=-1)
            builder.add_time(tides)

        if self._extra_periods:
            periods = torch.tensor(self._extra_periods, dtype=t_rel.dtype, device=t_rel.device)
            omega_extra = (2.0 * math.pi) / periods
            Zextra = t_rel.unsqueeze(-1) * omega_extra
            extra_feats = torch.cat([torch.sin(Zextra), torch.cos(Zextra)], dim=-1)
            builder.add_time(extra_feats)

        features = builder.build()
        self._layout_cache = builder.layout()
        return features

    def feature_layout(self) -> dict[str, int]:
        if self._layout_cache is None:
                                                                            
            device = self.B.device
            sample_lat = torch.zeros(1, dtype=torch.float32, device=device)
            sample_lon = torch.zeros(1, dtype=torch.float32, device=device)
            sample_t = torch.zeros(1, dtype=torch.float32, device=device)
            try:
                with torch.no_grad():
                    _ = self.forward(sample_lat, sample_lon, sample_t)
            except RuntimeError:
                                                                      
                rff_dim = 2 * self.B.shape[1]
                tide_dim = 2 * int(self.omegas.numel())
                time_dim = int(self.include_time_channel) + tide_dim
                return {"space": rff_dim, "time": time_dim, "order": ["space", "time"], "full": rff_dim + time_dim}
        return dict(self._layout_cache) if self._layout_cache is not None else {"full": 2 * self.B.shape[1]}
