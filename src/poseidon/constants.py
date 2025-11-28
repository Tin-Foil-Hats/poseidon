"""Centralized physical constants used across Poseidon."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TidalConstituent:
    name: str
    period_hours: float
    period_seconds: float
    angular_frequency: float        

    @classmethod
    def from_hours(cls, name: str, period_hours: float) -> "TidalConstituent":
        seconds = period_hours * 3600.0
        omega = 2.0 * math.pi / seconds
        return cls(name=name, period_hours=period_hours, period_seconds=seconds, angular_frequency=omega)

                                             
TIDAL_CONSTITUENTS: Tuple[TidalConstituent, ...] = (
    TidalConstituent.from_hours("M2", 12.4206012),
    TidalConstituent.from_hours("S2", 12.0),
    TidalConstituent.from_hours("N2", 12.65834751),
    TidalConstituent.from_hours("K2", 11.96723606),
    TidalConstituent.from_hours("K1", 23.93447213),
    TidalConstituent.from_hours("O1", 25.81934166),
    TidalConstituent.from_hours("P1", 24.06588971),
    TidalConstituent.from_hours("Q1", 26.86835672),
    TidalConstituent.from_hours("Mf", 327.8589682),
    TidalConstituent.from_hours("Mm", 661.3093695),
)

DEFAULT_SEMIDIURNAL_PERIODS_S: Tuple[float, ...] = tuple(
    c.period_seconds for c in TIDAL_CONSTITUENTS if c.name in {"M2", "S2", "N2", "K2"}
)

DEFAULT_DIURNAL_PERIODS_S: Tuple[float, ...] = tuple(
    c.period_seconds for c in TIDAL_CONSTITUENTS if c.name in {"K1", "O1", "P1", "Q1"}
)

DEFAULT_TIDAL_PERIODS_S: Tuple[float, ...] = DEFAULT_SEMIDIURNAL_PERIODS_S + DEFAULT_DIURNAL_PERIODS_S
