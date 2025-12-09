"""Public training utilities for Poseidon."""

from .callbacks import SetEpochOnIterable
from .perf_monitor import PerfCallback
from .lightning_datamodule import ShardedDataModule, create_datamodule_from_config
from .lit_module import LitRegressor
from .sample_weighting import (
    distance_to_coast_weight,
    cross_track_distance_weight,
    normalize_weights,
    prepare_loss_weight_config,
    compute_loss_weights,
)
from .lr_schedulers import LinearWarmupCosineAnnealingLR

__all__ = [
    "ShardedDataModule",
    "create_datamodule_from_config",
    "LitRegressor",
    "SetEpochOnIterable",
    "PerfCallback",
    "LinearWarmupCosineAnnealingLR",
    "distance_to_coast_weight",
    "cross_track_distance_weight",
    "normalize_weights",
    "prepare_loss_weight_config",
    "compute_loss_weights",
]
