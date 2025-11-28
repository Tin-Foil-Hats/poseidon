from pathlib import Path
from typing import Dict, Any, Iterable, Union
import numpy as np

REQUIRED_KEYS = ("lat", "lon", "t", "y")


def validate_shard(arrays: Dict[str, np.ndarray]) -> None:
    for key in REQUIRED_KEYS:
        if key not in arrays:
            raise ValueError(f"missing required key {key} in shard")
    n = None
    for key in REQUIRED_KEYS:
        v = arrays[key]
        if v.ndim != 1:
            raise ValueError(f"{key} must be 1D, got shape {v.shape}")
        if n is None:
            n = v.shape[0]
        elif v.shape[0] != n:
            raise ValueError("all required arrays must have the same length")


def load_shard(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    path = Path(path)
    data = dict(np.load(path))
    validate_shard(data)
    return data


def save_shard(path: Union[str, Path], arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    validate_shard(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def shard_length(arrays: Dict[str, np.ndarray]) -> int:
    return int(arrays[REQUIRED_KEYS[0]].shape[0])
