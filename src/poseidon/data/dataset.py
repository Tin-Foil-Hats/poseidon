from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import bisect
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .schema import load_shard


class ShardedFlatDataset(Dataset):
    def __init__(
        self,
        index: List[Dict[str, Any]],
        cache_size: int = 4,
        preload: bool = False,
        chunk_size: Optional[int] = None,
    ):
        self.index = index
        offsets = []
        total = 0
        for item in index:
            offsets.append(total)
            total += int(item["length"])
        self.offsets = offsets
        self.total_length = total
        self.preload = bool(preload)
        self.chunk_size = int(chunk_size) if chunk_size else 0
        self.cache_size: Optional[int] = None if self.preload else int(cache_size)
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._chunks: list[Dict[str, torch.Tensor]] | None = None
        self._chunk_len: int = 0
        self.chunked: bool = False

        if self.preload:
            for item in index:
                path = str(item["path"])
                if path not in self._cache:
                    self._cache[path] = load_shard(path)
        if self.chunk_size:
            self._build_chunks()

    def __len__(self) -> int:
        if self._chunks is not None:
            return self._chunk_len
        return self.total_length

    @property
    def is_chunked(self) -> bool:
        return self._chunks is not None and self._chunk_len > 0

    def _load_shard_cached(self, path: str) -> Dict[str, np.ndarray]:
        path = str(path)
        if path in self._cache:
            arrays = self._cache.pop(path)
            self._cache[path] = arrays
            return arrays
        arrays = load_shard(path)
        self._cache[path] = arrays
        if self.cache_size is not None:
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def _build_chunks(self) -> None:
        if self.chunk_size <= 0:
            return
        tensors: list[Dict[str, torch.Tensor]] = []
        buffer: Dict[str, list[torch.Tensor]] = {"lat": [], "lon": [], "t": [], "y": []}
        count = 0
        for item in self.index:
            arrays = self._load_shard_cached(item["path"])
            n = int(item.get("length", len(arrays["lat"])))
            lat = arrays["lat"]
            lon = arrays["lon"]
            t = arrays["t"]
            y = arrays["y"]
            for i in range(n):
                buffer["lat"].append(torch.as_tensor(lat[i], dtype=torch.float32))
                buffer["lon"].append(torch.as_tensor(lon[i], dtype=torch.float32))
                buffer["t"].append(torch.as_tensor(t[i], dtype=torch.float32))
                buffer["y"].append(torch.as_tensor(y[i], dtype=torch.float32))
                count += 1
                if count == self.chunk_size:
                    tensors.append({key: torch.stack(vals) for key, vals in buffer.items()})
                    buffer = {"lat": [], "lon": [], "t": [], "y": []}
                    count = 0
        if count:
            tensors.append({key: torch.stack(vals) for key, vals in buffer.items()})
        if tensors:
            self._chunks = tensors
            self._chunk_len = len(tensors)
            self.chunked = True

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._chunks is not None:
            return self._chunks[idx]
        if idx < 0:
            idx = self.total_length + idx
        if idx < 0 or idx >= self.total_length:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.offsets, idx) - 1
        base = self.offsets[shard_idx]
        item = self.index[shard_idx]
        local_idx = idx - base
        arrays = self._load_shard_cached(item["path"])
        lat = arrays["lat"][local_idx]
        lon = arrays["lon"][local_idx]
        t = arrays["t"][local_idx]
        y = arrays["y"][local_idx]
        return {
            "lat": torch.as_tensor(lat),
            "lon": torch.as_tensor(lon),
            "t": torch.as_tensor(t),
            "y": torch.as_tensor(y),
        }


class BatchShardIterable(IterableDataset):
    def __init__(self, shard_paths: List[Union[str, Path]], infinite: bool = True):
        super().__init__()
        self.shard_paths = [str(Path(p)) for p in shard_paths]
        self.infinite = infinite

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(0)
        while True:
            order = torch.randperm(len(self.shard_paths), generator=rng).tolist()
            for i in order:
                path = self.shard_paths[i]
                batch = torch.load(path)
                yield batch
            if not self.infinite:
                break


class PassShardDataset(Dataset):
    """Dataset that loads an entire shard (pass) per item."""

    def __init__(self, index: List[Dict[str, Any]]):
        self.index = list(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.index[idx]
        arrays = load_shard(item["path"])
        sample = {
            "lat": torch.as_tensor(arrays["lat"], dtype=torch.float32),
            "lon": torch.as_tensor(arrays["lon"], dtype=torch.float32),
            "t": torch.as_tensor(arrays["t"], dtype=torch.float32),
            "y": torch.as_tensor(arrays["y"], dtype=torch.float32),
        }
        if "cycle" in item:
            sample.setdefault("cycle", int(item["cycle"]))
        if "pass" in item:
            sample.setdefault("pass", int(item["pass"]))
        sample.setdefault("path", item.get("path"))
        sample.setdefault("length", int(item.get("length", sample["lat"].numel())))
        return sample
