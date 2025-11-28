                      
from pathlib import Path
from typing import Optional, Dict, Any, Literal, List

import glob
import json
import math
import os
import random

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from poseidon.data.dataset import ShardedFlatDataset, PassShardDataset
from poseidon.data.shards import load_index, split_by_cycle_pass

TimeStatsMode = Literal["none", "train", "all"]
TrainLoaderMode = Literal["auto", "flat", "stream"]

class _BatchShardIterable(IterableDataset):
    """
    Train iterator over pre-batched .pt shards.
    Each file: {'X':[K,B,3], 'Y':[K,B,1]} with X[...,0]=lat, 1=lon, 2=t.
    Supports micro-batching and epoch-aware shuffling.
    """

    def __init__(
        self,
        shard_dir: str,
        micro_bs: int = 0,
        shuffle_files: bool = True,
        shuffle_batches: bool = True,
        index_name: str = "batch_index.json",
        cache_files: bool = False,
    ):
        super().__init__()
        self.dir = str(shard_dir)
        self.paths = sorted(glob.glob(os.path.join(self.dir, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No .pt shards in {self.dir}")

        self.micro_bs = int(micro_bs) if micro_bs else 0
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_batches = bool(shuffle_batches)
        self.index_path = os.path.join(self.dir, index_name)
        self.cache_files = bool(cache_files)
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

        self._batches_per_file = self._load_or_build_index()

        if self.micro_bs > 0:
            d0 = torch.load(self.paths[0], map_location="cpu")
            B = int(d0["X"].shape[1]); del d0
            chunks = max(B // self.micro_bs, 1)
            self._total = int(sum(self._batches_per_file) * chunks)
        else:
            self._total = int(sum(self._batches_per_file))

        self._epoch = 0
        self._base_seed = int(os.environ.get("PL_GLOBAL_SEED", 0))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, worker_id: int = 0) -> random.Random:
        return random.Random(self._base_seed + self._epoch * 9973 + worker_id * 389)

    def _load_or_build_index(self):
        if os.path.isfile(self.index_path):
            j = json.loads(open(self.index_path).read())
            if j.get("paths") == self.paths:
                return list(map(int, j["batches_per_file"]))
        counts = []
        for p in self.paths:
            d = torch.load(p, map_location="cpu")
            counts.append(int(d["X"].shape[0]))      
            del d
        with open(self.index_path, "w") as f:
            json.dump({"paths": self.paths, "batches_per_file": counts}, f)
        return counts

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1

        rng = self._rng(worker_id)

        paths = self.paths[:]
        if self.shuffle_files:
            rng.shuffle(paths)

        if worker is not None:
            paths = paths[worker_id::worker_count]

        if not paths:
            return

        for p in paths:
            d = self._load_file(p)
            X, Y = d["X"], d["Y"]                                     
            K, B = X.shape[0], X.shape[1]

            order = list(range(K))
            if self.shuffle_batches:
                rng.shuffle(order)

            mb = self.micro_bs
            if mb <= 0 or mb >= B:
                for i in order:
                    Xi, Yi = X[i], Y[i].view(-1)
                    yield {
                        "lat": Xi[:, 0].contiguous(),
                        "lon": Xi[:, 1].contiguous(),
                        "t":   Xi[:, 2].contiguous(),
                        "y":   Yi.contiguous(),
                    }
            else:
                phase = (rng.randrange(max(B // mb, 1)) * mb) if (B % mb == 0 and B // mb > 0) else 0
                for i in order:
                    Xi, Yi = X[i], Y[i].view(-1)
                    for s in range(phase, phase + B, mb):
                        a = s % B
                        e = a + mb
                        if e > B: break
                        yield {
                            "lat": Xi[a:e, 0].contiguous(),
                            "lon": Xi[a:e, 1].contiguous(),
                            "t":   Xi[a:e, 2].contiguous(),
                            "y":   Yi[a:e].contiguous(),
                        }

    def _load_file(self, path: str) -> dict[str, torch.Tensor]:
        if self.cache_files:
            cached = self._cache.get(path)
            if cached is not None:
                return cached
        data = torch.load(path, map_location="cpu")
        if self.cache_files:
            self._cache[path] = data
        return data



class ShardedDataModule(LightningDataModule):
    def __init__(
        self,
        shards_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        time_stats: TimeStatsMode = "train",

        train_batchshard_dir: Optional[str] = None,
        train_micro_bs: int = 0,                                                           
        target_normalizer: Optional[Dict[str, Any]] = None,
        train_num_workers: Optional[int] = None,
        train_cache_shards: bool = False,
        val_full_pass: bool = False,
        test_full_pass: bool = False,
        preload_npz: bool = False,
        train_loader: TrainLoaderMode = "auto",
        val_chunk_size: int = 0,
        test_chunk_size: int = 0,
        train_time_scan_files: int | None = None,
    ):
        super().__init__()
        self.shards_dir = Path(shards_dir).resolve()                      
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.time_stats: TimeStatsMode = time_stats

        self.train_batchshard_dir = Path(train_batchshard_dir).resolve() if train_batchshard_dir else None
        self.train_micro_bs = int(train_micro_bs)
        self.train_num_workers = int(train_num_workers) if train_num_workers is not None else None
        self.train_cache_shards = bool(train_cache_shards)
        self.val_full_pass = bool(val_full_pass)
        self.test_full_pass = bool(test_full_pass)
        self.preload_npz = bool(preload_npz)
        if train_loader not in ("auto", "flat", "stream"):
            raise ValueError(f"Unsupported train_loader mode: {train_loader}")
        self.train_loader: TrainLoaderMode = train_loader
        self.val_chunk_size = int(val_chunk_size)
        self.test_chunk_size = int(test_chunk_size)
        self.train_time_scan_files = None if train_time_scan_files in (None, -1) else int(train_time_scan_files)

        tn_cfg = target_normalizer if target_normalizer is not None else {"type": "none"}
        if isinstance(tn_cfg, str):
            tn_cfg = {"type": tn_cfg}
        else:
            tn_cfg = dict(tn_cfg)
        self.target_normalizer_cfg: Dict[str, Any] = tn_cfg
        self.target_normalizer_kind = str(self.target_normalizer_cfg.get("type", "none")).lower()
        self.target_eps = float(self.target_normalizer_cfg.get("eps", 1e-6))
        self._provided_target_state = (
            dict(self.target_normalizer_cfg.get("state", {}))
            if isinstance(self.target_normalizer_cfg.get("state"), dict)
            else None
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.bbox_stats: Dict[str, float] = {}
        self.t_stats: Dict[str, float] | None = None                                      
        self.target_stats: Dict[str, float] | None = None

    def _resolve(self, p) -> Path:
        p = Path(p)
        if p.is_absolute():
            if p.exists():
                return p
            fallback = (self.shards_dir / p.name).resolve()
            if fallback.exists():
                return fallback
            return p
        s = p.as_posix()
        sd = self.shards_dir.as_posix().rstrip("/")
        if s.startswith(sd + "/") or s == sd:
            return Path(s).resolve()
        return (self.shards_dir / p).resolve()

    def _abs_split(self, lst) -> List[dict]:
        out = []
        for e in lst:
            e2 = dict(e)
            if "path" in e2:
                e2["path"] = str(self._resolve(e2["path"]))
            out.append(e2)
        return out

    def _scan_bbox_npz(self, shard_paths):
        lat_min, lat_max = +np.inf, -np.inf
        lon_min, lon_max = +np.inf, -np.inf
        for f in shard_paths:
            f = self._resolve(f)
            with np.load(f) as z:
                lat = z["lat"]; lon = z["lon"]
                lat_min = min(lat_min, float(lat.min()))
                lat_max = max(lat_max, float(lat.max()))
                lon_min = min(lon_min, float(lon.min()))
                lon_max = max(lon_max, float(lon.max()))
        self.bbox_stats = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    def _scan_time_npz(self, shard_paths) -> Dict[str, float]:
        mean = 0.0
        m2 = 0.0
        n = 0
        tmin = +np.inf
        tmax = -np.inf
        for f in shard_paths:
            f = self._resolve(f)
            with np.load(f) as z:
                t = z["t"].astype(np.float64, copy=False)
                if t.size == 0:
                    continue
                batch_n = int(t.size)
                batch_mean = float(t.mean())
                batch_var = float(t.var())                       
                delta = batch_mean - mean
                total = n + batch_n
                if total == 0:
                    continue
                mean += delta * batch_n / total
                m2 += batch_var * batch_n + delta * delta * n * batch_n / total
                n = total
                tmin = min(tmin, float(t.min()))
                tmax = max(tmax, float(t.max()))
        if n == 0:
            return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
        std = float(np.sqrt(max(m2 / n, 0.0)))
        return {"mean": mean, "std": std, "min": tmin, "max": tmax}

    def _scan_target_npz(self, shard_paths) -> Dict[str, float]:
        mean = 0.0
        m2 = 0.0
        n = 0
        for f in shard_paths:
            f = self._resolve(f)
            with np.load(f) as z:
                y = z["y"].astype(np.float64, copy=False)
                if y.size == 0:
                    continue
                batch_n = int(y.size)
                batch_mean = float(y.mean())
                batch_var = float(y.var())
                delta = batch_mean - mean
                total = n + batch_n
                if total == 0:
                    continue
                mean += delta * batch_n / total
                m2 += batch_var * batch_n + delta * delta * n * batch_n / total
                n = total
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        std = float(np.sqrt(max(m2 / n, 0.0)))
        return {"mean": mean, "std": std, "count": n}

    def _scan_stats_pt_train(self, scan_files: int | None = None) -> Dict[str, Dict[str, float]]:
        assert self.train_batchshard_dir is not None
        paths = sorted(glob.glob(os.path.join(str(self.train_batchshard_dir), "*.pt")))
        if scan_files is None or scan_files <= 0:
            selected = paths
        else:
            selected = paths[:min(scan_files, len(paths))]
        if not selected:
            return {"bbox": {}, "time": {"mean": 0.0, "std": 1.0, "min": None, "max": None}, "target": {"mean": 0.0, "std": 0.0, "count": 0}}
        lat_min = +1e9; lat_max = -1e9
        lon_min = +1e9; lon_max = -1e9
        mean = 0.0
        m2 = 0.0
        t_cnt = 0
        t_min = float("inf")
        t_max = float("-inf")
        y_mean = 0.0
        y_m2 = 0.0
        y_cnt = 0
        for p in selected:
            d = torch.load(p, map_location="cpu")
            X = d["X"].to(dtype=torch.float64)
            lat = X[...,0].reshape(-1).float()
            lon = X[...,1].reshape(-1).float()
            t   = X[...,2].reshape(-1)
            Y = d["Y"].to(dtype=torch.float64)
            y   = Y.reshape(-1)
            lat_min = min(lat_min, float(lat.min())); lat_max = max(lat_max, float(lat.max()))
            lon_min = min(lon_min, float(lon.min())); lon_max = max(lon_max, float(lon.max()))
            batch_n = int(t.numel())
            if batch_n == 0:
                continue
            batch_mean = float(t.mean())
            batch_var = float(t.var(unbiased=False))
            batch_min = float(t.min())
            batch_max = float(t.max())
            delta = batch_mean - mean
            total = t_cnt + batch_n
            if total == 0:
                continue
            mean += delta * batch_n / total
            m2 += batch_var * batch_n + delta * delta * t_cnt * batch_n / total
            t_cnt = total
            t_min = min(t_min, batch_min)
            t_max = max(t_max, batch_max)

            y_batch_n = int(y.numel())
            if y_batch_n > 0:
                y_batch_mean = float(y.mean())
                y_batch_var = float(y.var(unbiased=False))
                y_delta = y_batch_mean - y_mean
                y_total = y_cnt + y_batch_n
                if y_total > 0:
                    y_mean += y_delta * y_batch_n / y_total
                    y_m2 += y_batch_var * y_batch_n + y_delta * y_delta * y_cnt * y_batch_n / y_total
                    y_cnt = y_total
        if t_cnt == 0:
            t_mean = 0.0
            t_std = 1.0
            t_min_val = None
            t_max_val = None
        else:
            t_mean = mean
            t_std  = math.sqrt(max(m2 / t_cnt, 0.0))
            t_min_val = float(t_min)
            t_max_val = float(t_max)
        if y_cnt == 0:
            target_mean = 0.0
            target_std = 0.0
        else:
            target_mean = y_mean
            target_std = math.sqrt(max(y_m2 / y_cnt, 0.0))
        bbox = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}
        tstats = {"mean": t_mean, "std": t_std, "min": t_min_val, "max": t_max_val}
        target = {"mean": target_mean, "std": target_std, "count": y_cnt}
        return {"bbox": bbox, "time": tstats, "target": target}

              
    def setup(self, stage: Optional[str] = None):
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shard directory not found: {self.shards_dir}")

        index = load_index(self.shards_dir)
        splits = split_by_cycle_pass(index, val_ratio=self.val_ratio, test_ratio=self.test_ratio, seed=self.seed)
        splits_norm = {
            "train": self._abs_split(splits["train"]),
            "val":   self._abs_split(splits["val"]),
            "test":  self._abs_split(splits["test"]),
        }

        loader_mode = self.train_loader
        if loader_mode == "auto":
            use_stream = self.train_batchshard_dir is not None
        elif loader_mode == "flat":
            use_stream = False
        else:            
            use_stream = True

        if use_stream:
            if self.train_batchshard_dir is None:
                raise FileNotFoundError("train_loader='stream' requires train_batchshard_dir")
            if not self.train_batchshard_dir.exists():
                raise FileNotFoundError(f"train_batchshard_dir not found: {self.train_batchshard_dir}")
            self.train_ds = _BatchShardIterable(
                str(self.train_batchshard_dir), micro_bs=self.train_micro_bs,
                shuffle_files=True, shuffle_batches=True,
                cache_files=self.train_cache_shards
            )
        else:
            self.train_ds = ShardedFlatDataset(splits_norm["train"], preload=self.preload_npz)
        if self.val_full_pass:
            self.val_ds = PassShardDataset(splits_norm["val"])
        else:
            self.val_ds = ShardedFlatDataset(
                splits_norm["val"],
                preload=self.preload_npz,
                chunk_size=self.val_chunk_size if self.val_chunk_size > 0 else None,
            )

        if self.test_full_pass:
            self.test_ds = PassShardDataset(splits_norm["test"])
        else:
            self.test_ds = ShardedFlatDataset(
                splits_norm["test"],
                preload=self.preload_npz,
                chunk_size=self.test_chunk_size if self.test_chunk_size > 0 else None,
            )

        all_paths = [p.resolve() for p in self.shards_dir.glob("*.npz")]
        train_paths = [Path(e["path"]) for e in splits_norm["train"] if "path" in e]
        use_train = train_paths if train_paths else all_paths

        self._scan_bbox_npz(use_train)                                                      

        need_pt_scan = (
            use_stream
            and (
                self.time_stats == "train"
                or self.target_normalizer_kind not in {"none", "identity"}
            )
        )
        pt_stats = self._scan_stats_pt_train(scan_files=self.train_time_scan_files) if need_pt_scan else None

        if self.time_stats == "none":
            self.t_stats = None
        elif self.time_stats == "all":
            self.t_stats = self._scan_time_npz(all_paths)
        elif self.time_stats == "train" and use_stream:
            if pt_stats is None:
                pt_stats = self._scan_stats_pt_train(scan_files=self.train_time_scan_files)
            self.t_stats = pt_stats["time"]
        else:
            self.t_stats = self._scan_time_npz(use_train)

        target_state: Dict[str, Any]
        kind = self.target_normalizer_kind
        if kind in {"none", "identity"}:
            target_state = {"type": "none"}
        else:
            if use_stream:
                if pt_stats is None:
                    pt_stats = self._scan_stats_pt_train(scan_files=self.train_time_scan_files)
                target_stats = pt_stats.get("target", {"mean": 0.0, "std": 0.0, "count": 0})
            else:
                target_stats = self._scan_target_npz(use_train)
            if target_stats.get("count", 0) == 0:
                if self._provided_target_state is not None:
                    target_state = dict(self._provided_target_state)
                    target_state.setdefault("type", kind)
                    target_state.setdefault("eps", self.target_eps)
                else:
                    target_state = {"type": kind, "mean": 0.0, "std": 1.0, "eps": self.target_eps}
            else:
                mean = float(target_stats.get("mean", 0.0))
                std = float(target_stats.get("std", 0.0))
                target_state = {"type": kind, "mean": mean, "std": max(std, self.target_eps), "eps": self.target_eps}

        self.target_stats = target_state
        self.stats = {"bbox": self.bbox_stats, "time": self.t_stats, "target": self.target_stats}

        def _count_samples(items):
            return sum(int(e.get("length", 0)) for e in items)

        def _format_summary(ds, samples: int) -> str:
            if isinstance(ds, PassShardDataset):
                return f"{samples} samples/{len(ds)} passes"
            if isinstance(ds, ShardedFlatDataset):
                note = " (preload)" if getattr(ds, "preload", False) else ""
                return f"{samples} samples{note}"
            if isinstance(ds, _BatchShardIterable):
                return f"{samples} samples/{len(ds)} batches"
            return str(samples)

        train_samples = _count_samples(splits_norm["train"])
        val_samples = _count_samples(splits_norm["val"])
        test_samples = _count_samples(splits_norm["test"])

        print(
            "[ShardedDataModule] "
            f"Train={_format_summary(self.train_ds, train_samples)}  "
            f"Val={_format_summary(self.val_ds, val_samples)}  "
            f"Test={_format_summary(self.test_ds, test_samples)}  "
            f"Shards(npz)={len(all_paths)}  time_stats={self.time_stats}  "
            f"train_mode={'pt-batch' if use_stream else 'npz'}"
        )

             
    def _loader(self, ds, shuffle=False, collate_fn=None, batch_size=None):
        if isinstance(ds, PassShardDataset):
            batch_size = 1 if batch_size is None else batch_size
            collate_fn = collate_fn or self._pass_collate
            shuffle = False
        chunked = isinstance(ds, ShardedFlatDataset) and getattr(ds, "is_chunked", False)
        if chunked:
            bs = 1 if batch_size is None else max(int(batch_size), 1)
            collate_fn = collate_fn or self._chunk_collate
            shuffle = False
        else:
            bs = self.batch_size if batch_size is None else batch_size
        is_iterable = isinstance(ds, IterableDataset)
        if is_iterable:
            bs = None
            worker_hint = self.train_num_workers if self.train_num_workers is not None else self.num_workers
            num_workers = max(int(worker_hint), 0)
            persistent = num_workers > 0
            prefetch = None
        else:
            num_workers = self.num_workers
            persistent = self.num_workers > 0
            prefetch = 2 if self.num_workers > 0 else None

        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=(shuffle and not is_iterable),
            num_workers=num_workers,
            pin_memory=self.pin_memory and not is_iterable,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )

    @staticmethod
    def _pass_collate(batch):
        if not batch:
            raise ValueError("Empty batch encountered in pass collate.")
        if len(batch) != 1:
            raise ValueError("PassShardDataset expects batch_size=1.")
        return batch[0]

    @staticmethod
    def _chunk_collate(batch):
        if not batch:
            raise ValueError("Empty batch encountered in chunk collate.")
        if len(batch) == 1:
            sample = batch[0]
            out = {}
            for key, value in sample.items():
                if torch.is_tensor(value):
                    out[key] = value.reshape(-1).contiguous()
                else:
                    out[key] = value
            return out
        out = {}
        keys = batch[0].keys()
        for key in keys:
            values = []
            tensors = True
            for sample in batch:
                value = sample[key]
                if torch.is_tensor(value):
                    values.append(value.reshape(-1))
                else:
                    tensors = False
                    values.append(value)
            if tensors:
                out[key] = torch.cat(values, dim=0)
            else:
                out[key] = values
        return out

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


def create_datamodule_from_config(cfg: Dict[str, Any]):
    d = cfg.get("data", {})
    return ShardedDataModule(
        shards_dir=d.get("shards_dir", "../data/shards"),
        batch_size=d.get("batch_size", 64),
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        val_ratio=d.get("val_ratio", 0.1),
        test_ratio=d.get("test_ratio", 0.1),
        seed=d.get("seed", 42),
        time_stats=d.get("time_stats", "train"),
        train_batchshard_dir=d.get("train_batchshard_dir", None),
        train_micro_bs=d.get("train_micro_bs", 0),
        target_normalizer=d.get("target_normalizer"),
        train_num_workers=d.get("train_num_workers", None),
        train_cache_shards=d.get("train_cache_shards", False),
        val_full_pass=d.get("val_full_pass", False),
        test_full_pass=d.get("test_full_pass", False),
        preload_npz=d.get("preload_npz", False),
        train_loader=d.get("train_loader", "auto"),
        val_chunk_size=d.get("val_chunk_size", 0),
        test_chunk_size=d.get("test_chunk_size", 0),
        train_time_scan_files=d.get("train_time_scan_files"),
    )
