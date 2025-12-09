from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import hashlib
import json
import random
import re
import numpy as np
import torch

from .schema import load_shard, shard_length

PASS_RE = re.compile(r"_c(\d{3})_p(\d{3})_")


def parse_cycle_pass_from_name(p: Union[str, Path]) -> Tuple[int, int]:
    name = Path(p).name
    m = PASS_RE.search(name)
    if not m:
        raise ValueError(f"could not parse cycle/pass from name {name}")
    return int(m.group(1)), int(m.group(2))


def scan_shards(shards_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    shards_dir = Path(shards_dir)
    index: List[Dict[str, Any]] = []
    for path in sorted(shards_dir.glob("*.npz")):
        arrays = load_shard(path)
        n = shard_length(arrays)
        cy, pa = parse_cycle_pass_from_name(path)
        index.append(
            {
                "path": str(path),
                "length": int(n),
                "cycle": int(cy),
                "pass": int(pa),
            }
        )
    return index


def save_index(shards_dir: Union[str, Path], index: List[Dict[str, Any]], name: str = "index.json") -> Path:
    shards_dir = Path(shards_dir)
    out_path = shards_dir / name
    out_path.write_text(json.dumps(index))
    return out_path


def load_index(shards_dir: Union[str, Path], name: str = "index.json") -> List[Dict[str, Any]]:
    shards_dir = Path(shards_dir)
    idx_path = shards_dir / name
    if idx_path.exists():
        return json.loads(idx_path.read_text())
    index = scan_shards(shards_dir)
    save_index(shards_dir, index, name=name)
    return index


def split_by_group(
    index: List[Dict[str, Any]],
    group_keys: Tuple[str, ...],
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for item in index:
        key = tuple(item[k] for k in group_keys)
        groups.setdefault(key, []).append(item)
    group_keys_list = list(groups.keys())
    random.Random(seed).shuffle(group_keys_list)
    n_groups = len(group_keys_list)
    n_test = int(round(test_ratio * n_groups))
    n_val = int(round(val_ratio * n_groups))

    if test_ratio > 0.0 and n_test == 0 and n_groups > 0:
        n_test = 1
    if val_ratio > 0.0 and n_val == 0 and n_groups - n_test > 0:
        n_val = 1

    if n_val + n_test > n_groups:
                                                    
        if n_val >= n_test:
            n_val = max(n_val - 1, 0)
        else:
            n_test = max(n_test - 1, 0)

    n_train = max(n_groups - n_val - n_test, 0)

    if n_train == 0 and n_groups > 0:
                                            
        if n_val > n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = max(n_groups - n_val - n_test, 0)
    train_groups = set(group_keys_list[:n_train])
    val_groups = set(group_keys_list[n_train:n_train + n_val])
    test_groups = set(group_keys_list[n_train + n_val:])
    splits = {"train": [], "val": [], "test": []}
    for key, items in groups.items():
        if key in train_groups:
            splits["train"].extend(items)
        elif key in val_groups:
            splits["val"].extend(items)
        else:
            splits["test"].extend(items)
    return splits


def split_by_cycle_pass(
    index: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    for item in index:
        patched = dict(item)
        if "pass" not in patched and "pas" in patched:                                                                        
            patched["pass"] = patched.get("pas")
        if "length" not in patched:
            if "n" in patched:
                patched["length"] = int(patched["n"])            
            elif "count" in patched:
                patched["length"] = int(patched["count"])   
        normalized.append(patched)
    return split_by_group(normalized, ("cycle", "pass"), val_ratio, test_ratio, seed=seed)


def compute_index_signature(index: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for item in index:
        path = str(item.get("path", ""))
        cycle = item.get("cycle")
        pas = item.get("pass")
        length = item.get("length")
        rows.append(f"{path}|{cycle}|{pas}|{length}")
    if not rows:
        return "0"
    rows.sort()
    payload = "\n".join(rows).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def save_split_cache(
    path: Union[str, Path],
    *,
    splits: Dict[str, List[Dict[str, Any]]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    shards_dir: Union[str, Path],
    index_signature: str,
) -> Path:
    payload = {
        "meta": {
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "shards_dir": str(Path(shards_dir).resolve()),
            "index_signature": index_signature,
        },
        "splits": splits,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return out_path


def load_split_cache(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.is_file():
        return None
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return None

def _resolve_shard_path(candidate: Union[str, Path], root: Path) -> Path:
    path = Path(candidate)
    if path.is_absolute() and path.is_file():
        return path
    if not path.is_absolute():
        joined = (root / path).resolve()
        if joined.is_file():
            return joined
    alt = (root / path.name).resolve()
    if alt.is_file():
        return alt
    raise FileNotFoundError(str(candidate))


@dataclass
class ReshardResult:
    written: List[str]
    samples_loaded: int
    batches_written: int
    batch_size: int
    dropped_samples: int
    missing_shards: int
    source_shards: int
    split_groups: Dict[str, List[Tuple[int, int]]]
    clipped_samples: int = 0
    clip_bounds: Tuple[float, float] | None = None


def reshard_random_train(
    src_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    seed: int = 42,
    batch_size: int = 16384,
    batches_per_file: int = 64,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    clip_quantiles: Tuple[float, float] | None = None,
) -> ReshardResult:
    """Shuffle training shards and write pre-batched torch tensors."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batches_per_file <= 0:
        raise ValueError("batches_per_file must be positive")
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")
    if clip_quantiles is not None:
        q_lo, q_hi = clip_quantiles
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("clip_quantiles must satisfy 0 <= q_lo < q_hi <= 1")

    src_root = Path(src_dir).resolve()
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    index = scan_shards(src_root)
    splits = split_by_cycle_pass(index, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    train_shards = splits["train"]

    split_groups: Dict[str, List[Tuple[int, int]]] = {
        split: [(item["cycle"], item["pass"]) for item in items]
        for split, items in splits.items()
    }

    if not train_shards:
        raise RuntimeError(f"No training shards under {src_root}")

    lat_chunks: List[np.ndarray] = []
    lon_chunks: List[np.ndarray] = []
    t_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    missing = 0

    for item in train_shards:
        candidate = item.get("path")
        if candidate is None:
            continue
        try:
            shard_path = _resolve_shard_path(candidate, src_root)
        except FileNotFoundError:
            missing += 1
            continue
        arrays = load_shard(shard_path)
        lat_chunks.append(np.asarray(arrays["lat"], dtype=np.float32))
        lon_chunks.append(np.asarray(arrays["lon"], dtype=np.float32))
        t_chunks.append(np.asarray(arrays["t"], dtype=np.float32))
        y_chunks.append(np.asarray(arrays["y"], dtype=np.float32))

    if not lat_chunks:
        raise RuntimeError("No samples loaded. Check src_dir and shard filenames.")

    lat = np.concatenate(lat_chunks)
    lon = np.concatenate(lon_chunks)
    t = np.concatenate(t_chunks)
    y = np.concatenate(y_chunks)

    del lat_chunks, lon_chunks, t_chunks, y_chunks

    n_samples = int(lat.shape[0])
    perm = rng.permutation(n_samples)
    lat = lat[perm]
    lon = lon[perm]
    t = t[perm]
    y = y[perm]

    clipped = 0
    clip_bounds: Tuple[float, float] | None = None
    if clip_quantiles is not None:
        q_lo, q_hi = clip_quantiles
        lo_val, hi_val = np.quantile(y, [q_lo, q_hi])
        clip_bounds = (float(lo_val), float(hi_val))
        mask = (y >= lo_val) & (y <= hi_val)
        clipped = int((~mask).sum())
        if clipped:
            lat = lat[mask]
            lon = lon[mask]
            t = t[mask]
            y = y[mask]
            n_samples = int(lat.shape[0])
            if n_samples == 0:
                raise RuntimeError("All samples clipped; adjust clip_quantiles")

    B = int(batch_size)
    n_batches = n_samples // B
    if n_batches == 0:
        raise RuntimeError("Not enough samples to form a single batch; reduce batch_size")

    dropped = n_samples - n_batches * B

    written_paths: List[str] = []
    X_batches: List[torch.Tensor] = []
    Y_batches: List[torch.Tensor] = []
    file_idx = 0

    for batch_idx in range(n_batches):
        start = batch_idx * B
        stop = start + B
        x_np = np.stack((lat[start:stop], lon[start:stop], t[start:stop]), axis=1)
        y_np = y[start:stop, None]
        X_batches.append(torch.from_numpy(x_np.copy()).to(torch.float32))
        Y_batches.append(torch.from_numpy(y_np.copy()).to(torch.float32))

        if len(X_batches) == batches_per_file or batch_idx == n_batches - 1:
            out_path = out_root / f"batchshard_{file_idx:05d}.pt"
            torch.save({"X": torch.stack(X_batches, dim=0), "Y": torch.stack(Y_batches, dim=0)}, out_path)
            written_paths.append(str(out_path))
            X_batches.clear()
            Y_batches.clear()
            file_idx += 1

    return ReshardResult(
        written=written_paths,
        samples_loaded=n_samples,
        batches_written=n_batches,
        batch_size=B,
        dropped_samples=dropped,
        missing_shards=missing,
        source_shards=len(train_shards),
        split_groups=split_groups,
        clipped_samples=clipped,
        clip_bounds=clip_bounds,
    )
