import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from poseidon.data.schema import save_shard
from poseidon.data.shards import (
    scan_shards,
    load_index,
    split_by_cycle_pass,
    reshard_random_train,
)
from poseidon.data.dataset import ShardedFlatDataset, BatchShardIterable
from poseidon.data.transforms import TargetNormalizer
from poseidon.training.lightning_datamodule import ShardedDataModule, _BatchShardIterable
from poseidon.training.lit_module import LitRegressor


def test_schema_and_scan(tmp_path):
    n1 = 5
    n2 = 7
    shard1 = {
        "lat": np.linspace(0.0, 1.0, n1),
        "lon": np.linspace(10.0, 11.0, n1),
        "t": np.arange(n1, dtype=np.float64),
        "y": np.ones(n1, dtype=np.float32),
    }
    shard2 = {
        "lat": np.linspace(2.0, 3.0, n2),
        "lon": np.linspace(20.0, 21.0, n2),
        "t": np.arange(n2, dtype=np.float64),
        "y": 2.0 * np.ones(n2, dtype=np.float32),
    }
    p1 = tmp_path / "shard_c001_p001_test.npz"
    p2 = tmp_path / "shard_c001_p002_test.npz"
    save_shard(p1, shard1)
    save_shard(p2, shard2)
    index = scan_shards(tmp_path)
    assert len(index) == 2
    paths = {item["path"] for item in index}
    assert str(p1) in paths
    assert str(p2) in paths
    lengths = sorted(item["length"] for item in index)
    assert lengths == [n1, n2]
    cycles = {item["cycle"] for item in index}
    passes = {item["pass"] for item in index}
    assert cycles == {1}
    assert passes == {1, 2}


def test_split_and_dataset(tmp_path):
    n1 = 4
    n2 = 6
    shard1 = {
        "lat": np.linspace(0.0, 1.0, n1),
        "lon": np.linspace(10.0, 11.0, n1),
        "t": np.arange(n1, dtype=np.float64),
        "y": np.zeros(n1, dtype=np.float32),
    }
    shard2 = {
        "lat": np.linspace(2.0, 3.0, n2),
        "lon": np.linspace(20.0, 21.0, n2),
        "t": np.arange(n2, dtype=np.float64),
        "y": np.ones(n2, dtype=np.float32),
    }
    p1 = tmp_path / "shard_c001_p001_test.npz"
    p2 = tmp_path / "shard_c001_p002_test.npz"
    save_shard(p1, shard1)
    save_shard(p2, shard2)
    index = load_index(tmp_path)
    splits = split_by_cycle_pass(index, val_ratio=0.5, test_ratio=0.0, seed=0)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["test"]) == 0
    assert len(splits["train"]) + len(splits["val"]) == 2
    ds_train = ShardedFlatDataset(splits["train"])
    ds_val = ShardedFlatDataset(splits["val"])
    assert len(ds_train) + len(ds_val) == n1 + n2
    sample = ds_train[0]
    assert set(sample.keys()) == {"lat", "lon", "t", "y"}
    assert all(isinstance(v, torch.Tensor) for v in sample.values())


def test_sharded_flat_dataset_dataloader(tmp_path):
    def make_shard(offset, name):
        n = 4
        lat = (offset + np.arange(n, dtype=np.float32)).astype(np.float32)
        lon = lat + 100.0
        t = lat + 200.0
        y = lat + lon + 0.5 * t
        shard = {"lat": lat, "lon": lon, "t": t, "y": y}
        save_shard(tmp_path / name, shard)
        return lat, lon, t, y

    lat1, lon1, t1, y1 = make_shard(0.0, "shard_c001_p001_flat.npz")
    lat2, lon2, t2, y2 = make_shard(10.0, "shard_c001_p002_flat.npz")

    index = scan_shards(tmp_path)
    ds = ShardedFlatDataset(index)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    batches = list(loader)

    lat_all = torch.cat([b["lat"] for b in batches])
    lon_all = torch.cat([b["lon"] for b in batches])
    t_all = torch.cat([b["t"] for b in batches])
    y_all = torch.cat([b["y"] for b in batches])

    expected_lat = torch.from_numpy(np.concatenate([lat1, lat2]))
    expected_lon = torch.from_numpy(np.concatenate([lon1, lon2]))
    expected_t = torch.from_numpy(np.concatenate([t1, t2]))
    expected_y = torch.from_numpy(np.concatenate([y1, y2]))

    assert lat_all.dtype == torch.float32
    assert lon_all.dtype == torch.float32
    assert t_all.dtype == torch.float32
    assert y_all.dtype == torch.float32
    assert torch.allclose(lat_all, expected_lat)
    assert torch.allclose(lon_all, expected_lon)
    assert torch.allclose(t_all, expected_t)
    assert torch.allclose(y_all, expected_y)

    assert torch.allclose(lon_all, lat_all + 100.0)
    assert torch.allclose(y_all, lat_all + lon_all + 0.5 * t_all)


def test_batch_shard_iterable(tmp_path):
    batch1 = {
        "lat": torch.linspace(0.0, 1.0, 3),
        "lon": torch.linspace(10.0, 11.0, 3),
        "t": torch.arange(3, dtype=torch.float32),
        "y": torch.zeros(3),
    }
    batch2 = {
        "lat": torch.linspace(2.0, 3.0, 2),
        "lon": torch.linspace(20.0, 21.0, 2),
        "t": torch.arange(2, dtype=torch.float32),
        "y": torch.ones(2),
    }
    p1 = tmp_path / "batch_001.pt"
    p2 = tmp_path / "batch_002.pt"
    torch.save(batch1, p1)
    torch.save(batch2, p2)
    iterable = BatchShardIterable([p1, p2], infinite=False)
    batches = list(iterable)
    assert len(batches) == 2
    keys1 = set(batches[0].keys())
    keys2 = set(batches[1].keys())
    assert keys1 == {"lat", "lon", "t", "y"}
    assert keys2 == {"lat", "lon", "t", "y"}


def test_prebatched_iterable_microbatch(tmp_path, monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "0")
    data_dir = tmp_path / "batch_pt"
    data_dir.mkdir()

    lat = torch.linspace(0.0, 5.0, 6, dtype=torch.float32)
    lon = lat + 100.0
    t = lat + 200.0
    y = lat + lon + t

    X = torch.stack((lat, lon, t), dim=1).unsqueeze(0)             
    Y = y.view(1, 6, 1)

    torch.save({"X": X, "Y": Y}, data_dir / "batch_00000.pt")

    iterable = _BatchShardIterable(
        str(data_dir), micro_bs=2, shuffle_files=False, shuffle_batches=False, cache_files=False
    )

    batches = list(iter(iterable))
    assert len(batches) == 3
    for b in batches:
        assert set(b.keys()) == {"lat", "lon", "t", "y"}
        assert b["lat"].shape == (2,)
        assert b["lon"].shape == (2,)
        assert b["t"].shape == (2,)
        assert b["y"].shape == (2,)
        assert torch.allclose(b["lon"], b["lat"] + 100.0)
        assert torch.allclose(b["y"], b["lat"] + b["lon"] + b["t"])

    gathered_lat = torch.cat([b["lat"] for b in batches]).sort().values
    gathered_lon = torch.cat([b["lon"] for b in batches]).sort().values
    gathered_t = torch.cat([b["t"] for b in batches]).sort().values
    gathered_y = torch.cat([b["y"] for b in batches]).sort().values

    assert torch.allclose(gathered_lat, lat.sort().values)
    assert torch.allclose(gathered_lon, lon.sort().values)
    assert torch.allclose(gathered_t, t.sort().values)
    assert torch.allclose(gathered_y, y.sort().values)


def test_reshard_random_train(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    lat_values = []

    for i in range(3):
        n = 8
        start = i * n
        lat = np.arange(start, start + n, dtype=np.float32)
        lon = lat + 100.0
        t = lat + 200.0
        y = lat + 300.0
        shard = {"lat": lat, "lon": lon, "t": t, "y": y}
        save_shard(shard_dir / f"shard_c001_p{i+1:03d}_toy.npz", shard)
        lat_values.append(lat)

    out_dir = tmp_path / "batch_shards"
    result = reshard_random_train(
        src_dir=shard_dir,
        out_dir=out_dir,
        seed=0,
        batch_size=4,
        batches_per_file=2,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    assert result.samples_loaded == 24
    assert result.dropped_samples == 0
    assert result.missing_shards == 0
    assert result.batches_written == 6
    assert len(result.written) == 3
    assert result.samples_loaded - result.dropped_samples == result.batches_written * result.batch_size
    assert set(result.split_groups.keys()) == {"train", "val", "test"}
    assert len(result.split_groups["train"]) == 3
    assert len(result.split_groups["val"]) == 0
    assert len(result.split_groups["test"]) == 0

    observed = []
    for path in result.written:
        data = torch.load(path)
        assert set(data.keys()) == {"X", "Y"}
        X = data["X"]
        Y = data["Y"]
        assert X.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert X.ndim == 3 and X.shape[2] == 3
        assert Y.ndim == 3 and Y.shape[2] == 1
        flat_X = X.reshape(-1, 3)
        flat_Y = Y.reshape(-1)
        observed.append(flat_X[:, 0])
        lat_vals = flat_X[:, 0]
        lon_vals = flat_X[:, 1]
        t_vals = flat_X[:, 2]
        assert torch.allclose(lon_vals, lat_vals + 100.0)
        assert torch.allclose(t_vals, lat_vals + 200.0)
        assert torch.allclose(flat_Y, lat_vals + 300.0)

    observed_lat = torch.cat(observed)
    expected_lat = torch.from_numpy(np.concatenate(lat_values))
    assert observed_lat.shape[0] == expected_lat.shape[0]
    obs_sorted, _ = torch.sort(observed_lat)
    exp_sorted, _ = torch.sort(expected_lat)
    assert torch.allclose(obs_sorted, exp_sorted.to(torch.float32))

    out_dir2 = tmp_path / "batch_shards_split"
    result_split = reshard_random_train(
        src_dir=shard_dir,
        out_dir=out_dir2,
        seed=0,
        batch_size=4,
        batches_per_file=2,
        val_ratio=0.05,
        test_ratio=0.05,
    )

    assert len(result_split.split_groups["val"]) >= 1
    assert len(result_split.split_groups["test"]) >= 1
    assert len(result_split.split_groups["train"]) >= 1
    total_groups = sum(len(v) for v in result_split.split_groups.values())
    assert total_groups == 3


def test_target_normalizer_roundtrip(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    n = 32
    lat = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lon = lat + 10.0
    t = np.linspace(100.0, 200.0, n, dtype=np.float64)
    y = np.linspace(-5.0, 7.0, n, dtype=np.float32)
    shard = {"lat": lat, "lon": lon, "t": t, "y": y}
    save_shard(shard_dir / "shard_c001_p001_norm.npz", shard)

    dm = ShardedDataModule(
        shards_dir=str(shard_dir),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        val_ratio=0.0,
        test_ratio=0.0,
        target_normalizer={"type": "zscore"},
    )
    dm.setup("fit")

    state = dm.stats["target"]
    assert state["type"] == "zscore"
    assert state["std"] > 0.0

    normalizer = TargetNormalizer.from_dict(state)
    sample = torch.from_numpy(y)
    normalized = normalizer.transform(sample)
    zero = torch.tensor(0.0, dtype=normalized.dtype)
    one = torch.tensor(1.0, dtype=normalized.dtype)
    assert torch.isclose(normalized.mean(), zero, atol=1e-5)
    assert torch.isclose(normalized.std(unbiased=False), one, atol=1e-5)

    restored = normalizer.inverse(normalized)
    assert torch.allclose(restored, sample, atol=1e-5)


def test_sharded_datamodule_prebatched_pipeline(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()

    lat_parts = []
    lon_parts = []
    t_parts = []
    y_parts = []

    for i, (cyc, pas) in enumerate([(1, 1), (1, 2), (2, 1)], start=0):
        base = i * 10.0
        n = 6
        lat = (base + np.linspace(0.0, 1.0, n, dtype=np.float32)).astype(np.float32)
        lon = lat + 42.0
        t = lat + 1000.0
        y = lat + lon + 0.5 * t
        shard = {"lat": lat, "lon": lon, "t": t, "y": y}
        save_shard(shard_dir / f"shard_c{cyc:03d}_p{pas:03d}_pipe.npz", shard)
        lat_parts.append(lat)
        lon_parts.append(lon)
        t_parts.append(t)
        y_parts.append(y)

    all_lat = np.concatenate(lat_parts)
    all_lon = np.concatenate(lon_parts)
    all_t = np.concatenate(t_parts)
    all_y = np.concatenate(y_parts)

    batch_dir = tmp_path / "prebatched"
    reshard_random_train(
        src_dir=shard_dir,
        out_dir=batch_dir,
        seed=123,
        batch_size=6,
        batches_per_file=2,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    monkeypatch.setenv("PL_GLOBAL_SEED", "0")

    dm = ShardedDataModule(
        shards_dir=str(shard_dir),
        batch_size=6,
        num_workers=0,
        pin_memory=False,
        val_ratio=0.0,
        test_ratio=0.0,
        train_batchshard_dir=str(batch_dir),
        train_micro_bs=3,
        target_normalizer={"type": "zscore"},
    )
    dm.setup("fit")

    assert isinstance(dm.train_ds, _BatchShardIterable)
    assert dm.stats["bbox"]["lat_min"] == pytest.approx(float(all_lat.min()))
    assert dm.stats["bbox"]["lat_max"] == pytest.approx(float(all_lat.max()))
    assert dm.stats["bbox"]["lon_min"] == pytest.approx(float(all_lon.min()))
    assert dm.stats["bbox"]["lon_max"] == pytest.approx(float(all_lon.max()))

    t_stats = dm.stats["time"]
    assert t_stats is not None
    assert t_stats["mean"] == pytest.approx(float(all_t.mean()))
    assert t_stats["std"] == pytest.approx(float(all_t.std()), rel=1e-3, abs=1e-5)

    target_stats = dm.stats["target"]
    assert target_stats["type"] == "zscore"
    assert target_stats["mean"] == pytest.approx(float(all_y.mean()))
    assert target_stats["std"] == pytest.approx(float(all_y.std()), rel=1e-3, abs=1e-5)

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"lat", "lon", "t", "y"}
    assert batch["lat"].shape[0] == 3
    assert batch["lon"].shape[0] == 3
    assert batch["t"].shape[0] == 3
    assert batch["y"].shape[0] == 3
    assert batch["lat"].dtype == torch.float32
    assert batch["t"].dtype == torch.float32
    assert torch.isfinite(batch["y"]).all()
    assert torch.allclose(batch["lon"], batch["lat"] + 42.0, atol=1e-6)
    assert torch.allclose(batch["y"], batch["lat"] + batch["lon"] + 0.5 * batch["t"], atol=1e-6)

    lit_cfg = {
        "model": {
            "pe": {
                "type": "rect_baseline",
                "include_annual": False,
                "include_default_tides": False,
                "tidal_periods_s": [],
                "time_norm": "zscore",
            },
            "net": {"type": "mlp", "width": 8, "depth": 1, "act": "relu"},
            "loss": "mse",
        },
        "optim": {"lr": 1e-3},
    }

    module = LitRegressor(lit_cfg, dm.stats)
    loss = module.training_step({k: v.clone() for k, v in batch.items()}, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
