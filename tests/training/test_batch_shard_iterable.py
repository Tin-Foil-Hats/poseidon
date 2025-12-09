import os

import torch

from poseidon.training.lightning_datamodule import _BatchShardIterable


def _make_shard(dir_path: str) -> torch.Tensor:
    X = torch.zeros(1, 4, 3)
    X[0, :, 0] = torch.linspace(-1.0, 1.0, steps=4)  # lat
    X[0, :, 1] = torch.linspace(0.0, 3.0, steps=4)   # lon
    X[0, :, 2] = torch.tensor([0.0, 10.0, 20.0, 30.0])  # time

    Y = torch.linspace(-2.0, 2.0, steps=4).view(1, 4, 1)

    distance_to_coast = torch.linspace(0.0, 30.0, steps=4).view(1, 4, 1)
    cross_track_distance = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).view(1, 4, 1)

    shard = {
        "X": X,
        "Y": Y,
        "distance_to_coast": distance_to_coast,
        "cross_track_distance": cross_track_distance,
    }
    torch.save(shard, os.path.join(dir_path, "sample.pt"))
    return distance_to_coast.view(-1)


def test_batch_iterable_exposes_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "0")
    dist = _make_shard(tmp_path.as_posix())

    iterable = _BatchShardIterable(
        shard_dir=tmp_path.as_posix(),
        micro_bs=0,
        shuffle_files=False,
        shuffle_batches=False,
    )
    batch = next(iter(iterable))

    assert "distance_to_coast" in batch
    assert "cross_track_distance" in batch
    assert batch["distance_to_coast"].shape == batch["y"].shape
    assert torch.allclose(batch["distance_to_coast"], dist)


def test_batch_iterable_micro_batches_keep_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "0")
    _make_shard(tmp_path.as_posix())

    iterable = _BatchShardIterable(
        shard_dir=tmp_path.as_posix(),
        micro_bs=2,
        shuffle_files=False,
        shuffle_batches=False,
    )
    batch = next(iter(iterable))

    assert batch["distance_to_coast"].shape[0] == 2
    assert batch["cross_track_distance"].shape[0] == 2
