import torch

from poseidon.training import (
    compute_loss_weights,
    cross_track_distance_weight,
    distance_to_coast_weight,
    normalize_weights,
    prepare_loss_weight_config,
)


def test_distance_to_coast_linear_invert():
    distances = torch.tensor([0.0, 50.0, 100.0, 150.0, 200.0])
    weights = distance_to_coast_weight(
        distances,
        method="linear",
        min_km=0.0,
        max_km=200.0,
        invert=True,
    )
    assert torch.all(weights <= 1.0)
    assert torch.all(weights >= 0.0)
    assert weights[0] > weights[-1]


def test_distance_to_coast_exp_reversed():
    distances = torch.tensor([0.0, 25.0, 50.0])
    weights = distance_to_coast_weight(
        distances,
        method="exp",
        scale_km=25.0,
        invert=False,
    )
    assert weights[0] < weights[-1]
    assert torch.all(weights > 0.0)


def test_cross_track_gaussian_symmetry():
    distances = torch.tensor([-5.0, 0.0, 5.0])
    weights = cross_track_distance_weight(distances, method="gaussian", sigma_km=5.0)
    assert torch.isclose(weights[0], weights[-1])
    assert weights[1] >= weights[0]


def test_normalize_weights_mean():
    weights = torch.tensor([1.0, 2.0, 3.0])
    normalized = normalize_weights(weights, mode="mean")
    assert torch.isclose(normalized.mean(), torch.tensor(1.0))


def test_compute_loss_weights_mean_normalization():
    cfg = {
        "distance_to_coast": {
            "method": "linear",
            "min_km": 0.0,
            "max_km": 2.0,
            "invert": True,
            "clamp_min": 0.0,
            "normalize": "none",
        },
        "cross_track_distance": {
            "method": "linear",
            "sigma_km": 2.0,
            "clamp_min": 0.0,
            "scale": 2.0,
            "normalize": "none",
        },
        "combine": "product",
        "normalize": "mean",
        "apply_to": ["train"],
    }
    prepared = prepare_loss_weight_config(cfg)
    batch = {
        "distance_to_coast": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        "cross_track_distance": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
    }

    weights = compute_loss_weights(prepared, batch, device=torch.device("cpu"), stage="train")
    assert weights is not None
    assert weights.shape == (3,)

    expected = torch.tensor([2.4, 0.6, 0.0], dtype=torch.float32)
    assert torch.allclose(weights, expected, atol=1e-5)


def test_compute_loss_weights_stage_aliases():
    cfg = {
        "distance_to_coast": {
            "method": "linear",
            "min_km": 0.0,
            "max_km": 2.0,
            "invert": True,
            "clamp_min": 0.0,
        },
        "apply_to": ["val"],
    }
    prepared = prepare_loss_weight_config(cfg)
    batch = {
        "distance_to_coast": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }

    weights_val = compute_loss_weights(prepared, batch, device=torch.device("cpu"), stage="validate")
    assert weights_val is not None
    assert weights_val.shape == (2,)

    weights_train = compute_loss_weights(prepared, batch, device=torch.device("cpu"), stage="train")
    assert weights_train is None
