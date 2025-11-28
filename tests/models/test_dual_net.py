import torch

from poseidon.models.poseidon_model import build_model


def _context():
    return {
        "bbox": {"lat_min": 0.0, "lat_max": 10.0, "lon_min": -5.0, "lon_max": 5.0},
        "time": {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0},
    }


def test_dual_net_add_mode():
    cfg = {
        "pe": {
            "type": "rect_baseline",
            "include_annual": False,
            "tidal_periods_s": [44714.16, 43200.0],
            "time_norm": "zscore",
        },
        "net": {
            "type": "dual_net",
            "fusion": {"mode": "add"},
            "space_net": {"type": "mlp", "width": 16, "depth": 2},
            "time_net": {"type": "mlp", "width": 16, "depth": 2},
        },
        "loss": "mse",
    }

    model, _ = build_model(cfg, context=_context())
    batch = 8
    lat = torch.linspace(0.0, 10.0, batch)
    lon = torch.linspace(-5.0, 5.0, batch)
    t = torch.linspace(-1.0, 1.0, batch)

    out = model(lat, lon, t)
    assert out.shape == (batch, 1)


def test_dual_net_concat_mode():
    cfg = {
        "pe": {
            "type": "rect_baseline",
            "include_annual": True,
            "tidal_periods_s": [44714.16, 43200.0],
            "time_norm": "zscore",
        },
        "net": {
            "type": "dual_net",
            "fusion": {"mode": "concat", "width": 32, "depth": 2, "activation": "relu"},
            "space_net": {"type": "mlp", "width": 16, "depth": 2},
            "time_net": {"type": "mlp", "width": 16, "depth": 2},
        },
        "loss": "mse",
    }

    model, _ = build_model(cfg, context=_context())
    batch = 6
    lat = torch.rand(batch) * 10.0
    lon = torch.rand(batch) * 10.0 - 5.0
    t = torch.rand(batch) * 2.0 - 1.0

    out = model(lat, lon, t)
    assert out.shape == (batch, 1)