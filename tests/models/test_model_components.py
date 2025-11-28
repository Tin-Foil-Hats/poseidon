import math
import torch

from poseidon.models.losses import build_loss, list_registered_losses
from poseidon.models.nets import build_net, list_registered_nets
from poseidon.models.pe import build_pe, list_registered_pes
from poseidon.models.poseidon_model import build_model


def _sample_inputs(batch: int = 8):
    lat = torch.linspace(-5.0, 5.0, batch)
    lon = torch.linspace(10.0, 20.0, batch)
    t = torch.linspace(-2.0, 2.0, batch)
    return lat, lon, t


def _context():
    return {
        "bbox": {"lat_min": -10.0, "lat_max": 10.0, "lon_min": 5.0, "lon_max": 25.0},
        "time": {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0},
    }


def test_all_registered_nets_produce_expected_shape():
    lat, lon, t = _sample_inputs()
    in_dim = 4
    features = torch.stack([lat, lon, t, torch.ones_like(lat)], dim=-1)
    overrides = {
        "mlp": {"width": 16, "depth": 2},
        "resmlp": {"width": 16, "depth": 2},
        "siren": {"width": 16, "depth": 2, "omega0": 10.0, "omega0_hidden": 1.0},
        "siren_stable": {"width": 16, "depth": 2},
        "dual_net": {
            "layout": {"space": 2, "time": 2},
            "space_net": {"type": "mlp", "width": 8, "depth": 1},
            "time_net": {"type": "mlp", "width": 8, "depth": 1},
            "fusion": {"mode": "add"},
        },
        "dual_branch": {
            "layout": {"space": 2, "time": 2},
            "space_net": {"type": "mlp", "width": 8, "depth": 1},
            "time_net": {"type": "mlp", "width": 8, "depth": 1},
            "fusion": {"mode": "add"},
        },
        "two_tower": {
            "layout": {"space": 2, "time": 2},
            "space_net": {"type": "mlp", "width": 8, "depth": 1},
            "time_net": {"type": "mlp", "width": 8, "depth": 1},
            "fusion": {"mode": "add"},
        },
    }
    for name in list_registered_nets().keys():
        cfg = {"type": name, **overrides.get(name, {})}
        net = build_net(cfg, in_dim=in_dim)
        out = net(features)
        if isinstance(out, tuple):
            out, aux = out
            assert aux.shape == (features.shape[0],)
        assert out.shape == (features.shape[0],)


def test_registered_positional_encoders_match_feat_dim():
    lat, lon, t = _sample_inputs()
    ctx = _context()
    pe_names = list_registered_pes().keys()
    for name in pe_names:
        cfg = {"type": name}
        pe = build_pe(cfg, context=ctx)
        feats = pe(lat, lon, t)
        assert feats.shape == (lat.shape[0], pe.feat_dim())


def test_time_only_encoder_normalizes_time():
    lat, lon, t = _sample_inputs()
    ctx = _context()
    cfg = {"type": "time_only", "time_norm": "zscore"}
    pe = build_pe(cfg, context=ctx)
    feats = pe(lat, lon, t)
    assert feats.shape == (lat.shape[0], 1)
    expected = (t - ctx["time"]["mean"]) / ctx["time"]["std"]
    assert torch.allclose(feats.squeeze(-1), expected, atol=1e-6)

    pe_bias = build_pe({"type": "time_only", "include_bias": True}, context=ctx)
    feats_bias = pe_bias(lat, lon, t)
    assert feats_bias.shape == (lat.shape[0], 2)
    time_feat = feats_bias[:, 0]
    bias_feat = feats_bias[:, 1]
    assert torch.allclose(time_feat, t, atol=1e-6)
    assert torch.allclose(bias_feat, torch.ones_like(t), atol=1e-6)


def test_time_with_raw_encoder_uses_sincos_space():
    lat, lon, t = _sample_inputs()
    ctx = _context()
    pe = build_pe({"type": "time_with_raw", "time_norm": "minmax"}, context=ctx)
    feats = pe(lat, lon, t)
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    assert feats.shape == (lat.shape[0], 5)
    assert torch.allclose(feats[:, 0], torch.sin(lat_rad), atol=1e-6)
    assert torch.allclose(feats[:, 1], torch.cos(lat_rad), atol=1e-6)
    assert torch.allclose(feats[:, 2], torch.sin(lon_rad), atol=1e-6)
    assert torch.allclose(feats[:, 3], torch.cos(lon_rad), atol=1e-6)
    t_expected = (t - ctx["time"]["min"]) / (ctx["time"]["max"] - ctx["time"]["min"]) * 2.0 - 1.0
    assert torch.allclose(feats[:, 4], t_expected, atol=1e-6)

    pe_bias = build_pe({"type": "time_with_raw", "include_bias": True, "time_norm": "none"}, context=ctx)
    feats_bias = pe_bias(lat, lon, t)
    assert feats_bias.shape == (lat.shape[0], 6)
    assert torch.allclose(feats_bias[:, 0], torch.sin(lat_rad), atol=1e-6)
    assert torch.allclose(feats_bias[:, 1], torch.cos(lat_rad), atol=1e-6)
    assert torch.allclose(feats_bias[:, 2], torch.sin(lon_rad), atol=1e-6)
    assert torch.allclose(feats_bias[:, 3], torch.cos(lon_rad), atol=1e-6)
    assert torch.allclose(feats_bias[:, 4], t, atol=1e-6)
    assert torch.allclose(feats_bias[:, 5], torch.ones_like(t), atol=1e-6)


def test_time_with_harmonics_adds_expected_features():
    lat, lon, t = _sample_inputs()
    ctx = _context()
    cfg = {
        "type": "time_with_harmonics",
        "time_norm": "center",
        "include_bias": True,
        "fourier_bands": 2,
        "include_diurnal": True,
        "include_semidiurnal": False,
        "include_weekly": True,
        "include_annual": True,
        "extra_periods_s": [3600.0],
    }
    pe = build_pe(cfg, context=ctx)
    feats = pe(lat, lon, t)
    freq_count = int(pe.fourier_freqs.numel())
    period_count = int(pe.period_omegas.numel())
    expected_dim = 4 + 1 + 2 * freq_count + 2 * period_count + (1 if cfg["include_bias"] else 0)
    assert feats.shape == (lat.shape[0], expected_dim)
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    assert torch.allclose(feats[:, 0], torch.sin(lat_rad), atol=1e-6)
    assert torch.allclose(feats[:, 1], torch.cos(lat_rad), atol=1e-6)
    assert torch.allclose(feats[:, 2], torch.sin(lon_rad), atol=1e-6)
    assert torch.allclose(feats[:, 3], torch.cos(lon_rad), atol=1e-6)

    layout = pe.feature_layout()
    assert layout["space"] == 4
    assert layout["time"] == expected_dim - layout["space"] - (1 if cfg["include_bias"] else 0)

                                                    
    t_norm = t - ctx["time"]["mean"]
    start = layout["space"] + 1
    if freq_count > 0:
        scaled = t_norm[0] * pe.fourier_freqs
        sin_expected = torch.sin(scaled)
        cos_expected = torch.cos(scaled)
        sin_slice = slice(start, start + freq_count)
        cos_slice = slice(start + freq_count, start + 2 * freq_count)
        assert torch.allclose(feats[0, sin_slice], sin_expected, atol=1e-6)
        assert torch.allclose(feats[0, cos_slice], cos_expected, atol=1e-6)

                                              
    assert torch.allclose(feats[:, -1], torch.ones_like(lat), atol=1e-6)


def test_registered_losses_handle_scalar_and_gaussian_preds():
    pred = torch.tensor([1.0, -2.0])
    y = torch.tensor([0.5, -1.5])
    loss_names = list_registered_losses().keys()
    for name in loss_names:
        loss_fn = build_loss(name)
        base = loss_fn(pred, y)
        assert base.shape == pred.shape
        pred_tuple = (pred, torch.zeros_like(pred))
        aux = loss_fn(pred_tuple, y)
        assert aux.shape == pred.shape
    delta = 0.2
    delta_loss = build_loss({"type": "huber", "delta": delta})
    err = pred - y
    abs_err = torch.abs(err)
    expected = torch.where(abs_err < delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
    assert torch.allclose(expected, delta_loss(pred, y), atol=1e-6)


def test_locenc_model_output_dimensions():
    lat, lon, t = _sample_inputs()
    ctx = _context()
    cfg = {
        "pe": {"type": "fourier", "n_lat": 4, "n_lon": 4, "n_t": 2, "add_xyz": False},
        "net": {"type": "mlp", "width": 32, "depth": 2},
        "loss": "mse",
    }
    model, loss_fn = build_model(cfg, context=ctx)
    output = model(lat, lon, t)
    assert output.shape == (lat.shape[0], 1)
    assert loss_fn(output.squeeze(-1), torch.zeros_like(lat)).shape == lat.shape

    cfg_unc = {
        "pe": {"type": "fourier", "n_lat": 4, "n_lon": 4, "n_t": 2, "add_xyz": False},
        "net": {"type": "mlp", "width": 32, "depth": 2, "out_unc": True},
        "loss": "mse",
    }
    model_unc, _ = build_model(cfg_unc, context=ctx)
    output_unc = model_unc(lat, lon, t)
    assert isinstance(output_unc, tuple)
    mean, logvar = output_unc
    assert mean.shape == (lat.shape[0], 1)
    assert logvar.shape == (lat.shape[0],)

    cfg_time_raw = {
        "pe": {"type": "time_with_raw", "time_norm": "zscore"},
        "net": {"type": "mlp", "width": 16, "depth": 1},
        "loss": "mse",
    }
    model_time_raw, _ = build_model(cfg_time_raw, context=ctx)
    output_time_raw = model_time_raw(lat, lon, t)
    assert output_time_raw.shape == (lat.shape[0], 1)