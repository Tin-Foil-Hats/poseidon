import pytest
import torch

from poseidon.models.nets.deep_random_features import DeepRandomFeaturesNet


def _skip_if_optional_deps_missing():
    try:
        DeepRandomFeaturesNet(
            in_dim=3,
            spatial_dim=2,
            temporal_dim=1,
            num_layers=1,
            hidden_dim=16,
            bottleneck_dim=8,
        )
    except RuntimeError as exc:
        if "geometric-kernels" in str(exc):
            pytest.skip("geometric-kernels and scipy are required for DRF tests")
        raise


@pytest.mark.parametrize("combine", ["concat", "sum", "product"])
def test_forward_shapes_with_temporal(combine: str) -> None:
    _skip_if_optional_deps_missing()
    batch_size = 4
    model = DeepRandomFeaturesNet(
        in_dim=3,
        spatial_dim=2,
        temporal_dim=1,
        num_layers=2,
        hidden_dim=32,
        bottleneck_dim=16,
        combine=combine,
    ).eval()

    x = torch.tensor(
        [
            [-123.0, 45.0, 0.0],
            [10.0, -20.0, 1.5],
            [55.0, 10.0, -2.0],
            [179.0, -80.0, 0.75],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size,)
    assert torch.all(torch.isfinite(output))


def test_forward_without_temporal_branch() -> None:
    _skip_if_optional_deps_missing()
    model = DeepRandomFeaturesNet(
        in_dim=2,
        spatial_dim=2,
        temporal_dim=0,
        num_layers=2,
        hidden_dim=32,
        bottleneck_dim=16,
        combine="concat",
    ).eval()

    x = torch.tensor(
        [
            [0.0, 0.0],
            [45.0, 10.0],
            [-90.0, 5.0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (3,)
    assert torch.all(torch.isfinite(output))


def test_backward_pass_runs() -> None:
    _skip_if_optional_deps_missing()
    model = DeepRandomFeaturesNet(
        in_dim=3,
        spatial_dim=2,
        temporal_dim=1,
        num_layers=2,
        hidden_dim=16,
        bottleneck_dim=8,
    )

    x = torch.randn(5, 3, dtype=torch.float32)
    target = torch.randn(5, dtype=torch.float32)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    head_grad = model.head.weight.grad
    assert head_grad is not None
    assert torch.all(torch.isfinite(head_grad))
