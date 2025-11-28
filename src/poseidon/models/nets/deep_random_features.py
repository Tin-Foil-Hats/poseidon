"""Deep Random Features network (spherical Matérn random phase features).

This module ports the core architecture from Chen & Takao (2024)
(https://arxiv.org/abs/2412.11350) for use inside Poseidon. It supports
satellite-style inputs where the spatial coordinates are provided as
longitude/latitude pairs along with an optional temporal scalar. The network
alternates between frozen random feature maps (planar Matérn/Gaussian kernels)
and trainable linear heads, mirroring the design of the reference
implementation.

The spherical random phase feature map requires ``geometric-kernels`` and
``scipy``. If those packages are unavailable, the constructor raises a
``RuntimeError`` with installation guidance.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

try:                                                         
    from scipy.special import gegenbauer as scipy_gegenbauer                
except Exception:                                         
    scipy_gegenbauer = None

try:                                                     
    from geometric_kernels.spaces import Hypersphere                
    from geometric_kernels.kernels import MaternGeometricKernel                
except Exception:                    
    Hypersphere = None                
    MaternGeometricKernel = None                

from .base import NetBase, register_net

KernelKind = Literal["matern", "se", "sqexp", "gaussian"]


                                                                             
                                   
                                                                             
class _RFFLayer(nn.Module):
    """Frozen random Fourier feature block with a learnable output head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, amplitude: float = 1.0) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("RFFLayer dimensions must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.amplitude = float(amplitude)

        self.hidden = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.output = nn.Linear(self.hidden_dim, self.out_dim, bias=True)
        scale_val = math.sqrt(2.0) * self.amplitude / math.sqrt(self.hidden_dim)
        self.register_buffer("_scale", torch.tensor(scale_val, dtype=torch.float32), persistent=False)
        self.reset_parameters()

        self.hidden.weight.requires_grad_(False)
        self.hidden.bias.requires_grad_(False)

    def reset_parameters(self) -> None:                                         
        weight = self._sample_weight(self.hidden.weight.shape)
        if weight.dtype != self.hidden.weight.data.dtype:
            weight = weight.to(self.hidden.weight.data.dtype)
        self.hidden.weight.data.copy_(weight)
        self.hidden.bias.data.uniform_(0.0, 2.0 * math.pi)

        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(self.output.in_features))
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.hidden(x)
        feats = torch.cos(proj) * self._scale.to(dtype=x.dtype)
        return self.output(feats)

    def _sample_weight(self, shape: torch.Size) -> torch.Tensor:                               
        raise NotImplementedError


class _GaussianRFFLayer(_RFFLayer):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, lengthscale: float, amplitude: float) -> None:
        self.lengthscale = float(lengthscale)
        super().__init__(in_dim, hidden_dim, out_dim, amplitude=amplitude)

    def _sample_weight(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32) / self.lengthscale


class _MaternRFFLayer(_RFFLayer):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, lengthscale: float, nu: float, amplitude: float) -> None:
        self.lengthscale = float(lengthscale)
        self.nu = float(nu)
        super().__init__(in_dim, hidden_dim, out_dim, amplitude=amplitude)

    def _sample_weight(self, shape: torch.Size) -> torch.Tensor:
        dist = torch.distributions.StudentT(df=2.0 * self.nu, scale=torch.tensor(1.0 / self.lengthscale))
        return dist.sample(shape).to(torch.float32)


def _make_rff_layer(
    kind: KernelKind,
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    lengthscale: float,
    amplitude: float,
    nu: float,
) -> _RFFLayer:
    name = kind.lower()
    if name in {"se", "sqexp", "gaussian"}:
        return _GaussianRFFLayer(in_dim, hidden_dim, out_dim, lengthscale=lengthscale, amplitude=amplitude)
    if name == "matern":
        return _MaternRFFLayer(in_dim, hidden_dim, out_dim, lengthscale=lengthscale, nu=nu, amplitude=amplitude)
    raise ValueError(f"Unsupported kernel kind '{kind}'")


                                                                             
                                 
                                                                             
class RandomPhaseFeatureMap:
    """Random phase feature map for spherical Matérn Gaussian processes."""

    def __init__(self, num_features: int, *, nu: float, lengthscale: float, num_levels: int = 24) -> None:
        if scipy_gegenbauer is None or Hypersphere is None or MaternGeometricKernel is None:
            raise RuntimeError(
                "MaternRandomPhaseS2RFFLayer requires 'scipy' and 'geometric-kernels'. "
                "Install them via `pip install scipy geometric-kernels`."
            )

        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.num_features = int(num_features)
        self.nu = float(nu)
        self.lengthscale = float(lengthscale)
        self.num_levels = int(num_levels)

        sphere = Hypersphere(dim=2)
        kernel = MaternGeometricKernel(sphere, num=self.num_levels)
        eigenvalues = sphere.get_eigenvalues(self.num_levels)
        nu_arr = np.array([self.nu], dtype=np.float32)
        lengthscale_arr = np.array([self.lengthscale], dtype=np.float32)
        spectrum = kernel.spectrum(
            eigenvalues,
            nu=nu_arr,
            lengthscale=lengthscale_arr,
            dimension=sphere.dim,
        )
        spectrum = torch.as_tensor(spectrum, dtype=torch.float32).squeeze()
        self.spectrum = spectrum / torch.sum(spectrum)

        coeff_table = torch.zeros(self.num_levels, self.num_levels + 1, dtype=torch.float32)
        for level in range(self.num_levels):
            coeffs = scipy_gegenbauer(level, 0.5).coefficients                              
            coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
            coeff_table[level, -(level + 1) :] = coeffs
        self.gegenbauer_coeff_table = coeff_table

        self.levels = torch.multinomial(self.spectrum, num_samples=self.num_features, replacement=True)
        noise = torch.randn(self.num_features, 3, dtype=torch.float32)
        self.noise = noise / torch.linalg.norm(noise, dim=1, keepdim=True)

    @staticmethod
    def _polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for coeff in coeffs.transpose(0, 1):
            result = result * x + coeff.unsqueeze(1)
        return result

    @staticmethod
    def _addition_constant(levels: torch.Tensor) -> torch.Tensor:
        const = (2.0 * levels + 1.0) / (4.0 * math.pi)
        return const[:, None]

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        noise = self.noise.to(device)
        levels = self.levels.to(device)
        coeffs = self.gegenbauer_coeff_table.to(device)

        uxt = noise @ X.T          
        addition = self._addition_constant(levels)
        gegen = self._polyval(coeffs[levels], uxt)
        scaling = torch.sqrt(addition) / math.sqrt(self.num_features * 0.079)
        features = scaling * gegen
        return features.T          


def _spherical_to_cartesian(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


class MaternRandomPhaseS2RFFLayer(nn.Module):
    """Spherical Matérn random phase layer."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        lengthscale: float,
        nu: float,
        amplitude: float,
        lon_lat_inputs: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.lengthscale = float(lengthscale)
        self.nu = float(nu)
        self.amplitude = float(amplitude)
        self.lon_lat_inputs = bool(lon_lat_inputs)

        self.feature_map = RandomPhaseFeatureMap(
            num_features=self.hidden_dim, nu=self.nu, lengthscale=self.lengthscale
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        coords = x.to(device)
        if self.lon_lat_inputs:
            coords = _spherical_to_cartesian(coords[:, 0], coords[:, 1])
        features = self.feature_map(coords).to(device)
        features = features * self.amplitude
        return self.output_layer(features)


class SumFeatures(nn.Module):
    """Fuse planar RFFs and spherical random phase features."""

    def __init__(
        self,
        *,
        input_dim: int,
        spherical_dim: int,
        hidden_dim: int,
        output_dim: int,
        lengthscale: float,
        amplitude: float,
        planar_lengthscale: float,
        planar_amplitude: float,
        nu: float,
        lon_lat_inputs: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= spherical_dim:
            raise ValueError("input_dim must exceed spherical_dim")

        planar_dim = input_dim - spherical_dim
        self.lon_lat_inputs = bool(lon_lat_inputs)
        self.spherical_dim = int(spherical_dim)

        planar_block = _MaternRFFLayer(
            planar_dim,
            hidden_dim,
            output_dim,
            lengthscale=planar_lengthscale,
            nu=nu,
            amplitude=planar_amplitude,
        )
        self.planar_layer = planar_block.hidden
        self.planar_layer.weight.requires_grad_(False)
        self.planar_layer.bias.requires_grad_(False)
        self.planar_scale = math.sqrt(2.0) * planar_amplitude / math.sqrt(hidden_dim)

        self.spherical_layer = RandomPhaseFeatureMap(
            num_features=hidden_dim, nu=nu, lengthscale=lengthscale
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        planar = x[:, :-self.spherical_dim].to(device)
        spherical = x[:, -self.spherical_dim :].to(device)

        if self.lon_lat_inputs:
            spherical = _spherical_to_cartesian(spherical[:, 0], spherical[:, 1])

        planar_proj = self.planar_layer(planar)
        planar_feats = torch.cos(planar_proj) * self.planar_scale
        spherical_feats = self.spherical_layer(spherical).to(device)
        fused = planar_feats + spherical_feats
        return self.output_layer(fused)


                                                                             
                    
                                                                             

def _expand(value: float | Sequence[float], *, count: int, name: str) -> list[float]:
    if isinstance(value, Iterable) and not isinstance(value, (int, float)):
        expanded = [float(v) for v in value]
        if len(expanded) != count:
            raise ValueError(f"Expected {count} entries for '{name}', got {len(expanded)}")
        return expanded
    return [float(value)] * count


@register_net(name="drf", aliases=("deep_random_features", "deep_rff"))
class DeepRandomFeaturesNet(NetBase):
    """Deep random features network with spherical Matérn blocks."""

    def __init__(
        self,
        in_dim: int,
        *,
        spatial_dim: int = 2,
        temporal_dim: int = 1,
        num_layers: int = 3,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        kernel: KernelKind = "matern",
        temporal_kernel: KernelKind | None = None,
        lengthscale: float | Sequence[float] = 1.0,
        amplitude: float | Sequence[float] = 1.0,
        planar_lengthscale: float | Sequence[float] | None = None,
        planar_amplitude: float | Sequence[float] | None = None,
        temporal_lengthscale: float | None = None,
        temporal_amplitude: float | None = None,
        nu: float = 1.5,
        lon_lat_inputs: bool = True,
        spatial_in_degrees: bool = True,
        combine: Literal["concat", "sum", "product"] = "concat",
        spatial_order: Literal["lonlat", "latlon"] = "lonlat",
        output_dim: int = 1,
        use_spherical: bool = True,
        skip_input: bool = True,
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if spatial_dim <= 0:
            raise ValueError("spatial_dim must be positive")
        if spatial_dim + temporal_dim > in_dim:
            raise ValueError("spatial_dim + temporal_dim exceeds input dimensionality")
        if lon_lat_inputs and use_spherical and spatial_dim < 2:
            raise ValueError("lon_lat_inputs=True requires lon/lat inputs")

        self.spatial_dim = int(spatial_dim)
        self.temporal_dim = int(temporal_dim)
        self.num_layers = int(num_layers)
        self.bottleneck_dim = int(bottleneck_dim)
        self.lon_lat_inputs = bool(lon_lat_inputs)
        self.spatial_in_degrees = bool(spatial_in_degrees)
        if spatial_order not in {"lonlat", "latlon"}:
            raise ValueError("spatial_order must be 'lonlat' or 'latlon'")
        self.spatial_order = spatial_order
        self.combine = combine
        self.output_dimension = int(output_dim)

        self.use_spherical = bool(use_spherical)
        self.skip_input = bool(skip_input)
        spatial_lengthscales = _expand(lengthscale, count=num_layers, name="lengthscale")
        spatial_amplitudes = _expand(amplitude, count=num_layers, name="amplitude")
        planar_lengthscales = _expand(
            planar_lengthscale if planar_lengthscale is not None else lengthscale,
            count=max(num_layers - 1, 1),
            name="planar_lengthscale",
        )
        planar_amplitudes = _expand(
            planar_amplitude if planar_amplitude is not None else amplitude,
            count=max(num_layers - 1, 1),
            name="planar_amplitude",
        )

        temporal_kernel = temporal_kernel or kernel
        temporal_lengthscale = float(temporal_lengthscale or spatial_lengthscales[-1])
        temporal_amplitude = float(temporal_amplitude or spatial_amplitudes[-1])

        self.spatial_layers = nn.ModuleList()

        if self.use_spherical:
            self.s2_dim = 2 if self.lon_lat_inputs else 3
            for idx in range(num_layers):
                if idx == 0:
                    layer = MaternRandomPhaseS2RFFLayer(
                        hidden_dim=hidden_dim,
                        output_dim=bottleneck_dim,
                        lengthscale=spatial_lengthscales[idx],
                        nu=nu,
                        amplitude=spatial_amplitudes[idx],
                        lon_lat_inputs=self.lon_lat_inputs,
                    )
                else:
                    layer = SumFeatures(
                        input_dim=bottleneck_dim + self.s2_dim,
                        spherical_dim=self.s2_dim,
                        hidden_dim=hidden_dim,
                        output_dim=bottleneck_dim,
                        lengthscale=spatial_lengthscales[idx],
                        amplitude=spatial_amplitudes[idx],
                        planar_lengthscale=planar_lengthscales[idx - 1],
                        planar_amplitude=planar_amplitudes[idx - 1],
                        nu=nu,
                        lon_lat_inputs=self.lon_lat_inputs,
                    )
                self.spatial_layers.append(layer)
        else:
            self.s2_dim = 0
            current_dim = self.spatial_dim
            for idx in range(num_layers):
                layer = _make_rff_layer(
                    kernel,
                    in_dim=current_dim,
                    hidden_dim=hidden_dim,
                    out_dim=bottleneck_dim,
                    lengthscale=spatial_lengthscales[idx],
                    amplitude=spatial_amplitudes[idx],
                    nu=nu,
                )
                self.spatial_layers.append(layer)
                if self.skip_input and idx < num_layers - 1:
                    current_dim = bottleneck_dim + self.spatial_dim
                else:
                    current_dim = bottleneck_dim

        if self.temporal_dim > 0:
            self.temporal_layer = _make_rff_layer(
                temporal_kernel,
                in_dim=self.temporal_dim,
                hidden_dim=hidden_dim,
                out_dim=bottleneck_dim,
                lengthscale=temporal_lengthscale,
                amplitude=temporal_amplitude,
                nu=nu,
            )
        else:
            self.temporal_layer = None

        if combine == "concat":
            head_in = bottleneck_dim + (0 if self.temporal_dim == 0 else bottleneck_dim)
        elif combine in {"sum", "product"}:
            head_in = bottleneck_dim
        else:
            raise ValueError(f"Unsupported combine mode '{combine}'")
        self.head = nn.Linear(head_in, self.output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[..., : self.spatial_dim]
        temporal = x[..., self.spatial_dim : self.spatial_dim + self.temporal_dim]

        if self.lon_lat_inputs and self.spatial_in_degrees:
            spatial = torch.deg2rad(spatial)

        if self.spatial_order == "latlon" and spatial.shape[-1] >= 2:
            front = spatial[..., :2]
            rest = spatial[..., 2:]
            front = front[..., [1, 0]]
            spatial = torch.cat([front, rest], dim=-1) if rest.numel() else front

        spatial_feats = spatial
        spatial_base = spatial
        for idx, layer in enumerate(self.spatial_layers):
            spatial_feats = layer(spatial_feats)
            if idx < self.num_layers - 1:
                if self.use_spherical:
                    spatial_feats = torch.cat([spatial_feats, spatial_base], dim=-1)
                elif self.skip_input:
                    spatial_feats = torch.cat([spatial_feats, spatial_base], dim=-1)

        if self.temporal_layer is not None and self.temporal_dim > 0:
            temporal_feats = self.temporal_layer(temporal)
        else:
            temporal_feats = None

        if self.combine == "concat":
            if temporal_feats is None:
                fused = spatial_feats
            else:
                fused = torch.cat([spatial_feats, temporal_feats], dim=-1)
        elif self.combine == "sum":
            fused = spatial_feats if temporal_feats is None else spatial_feats + temporal_feats
        else:           
            fused = spatial_feats if temporal_feats is None else spatial_feats * temporal_feats

        out = self.head(fused)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def out_dim(self) -> int:                                           
        return self.output_dimension
