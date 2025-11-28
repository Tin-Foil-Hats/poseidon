from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import NetBase, get_net_class, register_net


def _normalize_cfg(cfg: Optional[Dict[str, object]]) -> Dict[str, object]:
    if cfg is None:
        return {"type": "mlp"}
    if isinstance(cfg, str):
        return {"type": cfg}
    return dict(cfg)


@register_net(name="dual_net", aliases=["dual_branch", "two_tower"])
class DualBranchNet(NetBase):
    """Two-branch wrapper that routes spatial/time features to separate nets.

    The encoder must expose its feature layout via ``feature_layout`` with at
    least ``{"space": int, "time": int}``. Each branch can reuse any
    registered Poseidon net (``mlp``, ``resmlp``, etc.), and the outputs are
    aggregated through either a simple sum (``mode=add``) or a small fusion MLP
    (``mode=concat``).
    """

    def __init__(
        self,
        in_dim: int,
        *,
        space_net: Optional[Dict[str, object]] = None,
        time_net: Optional[Dict[str, object]] = None,
        fusion: Optional[Dict[str, object]] = None,
        fusion_mode: str = "add",
        layout: Optional[Dict[str, object]] = None,
        space_dim: Optional[int] = None,
        time_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.space_cfg = _normalize_cfg(space_net)
        self.time_cfg = _normalize_cfg(time_net)
        self.fusion_cfg = dict(fusion or {})
        self.fusion_mode = str(self.fusion_cfg.get("mode", fusion_mode)).lower()
        if self.fusion_mode not in {"add", "concat"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        self.layout_override = dict(layout or {})
        if space_dim is not None:
            self.layout_override["space"] = int(space_dim)
        if time_dim is not None:
            self.layout_override["time"] = int(time_dim)

        self.space_net: Optional[nn.Module] = None
        self.time_net: Optional[nn.Module] = None
        self.fusion_head: Optional[nn.Module] = None
        self._space_slice: Optional[slice] = None
        self._time_slice: Optional[slice] = None
        self._layout_bound = False

    def bind_encoder_layout(self, layout: Optional[Dict[str, int]]) -> None:
        if layout is None:
            layout = {}
        layout = dict(layout)
        if self.layout_override:
            layout.update({k: v for k, v in self.layout_override.items() if isinstance(v, int)})
            order_override = self.layout_override.get("order")
            if order_override is not None:
                layout["order"] = order_override

        self._apply_layout(layout)

    def _apply_layout(self, layout: Dict[str, int]) -> None:
        if self._layout_bound:
            return

        if "space" not in layout or "time" not in layout:
            raise ValueError(
                "DualBranchNet requires encoder layout with 'space' and 'time' dimensions."
            )

        space_dim = int(layout["space"])
        time_dim = int(layout["time"])
        order = list(layout.get("order", ["space", "time"]))

        if set(order) >= {"space", "time"}:
            order = [k for k in order if k in {"space", "time"}]
        else:
            order = ["space", "time"]

        offset = 0
        slices: Dict[str, slice] = {}
        for key in order:
            dim = space_dim if key == "space" else time_dim
            slices[key] = slice(offset, offset + dim)
            offset += dim
        if offset > self.in_dim:
            raise ValueError("Encoder feature layout exceeds provided in_dim for DualBranchNet")

        self._space_slice = slices["space"]
        self._time_slice = slices["time"]

        if self.space_net is None:
            self.space_net = self._build_branch(self.space_cfg, space_dim)
        if self.time_net is None:
            self.time_net = self._build_branch(self.time_cfg, time_dim)

        if self.fusion_mode == "concat" and self.fusion_head is None:
            input_dim = self._branch_output_dim(self.space_net) + self._branch_output_dim(self.time_net)
            self.fusion_head = self._build_fusion_head(input_dim)

        self._layout_bound = True

    @staticmethod
    def _branch_output_dim(net: nn.Module) -> int:
        if hasattr(net, "out_dim"):
            try:
                return int(net.out_dim())                          
            except TypeError:
                pass
        return 1

    @staticmethod
    def _build_branch(cfg: Dict[str, object], in_dim: int) -> nn.Module:
        if in_dim <= 0:
            raise ValueError("Branch input dimension must be positive for DualBranchNet")
        cfg = dict(cfg)
        net_type = cfg.pop("type", "mlp")
        cls = get_net_class(str(net_type))
        return cls(in_dim=in_dim, **cfg)

    def _build_fusion_head(self, input_dim: int) -> nn.Module:
        activation = str(self.fusion_cfg.get("activation", "relu")).lower()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation, nn.ReLU)
        depth = int(self.fusion_cfg.get("depth", 1))
        width = int(self.fusion_cfg.get("width", max(input_dim, 1)))

        layers: list[nn.Module] = []
        d = input_dim
        if depth <= 0:
            layers.append(nn.Linear(d, 1))
        else:
            for _ in range(depth - 1):
                layers.append(nn.Linear(d, width))
                layers.append(act_cls())
                d = width
            layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _prepare_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 1:
            return output.unsqueeze(-1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._layout_bound and {"space", "time"}.issubset(self.layout_override.keys()):
            init_layout = {
                "space": int(self.layout_override["space"]),
                "time": int(self.layout_override["time"]),
            }
            if "order" in self.layout_override:
                init_layout["order"] = self.layout_override["order"]
            self._apply_layout(init_layout)

        if not self._layout_bound:
            raise RuntimeError("DualBranchNet must bind encoder layout before forward pass")
        assert self._space_slice is not None and self._time_slice is not None
        assert self.space_net is not None and self.time_net is not None

        space_feats = x[..., self._space_slice]
        time_feats = x[..., self._time_slice]

        space_out = self._prepare_output(self.space_net(space_feats))
        time_out = self._prepare_output(self.time_net(time_feats))

        if self.fusion_mode == "add":
            if space_out.shape != time_out.shape:
                raise ValueError("Add fusion requires branch outputs with identical shapes")
            fused = space_out + time_out
            return fused.reshape(fused.shape[0], -1).squeeze(-1)

        concat = torch.cat([space_out, time_out], dim=-1)
        if self.fusion_head is None:
            self.fusion_head = self._build_fusion_head(concat.size(-1)).to(
                device=concat.device, dtype=concat.dtype
            )
        out = self.fusion_head(concat)
        return out.reshape(out.shape[0], -1).squeeze(-1)