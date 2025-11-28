                 
import torch
import torch.nn as nn
from .pe import build_pe, PEBase
from .nets import build_net
from .losses import build_loss

class LocEncModel(nn.Module):
    """Wrapper pairing a positional encoder with a regression network."""
    def __init__(self, pe: PEBase, net: nn.Module):
        super().__init__()
        self.pe = pe
        self.net = net
        self._last_layout: dict | None = None

    def forward(self, lat, lon, t):
        """Encode geospatial inputs and return network predictions."""
        Z = self.pe(lat, lon, t)
        try:
            self._last_layout = self.pe.feature_layout()
        except Exception:
            self._last_layout = None
        out = self.net(Z)
        if isinstance(out, tuple):
            y, aux = out
            if y.ndim == 1: y = y.unsqueeze(-1)
            elif y.ndim == 2 and y.shape[-1] != 1: y = y[:, :1]
            return (y, aux)
        y = out
        if y.ndim == 1: y = y.unsqueeze(-1)
        elif y.ndim == 2 and y.shape[-1] != 1: y = y[:, :1]
        return y

    def feature_layout(self) -> dict | None:
        """Return the most recent feature layout emitted by the encoder."""
        if self._last_layout is not None:
            return self._last_layout
        try:
            return self.pe.feature_layout()
        except Exception:
            return None


def build_model(cfg: dict, context: dict):
    """Instantiate positional encoder, network, and loss from config.

    The context mapping supplies PE-specific statistics:
      bbox: {lat_min, lat_max, lon_min, lon_max}  # degrees
      time: {mean, std, min, max}                 # seconds
    """
    pe = build_pe(cfg["pe"], context=context)
    in_dim = pe.feat_dim()
    layout = pe.feature_layout()
    net_cfg = dict(cfg.get("net", {}))
    net = build_net(net_cfg, in_dim=in_dim)
    if hasattr(net, "bind_encoder_layout"):
        net.bind_encoder_layout(layout)
    loss_fn = build_loss(cfg.get("loss", "mse"))
    return LocEncModel(pe, net), loss_fn
