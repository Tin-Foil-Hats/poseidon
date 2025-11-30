"""Lightning module wrapper around Poseidon models."""

import torch
import torch.nn as nn
import lightning.pytorch as L

from poseidon.data.transforms import TargetNormalizer
from poseidon.models.poseidon_model import build_model
from poseidon.training.lr_schedulers import LinearWarmupCosineAnnealingLR

class LitRegressor(L.LightningModule):
    """Lightning module wrapping Poseidon models with gradient diagnostics."""
    def __init__(self, cfg, dm_stats, lr=1e-3, wd=1e-4):
        super().__init__()
        self.config = cfg
        context = {"bbox": dm_stats["bbox"], "time": dm_stats["time"]}
        model_cfg = cfg.get("model") if isinstance(cfg, dict) and "model" in cfg else cfg
        self.model, self.loss_fn = build_model(model_cfg, context=context)
        self.lr, self.wd = float(lr), float(wd)
        if isinstance(cfg, dict):
            self.optim_cfg = dict(cfg.get("optim", {}))
            self.reg_cfg = dict(cfg.get("regularization", {}))
        else:
            self.optim_cfg = {}
            self.reg_cfg = {}
        self.use_ema = bool(self.optim_cfg.get("ema", False))
        self.ema_decay = float(self.optim_cfg.get("ema_decay", 0.999))
        self.target_normalizer = TargetNormalizer.from_dict((dm_stats or {}).get("target"))
        self._feature_layout_logged = False
        self._grad_linear: nn.Module | None = None
        self._grad_slices: dict[str, slice] | None = None
        self._grad_accum: dict[str, float] = {}
        self._grad_counts = 0

    def training_step(self, batch, _):
        """Run a training iteration and log loss plus gradient group stats."""
        lat = batch["lat"].float()
        lon = batch["lon"].float()
        t   = batch.get("t", None)
        if t is not None: t = t.float()
        y   = batch["y"].float()

        pred = self.model(lat, lon, t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_raw = pred.reshape(-1)
        y_raw = y.reshape(-1)
        batch_size = max(int(y_raw.numel()), 1)
        # Model predictions are produced in physical target units even when the loss uses normalized values.
        pred_physical = pred_raw

        if self.target_normalizer.active:
            pred_for_loss = self.target_normalizer.transform(pred_raw)
            target_for_loss = self.target_normalizer.transform(y_raw)
        else:
            pred_for_loss = pred_raw
            target_for_loss = y_raw
        loss = self.loss_fn(pred_for_loss, target_for_loss).mean()

                                                                             
        lap_w = float(self.reg_cfg.get("laplace_weight", 0.0))
        if lap_w > 0.0:
            lap_pen = self._laplacian_penalty(lat, lon, pred_physical)
            loss = loss + lap_w * lap_pen
            self.log(
                "train_laplace_penalty",
                lap_pen.detach(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if not self._feature_layout_logged:
            layout = self.model.feature_layout() if hasattr(self.model, "feature_layout") else None
            if layout:
                print(f"[LitRegressor] feature layout: {layout}")
                self._feature_layout_logged = True
            self._prepare_grad_monitor(layout)
        else:
            self._prepare_grad_monitor(self.model.feature_layout() if hasattr(self.model, "feature_layout") else None)

                                                       
        diff = pred_physical - y_raw
        raw_mse = torch.mean(diff * diff)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )

        self.log(
            "train_loss_raw",
            raw_mse.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )

        rmse = torch.sqrt(raw_mse)
        self.log(
            "train_rmse",
            rmse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_fit_start(self):
        if self.use_ema:
            self.ema_params = [p.detach().clone() for p in self.model.parameters()]

    def validation_step(self, batch, _):
        """Compute validation loss/metrics for a single batch."""
        lat = batch["lat"].float()
        lon = batch["lon"].float()
        t   = batch.get("t", None)
        if t is not None: t = t.float()
        y   = batch["y"].float()

        pred = self.model(lat, lon, t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_raw = pred.reshape(-1)
        y_raw = y.reshape(-1)

        if self.target_normalizer.active:
            pred_for_loss = self.target_normalizer.transform(pred_raw)
            target_for_loss = self.target_normalizer.transform(y_raw)
        else:
            pred_for_loss = pred_raw
            target_for_loss = y_raw
        loss = self.loss_fn(pred_for_loss, target_for_loss).mean()
        batch_size = max(int(y_raw.numel()), 1)
        self.log("val_loss", loss, prog_bar=True, sync_dist=False, batch_size=batch_size)

        pred_physical = pred_raw
        diff = pred_physical - y_raw
        raw_mse = torch.mean(diff * diff)
        self.log(
            "val_loss_raw",
            raw_mse.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        rmse = torch.sqrt(raw_mse)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, _):
        """Evaluate on held-out test data using the configured loss."""
        lat = batch["lat"].float()
        lon = batch["lon"].float()
        t   = batch.get("t", None)
        if t is not None:
            t = t.float()
        y   = batch["y"].float()

        pred = self.model(lat, lon, t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_raw = pred.reshape(-1)
        y_raw = y.reshape(-1)

        if self.target_normalizer.active:
            pred_for_loss = self.target_normalizer.transform(pred_raw)
            target_for_loss = self.target_normalizer.transform(y_raw)
        else:
            pred_for_loss = pred_raw
            target_for_loss = y_raw
        loss = self.loss_fn(pred_for_loss, target_for_loss).mean()
        batch_size = max(int(y_raw.numel()), 1)
        self.log("test_loss", loss, prog_bar=True, sync_dist=False, batch_size=batch_size)

        pred_physical = pred_raw
        diff = pred_physical - y_raw
        raw_mse = torch.mean(diff * diff)
        self.log(
            "test_loss_raw",
            raw_mse.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        rmse = torch.sqrt(raw_mse)
        self.log("test_rmse", rmse, prog_bar=True, sync_dist=False, batch_size=batch_size)
        return loss

    def on_after_backward(self):
        """Accumulate gradient norms for feature groups after each backward pass."""
        super().on_after_backward()
        if not self._grad_slices or self._grad_linear is None:
            return
        weight_grad = self._grad_linear.weight.grad
        if weight_grad is None:
            return
        with torch.no_grad():
            for key, slc in self._grad_slices.items():
                if slc.stop <= slc.start:
                    continue
                group_grad = weight_grad[:, slc]
                if group_grad.numel() == 0:
                    continue
                value = torch.linalg.vector_norm(group_grad).detach().cpu().item()
                self._grad_accum[key] = self._grad_accum.get(key, 0.0) + value
            self._grad_counts += 1

    def on_train_epoch_end(self):
        """Log averaged gradient norms per feature group at epoch end."""
        super().on_train_epoch_end()
        if self._grad_counts and self._grad_accum:
            device = self._grad_linear.weight.device if self._grad_linear is not None else torch.device("cpu")
            for key, total in self._grad_accum.items():
                avg = total / self._grad_counts
                tensor = torch.tensor(avg, dtype=torch.float32, device=device)
                self.log(
                    f"grad_norm/{key}",
                    tensor,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            self._grad_accum = {key: 0.0 for key in self._grad_accum}
            self._grad_counts = 0

    def _prepare_grad_monitor(self, layout: dict | None):
        """Initialise gradient bookkeeping based on encoder feature layout."""
        if layout is None or self._grad_linear is not None:
            return
        backbone = self.model.net if hasattr(self.model, "net") else self.model
        first_linear = self._find_first_linear(backbone)
        if first_linear is None or first_linear.weight.ndim != 2:
            return
        full_dim = int(first_linear.weight.shape[1])
        layout_full = int(layout.get("full", full_dim))
        if full_dim != layout_full:
            print(
                f"[LitRegressor] Warning: feature layout full ({layout_full}) differs from first layer input ({full_dim}); using input width."
            )
        order = layout.get("order") if isinstance(layout, dict) else None
        if not order:
            order = [key for key in ("space", "time", "other") if isinstance(layout, dict) and key in layout]
        slices: dict[str, slice] = {}
        offset = 0
        for key in order:
            width = int(layout.get(key, 0)) if isinstance(layout, dict) else 0
            if width <= 0:
                continue
            end = min(offset + width, full_dim)
            slices[key] = slice(offset, end)
            offset = end
        if full_dim > offset:
            slices["residual"] = slice(offset, full_dim)
        if not slices:
            return
        self._grad_linear = first_linear
        self._grad_slices = slices
        self._grad_accum = {key: 0.0 for key in slices}

    def _find_first_linear(self, module: nn.Module) -> nn.Module | None:
        """Recursively return the first module with a 2D weight matrix."""
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            if module.weight.ndim == 2:
                return module
        if isinstance(module, nn.Linear):                                   
            return module
        for child in module.children():
            found = self._find_first_linear(child)
            if found is not None:
                return found
        return None

    def configure_optimizers(self):
        """Build optimizer and scheduler configuration from optim block."""
        lr = float(self.optim_cfg.get("lr", self.lr))
        wd = float(self.optim_cfg.get("wd", self.wd))
        betas = tuple(self.optim_cfg.get("betas", (0.9, 0.999)))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=betas)

        kind = str(self.optim_cfg.get("schedule", "cosine")).lower()

        def _max_epochs(default=50):
            return getattr(self.trainer, "max_epochs", default)

        def _total_steps(default=1000):
            ts = getattr(self.trainer, "estimated_stepping_batches", None)
            return int(ts) if ts and ts > 0 else default

        if kind in ("none", "constant"):
            return {"optimizer": opt}

        if kind in ("cosine", "cosine_anneal"):
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_max_epochs())
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        if kind in ("cosine_warmup", "warmup_cosine"):
            w_epochs = int(self.optim_cfg.get("warmup_epochs", 3))
            tmax = max(_max_epochs() - w_epochs, 1)
            warm = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=float(self.optim_cfg.get("warmup_start", 0.1)), total_iters=w_epochs
            )
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[w_epochs])
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        if kind in ("warmcosine", "linear_warmup_cosine"):
            max_epochs = int(self.optim_cfg.get("max_epochs", _max_epochs()))
            if max_epochs <= 0:
                max_epochs = max(_max_epochs(), 1)
            warm_epochs = int(self.optim_cfg.get("warmup_epochs", 10))
            if warm_epochs > max_epochs:
                warm_epochs = max_epochs
            if warm_epochs < 0:
                warm_epochs = 0
            warm_start = float(
                self.optim_cfg.get(
                    "warmup_start_lr",
                    self.optim_cfg.get("warmup_lr", self.optim_cfg.get("warmup_start", 0.0)),
                )
            )
            eta_min = float(self.optim_cfg.get("eta_min", 0.0))
            sched = LinearWarmupCosineAnnealingLR(
                opt,
                warmup_epochs=warm_epochs,
                max_epochs=max_epochs,
                warmup_start_lr=warm_start,
                eta_min=eta_min,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        if kind in ("plateau", "reduce_on_plateau", "reduce"):
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(self.optim_cfg.get("factor", 0.5)),
                patience=int(self.optim_cfg.get("patience", 5)),
                threshold=float(self.optim_cfg.get("threshold", 1e-4)),
                cooldown=int(self.optim_cfg.get("cooldown", 0)),
                min_lr=float(self.optim_cfg.get("min_lr", 1e-6)),
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val_loss"}}

        if kind in ("onecycle", "one_cycle"):
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=lr,
                total_steps=_total_steps(),
                pct_start=float(self.optim_cfg.get("pct_start", 0.1)),
                anneal_strategy="cos",
                div_factor=float(self.optim_cfg.get("div_factor", 25.0)),
                final_div_factor=float(self.optim_cfg.get("final_div_factor", 1e3)),
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if kind in ("step", "multistep"):
            gamma = float(self.optim_cfg.get("gamma", 0.5))
            milestones = self.optim_cfg.get(
                "milestones",
                [int(0.6 * _max_epochs()), int(0.8 * _max_epochs())],
            )
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        raise ValueError(f"Unknown schedule: {kind}")

    def _laplacian_penalty(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Approximate Laplacian smoothness on an irregular batch via kNN graph.

        Encourages local isotropy (reduces along-track striping) while keeping
        mesoscale/submesoscale detail by limiting the neighbourhood and length scale.
        """
        max_pts = int(self.reg_cfg.get("laplace_max_points", 512))
        k = int(self.reg_cfg.get("laplace_k", 8))
        length_km = float(self.reg_cfg.get("laplace_length_km", 50.0))
        eps = 1e-6

                                                                   
        lat = lat_deg.reshape(-1)
        lon = lon_deg.reshape(-1)
        vals = values.reshape(-1)
        n = lat.numel()
        if n == 0:
            return torch.tensor(0.0, device=lat.device, dtype=lat.dtype)
        if n > max_pts:
            idx = torch.randperm(n, device=lat.device)[:max_pts]
            lat, lon, vals = lat[idx], lon[idx], vals[idx]
            n = lat.numel()
        if n <= 1 or k <= 0:
            return torch.tensor(0.0, device=lat.device, dtype=lat.dtype)

                                                                      
        lat_r = torch.deg2rad(lat)
        lon_r = torch.deg2rad(lon)
        cos_lat = torch.cos(lat_r)
        x = cos_lat * torch.cos(lon_r)
        y = cos_lat * torch.sin(lon_r)
        z = torch.sin(lat_r)
        coords = torch.stack((x, y, z), dim=1)
        chord = torch.cdist(coords, coords)          
                                                                      
        arc = 2.0 * 6371.0 * torch.asin(torch.clamp(chord * 0.5, max=1.0))

                                            
        k_eff = min(k + 1, n)
        dist, idx = torch.topk(arc, k=k_eff, largest=False)
        dist = dist[:, 1:]             
        idx = idx[:, 1:]

        neighbor_vals = vals[idx]
        center_vals = vals.unsqueeze(1)
        diff = center_vals - neighbor_vals
                                                                          
        denom = (dist + eps) ** 2
        weights = torch.exp(-(dist / max(length_km, eps)) ** 2)
        penalty = (weights * (diff * diff) / denom).mean()
        return penalty
