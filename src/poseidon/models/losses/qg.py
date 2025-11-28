from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operators.differential import grad as gradient


class QGRegularization(nn.Module):
    def __init__(
        self, f: float = 1e-4, g: float = 9.81, Lr: float = 1.0, reduction: str = "mean"
    ):
        super().__init__()

        self.f = f
        self.g = g
        self.Lr = Lr
        self.reduction = reduction

    def forward(self, out, x):

        x = x.requires_grad_(True)

                           
        out_jac = gradient(out, x)
        assert out_jac.shape == x.shape

                          
        loss1 = _qg_term1(out_jac, x, self.f, self.g, self.Lr)
                          
        loss2 = _qg_term2(out_jac, self.f, self.g, self.Lr)

        loss = (loss1 + loss2).square()

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


class QGRegularizationFree(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-5,
        reduction: str = "mean",
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True)
        print(self.alpha, self.beta)
        self.eps = eps
        self.reduction = reduction

    def forward(self, out, x, alpha=None, beta=None):

        x = x.requires_grad_(True)

                           
        x_grad = gradient(out, x)
        assert x_grad.shape == x.shape

                          
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        alpha = F.softplus(alpha + 1e-5)
        beta = F.softplus(beta + 1e-5)
        t1, t2, t3 = _qg_loss_free(x_grad, x)

        loss = (t1 - (alpha * t2 - beta * t3)).square()

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


def qg_constants(f, g, L_r):
    c_1 = g / f
    c_2 = 1 / L_r**2
    c_3 = c_1 * c_2
    return c_1, c_2, c_3


def qg_loss(
    ssh, x, f: float = 1e-4, g: float = 9.81, Lr: float = 1.0, reduction: str = "mean"
):
    ssh_jac = gradient(ssh, x)
    assert ssh_jac.shape == x.shape

                      
    loss1 = _qg_term1(ssh_jac, x, f, g, Lr)
                      
    loss2 = _qg_term2(ssh_jac, f, g, Lr)

    loss = (loss1 + loss2).square()

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


def _qg_term1(u_grad, x_var, f: float = 1.0, g: float = 1.0, L_r: float = 1.0):
    x_var = x_var.requires_grad_(True)
    c_1, c_2, c_3 = qg_constants(f, g, L_r)

                                               
    u_x, u_y, u_t = torch.split(u_grad, [1, 1, 1], dim=1)

                       
    u_grad2 = gradient(u_grad, x_var)
    assert u_grad2.shape == x_var.shape

                                                       
    u_xx, u_yy, u_tt = torch.split(u_grad2, [1, 1, 1], dim=1)
    assert u_xx.shape == u_yy.shape == u_tt.shape

                                  
    u_lap = u_xx + u_yy
    assert u_lap.shape == u_xx.shape == u_yy.shape

                                 
    u_lap_grad = gradient(u_lap, x_var)
    assert u_lap_grad.shape == x_var.shape

                                   
    u_lap_grad_x, u_lap_grad_y, u_lap_grad_t = torch.split(u_lap_grad, [1, 1, 1], dim=1)
    assert u_lap_grad_x.shape == u_lap_grad_y.shape == u_lap_grad_t.shape

            
    loss = u_lap_grad_t + c_1 * u_x * u_lap_grad_y - c_1 * u_y * u_lap_grad_x
    assert loss.shape == u_lap_grad_t.shape == u_lap_grad_y.shape == u_lap_grad_x.shape

    return loss


def _qg_loss_free(u_grad, x_var):
                                               
    u_x, u_y, u_t = torch.split(u_grad, [1, 1, 1], dim=1)

                       
    u_grad2 = gradient(u_grad, x_var)
    assert u_grad2.shape == x_var.shape

                                                       
    u_xx, u_yy, u_tt = torch.split(u_grad2, [1, 1, 1], dim=1)
    assert u_xx.shape == u_yy.shape == u_tt.shape

                                  
    u_lap = u_xx + u_yy
    assert u_lap.shape == u_xx.shape == u_yy.shape

                                 
    u_lap_grad = gradient(u_lap, x_var)
    assert u_lap_grad.shape == x_var.shape

                                   
    u_lap_grad_x, u_lap_grad_y, u_lap_grad_t = torch.split(u_lap_grad, [1, 1, 1], dim=1)
    assert u_lap_grad_x.shape == u_lap_grad_y.shape == u_lap_grad_t.shape

            
    t1 = u_t

            
    t2 = u_lap_grad_t
    t3 = u_x * u_lap_grad_y - u_y * u_lap_grad_x
    assert (
        t1.shape
        == t2.shape
        == t3.shape
        == u_lap_grad_t.shape
        == u_lap_grad_y.shape
        == u_lap_grad_x.shape
    )

    return t1, t2, t3


def _qg_term2(u_grad, f: float = 1.0, g: float = 1.0, Lr: float = 1.0):
    _, c_2, c_3 = qg_constants(f, g, L_r)

                                               
    *_, u_t = torch.split(u_grad, [1, 1, 1], dim=1)

                      
    loss = -c_2 * u_t

    return loss
