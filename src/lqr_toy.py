from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .utils import build_mlp


@dataclass
class LQRConfig:
    a: float
    b: float
    q: float
    r: float
    T: float


def riccati_ode(t: float, P: np.ndarray, a: float, b: float, q: float, r: float) -> np.ndarray:
    dP = -(2.0 * a * P + q - (b ** 2 / r) * P ** 2)
    return dP


def solve_riccati(cfg: LQRConfig, P_T: float) -> Tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        riccati_ode,
        (cfg.T, 0.0),
        np.array([P_T], dtype=np.float64),
        args=(cfg.a, cfg.b, cfg.q, cfg.r),
        dense_output=True,
    )
    t = np.linspace(0.0, cfg.T, 200)
    P = sol.sol(t)[0]
    return t, P


def lqr_control(cfg: LQRConfig, P: np.ndarray, x: np.ndarray) -> np.ndarray:
    return -(cfg.b / cfg.r) * P * x


def hjb_residual(
    V: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: LQRConfig,
) -> torch.Tensor:
    V_t = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    V_x = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    u_star = -(cfg.b / (2.0 * cfg.r)) * V_x
    f = cfg.a * x + cfg.b * u_star
    running_cost = cfg.q * x ** 2 + cfg.r * u_star ** 2
    residual = V_t + running_cost + V_x * f
    return residual


class ValueNet(torch.nn.Module):
    def __init__(self, hidden_layers: int, hidden_units: int, activation: str = "tanh") -> None:
        super().__init__()
        self.net = build_mlp(2, hidden_layers, hidden_units, activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

