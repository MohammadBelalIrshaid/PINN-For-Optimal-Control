from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExoParams:
    I: float
    B: float
    m: float
    g: float
    l: float


def get_default_params() -> ExoParams:
    return ExoParams(I=0.25, B=0.1, m=5.0, g=9.81, l=0.25)


def knee_dynamics_numpy(x: np.ndarray, u: float, params: ExoParams) -> np.ndarray:
    q, qdot = float(x[0]), float(x[1])
    G = params.m * params.g * params.l * np.sin(q)
    dq = qdot
    dqdot = (u - params.B * qdot - G) / params.I
    return np.array([dq, dqdot], dtype=float)


class ResidualNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_u: torch.Tensor) -> torch.Tensor:
        return self.net(x_u)


def _knee_dynamics_torch_impl(
    x: torch.Tensor,
    u: torch.Tensor,
    params: ExoParams,
    residual_net: nn.Module | None = None,
    residual_guard: bool = False,
    guard_theta_min: float | None = None,
    guard_theta_max: float | None = None,
    guard_omega_min: float | None = None,
    guard_omega_max: float | None = None,
) -> torch.Tensor:
    q = x[..., 0]
    qdot = x[..., 1]

    if u.dim() == x.dim() and u.shape[-1] == 1:
        u = u.squeeze(-1)

    G = params.m * params.g * params.l * torch.sin(q)
    dq = qdot
    dqdot_nom = (u - params.B * qdot - G) / params.I
    dx_nom = torch.stack([dq, dqdot_nom], dim=-1)

    if residual_net is None:
        return dx_nom

    x_u = torch.stack([q, qdot, u], dim=-1)
    dx_res = residual_net(x_u)

    if residual_guard and None not in (guard_theta_min, guard_theta_max, guard_omega_min, guard_omega_max):
        in_box = (
            (q >= guard_theta_min)
            & (q <= guard_theta_max)
            & (qdot >= guard_omega_min)
            & (qdot <= guard_omega_max)
        )
        dx_res = dx_res * in_box.unsqueeze(-1).to(dx_res.dtype)

    return dx_nom + dx_res


def knee_dynamics_torch(
    x: torch.Tensor,
    u: torch.Tensor,
    params: ExoParams,
    residual_net: nn.Module | None = None,
    residual_guard: bool = False,
    guard_theta_min: float | None = None,
    guard_theta_max: float | None = None,
    guard_omega_min: float | None = None,
    guard_omega_max: float | None = None,
) -> torch.Tensor:
    return _knee_dynamics_torch_impl(
        x,
        u,
        params,
        residual_net=residual_net,
        residual_guard=residual_guard,
        guard_theta_min=guard_theta_min,
        guard_theta_max=guard_theta_max,
        guard_omega_min=guard_omega_min,
        guard_omega_max=guard_omega_max,
    )


def knee_dynamics_uncertain_torch(
    x: torch.Tensor,
    u: torch.Tensor,
    params: ExoParams,
    psi_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    q = x[..., 0]
    qdot = x[..., 1]

    if u.dim() == x.dim() and u.shape[-1] == 1:
        u = u.squeeze(-1)

    G = params.m * params.g * params.l * torch.sin(q)
    dq = qdot
    dqdot_nom = (u - params.B * qdot - G) / params.I

    if psi_fn is not None:
        dqdot = dqdot_nom + psi_fn(q, qdot)
    else:
        dqdot = dqdot_nom

    return torch.stack([dq, dqdot], dim=-1)
