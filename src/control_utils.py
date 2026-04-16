from __future__ import annotations

from typing import Tuple

import torch

from .knee_dynamics import ExoParams
from .utils import make_reference_functions


def normalize_state(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_scale: float,
    qd_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep normalization consistent across training/rollout.
    return q / q_scale, qd / qd_scale


def denormalize_state(
    qn: torch.Tensor,
    qdn: torch.Tensor,
    q_scale: float,
    qd_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return qn * q_scale, qdn * qd_scale


def optimal_torque(
    V_omega: torch.Tensor,
    params: ExoParams,
    w_u: float,
    cap_multiplier: bool = False,
    multiplier_cap: float = 1e3,
) -> torch.Tensor:
    # Guard w_u to avoid division blow-ups in control multiplier.
    w_u_safe = max(float(w_u), 1e-12)
    mult = 1.0 / (2.0 * params.I * w_u_safe)
    if cap_multiplier and abs(mult) > multiplier_cap:
        mult = float(multiplier_cap) if mult > 0 else -float(multiplier_cap)
    return -mult * V_omega


def apply_saturation(
    u_raw: torch.Tensor,
    u_max: float | None,
    method: str = "clamp",
    u_scale: float = 1.0,
) -> torch.Tensor:
    if u_max is None:
        return u_raw
    if method == "tanh":
        return u_max * torch.tanh(u_raw / max(u_scale, 1e-6))
    return torch.clamp(u_raw, -u_max, u_max)


def compute_control(
    V_omega: torch.Tensor,
    params: ExoParams,
    w_u: float,
    u_max: float | None = None,
    saturation_method: str = "clamp",
    u_scale: float = 1.0,
    cap_multiplier: bool = False,
    multiplier_cap: float = 1e3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_raw = optimal_torque(
        V_omega,
        params,
        w_u,
        cap_multiplier=cap_multiplier,
        multiplier_cap=multiplier_cap,
    )
    u_bounded = apply_saturation(u_raw, u_max, method=saturation_method, u_scale=u_scale)
    return u_raw, u_bounded


def build_reference_functions(
    cfg,
    t_ref=None,
    theta_ref_arr=None,
    theta_ref_dot_arr=None,
):
    # Single source of truth for reference signals.
    return make_reference_functions(cfg, t_ref, theta_ref_arr, theta_ref_dot_arr)
