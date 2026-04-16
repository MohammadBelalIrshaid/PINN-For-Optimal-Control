from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .control_utils import apply_saturation, compute_control, optimal_torque
from .knee_dynamics import ExoParams, knee_dynamics_torch
from .utils import build_mlp, sample_uniform


@dataclass
class HJBConfig:
    w_track: float
    w_omega: float
    w_u: float


class ValueNet(torch.nn.Module):
    def __init__(self, hidden_layers: int, hidden_units: int, activation: str = "tanh") -> None:
        super().__init__()
        self.net = build_mlp(3, hidden_layers, hidden_units, activation)

    def forward(self, theta: torch.Tensor, omega: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([theta, omega, t], dim=1)
        return self.net(x)


def compute_grads(
    V: torch.Tensor,
    theta: torch.Tensor,
    omega: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    V_t = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    V_theta = torch.autograd.grad(V.sum(), theta, create_graph=True)[0]
    V_omega = torch.autograd.grad(V.sum(), omega, create_graph=True)[0]
    return V_t, V_theta, V_omega


 


def _get_nested(cfg: Any, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def compute_total_loss(losses: Dict[str, torch.Tensor], cfg: Any) -> torch.Tensor:
    mode = _get_nested(cfg, "training", "mode", default="pure_hjb")

    L_hjb = losses["hjb"]
    L_term = losses["term"]
    L_data = losses.get("data", torch.zeros_like(L_hjb))
    L_traj = losses.get("traj", torch.zeros_like(L_hjb))
    L_sat = losses.get("sat", torch.zeros_like(L_hjb))
    L_res_reg = losses.get("res_reg", torch.zeros_like(L_hjb))

    lambda_hjb = float(_get_nested(cfg, "training", "lambda_hjb", default=1.0))
    lambda_term = float(_get_nested(cfg, "training", "lambda_term", default=1.0))
    lambda_data = float(_get_nested(cfg, "training", "lambda_data", default=0.0))
    lambda_traj = float(_get_nested(cfg, "training", "lambda_traj", default=0.0))
    lambda_sat = float(_get_nested(cfg, "training", "lambda_sat", default=0.0))
    lambda_res_reg = float(_get_nested(cfg, "training", "lambda_res_reg", default=0.0))

    if mode == "pure_hjb":
        total = lambda_hjb * L_hjb + lambda_term * L_term
    elif mode == "hjb_data":
        total = (
            lambda_hjb * L_hjb
            + lambda_term * L_term
            + lambda_data * L_data
            + lambda_sat * L_sat
        )
    elif mode == "hjb_data_traj":
        total = (
            lambda_hjb * L_hjb
            + lambda_term * L_term
            + lambda_data * L_data
            + lambda_traj * L_traj
            + lambda_sat * L_sat
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if lambda_res_reg > 0.0:
        total = total + lambda_res_reg * L_res_reg

    return total


def _enforce_bounds(
    samples: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    enforce_bounds: bool,
    reject_out_of_bounds: bool,
    max_resample_tries: int,
) -> tuple[np.ndarray, dict]:
    info = {
        "oob_ratio": 0.0,
        "clipped_ratio": 0.0,
        "oob_by_dim": np.zeros(samples.shape[1], dtype=np.float32),
    }
    if not enforce_bounds:
        return samples, info

    oob_mask = (samples < low) | (samples > high)
    oob_rows = oob_mask.any(axis=1)
    info["oob_ratio"] = float(oob_rows.mean()) if samples.size else 0.0
    info["oob_by_dim"] = oob_mask.mean(axis=0) if samples.size else info["oob_by_dim"]

    if reject_out_of_bounds and oob_rows.any():
        tries = 0
        while tries < max_resample_tries:
            tries += 1
            idx = np.where(oob_rows)[0]
            if idx.size == 0:
                break
            resampled = np.random.uniform(low, high, size=(idx.size, low.shape[0])).astype(np.float32)
            samples[idx] = resampled
            oob_mask = (samples < low) | (samples > high)
            oob_rows = oob_mask.any(axis=1)

    if oob_rows.any():
        samples = np.clip(samples, low, high)
        info["clipped_ratio"] = float(oob_rows.mean())
        info["oob_by_dim"] = oob_mask.mean(axis=0)

    return samples, info


def sample_collocation(cfg: Any, n: int, return_info: bool = False):
    theta_min = float(_get_nested(cfg, "collocation", "theta_min", default=-1.0))
    theta_max = float(_get_nested(cfg, "collocation", "theta_max", default=1.0))
    omega_min = float(_get_nested(cfg, "collocation", "omega_min", default=-3.0))
    omega_max = float(_get_nested(cfg, "collocation", "omega_max", default=3.0))
    t_min = float(_get_nested(cfg, "collocation", "t_min", default=0.0))
    t_max = float(_get_nested(cfg, "collocation", "t_max", default=1.0))
    samples = sample_uniform([theta_min, omega_min, t_min], [theta_max, omega_max, t_max], n)

    enforce_bounds = bool(_get_nested(cfg, "sample", "enforce_bounds", default=True))
    reject_out_of_bounds = bool(_get_nested(cfg, "sample", "reject_out_of_bounds", default=True))
    max_resample_tries = int(_get_nested(cfg, "sample", "max_resample_tries", default=50))
    low = np.array([theta_min, omega_min, t_min], dtype=np.float32)
    high = np.array([theta_max, omega_max, t_max], dtype=np.float32)
    samples, info = _enforce_bounds(
        samples,
        low,
        high,
        enforce_bounds=enforce_bounds,
        reject_out_of_bounds=reject_out_of_bounds,
        max_resample_tries=max_resample_tries,
    )
    if return_info:
        return samples, info
    return samples


def hjb_residual(
    theta: torch.Tensor,
    omega: torch.Tensor,
    t: torch.Tensor,
    V: torch.Tensor,
    V_t: torch.Tensor,
    V_theta: torch.Tensor,
    V_omega: torch.Tensor,
    params: ExoParams,
    cfg: HJBConfig,
    theta_ref: torch.Tensor,
    omega_ref: torch.Tensor | None = None,
    u_max: float | None = None,
    use_saturated_hjb: bool = True,
    saturation_method: str = "clamp",
    u_scale: float = 1.0,
    cap_multiplier: bool = False,
    multiplier_cap: float = 1e3,
    residual_net: torch.nn.Module | None = None,
    residual_guard: bool = False,
    guard_theta_min: float | None = None,
    guard_theta_max: float | None = None,
    guard_omega_min: float | None = None,
    guard_omega_max: float | None = None,
) -> torch.Tensor:
    u_raw, u_bounded = compute_control(
        V_omega,
        params,
        cfg.w_u,
        u_max=u_max,
        saturation_method=saturation_method,
        u_scale=u_scale,
        cap_multiplier=cap_multiplier,
        multiplier_cap=multiplier_cap,
    )
    u_used = u_bounded if use_saturated_hjb else u_raw
    x = torch.cat([theta, omega], dim=1)
    dx = knee_dynamics_torch(
        x,
        u_used,
        params,
        residual_net=residual_net,
        residual_guard=residual_guard,
        guard_theta_min=guard_theta_min,
        guard_theta_max=guard_theta_max,
        guard_omega_min=guard_omega_min,
        guard_omega_max=guard_omega_max,
    )
    omega_err = omega if omega_ref is None else (omega - omega_ref)
    running_cost = (
        cfg.w_track * (theta - theta_ref) ** 2
        + cfg.w_omega * omega_err ** 2
        + cfg.w_u * u_used ** 2
    )
    residual = V_t + running_cost + V_theta * dx[:, 0:1] + V_omega * dx[:, 1:2]
    return residual
