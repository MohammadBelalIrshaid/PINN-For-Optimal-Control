from __future__ import annotations

import math
import random
from typing import Iterable, Tuple

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mlp(
    input_dim: int,
    hidden_layers: int,
    hidden_units: int,
    activation: str = "tanh",
) -> torch.nn.Module:
    if activation == "tanh":
        act = torch.nn.Tanh
    elif activation == "relu":
        act = torch.nn.ReLU
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = [torch.nn.Linear(input_dim, hidden_units), act()]
    for _ in range(hidden_layers - 1):
        layers.append(torch.nn.Linear(hidden_units, hidden_units))
        layers.append(act())
    layers.append(torch.nn.Linear(hidden_units, 1))
    return torch.nn.Sequential(*layers)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def sample_uniform(
    low: Iterable[float],
    high: Iterable[float],
    n: int,
) -> np.ndarray:
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return np.random.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.gradient(x, dt, axis=0)
    return dx


def synthetic_reference(t: torch.Tensor, A: float = 0.5, period: float = 1.0) -> torch.Tensor:
    return A * torch.sin(2.0 * math.pi * t / period)


def synthetic_reference_dot(t: torch.Tensor, A: float = 0.5, period: float = 1.0) -> torch.Tensor:
    return A * (2.0 * math.pi / period) * torch.cos(2.0 * math.pi * t / period)


def _get_nested(cfg, *keys, default=None):
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


def sanitize_exo_params(params, cfg, warn_fn=print):
    # Safety: negative damping is unstable; optionally enforce positive B.
    enforce = bool(_get_nested(cfg, "params", "enforce_positive_damping", default=True))
    if enforce and hasattr(params, "B"):
        if params.B < 0:
            warn_fn(
                "[WARN] Damping B < 0; using abs(B) for stability. "
                "Set params.enforce_positive_damping=False to disable."
            )
            params.B = abs(params.B)
    return params


def make_reference_functions(
    cfg,
    t_ref: np.ndarray | None = None,
    theta_ref_arr: np.ndarray | None = None,
    theta_ref_dot_arr: np.ndarray | None = None,
) -> tuple:
    ref_block = _get_nested(cfg, "reference", default=None)
    if ref_block is None:
        use_synth = bool(_get_nested(cfg, "training", "use_synthetic_ref", default=True))
        ref_type = "synthetic" if use_synth else "epic_mean"
    else:
        ref_type = _get_nested(cfg, "reference", "type", default="synthetic")
    ref_type = str(ref_type).lower()

    if ref_type == "synthetic":
        kind = _get_nested(cfg, "reference", "synthetic", "kind", default="sine")
        amplitude = float(_get_nested(cfg, "reference", "synthetic", "amplitude", default=0.5))
        offset = float(_get_nested(cfg, "reference", "synthetic", "offset", default=0.0))
        period = float(_get_nested(cfg, "reference", "synthetic", "period", default=1.0))

        if kind == "half_sine":
            def theta_ref(t: torch.Tensor) -> torch.Tensor:
                return offset + amplitude * torch.sin(math.pi * t / period)

            def theta_ref_dot(t: torch.Tensor) -> torch.Tensor:
                return amplitude * (math.pi / period) * torch.cos(math.pi * t / period)
        else:
            def theta_ref(t: torch.Tensor) -> torch.Tensor:
                return offset + amplitude * torch.sin(2.0 * math.pi * t / period)

            def theta_ref_dot(t: torch.Tensor) -> torch.Tensor:
                return amplitude * (2.0 * math.pi / period) * torch.cos(2.0 * math.pi * t / period)

        return theta_ref, theta_ref_dot

    if ref_type in {"epic_mean", "siat_cycle"}:
        if t_ref is None or theta_ref_arr is None:
            raise ValueError(f"reference.type={ref_type} requires t_ref and theta_ref_arr.")
        if theta_ref_dot_arr is None:
            dt_grid = float(t_ref[1] - t_ref[0])
            theta_ref_dot_arr = np.gradient(theta_ref_arr, dt_grid).astype(np.float32)

        def theta_ref(t_tensor: torch.Tensor) -> torch.Tensor:
            t_np = t_tensor.detach().cpu().numpy().reshape(-1)
            ref = np.interp(t_np, t_ref, theta_ref_arr).astype(np.float32)
            return torch.tensor(ref, device=t_tensor.device).reshape(-1, 1)

        def theta_ref_dot(t_tensor: torch.Tensor) -> torch.Tensor:
            t_np = t_tensor.detach().cpu().numpy().reshape(-1)
            ref = np.interp(t_np, t_ref, theta_ref_dot_arr).astype(np.float32)
            return torch.tensor(ref, device=t_tensor.device).reshape(-1, 1)

        return theta_ref, theta_ref_dot

    raise ValueError(f"Unsupported reference.type: {ref_type}")


def get_reference(
    t: torch.Tensor,
    cfg,
    t_ref: np.ndarray | None = None,
    theta_ref_arr: np.ndarray | None = None,
    theta_ref_dot_arr: np.ndarray | None = None,
) -> torch.Tensor:
    theta_ref, _ = make_reference_functions(cfg, t_ref, theta_ref_arr, theta_ref_dot_arr)
    return theta_ref(t)


def get_reference_dot(
    t: torch.Tensor,
    cfg,
    t_ref: np.ndarray | None = None,
    theta_ref_arr: np.ndarray | None = None,
    theta_ref_dot_arr: np.ndarray | None = None,
) -> torch.Tensor:
    _, theta_ref_dot = make_reference_functions(cfg, t_ref, theta_ref_arr, theta_ref_dot_arr)
    return theta_ref_dot(t)
