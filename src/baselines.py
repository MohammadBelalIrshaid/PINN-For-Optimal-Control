from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.linalg import solve_discrete_are

from .knee_dynamics import ExoParams


def linearize_knee(
    params: Union[ExoParams, dict],
    theta0: float = 0.0,
    omega0: float = 0.0,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    def _get(p: Union[ExoParams, dict], name: str) -> float:
        return getattr(p, name) if hasattr(p, name) else p[name]

    I = _get(params, "I")
    B = _get(params, "B") if hasattr(params, "B") or "B" in params else _get(params, "b")
    m = _get(params, "m")
    g = _get(params, "g")
    l = _get(params, "l") if hasattr(params, "l") or "l" in params else _get(params, "ell")

    a11 = 0.0
    a12 = 1.0
    a21 = -(m * g * l * np.cos(theta0)) / I
    a22 = -(B / I)
    A = np.array([[a11, a12], [a21, a22]], dtype=np.float64)
    B = np.array([[0.0], [1.0 / I]], dtype=np.float64)

    Ad = np.eye(2) + dt * A
    Bd = dt * B
    return Ad, Bd


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K


def pd_controller(theta: float, omega: float, theta_ref: float, omega_ref: float, kp: float, kd: float) -> float:
    return -kp * (theta - theta_ref) - kd * (omega - omega_ref)
