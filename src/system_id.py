from __future__ import annotations

import numpy as np


def estimate_pendulum_params(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    u: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimates I, B, and mgl in:
        I*qddot + B*qdot + mgl*sin(q) = u
    using linear least squares.
    """
    q = q.reshape(-1)
    qdot = qdot.reshape(-1)
    qddot = qddot.reshape(-1)
    u = u.reshape(-1)

    A = np.column_stack([qddot, qdot, np.sin(q)])
    theta, *_ = np.linalg.lstsq(A, u, rcond=None)
    I_hat, B_hat, mgl_hat = theta
    return float(I_hat), float(B_hat), float(mgl_hat)
