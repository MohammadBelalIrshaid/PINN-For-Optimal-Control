import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.knee_dynamics import ExoParams, knee_dynamics_numpy, knee_dynamics_torch


def run_numpy_tests(params: ExoParams) -> None:
    x0 = np.array([0.0, 0.0], dtype=float)
    dx0 = knee_dynamics_numpy(x0, 0.0, params)
    print("NumPy q=0, qdot=0, u=0 ->", dx0)

    x1 = np.array([math.pi / 2.0, 0.0], dtype=float)
    dx1 = knee_dynamics_numpy(x1, 0.0, params)
    print("NumPy q=pi/2, qdot=0, u=0 ->", dx1)


def run_torch_tests(params: ExoParams) -> None:
    x0 = torch.tensor([[0.0, 0.0]])
    u0 = torch.tensor([0.0])
    dx0 = knee_dynamics_torch(x0, u0, params)
    print("Torch q=0, qdot=0, u=0 ->", dx0.squeeze(0).tolist())

    x1 = torch.tensor([[math.pi / 2.0, 0.0]])
    u1 = torch.tensor([[0.0]])
    dx1 = knee_dynamics_torch(x1, u1, params)
    print("Torch q=pi/2, qdot=0, u=0 ->", dx1.squeeze(0).tolist())


if __name__ == "__main__":
    params = ExoParams(I=0.25, B=0.1, m=5.0, g=9.81, l=0.25)
    run_numpy_tests(params)
    run_torch_tests(params)
