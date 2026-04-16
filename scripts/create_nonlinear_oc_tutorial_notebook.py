from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "07_nonlinear_optimal_control_pinn_tutorial.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md("""# 07 - Nonlinear Optimal Control with a Trajectory PINN

This tutorial solves a nonlinear optimal control problem for a torque-actuated pendulum using three viewpoints:

1. a trajectory PINN that learns state and control as continuous-time functions,
2. a coarse dynamic-programming / HJB-style benchmark on a discretized state-control grid,
3. a Pontryagin's Maximum Principle (PMP) solution via forward-backward sweep.

The notebook is written as a teaching example for an optimal control course. The emphasis is on the mathematical structure of the problem, how the losses are built, and what each method is really solving.
"""))

    cells.append(md("""## Roadmap

- Section A: state the nonlinear optimal control problem clearly.
- Section B: write the continuous-time mathematics.
- Section C: convert the problem into a trajectory PINN optimization.
- Section D: implement the PINN in PyTorch.
- Section E: inspect the learned state, control, losses, and objective value.
- Section F: build an approximate dynamic-programming benchmark on a coarse grid.
- Section G: build a PMP benchmark using forward-backward sweep.
- Section H: compare the three methods and discuss when each is natural.
"""))

    cells.append(code("""# Imports and reproducible setup
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

ROOT = Path(r\"e:/Optimal_Control/PINN/hjb_pinn_exoskeleton\")
OUTPUT_DIR = ROOT / \"results\" / \"figures\" / \"07_nonlinear_optimal_control_pinn_tutorial\"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(\"cpu\")
plt.rcParams.update({\"figure.figsize\": (8, 4.8), \"axes.grid\": True, \"font.size\": 11})

print(\"Device:\", DEVICE)
print(\"Figures will be saved to:\", OUTPUT_DIR)
"""))

    cells.append(md(r"""## A. Problem Statement

We study a damped, torque-actuated pendulum with state

\[
x(t)=\begin{bmatrix}
\theta(t) \\
\omega(t)
\end{bmatrix},
\]

where $\theta$ is the angle, $\omega$ is the angular velocity, and $u$ is the control torque.

The nonlinear dynamics are

\[
\dot{\theta} = \omega,
\qquad
\dot{\omega} = -\sin(\theta) - c\,\omega + u.
\]

The system is nonlinear because of the sine term. We stabilize toward $\theta_{ref}=0$, $\omega_{ref}=0$ using the nonlinear running cost

\[
L(x,u) = q_\theta(\theta-\theta_{ref})^2 + q_\omega \omega^2 + r u^2 + \alpha (\theta-\theta_{ref})^4.
\]

The quartic term makes the cost nonlinear as well. The terminal cost is

\[
\Phi(x(T)) = q_T(\theta(T)-\theta_{ref})^2 + q_{\omega T}\omega(T)^2.
\]

This is therefore a nonlinear optimal control problem because the dynamics contain $\sin(\theta)$ and the objective contains $(\theta-\theta_{ref})^4$.
"""))

    cells.append(md(r"""## B. Mathematical Formulation

We solve

\[
\min_{u(\cdot)} J(u)
\]

subject to

\[
\dot{x}(t)=f(x(t),u(t)), \qquad x(0)=x_0,
\]

with

\[
x(t)=\begin{bmatrix}\theta(t) \\ \omega(t)\end{bmatrix},
\qquad
f(x,u)=\begin{bmatrix}
\omega \\
-\sin(\theta)-c\omega + u
\end{bmatrix}.
\]

The cost is

\[
J(u)=\int_0^T L(x(t),u(t))\,dt + \Phi(x(T)).
\]

We use the nontrivial initial condition $\theta(0)=1.0$ rad and $\omega(0)=0$.
"""))

    cells.append(code("""# Problem parameters and common helpers
@dataclass
class PendulumOCConfig:
    c: float = 0.1
    q_theta: float = 1.0
    q_omega: float = 0.1
    r: float = 0.01
    alpha: float = 0.5
    q_T: float = 5.0
    q_omega_T: float = 1.0
    T: float = 3.0
    theta_ref: float = 0.0
    omega_ref: float = 0.0
    theta0: float = 1.0
    omega0: float = 0.0
    u_max: float = 2.0
    pinn_epochs: int = 2500
    pinn_lr: float = 2e-3
    pinn_hidden: int = 64
    pinn_layers: int = 3
    num_time_points: int = 201
    lambda_dyn: float = 50.0
    lambda_ic: float = 50.0
    lambda_cost: float = 1.0
    dp_n_theta: int = 61
    dp_n_omega: int = 61
    dp_n_time: int = 61
    pmp_iterations: int = 120
    pmp_relaxation: float = 0.25
    pmp_tol: float = 1e-4

cfg = PendulumOCConfig()


def make_time_grid(T: float, num_points: int, device: torch.device = DEVICE) -> torch.Tensor:
    t = torch.linspace(0.0, T, num_points, device=device).reshape(-1, 1)
    t.requires_grad_(True)
    return t


def pendulum_rhs_np(state: np.ndarray, u: float, cfg: PendulumOCConfig) -> np.ndarray:
    theta, omega = state
    return np.array([omega, -math.sin(theta) - cfg.c * omega + u], dtype=np.float64)


def pendulum_rhs_torch(theta: torch.Tensor, omega: torch.Tensor, u: torch.Tensor, cfg: PendulumOCConfig):
    dtheta = omega
    domega = -torch.sin(theta) - cfg.c * omega + u
    return dtheta, domega


def running_cost_np(theta, omega, u, cfg: PendulumOCConfig):
    err = theta - cfg.theta_ref
    return cfg.q_theta * err**2 + cfg.q_omega * omega**2 + cfg.r * u**2 + cfg.alpha * err**4


def terminal_cost_np(theta, omega, cfg: PendulumOCConfig):
    err = theta - cfg.theta_ref
    return cfg.q_T * err**2 + cfg.q_omega_T * omega**2


def running_cost_torch(theta, omega, u, cfg: PendulumOCConfig):
    err = theta - cfg.theta_ref
    return cfg.q_theta * err.pow(2) + cfg.q_omega * omega.pow(2) + cfg.r * u.pow(2) + cfg.alpha * err.pow(4)


def terminal_cost_torch(theta, omega, cfg: PendulumOCConfig):
    err = theta - cfg.theta_ref
    return cfg.q_T * err.pow(2) + cfg.q_omega_T * omega.pow(2)


def trapz_uniform_np(y: np.ndarray, dt: float) -> float:
    if y.size == 1:
        return float(y[0])
    return float(dt * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))


def trapz_uniform_torch(y: torch.Tensor, dt: float) -> torch.Tensor:
    if y.numel() == 1:
        return y.reshape(-1)[0]
    return dt * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def rk4_step(state: np.ndarray, u: float, dt: float, cfg: PendulumOCConfig) -> np.ndarray:
    k1 = pendulum_rhs_np(state, u, cfg)
    k2 = pendulum_rhs_np(state + 0.5 * dt * k1, u, cfg)
    k3 = pendulum_rhs_np(state + 0.5 * dt * k2, u, cfg)
    k4 = pendulum_rhs_np(state + dt * k3, u, cfg)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


print(cfg)
"""))
    cells.append(md(r"""## C. PINN Formulation

For this tutorial we use a trajectory PINN rather than a value-function PINN.

Instead of learning a value function $V(x,t)$, we learn three continuous-time functions:

\[
\theta_\phi(t), \qquad \omega_\phi(t), \qquad u_\psi(t).
\]

A single neural network can output all three.

The residual losses are

\[
\mathcal{L}_{dyn} = \left\|\frac{d\theta_\phi}{dt} - \omega_\phi\right\|_2^2 + \left\|\frac{d\omega_\phi}{dt} - \big(-\sin(\theta_\phi)-c\omega_\phi + u_\psi\big)\right\|_2^2,
\]

\[
\mathcal{L}_{ic} = |\theta_\phi(0)-\theta_0|^2 + |\omega_\phi(0)-\omega_0|^2,
\]

and the cost term is a quadrature approximation of

\[
\mathcal{L}_{cost} \approx \int_0^T L(x_\phi(t),u_\psi(t))\,dt + \Phi(x_\phi(T)).
\]

The total loss is

\[
\mathcal{L} = \lambda_{dyn}\mathcal{L}_{dyn} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{cost}\mathcal{L}_{cost}.
\]

This is useful pedagogically because it separates feasibility, boundary conditions, and optimality. For numerical stability we bound the control by $u(t)=u_{max}\tanh(\hat u(t))$.
"""))

    cells.append(code("""# PINN model and training utilities
class TrajectoryPINN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor):
        y = self.net(t)
        theta = y[:, 0:1]
        omega = y[:, 1:2]
        u_raw = y[:, 2:3]
        return theta, omega, u_raw


def build_network(cfg: PendulumOCConfig) -> nn.Module:
    return TrajectoryPINN(hidden_dim=cfg.pinn_hidden, num_hidden_layers=cfg.pinn_layers).to(DEVICE)


def predict_trajectory(net: nn.Module, t: torch.Tensor, cfg: PendulumOCConfig):
    theta, omega, u_raw = net(t / cfg.T)
    u = cfg.u_max * torch.tanh(u_raw)
    return theta, omega, u, u_raw


def compute_dynamics_residual(net: nn.Module, t: torch.Tensor, cfg: PendulumOCConfig):
    theta, omega, u, u_raw = predict_trajectory(net, t, cfg)
    theta_t = torch.autograd.grad(theta.sum(), t, create_graph=True)[0]
    omega_t = torch.autograd.grad(omega.sum(), t, create_graph=True)[0]
    dtheta_model, domega_model = pendulum_rhs_torch(theta, omega, u, cfg)
    res_theta = theta_t - dtheta_model
    res_omega = omega_t - domega_model
    return res_theta, res_omega, theta, omega, u, u_raw


def compute_cost(theta: torch.Tensor, omega: torch.Tensor, u: torch.Tensor, cfg: PendulumOCConfig):
    dt = cfg.T / (theta.shape[0] - 1)
    running = running_cost_torch(theta, omega, u, cfg)
    integral = trapz_uniform_torch(running.reshape(-1), dt)
    terminal = terminal_cost_torch(theta[-1], omega[-1], cfg)
    return integral + terminal.squeeze(), running.reshape(-1), terminal.squeeze()


def train_pinn(cfg: PendulumOCConfig):
    t = make_time_grid(cfg.T, cfg.num_time_points)
    net = build_network(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.pinn_lr)
    history = {key: [] for key in [\"epoch\", \"total\", \"dyn\", \"ic\", \"cost\", \"terminal\"]}

    for epoch in tqdm(range(1, cfg.pinn_epochs + 1), desc=\"Training trajectory PINN\"):
        optimizer.zero_grad()
        res_theta, res_omega, theta, omega, u, u_raw = compute_dynamics_residual(net, t, cfg)
        loss_dyn = res_theta.pow(2).mean() + res_omega.pow(2).mean()
        loss_ic = (theta[0] - cfg.theta0).pow(2) + (omega[0] - cfg.omega0).pow(2)
        loss_cost, running_cost, terminal = compute_cost(theta, omega, u, cfg)
        loss = cfg.lambda_dyn * loss_dyn + cfg.lambda_ic * loss_ic + cfg.lambda_cost * loss_cost
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            history[\"epoch\"].append(epoch)
            history[\"total\"].append(float(loss.detach().cpu()))
            history[\"dyn\"].append(float(loss_dyn.detach().cpu()))
            history[\"ic\"].append(float(loss_ic.detach().cpu()))
            history[\"cost\"].append(float(loss_cost.detach().cpu()))
            history[\"terminal\"].append(float(terminal.detach().cpu()))

    with torch.no_grad():
        theta, omega, u, u_raw = predict_trajectory(net, t, cfg)
        objective, running_cost, terminal = compute_cost(theta, omega, u, cfg)

    return {
        \"t\": t.detach().cpu().numpy().reshape(-1),
        \"theta\": theta.detach().cpu().numpy().reshape(-1),
        \"omega\": omega.detach().cpu().numpy().reshape(-1),
        \"u\": u.detach().cpu().numpy().reshape(-1),
        \"u_raw\": u_raw.detach().cpu().numpy().reshape(-1),
        \"running_cost\": running_cost.detach().cpu().numpy().reshape(-1),
        \"objective\": float(objective.detach().cpu()),
        \"terminal_cost\": float(terminal.detach().cpu()),
        \"history\": history,
        \"net\": net,
    }
"""))

    cells.append(md(r"""## D. PINN Implementation and Training

The implementation is deliberately explicit: `make_time_grid` builds the collocation grid, `compute_dynamics_residual` uses autograd to differentiate with respect to time, and `compute_cost` approximates the objective by trapezoidal quadrature. The network therefore learns an open-loop trajectory-control pair that is both feasible and low cost.
"""))

    cells.append(code("""# Train the trajectory PINN
pinn_result = train_pinn(cfg)

print(f\"PINN objective: {pinn_result['objective']:.4f}\")
print(f\"PINN terminal state: theta(T)={pinn_result['theta'][-1]:.4f}, omega(T)={pinn_result['omega'][-1]:.4f}\")
print(f\"PINN max |u|: {np.max(np.abs(pinn_result['u'])):.4f}\")
"""))

    cells.append(code("""# Plotting helpers for standalone publication-quality figures
def save_current_figure(filename: str):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches=\"tight\")
    plt.show()
    plt.close()
    return path


def plot_line(x, y, title, xlabel, ylabel, filename, label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend()
    return save_current_figure(filename)


def plot_multiple_lines(lines, title, xlabel, ylabel, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x, y, label in lines:
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return save_current_figure(filename)


def plot_phase(theta, omega, title, filename, label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theta, omega, label=label)
    ax.set_title(title)
    ax.set_xlabel(r\"$\\theta$ (rad)\")
    ax.set_ylabel(r\"$\\omega$ (rad/s)\")
    if label is not None:
        ax.legend()
    return save_current_figure(filename)


def plot_heatmap(x_grid, y_grid, values, title, xlabel, ylabel, filename, cbar_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(x_grid, y_grid, values.T, shading=\"auto\")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    return save_current_figure(filename)
"""))

    cells.append(md(r"""## E. Results from the PINN

We now inspect the learned state trajectory, control trajectory, phase portrait, running cost, and training curves. These plots answer two questions: did the network satisfy the physics, and did it produce a useful low-cost control strategy?
"""))

    cells.append(code("""# PINN result plots
plot_line(pinn_result['t'], pinn_result['theta'], 'PINN State Trajectory: Theta vs Time', 'time (s)', r'$\\theta(t)$ (rad)', 'pinn_theta_vs_time.png')
plot_line(pinn_result['t'], pinn_result['omega'], 'PINN State Trajectory: Omega vs Time', 'time (s)', r'$\\omega(t)$ (rad/s)', 'pinn_omega_vs_time.png')
plot_line(pinn_result['t'], pinn_result['u'], 'PINN Control Trajectory', 'time (s)', 'u(t) (Nm-equivalent)', 'pinn_control_vs_time.png')
plot_phase(pinn_result['theta'], pinn_result['omega'], 'PINN Phase Portrait', 'pinn_phase_portrait.png')
plot_line(pinn_result['history']['epoch'], pinn_result['history']['total'], 'PINN Training Loss: Total', 'epoch', 'loss', 'pinn_loss_total.png')
plot_multiple_lines([
    (pinn_result['history']['epoch'], pinn_result['history']['dyn'], 'dynamics'),
    (pinn_result['history']['epoch'], pinn_result['history']['ic'], 'initial condition'),
    (pinn_result['history']['epoch'], pinn_result['history']['cost'], 'cost objective'),
], 'PINN Training Loss Components', 'epoch', 'loss component', 'pinn_loss_components.png')
plot_line(pinn_result['t'], pinn_result['running_cost'], 'PINN Running Cost Over Time', 'time (s)', 'running cost', 'pinn_running_cost.png')
"""))
    cells.append(md(r"""## F. Approximate Dynamic Programming / HJB Benchmark

Because the pendulum has a 2D state, we can build a coarse finite-horizon dynamic-programming benchmark on a bounded state grid. This is not a full high-resolution HJB PDE solver. It is a practical tutorial benchmark based on a state grid over $(\theta,\omega)$, a small discrete control set, and backward dynamic programming in time.

At each time step we compute

\[
V_k(x) = \min_{u \in \mathcal{U}} \left[ \Delta t\,L(x,u) + V_{k+1}(x^+) \right],
\]

where $x^+$ is the next state after one short time step. This gives an approximate value function and an approximate feedback policy.
"""))

    cells.append(code("""# Approximate dynamic programming / HJB-style solver
def solve_dp_benchmark(cfg: PendulumOCConfig):
    theta_grid = np.linspace(-np.pi, np.pi, cfg.dp_n_theta)
    omega_grid = np.linspace(-3.0, 3.0, cfg.dp_n_omega)
    controls = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    t_grid = np.linspace(0.0, cfg.T, cfg.dp_n_time)
    dt = t_grid[1] - t_grid[0]

    TH, OM = np.meshgrid(theta_grid, omega_grid, indexing='ij')
    V = terminal_cost_np(TH, OM, cfg)
    value_slices = [None] * cfg.dp_n_time
    policy_slices = [None] * (cfg.dp_n_time - 1)
    value_slices[-1] = V.copy()

    for k in tqdm(range(cfg.dp_n_time - 2, -1, -1), desc='Backward DP'):
        interp = RegularGridInterpolator((theta_grid, omega_grid), V, bounds_error=False, fill_value=None)
        candidates = []
        for u in controls:
            theta_next = TH + dt * OM
            omega_next = OM + dt * (-np.sin(TH) - cfg.c * OM + u)
            theta_next = np.clip(theta_next, theta_grid[0], theta_grid[-1])
            omega_next = np.clip(omega_next, omega_grid[0], omega_grid[-1])
            pts = np.stack([theta_next.reshape(-1), omega_next.reshape(-1)], axis=1)
            next_val = interp(pts).reshape(TH.shape)
            stage = dt * running_cost_np(TH, OM, u, cfg)
            candidates.append(stage + next_val)
        stacked = np.stack(candidates, axis=0)
        argmin = np.argmin(stacked, axis=0)
        V = np.take_along_axis(stacked, argmin[None, ...], axis=0)[0]
        policy = controls[argmin]
        value_slices[k] = V.copy()
        policy_slices[k] = policy.copy()

    policy_interps = [RegularGridInterpolator((theta_grid, omega_grid), pol, bounds_error=False, fill_value=None) for pol in policy_slices]
    return {'theta_grid': theta_grid, 'omega_grid': omega_grid, 't_grid': t_grid, 'value_slices': value_slices, 'policy_slices': policy_slices, 'policy_interps': policy_interps, 'controls': controls, 'dt': dt}


def simulate_dp_policy(dp_result, cfg: PendulumOCConfig):
    t_grid = dp_result['t_grid']
    dt = dp_result['dt']
    theta_grid = dp_result['theta_grid']
    omega_grid = dp_result['omega_grid']
    state = np.array([cfg.theta0, cfg.omega0], dtype=np.float64)
    states = [state.copy()]
    controls = []
    for k in range(len(t_grid) - 1):
        theta = float(np.clip(state[0], theta_grid[0], theta_grid[-1]))
        omega = float(np.clip(state[1], omega_grid[0], omega_grid[-1]))
        u = float(dp_result['policy_interps'][k]([[theta, omega]])[0])
        state = rk4_step(state, u, dt, cfg)
        controls.append(u)
        states.append(state.copy())
    states = np.asarray(states)
    controls = np.asarray(controls)
    running = running_cost_np(states[:-1, 0], states[:-1, 1], controls, cfg)
    objective = trapz_uniform_np(running, dt) + terminal_cost_np(states[-1, 0], states[-1, 1], cfg)
    return {'t': t_grid, 'theta': states[:, 0], 'omega': states[:, 1], 'u': np.concatenate([controls, controls[-1:]]), 'objective': objective, 'running_cost': np.concatenate([running, running[-1:]])}


dp_result = solve_dp_benchmark(cfg)
dp_rollout = simulate_dp_policy(dp_result, cfg)

print(f\"DP/HJB benchmark objective: {dp_rollout['objective']:.4f}\")
print(f\"DP/HJB terminal state: theta(T)={dp_rollout['theta'][-1]:.4f}, omega(T)={dp_rollout['omega'][-1]:.4f}\")
print(f\"DP/HJB max |u|: {np.max(np.abs(dp_rollout['u'])):.4f}\")
"""))

    cells.append(code("""# HJB-style benchmark visualizations
plot_heatmap(dp_result['theta_grid'], dp_result['omega_grid'], dp_result['value_slices'][0], 'Approximate HJB Value Function at t = 0', r'$\\theta$ (rad)', r'$\\omega$ (rad/s)', 'hjb_value_function_heatmap.png', 'V(?, ?, 0)')
plot_heatmap(dp_result['theta_grid'], dp_result['omega_grid'], dp_result['policy_slices'][0], 'Approximate HJB Policy at t = 0', r'$\\theta$ (rad)', r'$\\omega$ (rad/s)', 'hjb_policy_heatmap.png', 'u*(?, ?, 0)')
"""))

    cells.append(md(r"""## G. Pontryagin's Maximum Principle (PMP) Benchmark

PMP gives a variational characterization of the optimum. The Hamiltonian is

\[
H(x,u,\lambda) = L(x,u) + \lambda_1\omega + \lambda_2\big(-\sin(\theta)-c\omega+u\big).
\]

The stationarity condition is

\[
\frac{\partial H}{\partial u} = 2ru + \lambda_2 = 0,
\qquad
u^* = -\frac{\lambda_2}{2r}.
\]

The costate equations are

\[
\dot{\lambda}_1 = -\left(2q_\theta\theta + 4\alpha\theta^3 - \lambda_2\cos\theta\right),
\qquad
\dot{\lambda}_2 = -\left(2q_\omega\omega + \lambda_1 - c\lambda_2\right).
\]

We solve the resulting two-point boundary-value problem by a forward-backward sweep, with a relaxed control update. For numerical comparability we project the control into $[-u_{max},u_{max}]$.
"""))

    cells.append(code("""# PMP forward-backward sweep
def forward_simulate_with_control(u_traj: np.ndarray, t_grid: np.ndarray, cfg: PendulumOCConfig):
    dt = t_grid[1] - t_grid[0]
    states = np.zeros((len(t_grid), 2), dtype=np.float64)
    states[0] = np.array([cfg.theta0, cfg.omega0], dtype=np.float64)
    for k in range(len(t_grid) - 1):
        states[k + 1] = rk4_step(states[k], float(u_traj[k]), dt, cfg)
    return states


def backward_costate(states: np.ndarray, t_grid: np.ndarray, cfg: PendulumOCConfig):
    dt = t_grid[1] - t_grid[0]
    lam = np.zeros((len(t_grid), 2), dtype=np.float64)
    theta_T, omega_T = states[-1]
    lam[-1, 0] = 2.0 * cfg.q_T * (theta_T - cfg.theta_ref)
    lam[-1, 1] = 2.0 * cfg.q_omega_T * (omega_T - cfg.omega_ref)

    def costate_rhs(state, lam_vec):
        theta, omega = state
        lam1, lam2 = lam_vec
        err = theta - cfg.theta_ref
        dlam1 = -(2.0 * cfg.q_theta * err + 4.0 * cfg.alpha * err**3 - lam2 * math.cos(theta))
        dlam2 = -(2.0 * cfg.q_omega * omega + lam1 - cfg.c * lam2)
        return np.array([dlam1, dlam2], dtype=np.float64)

    for k in range(len(t_grid) - 2, -1, -1):
        state_kp1 = states[k + 1]
        lam_kp1 = lam[k + 1]
        k1 = costate_rhs(state_kp1, lam_kp1)
        k2 = costate_rhs(state_kp1, lam_kp1 - 0.5 * dt * k1)
        k3 = costate_rhs(state_kp1, lam_kp1 - 0.5 * dt * k2)
        k4 = costate_rhs(state_kp1, lam_kp1 - dt * k3)
        lam[k] = lam_kp1 - (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return lam


def solve_pmp(cfg: PendulumOCConfig):
    t_grid = np.linspace(0.0, cfg.T, cfg.num_time_points)
    u = np.zeros_like(t_grid)
    objective_history = []

    for _ in tqdm(range(cfg.pmp_iterations), desc='PMP forward-backward sweep'):
        states = forward_simulate_with_control(u, t_grid, cfg)
        lam = backward_costate(states, t_grid, cfg)
        u_candidate = np.clip(-lam[:, 1] / (2.0 * cfg.r), -cfg.u_max, cfg.u_max)
        u_new = (1.0 - cfg.pmp_relaxation) * u + cfg.pmp_relaxation * u_candidate
        diff = np.max(np.abs(u_new - u))
        u = u_new
        running = running_cost_np(states[:, 0], states[:, 1], u, cfg)
        objective_history.append(trapz_uniform_np(running, t_grid[1] - t_grid[0]) + terminal_cost_np(states[-1, 0], states[-1, 1], cfg))
        if diff < cfg.pmp_tol:
            break

    states = forward_simulate_with_control(u, t_grid, cfg)
    running = running_cost_np(states[:, 0], states[:, 1], u, cfg)
    objective = trapz_uniform_np(running, t_grid[1] - t_grid[0]) + terminal_cost_np(states[-1, 0], states[-1, 1], cfg)
    return {'t': t_grid, 'theta': states[:, 0], 'omega': states[:, 1], 'u': u, 'running_cost': running, 'objective': objective, 'iterations': len(objective_history), 'objective_history': objective_history}


pmp_result = solve_pmp(cfg)

print(f\"PMP objective: {pmp_result['objective']:.4f}\")
print(f\"PMP terminal state: theta(T)={pmp_result['theta'][-1]:.4f}, omega(T)={pmp_result['omega'][-1]:.4f}\")
print(f\"PMP max |u|: {np.max(np.abs(pmp_result['u'])):.4f}\")
print(f\"PMP iterations: {pmp_result['iterations']}\")
"""))
    cells.append(md(r"""## H. Method Comparison

At this point we have three solutions:

- **PINN:** a mesh-free continuous-time trajectory-control approximation,
- **DP/HJB benchmark:** a coarse state-feedback approximation on a grid,
- **PMP:** a variational optimal-control solution from forward-backward sweep.

This comparison is useful because each method solves the same control problem through a different mathematical lens.
"""))

    cells.append(code("""# Comparison plots and summary metrics
plot_multiple_lines([
    (pinn_result['t'], pinn_result['theta'], 'PINN'),
    (dp_rollout['t'], dp_rollout['theta'], 'DP / HJB benchmark'),
    (pmp_result['t'], pmp_result['theta'], 'PMP'),
], 'Theta Trajectory: PINN vs HJB-style DP vs PMP', 'time (s)', r'$\\theta(t)$ (rad)', 'comparison_theta_pinn_hjb_pmp.png')

plot_multiple_lines([
    (pinn_result['t'], pinn_result['omega'], 'PINN'),
    (dp_rollout['t'], dp_rollout['omega'], 'DP / HJB benchmark'),
    (pmp_result['t'], pmp_result['omega'], 'PMP'),
], 'Omega Trajectory: PINN vs HJB-style DP vs PMP', 'time (s)', r'$\\omega(t)$ (rad/s)', 'comparison_omega_pinn_hjb_pmp.png')

plot_multiple_lines([
    (pinn_result['t'], pinn_result['u'], 'PINN'),
    (dp_rollout['t'], dp_rollout['u'], 'DP / HJB benchmark'),
    (pmp_result['t'], pmp_result['u'], 'PMP'),
], 'Control Comparison: PINN vs HJB-style DP vs PMP', 'time (s)', 'u(t) (Nm-equivalent)', 'comparison_control_pinn_hjb_pmp.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pinn_result['theta'], pinn_result['omega'], label='PINN')
ax.plot(dp_rollout['theta'], dp_rollout['omega'], label='DP / HJB benchmark')
ax.plot(pmp_result['theta'], pmp_result['omega'], label='PMP')
ax.set_title('Phase Portrait Comparison')
ax.set_xlabel(r'$\\theta$ (rad)')
ax.set_ylabel(r'$\\omega$ (rad/s)')
ax.legend()
save_current_figure('comparison_phase_pinn_hjb_pmp.png')

methods = []
for name, result in [('PINN', pinn_result), ('DP / HJB benchmark', dp_rollout), ('PMP', pmp_result)]:
    methods.append({'Method': name, 'Objective': float(result['objective']), 'theta(T)': float(result['theta'][-1]), 'omega(T)': float(result['omega'][-1]), 'Max |u|': float(np.max(np.abs(result['u']))),})
summary_df = pd.DataFrame(methods)
summary_df
"""))

    cells.append(code("""# Optional overall objective bar chart
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(summary_df['Method'], summary_df['Objective'])
ax.set_title('Objective Value Comparison')
ax.set_ylabel('objective J')
save_current_figure('objective_bar_chart.png')
"""))

    cells.append(md(r"""## Interpretation: Why PINNs, HJB, and PMP Each Matter

### Why PINNs are suitable here

PINNs are attractive in this example because they:

- enforce the differential equations directly through residual losses,
- work in continuous time without an explicit state-space mesh,
- incorporate initial conditions naturally,
- let us optimize feasibility and performance in one differentiable program.

### Potential PINN limitations

Common issues include loss-balancing sensitivity, training instability, local minima, no guarantee of global optimality, and difficulty when the optimal solution has sharp transitions.

### Why HJB / dynamic programming is valuable

HJB is the principled global-optimality framework for feedback control. It gives a value-function interpretation and a state-feedback policy, but suffers from the curse of dimensionality.

### Why PMP is valuable

PMP gives elegant necessary conditions and is efficient for low-dimensional smooth problems, but the forward-backward boundary-value structure can be sensitive to initialization and damping.
"""))

    cells.append(code("""# Final comparison table and method summary
method_table = pd.DataFrame([
    {'Method': 'PINN', 'Strengths': 'Mesh-free, continuous-time, flexible with differential constraints and objectives', 'Weaknesses': 'Loss balancing sensitivity, no global optimality guarantee, training can be delicate', 'Best use case': 'Continuous-time constrained learning or physics-informed trajectory optimization'},
    {'Method': 'HJB / DP', 'Strengths': 'Global optimality framework, feedback policy interpretation, value-function view', 'Weaknesses': 'Curse of dimensionality, grid cost grows quickly with state dimension', 'Best use case': 'Low-dimensional feedback optimal control with manageable state grids'},
    {'Method': 'PMP', 'Strengths': 'Elegant necessary conditions, efficient for smooth low-dimensional problems', 'Weaknesses': 'Boundary-value sensitivity, may require careful initialization and damping', 'Best use case': 'Smooth deterministic optimal control when adjoint equations are tractable'},
])

print('Compact objective summary')
for _, row in summary_df.iterrows():
    print(f\"{row['Method']}: objective={row['Objective']:.4f}, theta(T)={row['theta(T)']:.4f}, omega(T)={row['omega(T)']:.4f}, max|u|={row['Max |u|']:.4f}\")

summary_path = OUTPUT_DIR / 'method_objectives.csv'
summary_df.to_csv(summary_path, index=False)
print('Saved method summary to:', summary_path)

method_table
"""))

    cells.append(md(r"""## Final Conclusion

The PINN solves this nonlinear optimal control problem by learning continuous-time approximations of the state and control trajectories and minimizing a loss that combines dynamics residuals, initial-condition enforcement, and the control objective itself.

The notebook enforces the nonlinear pendulum ODE, the initial state, bounded control through a smooth `tanh` parameterization, and the continuous-time running plus terminal cost.

Comparing the three methods:

- **PINN** is mesh-free and flexible, which is useful when we want continuous-time function approximations with embedded differential constraints.
- **HJB / DP** gives a principled value-function view and approximate feedback policy, but the grid cost grows rapidly.
- **PMP** gives elegant necessary conditions and is often efficient in low dimension, but it becomes a forward-backward boundary-value problem.

The main lesson is that PINNs are useful when we want differentiable, continuous-time physics-constrained optimization, while HJB and PMP remain the more classical optimal-control lenses for global feedback structure and variational necessary conditions.
"""))

    nb.cells = cells
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.10'}
    return nb


if __name__ == '__main__':
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open('w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f'Wrote {NOTEBOOK_PATH}')
