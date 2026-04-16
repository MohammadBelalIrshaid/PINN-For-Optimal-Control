from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(r"e:/Optimal_Control/PINN/hjb_pinn_exoskeleton")
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n"))


def setup_cell(notebook_name: str, extra_imports: str = ""):
    return code(
        f"""
        from pathlib import Path
        import sys
        import math
        import numpy as np
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from scipy.integrate import solve_bvp, solve_ivp
        from scipy.interpolate import RegularGridInterpolator
        from tqdm import trange
        {extra_imports}

        ROOT = Path(r"e:/Optimal_Control/PINN/hjb_pinn_exoskeleton")
        if not ROOT.exists():
            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from src.utils import set_global_seed, get_device

        set_global_seed(42)
        device = get_device()
        plt.style.use("seaborn-v0_8-whitegrid")
        mpl.rcParams["figure.dpi"] = 140
        NOTEBOOK_NAME = "{notebook_name}"
        OUTDIR = ROOT / "notebooks" / "debug_outputs" / NOTEBOOK_NAME
        OUTDIR.mkdir(parents=True, exist_ok=True)

        def savefig(fig, name: str):
            fig.tight_layout()
            fig.savefig(OUTDIR / name, bbox_inches="tight")

        print("device:", device)
        """
    )


def notebook_02():
    cells = [
        md(
            """
            # Notebook 02: PINN for the Nonlinear Bratu Boundary-Value Problem

            This notebook solves the nonlinear Bratu problem

            \[
            y''(x) + \lambda e^{y(x)} = 0, \qquad x \in [0,1], \qquad y(0)=y(1)=0
            \]

            with a vanilla PINN and compares the learned solution against a classical numerical reference from `scipy.solve_bvp`.
            """
        ),
        md(
            """
            ## Why this example matters

            - This is a **nonlinear boundary-value problem** rather than an optimal control problem.
            - PINNs are natural here because they enforce the differential equation and the boundary conditions directly.
            - HJB/PMP are not the natural tools because there is no control input or performance index to optimize.
            """
        ),
        setup_cell("02_pinn_nonlinear_bvp_bratu"),
        code(
            """
            LAM = 1.0
            EPOCHS = 4000
            LR = 1e-3
            N_COLLOCATION = 128
            X_REF = np.linspace(0.0, 1.0, 400)

            def bratu_fun(x, y):
                return np.vstack((y[1], -LAM * np.exp(y[0])))

            def bratu_bc(ya, yb):
                return np.array([ya[0], yb[0]])

            x_mesh = np.linspace(0.0, 1.0, 40)
            y_guess = np.zeros((2, x_mesh.size))
            ref_solution = solve_bvp(bratu_fun, bratu_bc, x_mesh, y_guess, tol=1e-8, max_nodes=5000)
            assert ref_solution.success, ref_solution.message
            y_ref = ref_solution.sol(X_REF)[0]

            class BratuPINN(nn.Module):
                def __init__(self, hidden_width=64, hidden_depth=3):
                    super().__init__()
                    layers = [nn.Linear(1, hidden_width), nn.Tanh()]
                    for _ in range(hidden_depth - 1):
                        layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
                    layers.append(nn.Linear(hidden_width, 1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            def bratu_derivatives(model, x):
                y = model(x)
                dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
                ddy = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
                return y, dy, ddy
            """
        ),
        md(
            """
            ## Train the PINN

            The loss combines the Bratu residual over interior collocation points and the two Dirichlet boundary conditions.
            """
        ),
        code(
            """
            model = BratuPINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            history = {"total": [], "pde": [], "bc": []}

            for epoch in trange(EPOCHS, desc="Training Bratu PINN"):
                optimizer.zero_grad()

                x_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
                y_f, _, ddy_f = bratu_derivatives(model, x_f)
                residual = ddy_f + LAM * torch.exp(y_f)
                loss_pde = (residual ** 2).mean()

                x_b = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)
                y_b = model(x_b)
                loss_bc = (y_b ** 2).mean()

                loss = loss_pde + 10.0 * loss_bc
                loss.backward()
                optimizer.step()

                history["total"].append(float(loss.detach().cpu()))
                history["pde"].append(float(loss_pde.detach().cpu()))
                history["bc"].append(float(loss_bc.detach().cpu()))
            """
        ),
        md(
            """
            ## Compare against the classical reference

            We evaluate the learned solution, the pointwise residual, and the absolute error on a dense grid.
            """
        ),
        code(
            """
            x_plot = torch.linspace(0.0, 1.0, 400, device=device).reshape(-1, 1).requires_grad_(True)
            y_pred_t, _, ddy_pred_t = bratu_derivatives(model, x_plot)
            residual_plot = (ddy_pred_t + LAM * torch.exp(y_pred_t)).detach().cpu().numpy().reshape(-1)
            y_pred = y_pred_t.detach().cpu().numpy().reshape(-1)
            abs_error = np.abs(y_pred - y_ref)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8))

            axes[0, 0].plot(X_REF, y_ref, label="Reference", linewidth=2)
            axes[0, 0].plot(X_REF, y_pred, "--", label="PINN", linewidth=2)
            axes[0, 0].set_title("Bratu solution")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y(x)")
            axes[0, 0].legend()

            axes[0, 1].plot(X_REF, abs_error, color="tab:red")
            axes[0, 1].set_title("Absolute error")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("|y_PINN - y_ref|")

            axes[1, 0].plot(X_REF, residual_plot, color="tab:green")
            axes[1, 0].set_title("PDE residual")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("Residual")

            axes[1, 1].plot(history["total"], label="Total")
            axes[1, 1].plot(history["pde"], label="PDE")
            axes[1, 1].plot(history["bc"], label="BC")
            axes[1, 1].set_yscale("log")
            axes[1, 1].set_title("Training losses")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend()

            savefig(fig, "bratu_summary.png")
            plt.show()

            print(f"Max absolute error: {abs_error.max():.3e}")
            print(f"Mean absolute error: {abs_error.mean():.3e}")
            """
        ),
        md(
            """
            ## Key takeaway

            The Bratu problem is a clean example of a nonlinear BVP that a vanilla PINN can solve well. The formulation is naturally stated in terms of a differential equation plus boundary conditions, so a PINN is the right abstraction; HJB/PMP would only be an artificial reformulation here.
            """
        ),
    ]
    return cells


def notebook_03():
    cells = [
        md(
            """
            # Notebook 03: PINN for a Nonlinear Reaction-Diffusion PDE

            This notebook solves the one-dimensional Fisher-KPP equation

            \[
            u_t = D u_{xx} + r u(1-u), \qquad x \in [0,1], \ t \in [0,T]
            \]

            with a space-time PINN and compares it against a finite-difference reference solution.
            """
        ),
        md(
            """
            ## Why this is a natural PINN problem

            - The task is to solve a **nonlinear PDE directly in space-time**.
            - The unknown is the state field `u(x,t)`, not a control law.
            - HJB/PMP are not the natural framework because there is no underlying optimal control problem to solve.
            """
        ),
        setup_cell("03_pinn_nonlinear_reaction_diffusion"),
        code(
            """
            D = 0.01
            R = 2.0
            T_FINAL = 0.5
            NX = 201
            DX = 1.0 / (NX - 1)
            DT = 5e-4
            NT = int(T_FINAL / DT) + 1
            X = np.linspace(0.0, 1.0, NX)
            T_GRID = np.linspace(0.0, T_FINAL, NT)

            def initial_condition(x):
                return 0.35 * np.exp(-60.0 * (x - 0.35) ** 2)

            u_ref = np.zeros((NT, NX), dtype=np.float64)
            u_ref[0] = initial_condition(X)
            for n in range(NT - 1):
                u = u_ref[n]
                lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (DX ** 2)
                react = R * u[1:-1] * (1.0 - u[1:-1])
                u_next = u.copy()
                u_next[1:-1] = u[1:-1] + DT * (D * lap + react)
                u_next[0] = 0.0
                u_next[-1] = 0.0
                u_ref[n + 1] = np.clip(u_next, 0.0, 1.2)

            class FisherPINN(nn.Module):
                def __init__(self, hidden_width=64, hidden_depth=4):
                    super().__init__()
                    layers = [nn.Linear(2, hidden_width), nn.Tanh()]
                    for _ in range(hidden_depth - 1):
                        layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
                    layers.append(nn.Linear(hidden_width, 1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x, t):
                    return self.net(torch.cat([x, t], dim=1))

            def fisher_terms(model, x, t):
                u = model(x, t)
                u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
                residual = u_t - D * u_xx - R * u * (1.0 - u)
                return u, residual
            """
        ),
        md(
            """
            ## Train a space-time PINN

            The loss includes the PDE residual, two Dirichlet boundaries, and the smooth initial condition.
            """
        ),
        code(
            """
            model = FisherPINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            epochs = 3500
            history = {"total": [], "pde": [], "bc": [], "ic": []}

            for epoch in trange(epochs, desc="Training Fisher-KPP PINN"):
                optimizer.zero_grad()

                x_f = torch.rand(4000, 1, device=device, requires_grad=True)
                t_f = T_FINAL * torch.rand(4000, 1, device=device, requires_grad=True)
                _, residual = fisher_terms(model, x_f, t_f)
                loss_pde = (residual ** 2).mean()

                t_bc = T_FINAL * torch.rand(300, 1, device=device)
                x_left = torch.zeros_like(t_bc)
                x_right = torch.ones_like(t_bc)
                loss_bc = (model(x_left, t_bc) ** 2).mean() + (model(x_right, t_bc) ** 2).mean()

                x_ic = torch.rand(300, 1, device=device)
                t_ic = torch.zeros_like(x_ic)
                u_ic_target = torch.tensor(initial_condition(x_ic.detach().cpu().numpy()), dtype=torch.float32, device=device)
                loss_ic = ((model(x_ic, t_ic) - u_ic_target) ** 2).mean()

                loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
                loss.backward()
                optimizer.step()

                history["total"].append(float(loss.detach().cpu()))
                history["pde"].append(float(loss_pde.detach().cpu()))
                history["bc"].append(float(loss_bc.detach().cpu()))
                history["ic"].append(float(loss_ic.detach().cpu()))
            """
        ),
        code(
            """
            x_plot = np.linspace(0.0, 1.0, 201)
            t_plot = np.linspace(0.0, T_FINAL, 151)
            XX, TT = np.meshgrid(x_plot, t_plot)
            with torch.no_grad():
                x_tensor = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device)
                t_tensor = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device)
                u_pred = model(x_tensor, t_tensor).detach().cpu().numpy().reshape(TT.shape)

            ref_interp = RegularGridInterpolator((T_GRID, X), u_ref)
            u_ref_plot = ref_interp(np.column_stack([TT.reshape(-1), XX.reshape(-1)])).reshape(TT.shape)
            abs_error = np.abs(u_pred - u_ref_plot)

            x_res = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
            t_res = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
            _, residual_grid = fisher_terms(model, x_res, t_res)
            residual_grid = residual_grid.detach().cpu().numpy().reshape(TT.shape)

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            im0 = axes[0, 0].imshow(u_ref_plot, origin="lower", aspect="auto", extent=[0, 1, 0, T_FINAL], cmap="viridis")
            axes[0, 0].set_title("Reference solution")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("t")
            fig.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(u_pred, origin="lower", aspect="auto", extent=[0, 1, 0, T_FINAL], cmap="viridis")
            axes[0, 1].set_title("PINN prediction")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("t")
            fig.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(abs_error, origin="lower", aspect="auto", extent=[0, 1, 0, T_FINAL], cmap="magma")
            axes[0, 2].set_title("Absolute error")
            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("t")
            fig.colorbar(im2, ax=axes[0, 2])

            slice_times = [0.0, 0.15, 0.30, 0.45]
            for t_sel in slice_times:
                idx = np.argmin(np.abs(t_plot - t_sel))
                axes[1, 0].plot(x_plot, u_ref_plot[idx], linewidth=2, label=f"ref t={t_plot[idx]:.2f}")
                axes[1, 0].plot(x_plot, u_pred[idx], "--", linewidth=2, label=f"PINN t={t_plot[idx]:.2f}")
            axes[1, 0].set_title("Selected time slices")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("u(x,t)")
            axes[1, 0].legend(ncol=2, fontsize=8)

            im3 = axes[1, 1].imshow(residual_grid, origin="lower", aspect="auto", extent=[0, 1, 0, T_FINAL], cmap="coolwarm")
            axes[1, 1].set_title("PDE residual")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("t")
            fig.colorbar(im3, ax=axes[1, 1])

            axes[1, 2].plot(history["total"], label="Total")
            axes[1, 2].plot(history["pde"], label="PDE")
            axes[1, 2].plot(history["bc"], label="BC")
            axes[1, 2].plot(history["ic"], label="IC")
            axes[1, 2].set_yscale("log")
            axes[1, 2].set_title("Training losses")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].legend(fontsize=8)

            savefig(fig, "reaction_diffusion_summary.png")
            plt.show()

            print(f"Mean absolute error: {abs_error.mean():.3e}")
            print(f"Max absolute error: {abs_error.max():.3e}")
            """
        ),
        md(
            """
            ## Key takeaway

            The Fisher-KPP equation is a direct nonlinear PDE solve in space-time, which is exactly the setting where PINNs are conceptually natural. The method can learn a smooth reaction-diffusion field without introducing an artificial optimal-control layer.
            """
        ),
    ]
    return cells


def notebook_04():
    cells = [
        md(
            """
            # Notebook 04: Inverse PINN for Burgers Parameter Identification

            This notebook treats the viscosity coefficient `nu` in viscous Burgers' equation as an unknown and learns it jointly with the solution field from sparse observations.
            """
        ),
        md(
            """
            ## Why this is a strong PINN use case

            - The task is an **inverse problem**: infer a latent physical parameter while fitting a PDE solution.
            - PINNs naturally combine sparse data with the governing equation.
            - HJB/PMP are not the natural tools because the target is parameter recovery, not optimal control.
            """
        ),
        setup_cell("04_pinn_inverse_burgers_parameter_id"),
        code(
            """
            NU_TRUE = 0.05
            T_FINAL = 1.0
            NX = 201
            X = np.linspace(-1.0, 1.0, NX)
            DX = X[1] - X[0]
            T_EVAL = np.linspace(0.0, T_FINAL, 201)

            def burgers_initial(x):
                return -np.sin(np.pi * x)

            def burgers_rhs_factory(nu_value):
                def rhs(_, u_inner):
                    u = np.zeros(NX, dtype=np.float64)
                    u[1:-1] = u_inner
                    ux = (u[2:] - u[:-2]) / (2.0 * DX)
                    uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (DX ** 2)
                    return -u[1:-1] * ux + nu_value * uxx
                return rhs

            u0 = burgers_initial(X)
            ref_solution = solve_ivp(
                burgers_rhs_factory(NU_TRUE),
                (0.0, T_FINAL),
                u0[1:-1],
                t_eval=T_EVAL,
                method="Radau",
                rtol=1e-6,
                atol=1e-8,
            )
            assert ref_solution.success, ref_solution.message
            u_ref = np.zeros((T_EVAL.size, NX), dtype=np.float64)
            u_ref[:, 1:-1] = ref_solution.y.T

            rng = np.random.default_rng(42)
            n_obs = 300
            obs_t_idx = rng.integers(0, T_EVAL.size, size=n_obs)
            obs_x_idx = rng.integers(0, NX, size=n_obs)
            obs_t = T_EVAL[obs_t_idx]
            obs_x = X[obs_x_idx]
            obs_u = u_ref[obs_t_idx, obs_x_idx]

            class InverseBurgersPINN(nn.Module):
                def __init__(self, hidden_width=64, hidden_depth=4, nu_init=0.08):
                    super().__init__()
                    layers = [nn.Linear(2, hidden_width), nn.Tanh()]
                    for _ in range(hidden_depth - 1):
                        layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
                    layers.append(nn.Linear(hidden_width, 1))
                    self.net = nn.Sequential(*layers)
                    self.raw_nu = nn.Parameter(torch.tensor(np.log(np.exp(nu_init) - 1.0), dtype=torch.float32))

                def forward(self, x, t):
                    return self.net(torch.cat([x, t], dim=1))

                def nu(self):
                    return torch.nn.functional.softplus(self.raw_nu) + 1e-5

            def burgers_terms(model, x, t):
                u = model(x, t)
                u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
                residual = u_t + u * u_x - model.nu() * u_xx
                return u, residual
            """
        ),
        md(
            """
            ## Jointly learn the field and the unknown viscosity

            We combine sparse supervised observations with Burgers physics plus boundary and initial conditions.
            """
        ),
        code(
            """
            model = InverseBurgersPINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            epochs = 3000
            history = {"total": [], "physics": [], "data": [], "bc_ic": [], "nu": []}

            obs_x_t = torch.tensor(obs_x.reshape(-1, 1), dtype=torch.float32, device=device)
            obs_t_t = torch.tensor(obs_t.reshape(-1, 1), dtype=torch.float32, device=device)
            obs_u_t = torch.tensor(obs_u.reshape(-1, 1), dtype=torch.float32, device=device)

            for epoch in trange(epochs, desc="Training inverse Burgers PINN"):
                optimizer.zero_grad()

                x_f = -1.0 + 2.0 * torch.rand(5000, 1, device=device, requires_grad=True)
                t_f = T_FINAL * torch.rand(5000, 1, device=device, requires_grad=True)
                _, residual = burgers_terms(model, x_f, t_f)
                loss_physics = (residual ** 2).mean()

                u_obs_pred = model(obs_x_t, obs_t_t)
                loss_data = ((u_obs_pred - obs_u_t) ** 2).mean()

                t_bc = T_FINAL * torch.rand(400, 1, device=device)
                x_left = -torch.ones_like(t_bc)
                x_right = torch.ones_like(t_bc)
                loss_bc = (model(x_left, t_bc) ** 2).mean() + (model(x_right, t_bc) ** 2).mean()

                x_ic = -1.0 + 2.0 * torch.rand(400, 1, device=device)
                t_ic = torch.zeros_like(x_ic)
                u_ic_target = torch.tensor(burgers_initial(x_ic.detach().cpu().numpy()), dtype=torch.float32, device=device)
                loss_ic = ((model(x_ic, t_ic) - u_ic_target) ** 2).mean()

                loss_bc_ic = loss_bc + loss_ic
                loss = loss_physics + 20.0 * loss_data + 10.0 * loss_bc_ic
                loss.backward()
                optimizer.step()

                history["total"].append(float(loss.detach().cpu()))
                history["physics"].append(float(loss_physics.detach().cpu()))
                history["data"].append(float(loss_data.detach().cpu()))
                history["bc_ic"].append(float(loss_bc_ic.detach().cpu()))
                history["nu"].append(float(model.nu().detach().cpu()))
            """
        ),
        code(
            """
            x_plot = np.linspace(-1.0, 1.0, 201)
            t_plot = np.linspace(0.0, T_FINAL, 201)
            XX, TT = np.meshgrid(x_plot, t_plot)
            with torch.no_grad():
                x_tensor = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device)
                t_tensor = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device)
                u_pred = model(x_tensor, t_tensor).detach().cpu().numpy().reshape(TT.shape)

            ref_interp = RegularGridInterpolator((T_EVAL, X), u_ref)
            u_ref_plot = ref_interp(np.column_stack([TT.reshape(-1), XX.reshape(-1)])).reshape(TT.shape)
            abs_error = np.abs(u_pred - u_ref_plot)

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            im0 = axes[0, 0].imshow(u_ref_plot, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="coolwarm")
            axes[0, 0].set_title("Reference Burgers solution")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("t")
            fig.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(u_pred, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="coolwarm")
            axes[0, 1].set_title("PINN prediction")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("t")
            fig.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(abs_error, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="magma")
            axes[0, 2].scatter(obs_x, obs_t, s=8, c="white", edgecolor="black", linewidth=0.2, alpha=0.7)
            axes[0, 2].set_title("Absolute error + sparse observations")
            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("t")
            fig.colorbar(im2, ax=axes[0, 2])

            im3 = axes[1, 0].imshow(abs_error, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="viridis")
            axes[1, 0].set_title("Error heatmap")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("t")
            fig.colorbar(im3, ax=axes[1, 0])

            axes[1, 1].plot(history["nu"], label="Estimated nu")
            axes[1, 1].axhline(NU_TRUE, color="black", linestyle="--", label="nu_true")
            axes[1, 1].set_title("Viscosity estimate over epochs")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("nu")
            axes[1, 1].legend()

            axes[1, 2].plot(history["total"], label="Total")
            axes[1, 2].plot(history["physics"], label="Physics")
            axes[1, 2].plot(history["data"], label="Data")
            axes[1, 2].plot(history["bc_ic"], label="BC/IC")
            axes[1, 2].set_yscale("log")
            axes[1, 2].set_title("Training losses")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].legend(fontsize=8)

            savefig(fig, "inverse_burgers_summary.png")
            plt.show()

            print(f"nu_true = {NU_TRUE:.5f}")
            print(f"nu_hat  = {history['nu'][-1]:.5f}")
            print(f"Mean absolute field error: {abs_error.mean():.3e}")
            """
        ),
        md(
            """
            ## Key takeaway

            Learning an unknown coefficient from sparse data plus physics is one of the clearest strengths of PINNs. This problem is not naturally posed as an HJB/PMP optimal-control problem; the governing PDE and the parameter are the primary unknowns.
            """
        ),
    ]
    return cells


def notebook_05():
    cells = [
        md(
            """
            # Notebook 05: Where a Vanilla PINN Struggles — Low-Viscosity Burgers

            This notebook repeats the viscous Burgers setup with a much smaller viscosity. The resulting sharp fronts make the problem considerably harder for a standard PINN with a smooth MLP.
            """
        ),
        md(
            """
            ## Why this is an honest failure case

            - We keep a standard PINN architecture and train it fairly.
            - The low-viscosity regime develops steep gradients that challenge smooth neural approximators.
            - This illustrates spectral bias and the difficulty vanilla PINNs have with sharp, multiscale structure.
            """
        ),
        setup_cell("05_pinn_failure_low_viscosity_burgers"),
        code(
            """
            NU_LOW = 0.003
            T_FINAL = 1.0
            NX = 301
            X = np.linspace(-1.0, 1.0, NX)
            DX = X[1] - X[0]
            DT = 5e-4
            NT = int(T_FINAL / DT) + 1
            T_EVAL = np.linspace(0.0, T_FINAL, NT)

            def burgers_initial(x):
                return -np.sin(np.pi * x)

            u_ref = np.zeros((NT, NX), dtype=np.float64)
            u_ref[0] = burgers_initial(X)
            for n in range(NT - 1):
                u = u_ref[n]
                f = 0.5 * u ** 2
                wave_speed = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
                flux_half = 0.5 * (f[:-1] + f[1:]) - 0.5 * wave_speed * (u[1:] - u[:-1])
                diffusion = NU_LOW * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (DX ** 2)

                u_next = u.copy()
                u_next[1:-1] = u[1:-1] - (DT / DX) * (flux_half[1:] - flux_half[:-1]) + DT * diffusion
                u_next[0] = 0.0
                u_next[-1] = 0.0
                u_ref[n + 1] = u_next

            class VanillaBurgersPINN(nn.Module):
                def __init__(self, hidden_width=64, hidden_depth=4):
                    super().__init__()
                    layers = [nn.Linear(2, hidden_width), nn.Tanh()]
                    for _ in range(hidden_depth - 1):
                        layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
                    layers.append(nn.Linear(hidden_width, 1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x, t):
                    return self.net(torch.cat([x, t], dim=1))

            def burgers_terms(model, x, t):
                u = model(x, t)
                u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
                residual = u_t + u * u_x - NU_LOW * u_xx
                return u, residual
            """
        ),
        md(
            """
            ## Train the vanilla PINN fairly

            We use the same style of architecture as the inverse Burgers notebook. The point is not to sabotage the method; the point is to observe what happens when the solution contains steep fronts.
            """
        ),
        code(
            """
            model = VanillaBurgersPINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            epochs = 3000
            history = {"total": [], "physics": [], "bc_ic": []}

            for epoch in trange(epochs, desc="Training low-viscosity Burgers PINN"):
                optimizer.zero_grad()

                x_f = -1.0 + 2.0 * torch.rand(5000, 1, device=device, requires_grad=True)
                t_f = T_FINAL * torch.rand(5000, 1, device=device, requires_grad=True)
                _, residual = burgers_terms(model, x_f, t_f)
                loss_physics = (residual ** 2).mean()

                t_bc = T_FINAL * torch.rand(400, 1, device=device)
                x_left = -torch.ones_like(t_bc)
                x_right = torch.ones_like(t_bc)
                loss_bc = (model(x_left, t_bc) ** 2).mean() + (model(x_right, t_bc) ** 2).mean()

                x_ic = -1.0 + 2.0 * torch.rand(400, 1, device=device)
                t_ic = torch.zeros_like(x_ic)
                u_ic_target = torch.tensor(burgers_initial(x_ic.detach().cpu().numpy()), dtype=torch.float32, device=device)
                loss_ic = ((model(x_ic, t_ic) - u_ic_target) ** 2).mean()

                loss_bc_ic = loss_bc + loss_ic
                loss = loss_physics + 10.0 * loss_bc_ic
                loss.backward()
                optimizer.step()

                history["total"].append(float(loss.detach().cpu()))
                history["physics"].append(float(loss_physics.detach().cpu()))
                history["bc_ic"].append(float(loss_bc_ic.detach().cpu()))
            """
        ),
        code(
            """
            x_plot = np.linspace(-1.0, 1.0, 201)
            t_plot = np.linspace(0.0, T_FINAL, 201)
            XX, TT = np.meshgrid(x_plot, t_plot)
            with torch.no_grad():
                x_tensor = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device)
                t_tensor = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device)
                u_pred = model(x_tensor, t_tensor).detach().cpu().numpy().reshape(TT.shape)

            ref_interp = RegularGridInterpolator((T_EVAL, X), u_ref)
            u_ref_plot = ref_interp(np.column_stack([TT.reshape(-1), XX.reshape(-1)])).reshape(TT.shape)
            abs_error = np.abs(u_pred - u_ref_plot)

            x_res = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
            t_res = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
            _, residual_grid = burgers_terms(model, x_res, t_res)
            residual_grid = residual_grid.detach().cpu().numpy().reshape(TT.shape)

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            im0 = axes[0, 0].imshow(u_ref_plot, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="coolwarm")
            axes[0, 0].set_title("Reference low-viscosity solution")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("t")
            fig.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(u_pred, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="coolwarm")
            axes[0, 1].set_title("Vanilla PINN prediction")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("t")
            fig.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(abs_error, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="magma")
            axes[0, 2].set_title("Absolute error")
            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("t")
            fig.colorbar(im2, ax=axes[0, 2])

            for t_sel in [0.15, 0.35, 0.55, 0.75]:
                idx = np.argmin(np.abs(t_plot - t_sel))
                axes[1, 0].plot(x_plot, u_ref_plot[idx], linewidth=2, label=f"ref t={t_plot[idx]:.2f}")
                axes[1, 0].plot(x_plot, u_pred[idx], "--", linewidth=2, label=f"PINN t={t_plot[idx]:.2f}")
            axes[1, 0].set_title("Slices through the sharp-front region")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("u(x,t)")
            axes[1, 0].legend(ncol=2, fontsize=8)

            im3 = axes[1, 1].imshow(residual_grid, origin="lower", aspect="auto", extent=[-1, 1, 0, T_FINAL], cmap="viridis")
            axes[1, 1].set_title("Residual heatmap")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("t")
            fig.colorbar(im3, ax=axes[1, 1])

            axes[1, 2].plot(history["total"], label="Total")
            axes[1, 2].plot(history["physics"], label="Physics")
            axes[1, 2].plot(history["bc_ic"], label="BC/IC")
            axes[1, 2].set_yscale("log")
            axes[1, 2].set_title("Training losses")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].legend(fontsize=8)

            savefig(fig, "low_viscosity_burgers_summary.png")
            plt.show()

            print(f"Mean absolute field error: {abs_error.mean():.3e}")
            print(f"Max absolute field error: {abs_error.max():.3e}")
            """
        ),
        md(
            """
            ## Discussion: why vanilla PINNs struggle here

            The failure is not due to a dishonest setup; it reflects a real limitation. Low-viscosity Burgers develops sharp fronts, while standard PINNs have a bias toward smooth low-frequency representations. This spectral bias makes it difficult to resolve steep localized structure without additional techniques such as adaptive sampling, domain decomposition, Fourier features, or shock-aware formulations.
            """
        ),
        md(
            """
            ## Key takeaway

            Vanilla PINNs are not universally reliable. They can struggle on problems with shocks, sharp fronts, stiffness, or multiple active scales, even when the governing equation is known exactly.
            """
        ),
    ]
    return cells


def notebook_06():
    cells = [
        md(
            """
            # Notebook 06: Scope Comparison — PINNs vs HJB/PMP

            This notebook compares two different nonlinear problem classes:

            - **Part A:** a nonlinear optimal control problem, where PMP/HJB ideas are natural.
            - **Part B:** a nonlinear boundary-value problem, where a PINN is natural.

            The point is not that one framework is universally better. The point is that method choice should match the mathematical structure of the problem.
            """
        ),
        setup_cell("06_scope_comparison_pinn_vs_hjb_pmp", extra_imports="import pandas as pd"),
        md(
            """
            ## Part A: Nonlinear optimal control for a pendulum

            We solve a finite-horizon nonlinear pendulum regulation problem with a Pontryagin-style two-point boundary-value formulation. This is the right conceptual setting for HJB/PMP: the unknown is an optimal control policy or optimality system, not just a state field.
            """
        ),
        code(
            """
            T_FINAL = 2.0
            damping = 0.2
            r_u = 0.1
            q0 = 1.0
            w0 = 0.0
            qf_weight = 8.0
            wf_weight = 1.0

            t_mesh = np.linspace(0.0, T_FINAL, 160)

            def pmp_ode(t, Y):
                q, w, lam_q, lam_w = Y
                u = -lam_w / (2.0 * r_u)
                dq = w
                dw = -np.sin(q) - damping * w + u
                dlam_q = -2.0 * q + lam_w * np.cos(q)
                dlam_w = -0.2 * w - lam_q + damping * lam_w
                return np.vstack((dq, dw, dlam_q, dlam_w))

            def pmp_bc(ya, yb):
                return np.array([
                    ya[0] - q0,
                    ya[1] - w0,
                    yb[2] - 2.0 * qf_weight * yb[0],
                    yb[3] - 2.0 * wf_weight * yb[1],
                ])

            y_guess = np.zeros((4, t_mesh.size))
            y_guess[0] = q0 * np.exp(-t_mesh)
            pmp_solution = solve_bvp(pmp_ode, pmp_bc, t_mesh, y_guess, tol=1e-4, max_nodes=10000)
            assert pmp_solution.success, pmp_solution.message

            t_plot = np.linspace(0.0, T_FINAL, 400)
            q_opt, w_opt, lam_q_opt, lam_w_opt = pmp_solution.sol(t_plot)
            u_opt = -lam_w_opt / (2.0 * r_u)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(t_plot, q_opt, label="q(t)", linewidth=2)
            axes[0].plot(t_plot, w_opt, label="omega(t)", linewidth=2)
            axes[0].set_title("Optimal state trajectory")
            axes[0].set_xlabel("t")
            axes[0].legend()

            axes[1].plot(t_plot, u_opt, color="tab:red", linewidth=2)
            axes[1].set_title("Optimal control from PMP")
            axes[1].set_xlabel("t")
            axes[1].set_ylabel("u(t)")

            savefig(fig, "scope_control_example.png")
            plt.show()
            """
        ),
        md(
            """
            ## Part B: Nonlinear BVP solved with a PINN

            We now revisit a smaller Bratu problem. Here the unknown is a state function that must satisfy a nonlinear differential equation and two boundary conditions. A PINN is a direct fit for that structure.
            """
        ),
        code(
            """
            LAM = 1.0
            x_ref = np.linspace(0.0, 1.0, 300)

            def bratu_fun(x, y):
                return np.vstack((y[1], -LAM * np.exp(y[0])))

            def bratu_bc(ya, yb):
                return np.array([ya[0], yb[0]])

            ref_solution = solve_bvp(bratu_fun, bratu_bc, np.linspace(0.0, 1.0, 40), np.zeros((2, 40)), tol=1e-8, max_nodes=5000)
            assert ref_solution.success, ref_solution.message
            y_ref = ref_solution.sol(x_ref)[0]

            class SmallBratuPINN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(1, 48), nn.Tanh(),
                        nn.Linear(48, 48), nn.Tanh(),
                        nn.Linear(48, 48), nn.Tanh(),
                        nn.Linear(48, 1),
                    )

                def forward(self, x):
                    return self.net(x)

            def bratu_terms(model, x):
                y = model(x)
                dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
                ddy = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
                return y, ddy + LAM * torch.exp(y)

            model = SmallBratuPINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in trange(2500, desc="Training compact Bratu PINN"):
                optimizer.zero_grad()
                x_f = torch.rand(128, 1, device=device, requires_grad=True)
                _, residual = bratu_terms(model, x_f)
                loss_pde = (residual ** 2).mean()
                x_b = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)
                loss_bc = (model(x_b) ** 2).mean()
                loss = loss_pde + 10.0 * loss_bc
                loss.backward()
                optimizer.step()

            x_plot_t = torch.tensor(x_ref.reshape(-1, 1), dtype=torch.float32, device=device)
            y_pred = model(x_plot_t).detach().cpu().numpy().reshape(-1)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_ref, y_ref, label="Classical BVP solver", linewidth=2)
            ax.plot(x_ref, y_pred, "--", label="PINN", linewidth=2)
            ax.set_title("Bratu BVP: classical solver vs PINN")
            ax.set_xlabel("x")
            ax.set_ylabel("y(x)")
            ax.legend()
            savefig(fig, "scope_bratu_example.png")
            plt.show()
            """
        ),
        md(
            """
            ## Summary table

            The comparison is about *problem class* and *natural applicability*, not about declaring one method globally superior.
            """
        ),
        code(
            """
            summary_df = pd.DataFrame([
                {
                    "problem_type": "Nonlinear optimal control",
                    "state/control_dimension": "2 state, 1 control",
                    "outputs_sought": "Optimal trajectory and torque",
                    "natural_method": "HJB / PMP",
                    "main_limitation": "Curse of dimensionality or BVP sensitivity",
                },
                {
                    "problem_type": "Nonlinear BVP / PDE solve",
                    "state/control_dimension": "Field over space-time",
                    "outputs_sought": "State function satisfying PDE/BC/IC",
                    "natural_method": "PINN",
                    "main_limitation": "Training can be slow or optimization-sensitive",
                },
                {
                    "problem_type": "Inverse parameter identification",
                    "state/control_dimension": "Field + latent parameters",
                    "outputs_sought": "State and hidden coefficients",
                    "natural_method": "PINN",
                    "main_limitation": "Identifiability and data quality matter",
                },
            ])
            display(summary_df)
            """
        ),
        md(
            """
            ## Key takeaway

            - PINNs are powerful for nonlinear differential equations, boundary-value problems, and inverse problems.
            - HJB and PMP are specialized tools for optimal control.
            - Vanilla PINNs can struggle on shocks, stiff systems, and multiscale or high-frequency solutions.
            - Method choice should depend on the mathematical structure of the problem.
            """
        ),
    ]
    return cells


def write_notebook(name: str, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    path = NOTEBOOK_DIR / name
    nbf.write(nb, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook("02_pinn_nonlinear_bvp_bratu.ipynb", notebook_02())
    write_notebook("03_pinn_nonlinear_reaction_diffusion.ipynb", notebook_03())
    write_notebook("04_pinn_inverse_burgers_parameter_id.ipynb", notebook_04())
    write_notebook("05_pinn_failure_low_viscosity_burgers.ipynb", notebook_05())
    write_notebook("06_scope_comparison_pinn_vs_hjb_pmp.ipynb", notebook_06())
