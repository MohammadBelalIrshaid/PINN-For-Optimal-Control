# PINN for Optimal Control

Physics-informed neural network examples and research code for optimal control, with a main focus on Hamilton-Jacobi-Bellman-informed control for a knee exoskeleton tracking problem.

This repository combines:

- Tutorial notebooks for PINNs and nonlinear optimal control.
- A toy scalar LQR HJB-PINN example with an analytic solution.
- A nonlinear pendulum optimal control tutorial comparing PINN, coarse HJB-style dynamic programming, and Pontryagin's Maximum Principle.
- Research notebooks and source code for a single-joint knee exoskeleton control pipeline using SIAT-LLMD and EPIC-LAB gait data.

## Repository Layout

```text
hjb_pinn_exoskeleton/
├── configs/         # YAML configuration for toy and knee experiments
├── notebooks/
│   ├── Tutorials/   # teaching notebooks for PINNs and optimal control
│   └── Research/    # biomechanics / exoskeleton workflow notebooks
├── scripts/         # helper scripts for notebook generation and testing
├── src/             # reusable Python modules
├── checkpoints/     # local trained model weights (ignored in git)
├── data/            # local processed datasets (ignored in git)
├── results/         # local generated figures / outputs (ignored in git)
├── requirements.txt
├── run_all.bat
└── run_all.sh
```

## What Is In This Project

### 1. Tutorial material

`notebooks/Tutorials/` contains standalone notebooks that explain core PINN and optimal control ideas:

- `01_toy_lqr_hjb_pinn.ipynb`: scalar LQR solved with an HJB-PINN and compared to the Riccati solution.
- `02_pinn_nonlinear_bvp_bratu.ipynb`: PINN for a nonlinear boundary-value problem.
- `03_pinn_nonlinear_reaction_diffusion.ipynb`: PINN for a nonlinear PDE.
- `04_pinn_inverse_burgers_parameter_id.ipynb`: inverse PINN for parameter identification.
- `05_pinn_failure_low_viscosity_burgers.ipynb`: PINN failure mode example.
- `06_scope_comparison_pinn_vs_hjb_pmp.ipynb`: conceptual comparison of PINN, HJB, and PMP.
- `07_nonlinear_optimal_control_pinn_tutorial.ipynb`: damped pendulum trajectory optimization with PINN, HJB-style DP, and PMP.

### 2. Exoskeleton research workflow

`notebooks/Research/` contains the end-to-end research pipeline:

- `02_knee_model_setup.ipynb`: single-DOF knee model and parameter setup.
- `03_siat_preprocessing.ipynb`: SIAT-LLMD preprocessing.
- `04_epic_preprocessing.ipynb`: EPIC-LAB preprocessing.
- `05_hjb_pinn_knee_training.ipynb`: HJB-PINN training for knee control.
- `06_baselines_and_evaluation.ipynb`: baseline controllers and metrics.
- `07_visualization_and_plots.ipynb`: publication-style plots and summaries.

### 3. Reusable source code

`src/` includes the reusable implementation:

- `lqr_toy.py`: toy LQR definitions, Riccati solver, HJB residual, and value network.
- `hjb_pinn.py`: value-function PINN components for the knee control problem.
- `knee_dynamics.py`: single-joint knee / exoskeleton dynamics.
- `datasets.py`: dataset loading and utilities.
- `baselines.py`: baseline control methods.
- `control_utils.py`, `system_id.py`, `utils.py`: support functions.

## Core Problem

The main research problem is optimal mid-level torque control for a single-joint knee exoskeleton. The controller is learned through a value function approximation constrained by the HJB equation, with additional data and trajectory terms.

At a high level, the training objective combines:

- HJB residual minimization.
- Terminal condition enforcement.
- Supervised torque matching on SIAT-LLMD.
- Trajectory tracking against EPIC-LAB reference motion.

## Data

This project expects access to two external gait datasets:

- `SIAT-LLMD`: knee angle and knee torque measurements.
- `EPIC-LAB`: knee angle trajectories used as reference motion.

The public repository does not need to include those datasets. Update the local paths in [`configs/knee_config.yaml`](configs/knee_config.yaml) before running the research notebooks.

Current config keys:

- `data_paths.epic_lab_base`
- `data_paths.siat_llmd_base`

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `tqdm`

## How To Run

### Quick start

1. Install dependencies.
2. Update dataset paths in `configs/knee_config.yaml`.
3. Run the tutorial notebooks in order if you want the teaching progression.
4. Run the research notebooks in order if you want the exoskeleton pipeline.

### Batch execution

Windows:

```bat
run_all.bat
```

Unix-like shell:

```bash
./run_all.sh
```

## Configuration Notes

Two main configs are included:

- [`configs/lqr_config.yaml`](configs/lqr_config.yaml): toy scalar LQR problem.
- [`configs/knee_config.yaml`](configs/knee_config.yaml): exoskeleton dynamics, data paths, loss weights, collocation bounds, controller saturation, debugging options, and rollout settings.

Important configuration sections in the knee setup:

- `cost_weights`
- `loss`
- `training`
- `control`
- `collocation`
- `rollout`
- `baseline`
- `debug`

## Outputs

When you run the notebooks locally, the project can generate:

- processed datasets in `data/`
- trained weights in `checkpoints/`
- figures, comparison tables, and animations in `results/`

These directories are treated as local working artifacts rather than source files for the public repository.

## Reproducibility

- Random seeds are set through utilities in `src/utils.py`.
- YAML config files capture experiment settings.
- The tutorial notebooks now include generated figures and animations for system motion under PINN control.

## Suggested Citation Context

If you use this repository in a report, thesis, or paper, describe it as:

> a PINN-based optimal control codebase combining tutorial examples, HJB-informed value-function learning, and a single-joint knee exoskeleton tracking application using gait data

## License / Data Use

Before sharing or redistributing results derived from SIAT-LLMD or EPIC-LAB, verify the usage terms for those datasets. This repository contains code and notebooks; dataset licensing remains governed by the original data providers.
