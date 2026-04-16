from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import finite_difference


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


def find_csv_files(
    base_path: str,
    name_contains: str,
    exclude_dirs: Iterable[str] | None = None,
    exclude_name_contains: Iterable[str] | None = None,
) -> List[Path]:
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")
    exclude_dirs_l = [d.lower() for d in (exclude_dirs or [])]
    exclude_name_l = [s.lower() for s in (exclude_name_contains or [])]

    matches: List[Path] = []
    for p in base.rglob("*.csv"):
        name = p.name
        if name_contains not in name:
            continue
        parts_l = [part.lower() for part in p.parts]
        if any(excl in parts_l for excl in exclude_dirs_l):
            continue
        name_l = name.lower()
        if any(excl in name_l for excl in exclude_name_l):
            continue
        matches.append(p)
    return matches


def _normalize_col(name: str) -> str:
    return " ".join(name.strip().lower().split())


def resolve_column(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target

    norm_map = {_normalize_col(c): c for c in df.columns}
    target_norm = _normalize_col(target)
    if target_norm in norm_map:
        return norm_map[target_norm]

    tokens = [t for t in target_norm.replace(":", " ").split(" ") if t]
    candidates = [
        c for c in df.columns if all(tok in _normalize_col(c) for tok in tokens)
    ]
    if len(candidates) == 1:
        return candidates[0]

    preview = ", ".join(list(df.columns)[:20])
    raise KeyError(
        f"Missing column: {target}. Candidates: {candidates}. Available columns (first 20): {preview}"
    )


def extract_columns(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    cols = []
    for col in columns:
        resolved = resolve_column(df, col)
        cols.append(df[resolved].to_numpy())
    return np.stack(cols, axis=1).astype(np.float32)


def normalize_time(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def segment_gait_cycles(
    signal: np.ndarray,
    heel_strike_indices: Optional[Iterable[int]] = None,
) -> List[np.ndarray]:
    if heel_strike_indices is None:
        return [signal]
    indices = list(heel_strike_indices)
    cycles = []
    for i in range(len(indices) - 1):
        cycles.append(signal[indices[i] : indices[i + 1]])
    return cycles


@dataclass
class SiatSample:
    theta: np.ndarray
    omega: np.ndarray
    torque: np.ndarray
    time: np.ndarray


def load_siat_dataset(
    base_path: str,
    kinematic_col: str,
    kinetic_col: str,
    file_filter: str = "WAK",
    dt: Optional[float] = None,
    strict: bool = False,
) -> List[SiatSample]:
    files = find_csv_files(
        base_path,
        file_filter,
        exclude_dirs=["labels"],
        exclude_name_contains=["label"],
    )
    samples: List[SiatSample] = []
    skipped: List[Path] = []
    for f in files:
        df = pd.read_csv(f)
        try:
            data = extract_columns(df, [kinematic_col, kinetic_col])
        except KeyError:
            skipped.append(f)
            if strict:
                raise
            continue
        theta = data[:, 0]
        torque = data[:, 1]
        dt_local = 1.0 if dt is None else dt
        omega = finite_difference(theta, dt_local)
        time = normalize_time(len(theta))
        samples.append(SiatSample(theta=theta, omega=omega, torque=torque, time=time))

    if skipped:
        print(f"Skipped {len(skipped)} file(s) missing required columns.")
        print("Example skipped:", skipped[0])

    return samples


def load_siat_for_sysid(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns concatenated arrays of (q, qdot, qddot, u_exo) for system ID.

    Notes:
      - Assumes SIAT CSVs contain knee angle and exo torque columns.
      - Uses numerical differentiation for qdot and qddot.
      - If a sampling dt is available, pass it via config (data_paths.siat_dt).
    """
    base_path = _get_nested(config, "data_paths", "siat_llmd_base")
    if base_path is None:
        raise ValueError("Missing data_paths.siat_llmd_base in config.")

    kinematic_col = _get_nested(
        config,
        "siat",
        "kinematic_col",
        default="Kinematic: right knee flexion angle",
    )
    kinetic_col = _get_nested(
        config,
        "siat",
        "kinetic_col",
        default="Kinetic: right knee flexion torque",
    )
    file_filter = _get_nested(config, "siat", "file_filter", default="WAK")
    dt = _get_nested(config, "data_paths", "siat_dt", default=None)

    samples = load_siat_dataset(
        base_path=base_path,
        kinematic_col=kinematic_col,
        kinetic_col=kinetic_col,
        file_filter=file_filter,
        dt=dt,
        strict=False,
    )

    q_list: List[np.ndarray] = []
    qdot_list: List[np.ndarray] = []
    qddot_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []

    for s in samples:
        q = s.theta
        qdot = s.omega
        dt_local = 1.0 if dt is None else float(dt)
        qddot = np.gradient(qdot, dt_local).astype(np.float32)

        q_list.append(q)
        qdot_list.append(qdot)
        qddot_list.append(qddot)
        u_list.append(s.torque)

    if not q_list:
        raise RuntimeError("No SIAT samples found for system identification.")

    q = np.concatenate(q_list, axis=0)
    qdot = np.concatenate(qdot_list, axis=0)
    qddot = np.concatenate(qddot_list, axis=0)
    u = np.concatenate(u_list, axis=0)
    return q, qdot, qddot, u


@dataclass
class EpicSample:
    theta: np.ndarray
    time: np.ndarray


def load_epic_dataset(
    base_path: str,
    angle_col: str,
    file_filter: str = "gon",
) -> List[EpicSample]:
    files = find_csv_files(base_path, file_filter)
    samples: List[EpicSample] = []
    for f in files:
        df = pd.read_csv(f)
        theta = extract_columns(df, [angle_col])[:, 0]
        time = normalize_time(len(theta))
        samples.append(EpicSample(theta=theta, time=time))
    return samples


class SiatTorqueDataset(Dataset):
    def __init__(self, samples: List[SiatSample]) -> None:
        self.theta = np.concatenate([s.theta for s in samples], axis=0)
        self.omega = np.concatenate([s.omega for s in samples], axis=0)
        self.torque = np.concatenate([s.torque for s in samples], axis=0)
        self.time = np.concatenate([s.time for s in samples], axis=0)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.theta[idx], dtype=torch.float32),
            torch.tensor(self.omega[idx], dtype=torch.float32),
            torch.tensor(self.time[idx], dtype=torch.float32),
            torch.tensor(self.torque[idx], dtype=torch.float32),
        )


class EpicTrajectoryDataset(Dataset):
    def __init__(self, samples: List[EpicSample]) -> None:
        self.theta = np.concatenate([s.theta for s in samples], axis=0)
        self.time = np.concatenate([s.time for s in samples], axis=0)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.theta[idx], dtype=torch.float32),
            torch.tensor(self.time[idx], dtype=torch.float32),
        )
