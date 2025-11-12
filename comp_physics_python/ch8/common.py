"""Common utilities for Chapter 8 molecular-dynamics examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MDConfig:
    n_atoms: int
    density: float
    temperature: float
    time_step: float
    equilibration_steps: int
    production_steps: int
    cutoff: float
    box_length: float | None = None

    def __post_init__(self) -> None:
        if self.box_length is None:
            self.box_length = (self.n_atoms / self.density) ** (1.0 / 3.0)
        self.half_box = 0.5 * self.box_length


def make_fcc_lattice(n_atoms: int, box: float) -> np.ndarray:
    cells = int(round((n_atoms / 4) ** (1.0 / 3.0)))
    if 4 * cells**3 != n_atoms:
        raise ValueError("n_atoms must be 4 * cells^3 for the FCC lattice")
    a = box / cells
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    positions = []
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                origin = np.array([ix, iy, iz], dtype=float)
                positions.extend(a * (origin + basis))
    return np.array(positions)


def maxwell_velocities(n_atoms: int, temperature: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, np.sqrt(temperature), size=(n_atoms, 3))
    v -= v.mean(axis=0, keepdims=True)
    current_temp = (np.sum(v**2) / (3 * n_atoms))
    v *= np.sqrt(temperature / current_temp)
    return v


def minimum_image(displacements: np.ndarray, box_length: float) -> np.ndarray:
    return displacements - box_length * np.rint(displacements / box_length)


def velocity_verlet(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    box: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = positions + velocities * dt + 0.5 * forces * dt * dt
    positions %= box
    velocities = velocities + 0.5 * forces * dt
    return positions, velocities
