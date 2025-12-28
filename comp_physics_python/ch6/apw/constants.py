"""Constants and lattice metadata for the copper APW example."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class APWParameters:
    """Holds all scalar parameters used by the original FORTRAN code."""

    number_k: int = 27
    max_l: int = 5
    max_sol: int = 1000
    muffin_tin_radius: float = 2.41191
    lattice_constant: float = 6.8219117
    step_length: float = 0.02  # fractional length between k-points
    energy_min: float = -0.04
    energy_max: float = 0.34
    energy_samples: int = 100
    data_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "comp_physics"
        / "comp_physics_textbook_code"
        / "4560_ch6"
        / "ch6"
        / "apw"
    )

    @property
    def pi(self) -> float:
        return float(np.pi)

    @property
    def reciprocal_lattice_constant(self) -> float:
        return 2.0 * self.pi / self.lattice_constant

    @property
    def unit_cell_volume(self) -> float:
        return (self.lattice_constant ** 3) / 4.0

    @property
    def default_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Default Γ→K line used in the FORTRAN example."""

        return self.gamma_point, self.k_point

    @property
    def gamma_point(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    @property
    def l_point(self) -> np.ndarray:
        return np.array([0.5, 0.5, 0.5], dtype=float)

    @property
    def k_point(self) -> np.ndarray:
        return np.array([0.75, 0.75, 0.0], dtype=float)

    @property
    def x_point(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    @property
    def w_point(self) -> np.ndarray:
        return np.array([1.0, 0.5, 0.0], dtype=float)

    @property
    def u_point(self) -> np.ndarray:
        return np.array([1.0, 0.25, 0.25], dtype=float)

    def clamp_energy_samples(self, samples: int | None = None) -> int:
        return samples if samples is not None else self.energy_samples
