"""Silicon-specific constants for the semi-empirical pseudopotential model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PseudoParameters:
    """All knobs for the silicon pseudopotential calculation."""

    number_ka: int = 27
    number_kb: int = 113
    max_level: int = 10
    reciprocal_lattice_constant: float = 0.6088  # 2*pi/a (Bohr^-1)
    step_length: float = 0.03  # fractional length inside the IBZ
    data_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "comp_physics"
        / "comp_physics_textbook_code"
        / "4560_ch6"
        / "ch6"
        / "apw"
    )

    # Fourier components V3, V8, V11 (Hartree) used in pseudo.f
    pseudopotential_constants: tuple[float, float, float] = (
        -0.2241,
        0.0551,
        0.0724,
    )

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
    def u_point(self) -> np.ndarray:
        return np.array([1.0, 0.25, 0.25], dtype=float)

    @property
    def default_line(self) -> tuple[np.ndarray, np.ndarray]:
        return self.l_point, self.gamma_point

    @property
    def kvectors_path(self) -> Path:
        return self.data_root / "KVectors"
