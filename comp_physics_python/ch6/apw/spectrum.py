"""High-level APW spectrum driver (Γ→K line by default)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .constants import APWParameters
from .kpoints import interpolate_line
from .matrices import StructureMatrices, build_hamiltonian, determinant
from .potential import LogPotential


@dataclass
class SpectrumSample:
    """Energies (roots/minima) recorded for a single k-point."""

    index: int
    k_point: np.ndarray
    arclength: float
    energies: List[float]


@dataclass
class APWSolver:
    """User-facing helper mirroring logapw.f's workflow."""

    params: APWParameters
    potential: LogPotential
    k_vectors: np.ndarray

    def __post_init__(self) -> None:
        self._structure = StructureMatrices(self.params, self.k_vectors)

    def spectrum_at_k(
        self,
        k_point: np.ndarray,
        energy_min: float | None = None,
        energy_max: float | None = None,
        samples: int | None = None,
    ) -> List[float]:
        emin = energy_min if energy_min is not None else self.params.energy_min
        emax = energy_max if energy_max is not None else self.params.energy_max
        num = samples if samples is not None else self.params.energy_samples
        energies = np.linspace(emin, emax, num, endpoint=True)
        mat_a, mat_b, mat_c = self._structure.build(k_point)

        values: list[float] = []
        prev_det = None
        prev_energy = None
        prevprev_det = None
        prevprev_energy = None

        for energy in energies:
            chi_r, chi_dot = self.potential.atom_integrals(
                energy, self.params.max_l, self.params.max_sol
            )
            h_mat = build_hamiltonian(energy, mat_a, mat_b, mat_c, chi_r, chi_dot)
            det_val = determinant(h_mat)

            if prev_det is not None and prev_energy is not None:
                if det_val == 0.0:
                    values.append(float(energy))
                elif det_val * prev_det < 0.0:
                    slope = det_val - prev_det
                    if abs(slope) > 1e-14:
                        root = prev_energy - prev_det * (energy - prev_energy) / slope
                        values.append(float(root))
                elif prevprev_det is not None:
                    if (
                        (prev_det - det_val) * (prev_det - prevprev_det) > 0.0
                        and abs(prev_det) < abs(det_val)
                        and abs(prev_det) < abs(prevprev_det)
                    ):
                        xs = np.array(
                            [prevprev_energy, prev_energy, energy], dtype=float
                        )
                        ys = np.array([prevprev_det, prev_det, det_val], dtype=float)
                        coeffs = np.polyfit(xs, ys, 2)
                        a, b, _ = coeffs
                        if abs(a) > 1e-12:
                            vertex = -b / (2.0 * a)
                            if xs[0] <= vertex <= xs[-1]:
                                values.append(float(vertex))
            prevprev_det = prev_det
            prevprev_energy = prev_energy
            prev_det = det_val
            prev_energy = energy
        return sorted(values)

    def line_spectrum(
        self,
        first_point: np.ndarray,
        second_point: np.ndarray,
        step_length: float | None = None,
        energy_min: float | None = None,
        energy_max: float | None = None,
        samples: int | None = None,
    ) -> Iterable[SpectrumSample]:
        step = step_length if step_length is not None else self.params.step_length
        emin = energy_min if energy_min is not None else self.params.energy_min
        emax = energy_max if energy_max is not None else self.params.energy_max
        sample_count = samples if samples is not None else self.params.energy_samples

        arc = 0.0
        prev = None
        for idx, k_point in enumerate(
            interpolate_line(first_point, second_point, step_length=step)
        ):
            if prev is not None:
                arc += float(np.linalg.norm(k_point - prev))
            energies = self.spectrum_at_k(
                k_point,
                energy_min=emin,
                energy_max=emax,
                samples=sample_count,
            )
            yield SpectrumSample(index=idx, k_point=k_point, arclength=arc, energies=energies)
            prev = k_point.copy()
