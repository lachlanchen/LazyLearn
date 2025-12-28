"""High-level spectrum driver for the semi-empirical pseudopotential."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from ..apw.kpoints import interpolate_line
from .constants import PseudoParameters
from .hamiltonian import PseudoHamiltonianBuilder
from .lowdin import full_diagonalisation, lowdin_levels


@dataclass
class BandSample:
    index: int
    k_point: np.ndarray
    arclength: float
    energies: List[float]


class PseudoSolver:
    """Encapsulates the full-diagonalisation + Löwdin workflow."""

    def __init__(self, params: PseudoParameters) -> None:
        self.params = params
        self.builder = PseudoHamiltonianBuilder(params)

    def spectrum_at_k(
        self,
        k_point: np.ndarray,
        previous_levels: Sequence[float] | None,
    ) -> list[float]:
        h_matrix = self.builder.build(k_point)
        if previous_levels is None:
            return full_diagonalisation(h_matrix, self.params.max_level)
        return lowdin_levels(
            h_matrix,
            list(previous_levels),
            self.params.number_ka,
        )

    def line_spectrum(
        self,
        first_point: np.ndarray,
        second_point: np.ndarray,
    ) -> Iterable[BandSample]:
        arc = 0.0
        prev_point = None
        previous_levels: list[float] | None = None
        for idx, k_point in enumerate(
            interpolate_line(
                first_point,
                second_point,
                step_length=self.params.step_length,
            )
        ):
            if prev_point is not None:
                arc += float(np.linalg.norm(k_point - prev_point))
            energies = self.spectrum_at_k(k_point, previous_levels)
            yield BandSample(index=idx, k_point=k_point, arclength=arc, energies=energies)
            previous_levels = energies
            prev_point = k_point.copy()
