"""Löwdin partitioning utilities."""

from __future__ import annotations

import numpy as np


def full_diagonalisation(h_matrix: np.ndarray, max_level: int) -> list[float]:
    """Return the lowest `max_level` eigenvalues of the full Hamiltonian."""

    eigvals = np.linalg.eigh(h_matrix)[0]
    return eigvals[:max_level].tolist()


def lowdin_matrix(
    h_matrix: np.ndarray,
    energy_guess: float,
    number_ka: int,
) -> np.ndarray:
    """Return U(E) for a given energy guess."""

    h_aa = h_matrix[:number_ka, :number_ka]
    h_ab = h_matrix[:number_ka, number_ka:]
    diag_bb = np.diag(h_matrix)[number_ka:]
    denom = energy_guess - diag_bb
    # Avoid division by zero (degenerate cases) by nudging denominators.
    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10 + 1e-10, denom)
    correction = (h_ab * (1.0 / denom)) @ h_ab.T
    return h_aa + correction


def lowdin_levels(
    h_matrix: np.ndarray,
    previous_levels: list[float],
    number_ka: int,
) -> list[float]:
    """Update the eigenvalues band-by-band using Löwdin matrices."""

    new_levels: list[float] = []
    for level_index, energy in enumerate(previous_levels):
        u_matrix = lowdin_matrix(h_matrix, energy, number_ka)
        eigvals = np.linalg.eigh(u_matrix)[0]
        if level_index >= eigvals.size:
            raise ValueError(
                f"Requested level {level_index} but Löwdin matrix has only {eigvals.size} bands"
            )
        new_levels.append(float(eigvals[level_index]))
    return new_levels
