"""FORTRAN-style Numerov integrator with optional singular handling."""

from __future__ import annotations

import numpy as np


def numerov(
    delta: float,
    start_i: int,
    end_i: int,
    f_array: np.ndarray,
    phi_start: float,
    phi_next: float,
    singular: bool,
) -> np.ndarray:
    """Integrate psi'' = f psi on an arbitrary grid index interval.

    Parameters mirror ``numerov.f`` exactly.  ``f_array`` is treated as a
    1-indexed array so callers can keep the original indexing scheme.
    """

    if start_i <= 0 or end_i <= 0:
        raise ValueError("Indices must be >= 1 for the FORTRAN-style grid")
    if start_i >= len(f_array) or end_i >= len(f_array):
        raise ValueError("Indices exceed the provided f_array length")

    solution = np.zeros_like(f_array, dtype=float)
    istep = -1 if delta < 0.0 else 1
    delta_sq = delta * delta
    fac = delta_sq / 12.0

    if not singular:
        solution[start_i] = phi_start
        w_prev = (1.0 - fac * f_array[start_i]) * phi_start
    else:
        w_prev = phi_start
        solution[start_i] = phi_start

    phi = phi_next
    next_index = start_i + istep
    solution[next_index] = phi_next
    w_val = (1.0 - fac * f_array[next_index]) * phi

    rng = (
        range(start_i + istep, end_i, istep)
        if istep > 0
        else range(start_i + istep, end_i, istep)
    )
    for i in rng:
        w_next = 2.0 * w_val - w_prev + delta_sq * phi * f_array[i]
        w_prev = w_val
        w_val = w_next
        denom = 1.0 - fac * f_array[i + istep]
        if abs(denom) < 1e-12:
            raise ZeroDivisionError("Numerov denominator vanished; adjust grid")
        phi = w_val / denom
        solution[i + istep] = phi
    return solution
