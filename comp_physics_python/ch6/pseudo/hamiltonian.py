"""Plane-wave Hamiltonian builder for the semi-empirical pseudopotential."""

from __future__ import annotations

import math

import numpy as np

from ..apw.kpoints import load_k_vectors
from .constants import PseudoParameters


def _delta_metric(delta: np.ndarray) -> int:
    i, j, k = int(delta[0]), int(delta[1]), int(delta[2])
    return 3 * (i * i + j * j + k * k) - 2 * (i * j + j * k + i * k)


class PseudoHamiltonianBuilder:
    """Builds H(k) following pseudo.f (semi-empirical silicon)."""

    def __init__(self, params: PseudoParameters) -> None:
        self.params = params
        self.k_vectors = load_k_vectors(params.kvectors_path, params.number_kb)
        self._pseud_map = {
            3: params.pseudopotential_constants[0],
            8: params.pseudopotential_constants[1],
            11: params.pseudopotential_constants[2],
        }

    def build(self, k_point: np.ndarray) -> np.ndarray:
        """Return the NumberKAB x NumberKAB Hamiltonian matrix at k."""

        n = self.params.number_kb
        h_mat = np.zeros((n, n), dtype=float)
        rec = self.params.reciprocal_lattice_constant
        pi_quarter = math.pi / 4.0

        for i in range(n):
            kv_i = self.k_vectors[i]
            # Diagonal kinetic term
            k_tot = np.array(
                [
                    k_point[0] - kv_i[0] + kv_i[1] + kv_i[2],
                    k_point[1] + kv_i[0] - kv_i[1] + kv_i[2],
                    k_point[2] + kv_i[0] + kv_i[1] - kv_i[2],
                ],
                dtype=float,
            )
            norm_sq = float(np.dot(k_tot, k_tot))
            h_mat[i, i] = rec * rec * norm_sq

            for j in range(i):
                delta = kv_i - self.k_vectors[j]
                metric = _delta_metric(delta)
                coeff = self._pseud_map.get(metric, 0.0)
                if coeff == 0.0:
                    continue
                phase = pi_quarter * float(np.sum(delta))
                value = coeff * math.cos(phase)
                h_mat[i, j] = value
                h_mat[j, i] = value
        return h_mat
