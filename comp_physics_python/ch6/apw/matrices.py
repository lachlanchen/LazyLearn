"""Matrix assembly for the APW Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import APWParameters
from .kpoints import cubic_metric
from .special_functions import legendre_p, spherical_bessel_j


def _inner_prod(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))


@dataclass
class StructureMatrices:
    """Energy-independent matrices A/B/C for a single k-point."""

    params: APWParameters
    k_vectors: np.ndarray  # shape (number_k, 3)

    def build(self, k_point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.params.number_k
        lmax = self.params.max_l
        mat_a = np.zeros((n, n))
        mat_b = np.zeros((n, n))
        mat_c = np.zeros((n, n, lmax + 1))

        prefac = (
            2.0
            * np.pi
            * self.params.muffin_tin_radius
            * self.params.muffin_tin_radius
            / self.params.unit_cell_volume
        )
        rec_lat = self.params.reciprocal_lattice_constant

        for i in range(n):
            kv_i = self.k_vectors[i]
            for j in range(i + 1):
                kv_j = self.k_vectors[j]
                delta = kv_i - kv_j
                dist = np.sqrt(cubic_metric(delta)) * rec_lat
                k_tot = self._k_total(k_point, kv_i)
                q_tot = self._k_total(k_point, kv_j)
                k_dot_q = _inner_prod(k_tot, q_tot) * rec_lat * rec_lat
                k_norm = np.sqrt(_inner_prod(k_tot, k_tot)) * rec_lat
                q_norm = np.sqrt(_inner_prod(q_tot, q_tot)) * rec_lat

                x = dist * self.params.muffin_tin_radius
                if x > 1e-8:
                    j1 = ((np.sin(x) / x) - np.cos(x)) / x
                    j1 /= dist
                else:
                    j1 = self.params.muffin_tin_radius / 3.0
                mat_a[i, j] = -2.0 * prefac * j1
                mat_b[i, j] = mat_a[i, j] * 0.5 * k_dot_q

                arg1 = q_norm * self.params.muffin_tin_radius
                arg2 = k_norm * self.params.muffin_tin_radius
                if k_norm > 1e-8 and q_norm > 1e-8:
                    arg3 = k_dot_q / (k_norm * q_norm)
                else:
                    arg3 = 1.0 - 1e-10

                for ell in range(lmax + 1):
                    jl1 = (
                        spherical_bessel_j(ell, arg1)
                        if arg1 > 1e-8
                        else (1.0 if ell == 0 else 0.0)
                    )
                    jl2 = (
                        spherical_bessel_j(ell, arg2)
                        if arg2 > 1e-8
                        else (1.0 if ell == 0 else 0.0)
                    )
                    mat_c[i, j, ell] = (2 * ell + 1) * prefac * legendre_p(
                        ell, arg3
                    ) * jl1 * jl2
                if i == j:
                    mat_a[i, i] += 1.0
                    mat_b[i, i] += 0.5 * k_dot_q
                else:
                    mat_a[j, i] = mat_a[i, j]
                    mat_b[j, i] = mat_b[i, j]
                    mat_c[j, i, :] = mat_c[i, j, :]
        return mat_a, mat_b, mat_c

    @staticmethod
    def _k_total(k_point: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
        return np.array(
            [
                k_point[0] - k_vec[0] + k_vec[1] + k_vec[2],
                k_point[1] + k_vec[0] - k_vec[1] + k_vec[2],
                k_point[2] + k_vec[0] + k_vec[1] - k_vec[2],
            ],
            dtype=float,
        )


def build_hamiltonian(
    energy: float,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    mat_c: np.ndarray,
    chi_r: np.ndarray,
    chi_dot: np.ndarray,
) -> np.ndarray:
    """Return the APW Hamiltonian H(E) = -E A + B + sum_l C_l * (chi'/chi)."""

    ratios = np.divide(
        chi_dot,
        chi_r,
        out=np.zeros_like(chi_dot),
        where=np.abs(chi_r) > 1e-14,
    )
    h_matrix = -energy * mat_a + mat_b
    for ell in range(ratios.size):
        h_matrix = h_matrix + mat_c[:, :, ell] * ratios[ell]
    # force Hermiticity to counter tiny FP drift
    return 0.5 * (h_matrix + h_matrix.T)


def determinant(h_matrix: np.ndarray) -> float:
    """Return det(H-EI) via direct LU (mirrors DGETRF)."""

    val = np.linalg.det(h_matrix)
    return float(np.real_if_close(val))
