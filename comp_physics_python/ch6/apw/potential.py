"""Logarithmic-grid muffin-tin potential loader (``potential`` file)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .numerov import numerov


def _read_potential_file(path: Path) -> tuple[float, float, int, np.ndarray]:
    text = path.read_text(encoding="utf-8").replace("D", "E")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Potential file {path} is empty")
    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"First line of {path} missing metadata")
    r_mt = float(header[0])
    delta = float(header[1])
    pot_num = int(float(header[2]))
    body = " ".join(lines[1:])
    values = np.fromstring(body, sep=" ")
    if values.size < pot_num:
        raise ValueError(
            f"Potential file {path} has {values.size} entries, needs {pot_num}"
        )
    return r_mt, delta, pot_num, values[:pot_num]


@dataclass
class LogPotential:
    """Encapsulates the logarithmic potential table used by ``logapw.f``."""

    radius: float
    delta: float
    pot_num: int
    potential: np.ndarray  # 1-indexed; entry 0 reserved
    int_points: np.ndarray
    deriv_points: np.ndarray
    mult_array: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "LogPotential":
        r_mt, delta, pot_num, values = _read_potential_file(path)
        potential = np.zeros(pot_num + 1)
        potential[1:] = values
        int_points = np.zeros(pot_num + 1)
        deriv_points = np.zeros(pot_num + 1)
        mult_array = np.zeros(pot_num + 1)

        r0 = r_mt / (np.exp(delta * (pot_num - 1)) - 1.0)
        for i in range(1, pot_num + 1):
            expon = np.exp(delta * (i - 1))
            int_points[i] = r0 * (expon - 1.0)
            deriv_points[i] = (delta * expon * r0) ** 2
            mult_array[i] = np.sqrt(expon)
        return cls(
            radius=r_mt,
            delta=delta,
            pot_num=pot_num,
            potential=potential,
            int_points=int_points,
            deriv_points=deriv_points,
            mult_array=mult_array,
        )

    def fill_f_array(self, ell: int, energy: float) -> np.ndarray:
        """Return the Numerov RHS array F(r, ell, E) on the log grid."""

        f_arr = np.zeros(self.pot_num + 1)
        delta_sq = self.delta * self.delta
        safe_r = np.clip(self.int_points, 1e-12, None)
        lterm = ell * (ell + 1.0)
        f_arr[2:] = self.deriv_points[2:] * (
            lterm / (safe_r[2:] ** 2) - 2.0 * (self.potential[2:] / safe_r[2:] + energy)
        )
        f_arr[2:] += 0.25 * delta_sq
        return f_arr

    def atom(
        self,
        energy: float,
        ell: int,
        max_sol: int,
    ) -> tuple[float, float]:
        """Replicate ``Atom``: return (phi(R_MT), phi'(R_MT))."""

        f_arr = np.zeros(max_sol + 2)
        vals = self.fill_f_array(ell, energy)
        f_arr[: vals.size] = vals

        phi_start = self.int_points[2] ** (ell + 1) / self.mult_array[2]
        phi_next = self.int_points[3] ** (ell + 1) / self.mult_array[3]
        sol = numerov(
            delta=1.0,
            start_i=2,
            end_i=self.pot_num,
            f_array=f_arr,
            phi_start=phi_start,
            phi_next=phi_next,
            singular=False,
        )
        sol[self.pot_num] *= self.mult_array[self.pot_num]
        sol[self.pot_num - 1] *= self.mult_array[self.pot_num - 1]
        phi_max = sol[self.pot_num] / self.radius
        step = self.int_points[self.pot_num] - self.int_points[self.pot_num - 1]
        deriv = (sol[self.pot_num] - sol[self.pot_num - 1]) / step
        deriv += 0.125 * step * (
            3.0 * f_arr[self.pot_num] * sol[self.pot_num]
            + f_arr[self.pot_num - 1] * sol[self.pot_num - 1]
        )
        deriv = (deriv - phi_max) / self.radius
        return phi_max, deriv

    def atom_integrals(self, energy: float, max_l: int, max_sol: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ChiR[l], ChiDotR[l] arrays for 0 <= l <= max_l."""

        chi_r = np.zeros(max_l + 1)
        chi_dot = np.zeros(max_l + 1)
        for ell in range(max_l + 1):
            chi_r[ell], chi_dot[ell] = self.atom(energy, ell, max_sol)
        return chi_r, chi_dot
