"""Variational solver for the infinite square well (Thijssen Chapter 3.2.1).

The original FORTRAN files under ``comp_physics_textbook_code/4557_ch3/ch3/deepwell``
construct polynomial basis functions \(\phi_k(x) = (x^2-1)x^k\) on the interval
``x ∈ [-1, 1]`` and evaluate the Hamiltonian/overlap integrals analytically.  The
resulting generalised eigenvalue problem reproduces the familiar
\(E_n = n^2\pi^2/4\) spectrum as the basis size grows.  This module ports that
workflow to Python.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from comp_physics_python.ch3._linalg import generalized_eigh  # type: ignore
else:
    from ._linalg import generalized_eigh


@dataclass
class DeepWellConfig:
    """Configuration of the polynomial basis."""

    n_basis: int = 12
    grid_points: int = 60  # number of x samples per eigenfunction when exporting

    def __post_init__(self) -> None:
        if self.n_basis < 2:
            raise ValueError("Need at least two basis functions for the variational ansatz")
        if self.grid_points < 2:
            raise ValueError("grid_points must be >= 2")


def _overlap_element(i: int, j: int) -> float:
    s = i + j
    if s % 2 != 0:
        return 0.0
    return 2.0 / (s + 5.0) - 4.0 / (s + 3.0) + 2.0 / (s + 1.0)


def _hamilton_element(i: int, j: int) -> float:
    s = i + j
    if s % 2 != 0:
        return 0.0
    return 8.0 * (i + j + 2.0 * i * j - 1.0) / ((s + 3.0) * (s + 1.0) * (s - 1.0))


def build_matrices(cfg: DeepWellConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = cfg.n_basis
    S = np.zeros((n, n), dtype=float)
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            S[i, j] = _overlap_element(i, j)
            H[i, j] = _hamilton_element(i, j)
    return H, S


def solve_deep_well(cfg: DeepWellConfig) -> tuple[np.ndarray, np.ndarray]:
    H, S = build_matrices(cfg)
    evals, evecs = generalized_eigh(H, S)
    return evals, evecs


def basis_functions(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    basis = np.empty((n, x.size), dtype=float)
    poly = x * x - 1.0
    for k in range(n):
        basis[k] = np.power(x, k) * poly
    return basis


def eigenfunctions_on_grid(evecs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    phi = basis_functions(grid, evecs.shape[0])
    return evecs.T @ phi


def write_eigenfunctions(path: Path, grid: np.ndarray, psi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i in range(psi.shape[0]):
            for x, val in zip(grid, psi[i]):
                fh.write(f"{x: .10f} {val: .10f}\n")
            fh.write("\n")


def run_cli(args: argparse.Namespace) -> None:
    cfg = DeepWellConfig(n_basis=args.n_basis, grid_points=args.grid_points)
    evals, evecs = solve_deep_well(cfg)

    pi = np.pi
    print("Variational    Exact")
    for idx, val in enumerate(evals):
        exact = pi * pi / 4.0 * (idx + 1) ** 2
        print(f"{val:10.6f}  {exact:10.6f}")

    grid = np.linspace(-1.0, 1.0, cfg.grid_points * 2 + 1)
    psi = eigenfunctions_on_grid(evecs, grid)
    write_eigenfunctions(Path(args.out), grid, psi)
    print(f"Eigenfunctions written to {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-basis", type=int, default=12, help="number of polynomial basis states (N in the book)")
    parser.add_argument(
        "--grid-points", type=int, default=60, help="half the number of grid points per eigenfunction (matches FORTRAN PtNum)"
    )
    parser.add_argument("--out", type=str, default="EigVecs_py", help="output file for eigenfunctions")
    return parser


if __name__ == "__main__":
    run_cli(build_arg_parser().parse_args())
