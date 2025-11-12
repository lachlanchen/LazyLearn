"""Hydrogen atom in a minimal Gaussian basis (Thijssen Chapter 3.2.2)."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from comp_physics_python.ch3._linalg import generalized_eigh  # type: ignore
else:
    from ._linalg import generalized_eigh


def _default_alphas() -> tuple[float, ...]:
    return (13.00773, 1.962079, 0.444529, 0.1219492)


@dataclass
class HydrogenGaussianConfig:
    Z: float = 1.0
    alphas: tuple[float, ...] = field(default_factory=_default_alphas)

    def __post_init__(self) -> None:
        if self.Z <= 0:
            raise ValueError("Z must be positive")
        # Scale exponents with Z^2 as in the FORTRAN program
        z2 = self.Z * self.Z
        self.alphas = tuple(a * z2 for a in self.alphas)
        self.pi = float(np.pi)

    @property
    def n(self) -> int:
        return len(self.alphas)


def overlap_matrix(cfg: HydrogenGaussianConfig) -> np.ndarray:
    n = cfg.n
    S = np.zeros((n, n), dtype=float)
    for r in range(n):
        for s in range(n):
            denom = cfg.alphas[r] + cfg.alphas[s]
            factor = cfg.pi / denom
            S[r, s] = factor * np.sqrt(factor)
    return S


def kinetic_element(a: float, b: float, pi: float) -> float:
    alph = a + b
    factor = pi / alph
    return 3.0 * factor * np.sqrt(factor) * a * b / alph


def coulomb_element(a: float, b: float, pi: float, Z: float) -> float:
    return -2.0 * Z * pi / (a + b)


def hamilton_matrix(cfg: HydrogenGaussianConfig) -> np.ndarray:
    n = cfg.n
    H = np.zeros((n, n), dtype=float)
    for r in range(n):
        for s in range(n):
            H[r, s] = kinetic_element(cfg.alphas[r], cfg.alphas[s], cfg.pi) + coulomb_element(
                cfg.alphas[r], cfg.alphas[s], cfg.pi, cfg.Z
            )
    return H


def solve_hydrogen(cfg: HydrogenGaussianConfig) -> tuple[np.ndarray, np.ndarray]:
    H = hamilton_matrix(cfg)
    S = overlap_matrix(cfg)
    evals, evecs = generalized_eigh(H, S)
    return evals, evecs


def radial_wavefunction(cfg: HydrogenGaussianConfig, coeffs: np.ndarray, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    psi = np.zeros_like(r)
    for alpha, c in zip(cfg.alphas, coeffs):
        psi += c * np.exp(-alpha * r * r)
    return psi


def run_cli(args: argparse.Namespace) -> None:
    cfg = HydrogenGaussianConfig(Z=args.Z)
    evals, evecs = solve_hydrogen(cfg)
    exact = -0.5 * cfg.Z * cfg.Z

    print(f"Lowest eigenvalue (variational) = {evals[0]: .8f}")
    print(f"Exact energy                     = {exact: .8f}")

    grid = np.linspace(0.0, 4.0 / cfg.Z, args.samples)
    psi = radial_wavefunction(cfg, evecs[:, 0], grid)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([grid, psi]), fmt="%12.6f")
    print(f"Ground-state radial wave function sampled to {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Z", type=float, default=1.0, help="nuclear charge")
    parser.add_argument("--samples", type=int, default=101, help="number of radial points for the output wave function")
    parser.add_argument("--out", type=str, default="WaveFunc_py", help="output file (matches FORTRAN default)")
    return parser


if __name__ == "__main__":
    run_cli(build_arg_parser().parse_args())
