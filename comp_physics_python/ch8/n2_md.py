"""Rigid-diatomic molecular dynamics for N2 (Thijssen Problem 8.7)."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from comp_physics_python.ch8.common import (
        MDConfig,
        make_fcc_lattice,
        maxwell_velocities,
        minimum_image,
        velocity_verlet,
    )  # type: ignore
else:  # pragma: no cover
    from .common import MDConfig, make_fcc_lattice, maxwell_velocities, minimum_image, velocity_verlet


@dataclass
class N2Config(MDConfig):
    bond_length: float = 1.1
    bond_k: float = 5.0


def initialise_pairs(config: N2Config) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = config.n_atoms
    if n_atoms % 2 != 0:
        raise ValueError("n_atoms must be even (two atoms per molecule)")
    com = make_fcc_lattice(n_atoms // 2, config.box_length)
    orientations = maxwell_velocities(n_atoms // 2, 1.0)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    pos = np.zeros((n_atoms, 3))
    pos[0::2] = (com - 0.5 * config.bond_length * orientations) % config.box_length
    pos[1::2] = (com + 0.5 * config.bond_length * orientations) % config.box_length
    vel = maxwell_velocities(n_atoms, config.temperature)
    return pos, vel


def intermolecular_forces(positions: np.ndarray, box: float, cutoff: float) -> tuple[np.ndarray, float]:
    n_atoms = positions.shape[0]
    forces = np.zeros_like(positions)
    pot = 0.0
    cutoff2 = cutoff * cutoff
    for i in range(n_atoms - 1):
        disp = positions[i + 1 :] - positions[i]
        disp = minimum_image(disp, box)
        r2 = np.sum(disp * disp, axis=1)
        mask = r2 < cutoff2
        if not np.any(mask):
            continue
        r2 = r2[mask]
        vec = disp[mask]
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        force_scalar = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
        forces[i] += np.sum(force_scalar[:, None] * vec, axis=0)
        forces[i + 1 :][mask] -= force_scalar[:, None] * vec
        pot += np.sum(4.0 * (inv_r12 - inv_r6))
    return forces, pot


def bond_forces(positions: np.ndarray, config: N2Config) -> tuple[np.ndarray, float]:
    forces = np.zeros_like(positions)
    energy = 0.0
    for i in range(0, config.n_atoms, 2):
        r12 = positions[i + 1] - positions[i]
        r12 = minimum_image(r12, config.box_length)
        dist = np.linalg.norm(r12)
        stretch = dist - config.bond_length
        if dist == 0:
            continue
        f_mag = -config.bond_k * stretch / dist
        f_vec = f_mag * r12
        forces[i] -= f_vec
        forces[i + 1] += f_vec
        energy += 0.5 * config.bond_k * stretch * stretch
    return forces, energy


def run_n2_md(config: N2Config, seed: int | None = None) -> dict:
    positions, velocities = initialise_pairs(config)
    inter_forces, pot = intermolecular_forces(positions, config.box_length, config.cutoff)
    bond_force, bond_energy = bond_forces(positions, config)
    forces = inter_forces + bond_force

    def integrate(steps: int, rescale: bool):
        nonlocal forces, pot, bond_energy
        obs = []
        for step in range(1, steps + 1):
            positions[:], velocities[:] = velocity_verlet(
                positions, velocities, forces, config.time_step, config.box_length
            )
            inter_forces, pot = intermolecular_forces(positions, config.box_length, config.cutoff)
            bond_force, bond_energy = bond_forces(positions, config)
            new_forces = inter_forces + bond_force
            velocities[:] += 0.5 * (forces + new_forces) * config.time_step
            forces = new_forces
            kin = 0.5 * np.sum(velocities * velocities)
            obs.append((kin, pot + bond_energy))
            if rescale and step % 50 == 0:
                velocities[:] = maxwell_velocities(config.n_atoms, config.temperature)
        return obs

    integrate(config.equilibration_steps, True)
    prod = integrate(config.production_steps, False)
    kin = np.array([k for k, _ in prod])
    pot = np.array([p for _, p in prod])
    T = 2 * kin / (3 * config.n_atoms)
    return {
        "temperature": T.mean(),
        "potential": pot.mean(),
        "kinetic": kin.mean(),
        "positions": positions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512, help="number of atoms (even)")
    parser.add_argument("--density", type=float, default=0.7)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--equil", type=int, default=500)
    parser.add_argument("--prod", type=int, default=2000)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--bond", type=float, default=1.1)
    args = parser.parse_args(argv)

    cfg = N2Config(
        n_atoms=args.n,
        density=args.density,
        temperature=args.temp,
        time_step=args.dt,
        equilibration_steps=args.equil,
        production_steps=args.prod,
        cutoff=args.cutoff,
        bond_length=args.bond,
    )
    result = run_n2_md(cfg)
    print(f"Average temperature {result['temperature']:.3f}")
    print(f"Average potential {result['potential']:.3f}")


if __name__ == "__main__":
    main()
