"""Molecular dynamics for argon in Lennard-Jones units (Thijssen §8.4)."""

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
class Observables:
    kinetic: float
    potential: float
    virial: float


def lennard_jones_forces(positions: np.ndarray, box: float, cutoff: float) -> tuple[np.ndarray, Observables]:
    n_atoms = positions.shape[0]
    forces = np.zeros_like(positions)
    pot = 0.0
    virial = 0.0
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
        virial += np.sum(force_scalar * r2)
    return forces, Observables(kinetic=0.0, potential=pot, virial=virial)


def run_md(
    config: MDConfig,
    seed: int | None = None,
    dump_stride: int = 100,
) -> dict:
    positions = make_fcc_lattice(config.n_atoms, config.box_length)
    velocities = maxwell_velocities(config.n_atoms, config.temperature, seed=seed)
    forces, _ = lennard_jones_forces(positions, config.box_length, config.cutoff)

    def integrate(n_steps: int, scale_vel: bool) -> list[Observables]:
        obs_list: list[Observables] = []
        for step in range(1, n_steps + 1):
            positions[:], velocities[:] = velocity_verlet(
                positions, velocities, forces, config.time_step, config.box_length
            )
            new_forces, obs = lennard_jones_forces(positions, config.box_length, config.cutoff)
            velocities[:] += 0.5 * (forces + new_forces) * config.time_step
            forces[:] = new_forces
            obs.kinetic = 0.5 * np.sum(velocities * velocities)
            obs_list.append(obs)
            if scale_vel and step % 50 == 0:
                velocities[:] = maxwell_velocities(config.n_atoms, config.temperature, seed=None)
        return obs_list

    integrate(config.equilibration_steps, scale_vel=True)
    production = integrate(config.production_steps, scale_vel=False)
    kinetic = np.array([obs.kinetic for obs in production])
    potential = np.array([obs.potential for obs in production])
    total = kinetic + potential
    temperature = (2.0 * kinetic) / (3.0 * config.n_atoms)
    pressure = (
        temperature * config.n_atoms + np.array([obs.virial for obs in production]) / 3.0
    ) / (config.box_length**3)

    return {
        "temperature": temperature.mean(),
        "kinetic": kinetic.mean(),
        "potential": potential.mean(),
        "total": total.mean(),
        "pressure": pressure.mean(),
        "positions": positions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256, help="number of atoms (must be 4*cells^3)")
    parser.add_argument("--density", type=float, default=0.8442)
    parser.add_argument("--temp", type=float, default=0.722)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--equil", type=int, default=500)
    parser.add_argument("--prod", type=int, default=2000)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = MDConfig(
        n_atoms=args.n,
        density=args.density,
        temperature=args.temp,
        time_step=args.dt,
        equilibration_steps=args.equil,
        production_steps=args.prod,
        cutoff=args.cutoff,
    )
    result = run_md(cfg, seed=args.seed)
    print(f"Average T = {result['temperature']:.3f}")
    print(f"Average potential = {result['potential']:.3f}")
    print(f"Average total = {result['total']:.3f}")
    out = pathlib.Path("ar_final_positions.npy")
    np.save(out, result["positions"])
    print(f"Saved final configuration to {out}")


if __name__ == "__main__":
    main()
