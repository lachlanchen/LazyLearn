"""Command-line interface for the Chapter 6 APW solver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

from .constants import APWParameters
from .kpoints import load_k_vectors
from .potential import LogPotential
from .spectrum import APWSolver


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augmented Plane Wave band-structure solver for the copper example "
            "in Thijssen's Computational Physics (Sec. 6.5.2)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing potential/KVectors (defaults to the FORTRAN tree).",
    )
    parser.add_argument(
        "--line",
        default="Gamma-K",
        choices=["Gamma-K", "Gamma-X", "X-W", "X-U", "K-Gamma", "L-Gamma"],
        help="High-symmetry path to scan inside the fcc Brillouin zone.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of energy samples per determinant scan (default 100).",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="Lower bound of the determinant scan window in Hartree.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="Upper bound of the determinant scan window in Hartree.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="IBZ step length used when interpolating between k-points (default 0.02).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file storing (segment_index, kx, ky, kz, energy) rows.",
    )
    return parser.parse_args(argv)


def build_params_and_solver(args: argparse.Namespace) -> Tuple[APWParameters, APWSolver]:
    base = APWParameters()
    data_root = (args.data_root or base.data_root).expanduser().resolve()
    potential = LogPotential.from_file(data_root / "potential")
    kwargs = {
        "data_root": data_root,
        "muffin_tin_radius": potential.radius,
    }
    if args.samples:
        kwargs["energy_samples"] = args.samples
    if args.energy_min is not None:
        kwargs["energy_min"] = args.energy_min
    if args.energy_max is not None:
        kwargs["energy_max"] = args.energy_max
    if args.step is not None:
        kwargs["step_length"] = args.step
    params = APWParameters(**kwargs)
    k_vectors = load_k_vectors(data_root / "KVectors", params.number_k)
    solver = APWSolver(
        params=params,
        potential=potential,
        k_vectors=k_vectors,
    )
    return params, solver


def pick_line(params: APWParameters, label: str) -> Tuple:
    lookup = {
        "Gamma-K": (params.gamma_point, params.k_point),
        "Gamma-X": (params.gamma_point, params.x_point),
        "X-W": (params.x_point, params.w_point),
        "X-U": (params.x_point, params.u_point),
        "K-Gamma": (params.k_point, params.gamma_point),
        "L-Gamma": (params.l_point, params.gamma_point),
    }
    if label not in lookup:
        raise ValueError(f"Unknown high-symmetry line {label}")
    return lookup[label]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    params, solver = build_params_and_solver(args)
    first, second = pick_line(params, args.line)
    samples = list(
        solver.line_spectrum(
            first,
            second,
            step_length=args.step,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
            samples=args.samples,
        )
    )

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.output.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["segment", "kx", "ky", "kz", "energy_Hartree"])
    else:
        csv_file = None

    try:
        for sample in samples:
            header = (
                f"k[{sample.index:03d}] (|Δk|={sample.arclength:.4f}): "
                f"{sample.k_point[0]:+.5f} {sample.k_point[1]:+.5f} {sample.k_point[2]:+.5f}"
            )
            print(header)
            if not sample.energies:
                print("  (no roots/minima within the chosen energy window)")
                continue
            for idx, energy in enumerate(sample.energies, start=1):
                print(f"  band {idx:02d}: {energy: .6f} Ha")
                if writer is not None:
                    writer.writerow(
                        [
                            sample.index,
                            f"{sample.k_point[0]:.8f}",
                            f"{sample.k_point[1]:.8f}",
                            f"{sample.k_point[2]:.8f}",
                            f"{energy:.10f}",
                        ]
                    )
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":  # pragma: no cover
    main()
