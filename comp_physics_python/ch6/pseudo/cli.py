"""CLI for the semi-empirical silicon pseudopotential solver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

from .constants import PseudoParameters
from .spectrum import PseudoSolver


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-empirical pseudopotential band structure (Thijssen §6.7.1)."
    )
    parser.add_argument(
        "--line",
        default="L-Gamma",
        choices=["L-Gamma", "Gamma-X", "X-U", "K-Gamma"],
        help="High-symmetry line to sample inside the fcc Brillouin zone.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=8,
        help="Number of bands to print/save at each k-point (<= 10).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="Override the default IBZ step length (0.03).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file receiving (segment,kx,ky,kz,band,energy) rows.",
    )
    return parser.parse_args(argv)


def pick_line(params: PseudoParameters, label: str) -> Tuple:
    lookup = {
        "L-Gamma": (params.l_point, params.gamma_point),
        "Gamma-X": (params.gamma_point, params.x_point),
        "X-U": (params.x_point, params.u_point),
        "K-Gamma": (params.k_point, params.gamma_point),
    }
    return lookup[label]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    params = PseudoParameters()
    if args.step is not None:
        params = PseudoParameters(step_length=args.step)
    solver = PseudoSolver(params)
    first, second = pick_line(params, args.line)
    samples = list(solver.line_spectrum(first, second))

    writer = None
    csv_file = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.output.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["segment", "kx", "ky", "kz", "band_index", "energy_Hartree"])

    try:
        for sample in samples:
            header = (
                f"k[{sample.index:03d}] (|Δk|={sample.arclength:.4f}): "
                f"{sample.k_point[0]:+.5f} {sample.k_point[1]:+.5f} {sample.k_point[2]:+.5f}"
            )
            print(header)
            for band_idx, energy in enumerate(sample.energies[: args.levels], start=1):
                print(f"  band {band_idx:02d}: {energy: .6f} Ha")
                if writer is not None:
                    writer.writerow(
                        [
                            sample.index,
                            f"{sample.k_point[0]:.8f}",
                            f"{sample.k_point[1]:.8f}",
                            f"{sample.k_point[2]:.8f}",
                            band_idx,
                            f"{energy:.10f}",
                        ]
                    )
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":  # pragma: no cover
    main()
