"""Reciprocal-lattice helpers: K-vector list + IBZ line sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


def load_k_vectors(path: Path, number: int) -> np.ndarray:
    """Return the first ``number`` rows from the KVectors table."""

    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            rows.append([int(parts[0]), int(parts[1]), int(parts[2])])
            if len(rows) == number:
                break
    if len(rows) < number:
        raise ValueError(
            f"KVectors file {path} contained {len(rows)} rows, need {number}"
        )
    return np.asarray(rows, dtype=int)


def cubic_metric(delta: np.ndarray) -> float:
    """Return the metric used in FillDist (squared fcc reciprocal length)."""

    i, j, k = delta
    return float(3 * (i * i + j * j + k * k) - 2 * (i * j + j * k + i * k))


def interpolate_line(
    first_point: np.ndarray,
    second_point: np.ndarray,
    step_length: float,
) -> Iterable[np.ndarray]:
    """Yield equally spaced points between two IBZ locations."""

    diff = second_point - first_point
    length = np.linalg.norm(diff)
    num = max(1, int(round(length / step_length)))
    steps = diff / num
    for t in range(num + 1):
        yield first_point + t * steps
