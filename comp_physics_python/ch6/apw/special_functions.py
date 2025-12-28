"""Spherical Bessel and Legendre polynomials via upward recursion."""

from __future__ import annotations

import math
from functools import lru_cache


def _safe_div(num: float, denom: float) -> float:
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Denominator vanished in special function helper")
    return num / denom


@lru_cache(maxsize=None)
def spherical_bessel_j(l: int, x: float) -> float:
    """Return j_l(x) with the same recurrence as ``special.f``."""

    if l < 0:
        raise ValueError("Order l must be non-negative")
    if abs(x) < 1e-8:
        if l == 0:
            return 1.0
        if l == 1:
            return x / 3.0
        return 0.0
    if l == 0:
        return math.sin(x) / x
    if l == 1:
        return math.sin(x) / (x * x) - math.cos(x) / x
    j_l_minus_2 = math.sin(x) / x
    j_l_minus_1 = math.sin(x) / (x * x) - math.cos(x) / x
    for n in range(2, l + 1):
        j_l = ((2 * n - 1) / x) * j_l_minus_1 - j_l_minus_2
        j_l_minus_2, j_l_minus_1 = j_l_minus_1, j_l
    return j_l_minus_1


@lru_cache(maxsize=None)
def legendre_p(l: int, x: float) -> float:
    """Return P_l(x) using the textbook upward recursion."""

    if l < 0:
        raise ValueError("Order l must be non-negative")
    if l == 0:
        return 1.0
    if l == 1:
        return x
    p_l_minus_2 = 1.0
    p_l_minus_1 = x
    for n in range(2, l + 1):
        p_l = ((2 * n - 1) * x * p_l_minus_1 - (n - 1) * p_l_minus_2) / n
        p_l_minus_2, p_l_minus_1 = p_l_minus_1, p_l
    return p_l_minus_1
