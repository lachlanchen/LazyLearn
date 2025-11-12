"""Small linear-algebra helpers for the chapter 3 translations."""

from __future__ import annotations

import numpy as np


def generalized_eigh(h_mat: np.ndarray, s_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve H c = E S c for symmetric H and SPD S.

    The implementation mirrors the LAPACK routine used by the FORTRAN sources by
    first performing a Cholesky factorisation of ``S`` and then solving the
    reduced standard eigenvalue problem ``L^{-1} H L^{-T}``.

    Returns
    -------
    eigenvalues : ndarray, shape (n,)
        Sorted ascending.
    eigenvectors : ndarray, shape (n, n)
        Columns contain the corresponding coefficients in the original basis.
    """

    h = np.array(h_mat, dtype=float, copy=True)
    s = np.array(s_mat, dtype=float, copy=True)

    if h.shape != s.shape or h.shape[0] != h.shape[1]:
        raise ValueError("H and S must be square matrices of the same size")

    # Cholesky factorisation S = L L^T
    L = np.linalg.cholesky(s)
    L_inv = np.linalg.inv(L)

    reduced = L_inv @ h @ L_inv.T
    evals, evecs_std = np.linalg.eigh(reduced)

    evecs = L_inv.T @ evecs_std

    # Normalise the eigenvectors with respect to S so that c^T S c = 1
    gram = evecs.T @ s @ evecs
    norms = np.sqrt(np.clip(np.diag(gram), a_min=1e-15, a_max=None))
    evecs /= norms

    return evals, evecs
