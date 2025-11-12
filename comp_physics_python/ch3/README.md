# Chapter 3 – Variational examples in Python

This folder reimplements the Section 3 examples from Thijssen’s *Computational
Physics* using modern Python/numpy tooling.  The goal is a line‑for‑line remake
of the original FORTRAN routines under
`comp_physics/comp_physics_textbook_code/4557_ch3/ch3`, plus extensive notes on
what each formula does.

## 1. Infinite square well with polynomial basis (`deep_well_variational.py`)

The FORTRAN program `deepwell.f` spans the 1D infinite well on the interval
`x ∈ [-1, 1]` with basis functions

\[
\phi_k(x) = (x^2 - 1) x^k, \qquad k = 0, …, N-1.
\]

Because the Hamiltonian and overlap integrals are analytic, the matrix elements
can be written explicitly (and vanish when \(k + k'\) is odd):

\[
S_{kk'} = \int_{-1}^{1} \phi_k \phi_{k'}\,\mathrm{d}x
         = \frac{2}{k+k'+5} - \frac{4}{k+k'+3} + \frac{2}{k+k'+1},
\]

\[
H_{kk'} = \int_{-1}^{1} \phi_k (-\tfrac{1}{2}\nabla^2)\phi_{k'}\,\mathrm{d}x
         = \frac{8\,(k+k' + 2kk' - 1)}{(k+k'+3)(k+k'+1)(k+k'-1)}.
\]

The generalised eigenvalue problem `H c = E S c` is solved through an explicit
Cholesky reduction (mirroring the LAPACK call hidden inside `geneig.f`).  The
returned eigenvalues approach the exact \(E_n = \frac{\pi^2}{4}(n+1)^2\) as `n_basis`
 grows, and `eigenfunctions_on_grid` reconstructs the spatial wavefunctions just
like the FORTRAN loop that wrote `EigVecs`.

Usage:

```bash
python comp_physics_python/ch3/deep_well_variational.py --n-basis 16 --grid-points 80 --out EigVecs_py
```

Inside Python:

```python
from comp_physics_python.ch3.deep_well_variational import DeepWellConfig, solve_deep_well
cfg = DeepWellConfig(n_basis=14)
evals, evecs = solve_deep_well(cfg)
```

## 2. Hydrogen atom with Gaussian primitives (`hydrogen_gaussians.py`)

`Hatom.f` constructs a four‑function Gaussian basis for the hydrogen atom and
computes the Hamiltonian/overlap matrices analytically as well:

- overlap
  \(S_{rs} = \pi / (\alpha_r + \alpha_s) \sqrt{\pi/(\alpha_r + \alpha_s)}\),
- kinetic
  \(T_{rs} = 3 S_{rs} \alpha_r \alpha_s / (\alpha_r + \alpha_s)\),
- Coulomb
  \(V_{rs} = -2 Z \pi / (\alpha_r + \alpha_s)\).

The exponents \(\alpha_r\) are scaled by \(Z^2\) so that different hydrogenic
ions can be handled without changing the primitive set.  The same Cholesky based
solver provides the variational eigenvalues and eigenvectors.  The ground-state
wave function is sampled on `r ∈ [0, 4/Z]` via
\(\psi(r) = \sum_r c_r e^{-\alpha_r r^2}\) and written to `WaveFunc_py`, matching
Thijssen’s output format.

Usage:

```bash
python comp_physics_python/ch3/hydrogen_gaussians.py --Z 1.0 --samples 201 --out WaveFunc_py
```

From Python:

```python
from comp_physics_python.ch3.hydrogen_gaussians import HydrogenGaussianConfig, solve_hydrogen
cfg = HydrogenGaussianConfig(Z=2.0)
evals, evecs = solve_hydrogen(cfg)
```

## Linear algebra helper (`_linalg.py`)

Both scripts call `generalized_eigh`, a tiny wrapper around `numpy.linalg` that
performs a Cholesky factorisation of the overlap matrix and reduces the
problem to a standard eigenvalue solve.  This mimics the LAPACK `dsygv` call in
`geneig.f` while keeping the code dependency‑free.
