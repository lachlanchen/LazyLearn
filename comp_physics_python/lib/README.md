# Shared Library (4570\_lib)

`4570_lib/lib/src` hosts the reusable building blocks referenced throughout the
textbook:

| Folder | Purpose |
| --- | --- |
| `matrix/` | Dense/sparse linear algebra utilities (Gauss elimination, eigen-decompositions, Gram–Schmidt). |
| `numerov/` | The general Numerov integrator used from Chapters 2, 5, and 6. |
| `rangen/` | Combined linear-congruential + shift-register random-number generators. |
| `XPS/`, `XPS_1/` | Pseudopotential support (Angular integrals, Bessel transforms) for X-ray photoelectron simulations. |

The Python port keeps these in `comp_physics_python.lib` so all chapter modules
can import the same tested routines.

## 1. Linear algebra helpers (`matrix/`)

Key routines:

* `gauss.f` – Gaussian elimination with partial pivoting, solving $A x = b$ and
  returning the determinant (used in §6 and §11).
* `jacobi.f` – Jacobi eigenvalue algorithm for symmetric matrices (used in the
  transfer-matrix chapter).
* `gramschmidt.f` – Modified Gram–Schmidt orthonormalisation, required by the
  Car–Parrinello constraint solver.

Python counterparts will live in `matrix_utils.py`, wrapping NumPy/SciPy while
keeping fallbacks for environments without BLAS.

## 2. Numerov integrator (`numerov/`)

`numerov.f` integrates second-order linear ODEs
$y''(x) = f(x) y(x)$ on arbitrary grids with optional singular handling at the
origin. Chapter 6 already reuses this routine verbatim; the general template is

$$
w_{n+1} = 2 w_n - w_{n-1} + h^2 f(x_n) y_n, \qquad
y_{n+1} = \frac{w_{n+1}}{1 - \tfrac{h^2}{12} f(x_{n+1})},
$$

with $w_n = \left(1 - \tfrac{h^2}{12} f(x_n)\right) y_n$. The Python
implementation (`numerov.py`) already exists and matches the FORTRAN interface.

## 3. Random-number generators (`rangen/`)

`rangen` implements:

* **L’Ecuyer combined LCG:** period ≈ $2^{61}$, used in Monte Carlo chapters.
* **Shift-register generator:** XOR feedback on 32-bit words for decorrelated
  streams.
* **Normal deviates:** Box–Muller transform and the polar (Marsaglia) method.

Python’s `numpy.random.Generator` will act as the default backend, but the port
will also include a faithful reproduction of the LCG so regression tests can
match the original bit sequences.

## 4. XPS support libraries (`XPS/`, `XPS_1/`)

These directories supply:

* Radial Bessel transforms $\int_0^\infty r^2 R_{nl}(r) j_l(kr) \, \mathrm{d}r$.
* Angular coupling coefficients (Clebsch–Gordan tables).
* Tabulated atomic form factors.

Even though the LazyPhysicsAndChemistry repo does not yet expose XPS examples,
the shared data structures are useful for future spectroscopy notebooks. The
Python port will wrap them as `xps.py`, offering interpolation-ready tables.

## Usage

The new shared module will expose:

```python
from comp_physics_python.lib import matrix_utils, rng, numerov, xps
```

ensuring every chapter uses the same tested code rather than reimplementing
NumPy/SciPy glue ad hoc.
