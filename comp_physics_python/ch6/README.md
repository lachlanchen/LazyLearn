# Chapter 6 – APW band structure

This folder ports `logapw.f` (Sec. 6.5.2 of *Computational Physics*) to
pure Python. The new solver lives under `apw/` and mirrors the FORTRAN logic:
logarithmic radial grids, Numerov integration, structure-matrix assembly, and
the determinant scan that locates copper’s energy bands along high-symmetry
lines of the fcc Brillouin zone.

## Method recap

Inside each muffin tin the radial Schrödinger equation

$$
\left[\frac{\mathrm{d}^2}{\mathrm{d}r^2} - \frac{\ell(\ell+1)}{r^2}
  + 2 \Bigl(\frac{V(r)}{r} + E\Bigr) \right] u_{\ell}(r) = 0
$$

is solved on a logarithmic grid. The Numerov update is identical to
`numerov.f`, so the boundary values
$\chi_\ell = u_\ell(R_\text{MT})$ and
$\dot{\chi}_\ell = u'_\ell(R_\text{MT})$ match the textbook expressions.
Outside the muffin tin, the APW basis is constructed from plane waves
$\exp(i(\mathbf{k}+\mathbf{K})\cdot\mathbf{r})$ and the matching condition at
$r = R_\text{MT}$ leads to the energy-dependent Hamiltonian

$$
H_{\mathbf{K}\mathbf{K}'}(E) = -E\,A_{\mathbf{K}\mathbf{K}'} + B_{\mathbf{K}\mathbf{K}'} +
\sum_{\ell=0}^{\ell_\text{max}} C_{\mathbf{K}\mathbf{K}'}^{(\ell)}
\frac{\dot{\chi}_\ell}{\chi_\ell},
$$

with the energy-independent matrices $A$, $B$, $C^{(\ell)}$ built from
spherical Bessel functions and Legendre polynomials as in Eq. (6.28). The band
energies are the roots of $\det H(E)$; we detect them by scanning $E$,
interpolating sign changes (band crossings) and local minima (flat bands).

## Code layout

| File | Purpose |
| --- | --- |
| `constants.py` | All lattice/muffin-tin metadata (`R_\text{MT}`, $a$, default k-paths, energy window). |
| `kpoints.py` | Reads the `KVectors` table and interpolates IBZ lines. |
| `special_functions.py` | Reimplements the `special.f` spherical Bessel and Legendre routines. |
| `numerov.py` | One-to-one port of `numerov.f` (FORTRAN indexing preserved). |
| `potential.py` | Loads the logarithmic potential and exposes `atom_integrals(E)` for all $\ell$. |
| `matrices.py` | Builds the $A/B/C$ structure matrices and the Hamiltonian $H(E)$. |
| `spectrum.py` | High-level driver that scans $\det H(E)$ along a k-path. |
| `cli.py` | Command-line interface (`python -m comp_physics_python.ch6.apw.cli ...`). |

All modules accept the original data files (`potential`, `KVectors`, etc.) out
of `comp_physics/comp_physics_textbook_code/4560_ch6/ch6/apw`. You may point the
CLI to another folder with `--data-root`.

## Usage

```bash
conda activate quantum  # or reuse the repo-wide environment

# Γ → K path with the default energy window [-0.04, 0.34] Hartree
python -m comp_physics_python.ch6.apw.cli --line Gamma-K

# Denser determinant sampling plus CSV output for plotting
python -m comp_physics_python.ch6.apw.cli \
  --line Gamma-K --samples 400 --output docs/data/cu_gamma_k.csv
```

Each k-point prints the bands it found inside the chosen energy window. When
`--output` is supplied the same data are written to a tidy CSV table for the
LazyLearn docs site.

## Notes on accuracy

* The logarithmic radial grid, potential table, and muffin-tin radius are
  imported verbatim from the FORTRAN distribution, so the boundary ratios
  $\dot{\chi}_\ell / \chi_\ell$ match to machine precision.
* Determinants are evaluated with `numpy.linalg.det`, mirroring the BLAS/LAPACK
  `DGETRF` flow. For stability we explicitly enforce Hermiticity on the
  Hamiltonian (`(H + H^\top)/2`).
* The quadratic interpolation that adds near-zero minima follows the textbook
  recipe but uses `numpy.polyfit` on three consecutive samples. This removes
  the occasional false positives seen in the original `logapw.f` when the
  determinant is nearly singular.

## TODO / extensions

1. Cache the $\chi_\ell(E)$ ratios during an energy scan (memoisation or a
   spline fit) to avoid re-integrating the radial equation at every sample.
2. Add GaussView-friendly `.log` writers so the copper spectrum rendered by the
   CLI can be compared directly with the historic `spectrum` text file.
3. Expose extra k-paths (Γ→X→W→K→Γ→L) as a single run and emit JSON for the
   `docs/` website.

## Semi-empirical pseudopotentials

Section 6.7.1 solves the silicon band structure with non-local
semi-empirical pseudopotentials. The plane-wave Hamiltonian in a basis
$\{\vert \mathbf{k} + \mathbf{K} \rangle\}$ reads

$$
H_{\mathbf{K}\mathbf{K}'}(\mathbf{k}) =
\delta_{\mathbf{K}\mathbf{K}'} \, \vert \mathbf{k}+\mathbf{K} \vert^2
+ V(\mathbf{K}-\mathbf{K}'),
$$

where the pseudopotential Fourier components depend only on
$\vert \mathbf{K}-\mathbf{K}' \vert$:

$$
V(\mathbf{G}) = V_3 \cos\left(\tfrac{\pi}{4}(G_x+G_y+G_z)\right)
\quad \text{if } \vert \mathbf{G} \vert^2 = 3,
$$

with similar constants $V_8$, $V_{11}$ for $\vert \mathbf{G} \vert^2 = 8, 11$
(all other Fourier components vanish). The current implementation keeps the
27 smallest $\mathbf{K}$ vectors explicitly and treats the remaining 86 plane
waves perturbatively via Löwdin partitioning. For a given trial energy $E$ the
effective Hamiltonian is

$$
U(E) = H_{AA} + H_{AB} (E - H_{BB})^{-1} H_{BA},
$$

where the $A$ block indexes the explicit subspace. Self-consistency is achieved
by iterating $E$ band-by-band along a k-path, seeding each step with the
previous point’s eigenvalue.

### Code layout (`pseudo/`)

| File | Purpose |
| --- | --- |
| `constants.py` | Silicon-specific metadata (reciprocal lattice constant, $V_3$, $V_8$, $V_{11}$, high-symmetry points). |
| `hamiltonian.py` | Builds the plane-wave Hamiltonian block $H(\mathbf{k})$ and encodes the distance→potential lookup. |
| `lowdin.py` | Löwdin partitioning helpers (`U(E)` assembly, eigenvalue selection per band). |
| `spectrum.py` | Driver that walks high-symmetry lines, runs the initial full diagonalisation, then applies Löwdin updates point by point. |
| `cli.py` | Command-line interface (`python -m comp_physics_python.ch6.pseudo.cli --line L-Gamma`). |

### Usage

```bash
# Default: L → Γ segment with step length 0.03 (fraction of |b_i|)
python -m comp_physics_python.ch6.pseudo.cli

# Include Γ → X afterwards and dump the first 6 bands to CSV
python -m comp_physics_python.ch6.pseudo.cli \
  --line Gamma-X --levels 6 --output docs/data/si_gamma_x.csv
```

The CLI prints each k-point and the requested number of bands. When `--output`
is set, the same data are written as tidy CSV rows for downstream plotting.

### Notes

* `KVectors` are imported verbatim from the Fortran source, guaranteeing
  identical $\mathbf{K}$ orderings and symmetry.
* The Löwdin correction uses vectorised NumPy algebra:
  $H_{AB} (E - H_{BB})^{-1} H_{BA}$ is evaluated in a single matrix multiply,
  matching Eq. (6.55).
* The level-by-level iteration (`Diag(Level)` in the FORTRAN) is preserved,
  ensuring the Python results trace the same energy sheets as the book figures.

### Future work

1. Add multi-segment scans (Γ→X→U→K→Γ→L) mirroring Fig. 6.12.
2. Surface the eigenvectors so effective masses can be computed directly.
3. Hook into the docs site to overlay APW vs pseudo bands for silicon.
