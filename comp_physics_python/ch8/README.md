# Chapter 8 – Molecular Dynamics in Python

These scripts reproduce the molecular-dynamics workflows from Thijssen’s
*Computational Physics* (ch. 8), replacing the original Fortran codes under
`comp_physics/comp_physics_textbook_code/4561_ch8`.  Everything runs in reduced
Lennard–Jones units so the results match the book’s numbers when you pick the
same densities and temperatures.

## 1. Argon (monatomic) – `ar_md.py`

Equations of motion: Newton’s second law with the pairwise Lennard–Jones (12–6)
potential truncated at `r_cut`:

$$
V(r) = 4\varepsilon \Big[ (\sigma/r)^{12} - (\sigma/r)^6 \Big],\qquad
\mathbf{F}_{ij} = 24\varepsilon \frac{\mathbf{r}_{ij}}{r^2}
 \Big[ 2(\sigma/r)^{12} - (\sigma/r)^6 \Big].
$$

The code uses:

- An FCC lattice for the initial positions (same as `md.F90`).
- Maxwell–Boltzmann velocities with a velocity-rescaling pre-equilibration stage.
- Periodic boundary conditions with the minimum-image convention.
- Velocity-Verlet time integration.
- Running estimates of temperature, potential/kinetic energy, and pressure via
  the virial expression.

Example (1000 equilibration steps + 4000 production steps):

```bash
python comp_physics_python/ch8/ar_md.py --n 256 --density 0.8442 --temp 0.722 \
       --dt 0.004 --equil 1000 --prod 4000 --cutoff 2.5
```

## 2. Nitrogen (rigid diatomic) – `n2_md.py`

Problem 8.7 treats N₂ as rigid dumbbells of length `ℓ`.  The Python version keeps
all translational DOF explicit (two atoms per molecule) and enforces the bond
length after every Verlet update.

Interactions between different atoms again follow the LJ force above; enforcing
rigidity ensures intramolecular distances stay equal to `ℓ` while momenta remain
center-of-mass conserving.

Usage:

```bash
python comp_physics_python/ch8/n2_md.py --n 512 --density 0.7 \
       --temp 1.0 --dt 0.003 --equil 500 --prod 3000 --bond 1.1
```

Both scripts dump summary data (energies, temperature) to stdout and
`ar_md.py` also writes the final configuration to `ar_final_positions.npy` for
post-processing.
