# Chapter 14 – Lattice Boltzmann Fluids

`4567_ch14/ch14` contains the single program `lb.F90` plus the helper
`setcol.c`. Together they implement the BGK (Bhatnagar–Gross–Krook) lattice
Boltzmann method for incompressible flow in two dimensions, matching §14.2 of
the book.

## 1. Discrete velocity set

The code uses the $D2Q9$ stencil with velocities

$$
\mathbf{c}_0 = (0,0), \quad
\mathbf{c}_{1,3} = (\pm 1, 0), \quad
\mathbf{c}_{2,4} = (0, \pm 1), \quad
\mathbf{c}_{5-8} = (\pm 1, \pm 1),
$$

and weights
$w_0 = \tfrac{4}{9}$, $w_{1-4} = \tfrac{1}{9}$, $w_{5-8} = \tfrac{1}{36}$.
Distribution functions $f_i(\mathbf{x}, t)$ evolve according to

$$
f_i(\mathbf{x} + \mathbf{c}_i, t+1) =
f_i(\mathbf{x}, t) - \frac{1}{\tau}
\bigl(f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\rho, \mathbf{u})\bigr),
$$

with the BGK relaxation time $\tau$. The equilibrium populations are

$$
f_i^{\text{eq}} = w_i \rho \left[
1 + 3 (\mathbf{c}_i \cdot \mathbf{u})
+ \frac{9}{2} (\mathbf{c}_i \cdot \mathbf{u})^2
- \frac{3}{2} \mathbf{u}^2
\right].
$$

Macrosopic density and velocity follow from the moment sums:
$\rho = \sum_i f_i$ and $\rho \mathbf{u} = \sum_i \mathbf{c}_i f_i$.

## 2. Forcing and boundary conditions

`setcol.c` seeds the initial obstacle mask (e.g., a cylinder for the von
Kármán vortex street). Bounce-back boundaries enforce zero velocity at the
solid walls by reflecting populations:

$$
f_i(\mathbf{x}_b, t+1) = f_{i^\ast}^\ast(\mathbf{x}_b, t),
$$

where $i^\ast$ indexes the opposite direction. Pressure-driven inflow is
implemented via the Zou/He velocity boundary condition, solving for the unknown
populations at the inlet/outlet after prescribing $\rho$ or $u_x$.

## 3. Viscosity and stability

The kinematic viscosity is $\nu = c_s^2 (\tau - 1/2)$ with $c_s^2 = 1/3$.
`lb.F90` takes $\tau$ (or equivalently $\nu$) as input and checks the Mach
number $Ma = \vert \mathbf{u} \vert / c_s$ to keep it $< 0.3$. Reynolds numbers
are reported via $\mathrm{Re} = U L / \nu$ where $U$ is the inlet velocity and
$L$ the cylinder diameter.

## Python port

The Python module will offer:

* `lattice.py` – discretisation metadata (velocity set, weights, bounce-back map).
* `lbm.py` – main collide-and-stream loop with optional forcing (Guo forcing)
  and profile sampling.
* `boundaries.py` – inlet/outlet and solid-wall enforcement.
* `diagnostics.py` – lift/drag coefficient computation and vortex shedding FFTs.

All steps retain the exact algebra used in `lb.F90`, enabling bitwise
comparison of diagnostic curves between Fortran and Python for the canonical
cylinder benchmark.
