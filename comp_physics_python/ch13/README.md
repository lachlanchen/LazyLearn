# Chapter 13 – Finite-Element & Multiscale Methods

`4566_ch13/ch13` contains four families of finite-element programs:

1. `simplefem/` – Poisson and Helmholtz solvers on triangular meshes.
2. `rivaradir/` – Ritz variational formulation with adaptive refinement.
3. `femdyn/` – explicit time integration of elastic membranes (wave equation).
4. `multiscale/` – Heterogeneous multiscale method (HMM) coupling micro and
   macro elasticity.

This README outlines the mathematical formulation each Fortran code implements.

## 1. Simple FEM (`simplefem/`)

### 1.1 Weak form

For the scalar Poisson problem

$$
-\nabla \cdot (k \nabla u) = f \quad \text{in } \Omega,
$$

with Dirichlet BCs on $\partial \Omega$, the Galerkin formulation seeks
$u_h = \sum_j u_j \phi_j$ such that

$$
\sum_j u_j \int_{\Omega} k \, \nabla \phi_j \cdot \nabla \phi_i \,\mathrm{d}\Omega
= \int_{\Omega} f \phi_i \,\mathrm{d}\Omega.
$$

`simplefem` assembles the stiffness matrix element-wise on linear triangles:

$$
\mathbf{K}_e = \frac{k_e}{4 A_e}
\begin{pmatrix}
b_1^2 + c_1^2 & b_1 b_2 + c_1 c_2 & b_1 b_3 + c_1 c_3 \\
\cdots & \cdots & \cdots \\
\cdots & \cdots & \cdots
\end{pmatrix},
$$

where $b_i, c_i$ are coefficients from the triangle shape functions. The RHS is
integrated via midpoint quadrature. Dirichlet nodes are enforced by row/column
elimination, matching §13.1.

### 1.2 Error estimators

The Fortran code computes the $L^2$ error against analytical solutions and
optionally a residual-based estimator per element:

$$
\eta_e^2 = h_e^2 \Vert f + \nabla \cdot (k \nabla u_h) \Vert_{L^2(\Omega_e)}^2.
$$

The Python version will expose both, enabling adaptive refinement loops.

## 2. Ritz variational + adaptive refinement (`rivaradir/`)

The Ritz method minimises the functional

$$
\mathcal{F}[u] = \frac{1}{2} \int_\Omega k (\nabla u)^2 \,\mathrm{d}\Omega
- \int_\Omega f u \,\mathrm{d}\Omega,
$$

in a basis of hat functions defined on a refinement tree. `rivaradir` keeps the
same stiffness assembly as `simplefem`, but after each solve it marks the
largest-error elements (according to $\eta_e$) and performs red-green refinement.

Key maths to capture in Python:

- Hierarchical basis functions that keep parent/child relations.
- Energy-norm estimator $\eta = (\sum_e \eta_e^2)^{1/2}$ driving refinement.

## 3. Dynamic FEM (`femdyn/`)

`femdyn` solves the membrane wave equation

$$
\rho \frac{\partial^2 u}{\partial t^2}
 = \nabla \cdot (T \nabla u) + f,
$$

with lumped mass matrix $\mathbf{M} = \operatorname{diag}(m_i)$ so that the
explicit central-difference update reads

$$
u_i^{n+1} = 2u_i^{n} - u_i^{n-1} + \frac{\Delta t^2}{m_i}
\left(f_i^n - \sum_j K_{ij} u_j^{n}\right).
$$

The stability condition
$\Delta t < 2 / \sqrt{\lambda_\text{max}(\mathbf{M}^{-1} \mathbf{K})}$ is checked
in the initialisation routines. Damping can be added via a Rayleigh term
$\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$.

## 4. Heterogeneous multiscale method (`multiscale/`)

The multiscale code couples a coarse grid displacement $U_H$ with microscopic
cell problems that estimate the effective stiffness tensor. For each macro cell
$K$:

1. Solve the micro problem
   $-\nabla \cdot \bigl(C(x / \varepsilon) \nabla \chi^{(K)}\bigr) = 0$ on a
   representative cell with periodic BCs.
2. Estimate the homogenised tensor
   $C^\text{eff}_{ijkl} = \int_Y C_{ijpq}
   (\delta_{pk} + \partial_{y_p} \chi_k)
   (\delta_{ql} + \partial_{y_q} \chi_l) \,\mathrm{d}y$.
3. Assemble the macro stiffness with $C^\text{eff}$ and solve
   $-\nabla \cdot (C^\text{eff} \nabla U_H) = F$.

The Fortran implementation iterates steps 1–3, updating $C^\text{eff}$ until the
energy difference between consecutive iterations falls below tolerance.

## Python implementation sketch

- `meshes.py` – mesh loader + refinement utilities shared across submodules.
- `poisson_fem.py` – reproduces `simplefem` (assembly, boundary conditions,
  error estimators).
- `ritz_adaptive.py` – hierarchical basis, indicator-driven refinement loop.
- `wave_fem.py` – lumped-mass explicit integrator with CFL checks.
- `hmm.py` – orchestrates micro/macro problems with numpy/scipy linear solves.

Each solver will emit intermediate diagnostics (energy, estimators) so the
LazyLearn docs can trace convergence exactly as in chapter figures.
