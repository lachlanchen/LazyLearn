# Chapter 12 – Quantum Monte Carlo & Stochastic Evolution

The Fortran sources under `4565_ch12/ch12` implement four distinct numerical
techniques:

1. `vmc/` – Variational Monte Carlo for atoms/molecules.
2. `dmc/` – Diffusion Monte Carlo with branching walkers.
3. `pimc/` – Path-Integral Monte Carlo for bosons.
4. `FokPlanck/` – Fokker–Planck solvers for harmonic and helium systems.

This README records the maths that the Python ports will reproduce.

## 1. Variational Monte Carlo (`vmc/`)

Given a trial wavefunction $\Psi_T(\mathbf{R}; \boldsymbol{\alpha})$, VMC
estimates the energy

$$
E[\boldsymbol{\alpha}] = \frac{\int \mathrm{d}\mathbf{R} \,
\vert \Psi_T(\mathbf{R}) \vert^2 \, E_L(\mathbf{R})}
{\int \mathrm{d}\mathbf{R} \, \vert \Psi_T(\mathbf{R}) \vert^2},
\qquad
E_L = \frac{H \Psi_T}{\Psi_T},
$$

by sampling configurations distributed according to
$\vert \Psi_T \vert^2$ with a Metropolis walk. The drifted Metropolis proposal
used in `vmc.f90` is

$$
\mathbf{R}' = \mathbf{R} + \chi + \tau \, \mathbf{F}(\mathbf{R}), \qquad
\mathbf{F} = 2 \nabla \ln \Psi_T,
$$

accepted with the Green-function weight (Eq. 12.12). The derivatives
$\partial E/\partial \alpha_k$ employ the covariance estimator

$$
\frac{\partial E}{\partial \alpha_k} = 2
\left\langle \left(E_L - \langle E_L \rangle\right)
\frac{\partial \ln \Psi_T}{\partial \alpha_k} \right\rangle.
$$

The Python module will keep both the simple Metropolis and the drifted Langevin
move, exposing automatic differentiation hooks for the parameter updates.

## 2. Diffusion Monte Carlo (`dmc/`)

Imaginary-time projection solves

$$
-\frac{\partial f}{\partial \tau} = (H - E_T) f,
$$

whose Green’s function is approximated as the product of diffusion,
drift, and branching:

- **Diffusion:** $\mathbf{R}' = \mathbf{R} + \sqrt{\tau} \, \eta$.
- **Drift:** $\mathbf{F} = 2 \nabla \ln \Psi_T$, as in VMC, but evaluated at the
  midpoint $\bar{\mathbf{R}} = (\mathbf{R}+\mathbf{R}')/2$.
- **Branching weight:** $w = \exp[-\tau (E_L(\mathbf{R}') + E_L(\mathbf{R}) - 2E_T)/2]$.

Walkers are replicated or removed according to $w$, keeping the population near
the target size by adjusting $E_T$ via Eq. (12.28):

$$
E_T(\tau + \Delta \tau) = \bar{E}_L(\tau) - \frac{\alpha}{\Delta \tau}
\ln \frac{N_w(\tau)}{N_0}.
$$

The Fortran code stores local energies in `Energy.dat`; the Python adaptation
will stream the same observables plus variance estimates to make regression
testing easier.

## 3. Path-Integral Monte Carlo (`pimc/`)

`pimc` discretises imaginary time into $M$ Trotter slices of width
$\Delta \tau = \beta / M$. The $N$-boson partition function becomes a classical
polymer integral with action

$$
S = \sum_{k=1}^{M} \sum_{i=1}^{N}
\frac{(\mathbf{r}_{i,k} - \mathbf{r}_{i,k+1})^2}{4 \lambda \Delta \tau}
+ \Delta \tau \, V(\mathbf{r}_{i,k}),
$$

where $\lambda = \hbar^2 / 2m$. Exchange permutations are sampled explicitly via
the primitive estimator used in §12.4. The Metropolis ratio for a single-bead
move is

$$
P_\text{acc} = \min\left\{1,
\exp\left[-\Delta S\right]\right\},
$$

with $\Delta S$ computed from the kinetic links touching the moved bead plus
the local potential term. The Python code will generalise this to multi-bead
and bisection moves while keeping the primitive estimators so the results can be
validated against the Fortran outputs.

## 4. Fokker–Planck solvers (`FokPlanck/`)

The harmonic and helium examples integrate the one-dimensional Fokker–Planck
equation

$$
\frac{\partial P}{\partial t} = -\frac{\partial}{\partial x}
\left[ A(x) P \right]
+ \frac{\partial^2}{\partial x^2} \left[ D(x) P \right],
$$

using a Crank–Nicolson discretisation:

$$
\left(1 - \frac{\Delta t}{2} \hat{L}\right) P^{n+1} =
\left(1 + \frac{\Delta t}{2} \hat{L}\right) P^{n},
$$

with tridiagonal $\hat{L}$ assembled from drift $A(x)$ and diffusion $D(x)$.
Probability conservation is enforced by renormalising $P$ after each timestep.
The helium variant couples two coordinates and uses an alternating-direction
implicit (ADI) scheme to maintain stability.

## Python roadmap

Each subfolder of `comp_physics_python/ch12` will target one algorithm:

- `vmc.py` – reusable Metropolis + drift moves, local-energy estimators, and
  automatic differentiation of trial wavefunctions.
- `dmc.py` – walker object with branching, population control, and mixed
  estimators (energy, radius).
- `pimc.py` – bosonic worldline sampler with permutation updates and energy
  estimators (kinetic via virial theorem, potential via slice averaging).
- `fokker_planck.py` – Crank–Nicolson / ADI solvers for general drift/diffusion
  coefficients, exposing diagnostics such as current conservation.
