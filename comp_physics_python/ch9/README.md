# Chapter 9 – Time-Dependent Electronic Structure

This chapter contains two independent codes in the textbook sources:

1. `carpar/` – a plane-wave Car–Parrinello molecular dynamics (CPMD) driver
   with norm-conserving pseudopotentials, fast Fourier transform grids, and a
   RATTLE constraint to keep the Kohn–Sham orbitals orthonormal.
2. `hf_dynamics/` – the two-site hydrogen Hartree–Fock molecular dynamics
   example from §9.3, written in an atom-centred Gaussian basis and integrated
   with a damped Verlet scheme.

The Python port will keep the same split: a CPMD engine that talks to FFT PW
data and a tight-binding/cluster TDHF integrator for minimal systems.

## 1. Car–Parrinello CPMD (`carpar/`)

### 1.1 Lagrangian and equations of motion

The CPMD Lagrangian used in `main.f90` is

$$
\mathcal{L} = \sum_I \frac{1}{2} M_I \vert \dot{\mathbf{R}}_I \vert^2
+ \mu \sum_n \langle \dot{\psi}_n \vert \dot{\psi}_n \rangle
- E_\text{KS}[\{\psi_n\}, \{\mathbf{R}_I\}]
+ \sum_{mn} \Lambda_{mn} \left(\langle \psi_m \vert \psi_n \rangle - \delta_{mn}\right),
$$

with fictitious electronic mass $\mu$, ion coordinates $\mathbf{R}_I$, and
Lagrange multipliers $\Lambda_{mn}$ enforcing orbital orthonormality.
Functional derivatives give two coupled equations of motion:

$$
\mu \, \ddot{\psi}_n = -\frac{\delta E_\text{KS}}{\delta \psi_n^\ast}
 + \sum_m \Lambda_{mn} \psi_m,
\qquad
M_I \, \ddot{\mathbf{R}}_I = -\frac{\partial E_\text{KS}}{\partial \mathbf{R}_I}.
$$

The Kohn–Sham energy decomposes into kinetic, Coulomb, electron–ion, and
exchange–correlation (LDA) pieces:

$$
E_\text{KS} = \sum_n \langle \psi_n \vert -\tfrac{1}{2}\nabla^2 \vert \psi_n \rangle
+ \frac{1}{2} \iint \frac{\rho(\mathbf{r}) \rho(\mathbf{r}')}{\vert \mathbf{r}-\mathbf{r}'\vert}
\, \mathrm{d}\mathbf{r}\,\mathrm{d}\mathbf{r}'
+ E_\text{ion-ion} + E_\text{xc}[\rho].
$$

`force.f90` evaluates the ionic forces via Hellmann–Feynman terms plus the
Pulay correction induced by the plane-wave cutoff.

### 1.2 Discretisation

- **Plane-wave grid:** `InitParams` chooses a cubic FFT grid of size
  `GridSize = min(2^n, 3^n, 5^n)` so that all reciprocal lattice vectors with
  $\vert \mathbf{G} \vert \le G_\text{max} = \sqrt{2E_\text{cut}}$ fit exactly.
- **Density:** reciprocal-space densities $\rho(\mathbf{G})$ are collected,
  inverse-transformed to $\rho(\mathbf{r})$, and used for Hartree and XC
  potentials.
- **Pseudopotential:** `pseudo.f90` implements norm-conserving non-local Kleinman–Bylander
  projectors $V_\text{NL} = \sum_{lm} \vert \beta_{lm} \rangle D_{lm} \langle \beta_{lm} \vert$.

### 1.3 RATTLE constraint (orthonormality)

`Rattle(...)` enforces $\langle \psi_m \vert \psi_n \rangle = \delta_{mn}$ after
each velocity-Verlet propagation of the orbital coefficients $c_{n\mathbf{G}}$.
Given tentative coefficients $\tilde{c}$, the constraint correction solves

$$
\mathbf{S} \, (\tilde{c} + \Delta c) = \mathbf{I}, \qquad
\mathbf{S}_{mn} = \langle \psi_m \vert \psi_n \rangle,
$$

which leads to the linear system for the Lagrange multipliers used in the
Fortran code. The Python port will reuse the same projection, relying on NumPy
linear algebra instead of the hand-written Gauss–Jordan eliminator.

### 1.4 Integrator

The main time stepper is velocity-Verlet for both ions and electronic
coefficients, with two time-step parameters:

- `TimeStepOrt` – short electronic step used during the orthogonalisation stage.
- `TimeStepCP` – full CPMD step for coupled ion+electron propagation.

In Python we will expose both and keep identical ordering:

1. Half-step update of velocities $\dot{\psi}_n$ and $\dot{\mathbf{R}}_I$.
2. Drift the coordinates.
3. Rebuild densities, potentials, and forces (`Calc_OrbForce`, `Calc_IonForce`).
4. Apply RATTLE.
5. Complete the velocity step.

## 2. TD Hartree–Fock dynamics (`hf_dynamics/`)

### 2.1 Basis and overlap

`elec_md.f` and `nucl_md.f` use the contracted Gaussian basis set listed in
`Alpha`. The single-electron basis functions are
$\phi_\mu(\mathbf{r}) = \exp(-\alpha_\mu \vert \mathbf{r}-\mathbf{R}_A \vert^2)$,
centred on either nucleus. The overlap matrix is

$$
S_{\mu\nu} = \langle \phi_\mu \vert \phi_\nu \rangle =
\left( \frac{\pi}{\alpha_\mu + \alpha_\nu} \right)^{3/2}
\exp\Bigl(-\frac{\alpha_\mu \alpha_\nu}{\alpha_\mu + \alpha_\nu} R_{AB}^2\Bigr).
$$

Similar closed-form expressions are used for kinetic, nuclear attraction, and
two-electron Coulomb integrals; they are precomputed in `CalcHamilton` and
`BuildSuper`.

### 2.2 Equations of motion for the coefficients

Section §9.3 derives the constrained dynamics for the MO coefficients
$\mathbf{c}(t)$ in a non-orthogonal basis. In matrix form:

$$
\mathbf{S} \ddot{\mathbf{c}} = -\frac{\partial E_\text{HF}}{\partial \mathbf{c}}
- \gamma \, \mathbf{S} \dot{\mathbf{c}}
+ \sum_{\lambda} \Lambda_\lambda \frac{\partial}{\partial \mathbf{c}}
  \left(\mathbf{c}^\mathsf{T} \mathbf{S} \mathbf{c} - 1\right),
$$

with friction coefficient $\gamma$ adding numerical damping. `Verlet` implements
the discrete update (eq. 9.31):

$$
\mathbf{c}_{n+1} = \frac{2\mathbf{c}_n - (1-\gamma)\mathbf{c}_{n-1}
- h^2 \, \mathbf{F}(\mathbf{c}_n)}{1+\gamma},
$$

where $\mathbf{F} = \partial E_\text{HF}/\partial \mathbf{c}$ is constructed
from the Fock matrix. Because the basis is not orthonormal the new coefficients
must satisfy the constraint
$\mathbf{c}_{n+1}^\mathsf{T} \mathbf{S} \mathbf{c}_{n+1} = 1$, enforced by the
`Normalise` routine which solves the quadratic equation for the Lagrange
multiplier in Eq. (9.32).

### 2.3 Nuclear motion

`nucl_md.f` integrates Newton’s equations for the internuclear distance $R$ in
the hydrogen dimer using the same Verlet step:

$$
M \ddot{R} = -\frac{\mathrm{d}E_\text{BO}(R)}{\mathrm{d}R},
$$

where $E_\text{BO}(R)$ is the Born–Oppenheimer energy computed from the
HF electronic solver. Forces are obtained via numerical differentiation of the
electronic energy; the program prints $R(t)$ and the ionic potential energy
curve, matching Fig. 9.6 in the book.

### 2.4 Python port plan

The Python chapter will expose two modules:

- `carpar.py` – wraps FFT grids (NumPy FFTs), ultrasoft pseudopotential parser,
  CPMD integrator, and Gauss–Legendre time stepping for ionic dynamics.
- `hf_td.py` – reproduces the damped Verlet evolution of MO coefficients and
  couples it to a one-dimensional nuclear coordinate integrator.

Both modules will share utility layers for Gaussian integral evaluation and
for time-reversible integrators so that later chapters (e.g. nuclear dynamics)
can reuse them.
