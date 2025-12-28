# Chapter 11 – Transfer Matrices & Spin Chains

Chapter 11 covers two families of lattice models:

1. `transfermat/` – classical partition functions computed via $2 \times 2$
   transfer matrices (Ising model in a field, anisotropic couplings, and
   correlation length extraction).
2. `SpinChains/` – quantum $S=\tfrac{1}{2}$ chains solved by exact diagonalisation
   (`diagChain/`) and density matrix renormalisation group (`dmrg/`).

The Python port will bundle both under `comp_physics_python.ch11`, keeping a
shared linear-algebra backend (NumPy/SciPy) and reusable observables.

## 1. Transfer matrices (`transfermat/`)

### 1.1 Transfer matrix construction

For a 1D classical Ising model with Hamiltonian

$$
\mathcal{H} = -J \sum_{i} s_i s_{i+1} - h \sum_i s_i, \qquad s_i = \pm 1,
$$

the partition function on $N$ sites can be written as
$Z = \operatorname{Tr} \mathbf{T}^N$ with transfer matrix

$$
\mathbf{T} = \begin{pmatrix}
e^{\beta(J+h)} & e^{-\beta J} \\
e^{-\beta J} & e^{\beta(J-h)}
\end{pmatrix}.
$$

`tm.f` generalises this to anisotropic couplings and complex fields. The free
energy per spin is governed by the largest eigenvalue $\lambda_0$:

$$
f = -\frac{1}{\beta} \ln \lambda_0,
$$

and the correlation length follows from the ratio of the first two eigenvalues:

$$
\xi^{-1} = \ln \left(\frac{\lambda_0}{\lambda_1}\right).
$$

### 1.2 Observables

Derivatives of $\ln \lambda_0$ give magnetisation and susceptibility:

$$
m = \frac{1}{N} \frac{\partial \ln Z}{\partial (\beta h)} =
\frac{1}{\lambda_0}\frac{\partial \lambda_0}{\partial (\beta h)},
\qquad
\chi = \frac{\partial m}{\partial h}.
$$

`tm.f` evaluates these derivatives numerically by perturbing $h$ and $J$. The
Python port will expose analytical derivatives via automatic differentiation,
but we will keep the finite-difference pathway for parity with the Fortran.

## 2. Quantum spin chains (`SpinChains/`)

### 2.1 Exact diagonalisation (`diagChain/`)

`diagChain` diagonalises finite $S=\tfrac{1}{2}$ Heisenberg chains with

$$
H = J \sum_{i=1}^{L} \mathbf{S}_i \cdot \mathbf{S}_{i+1}
+ h \sum_i S_i^z,
$$

employing sparse basis states labelled by bit strings. The Hamiltonian matrix is
built in the $S_z$-conserving basis and diagonalised with LAPACK (dense) for
$L \lesssim 16$. Observables:

* Ground-state energy density $e_0 = E_0/L$.
* Spin–spin correlators $C(r) = \langle S_i^z S_{i+r}^z \rangle$ via exact
  state vectors.

### 2.2 Density Matrix Renormalisation Group (`dmrg/`)

`dmrg` implements the finite-system DMRG algorithm:

1. **Warm-up:** grow left/right blocks by appending one site at a time, keeping
   the dominant $m$ eigenvectors of the reduced density matrix.
2. **Superblock diagonalisation:** solve
   $H \vert \Psi \rangle = E \vert \Psi \rangle$ for the current four-block
   structure using Lanczos.
3. **Density matrix:** trace out the environment block to obtain
   $\rho_\text{left} = \operatorname{Tr}_\text{right} \vert \Psi \rangle \langle \Psi \vert$,
   diagonalise it, and keep the $m$ largest eigenvectors as the new basis.
4. **Sweep:** move the boundary between system/environment to converge
   observables.

The underlying math is set by the Schmidt decomposition
$\vert \Psi \rangle = \sum_{\alpha} \lambda_\alpha
\vert \phi_\alpha^\text{L} \rangle \vert \phi_\alpha^\text{R} \rangle$; truncating
to the $m$ largest $\lambda_\alpha$ minimises the discarded weight
$\epsilon = \sum_{\alpha>m} \lambda_\alpha^2$.

### 2.3 Python design

The Python module will ship two entry points:

* `transfer_matrix.py` – builds arbitrary $2 \times 2$ matrices, computes
  $\lambda_{0,1}$, magnetisation, susceptibility, and correlation lengths as a
  function of $(J, h, T)$.
* `spin_chain.py` – exposes both ED and DMRG solvers, sharing tensor-function
  helpers (Kronecker products, reduced density matrices, Lanczos iterations).

Both will emit numerical data matching the book’s figures so the docs site can
compare Fortran and Python results side-by-side.
