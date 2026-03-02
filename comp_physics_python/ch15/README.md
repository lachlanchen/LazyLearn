# Chapter 15 – Advanced Spin Monte Carlo

`4568_ch15/ch15` contains the Monte Carlo algorithms discussed in §15:

- `Wolff.F90`, `wolff_xy.F90` – single-cluster Wolff updates for Ising and XY
  models.
- `SW.F90` – Swendsen–Wang multi-cluster algorithm.
- `BackTrack.F90` – backtracking cluster builder used to illustrate percolation.
- `hosh_kop.f90` – Hoshen–Kopelman cluster labelling for analysing percolation.
- `xy.F90` – Metropolis XY model for comparison.

This README summarises the mathematics that will inform the Python rewrite.

## 1. Cluster representation of the Ising model

Starting from the Ising Hamiltonian $H = -J \sum_{\langle ij \rangle} s_i s_j$,
the Fortuin–Kasteleyn (FK) representation introduces bond variables $b_{ij}$
that are set to 1 with probability

$$
p_{ij} = 1 - e^{-2\beta J} \quad \text{if } s_i = s_j, \qquad
p_{ij} = 0 \text{ otherwise}.
$$

Clusters of spins connected by occupied bonds can be flipped as a whole without
violating detailed balance. This underlies the Swendsen–Wang (SW) and Wolff
algorithms.

## 2. Swendsen–Wang algorithm (`SW.F90`)

Each sweep performs:

1. **Bond activation:** For every nearest-neighbour pair with equal spins,
   activate the bond with probability $p_{ij}$.
2. **Cluster identification:** Label connected components via Hoshen–Kopelman.
3. **Cluster flipping:** Assign a new spin $s = \pm 1$ to each cluster with
   probability $1/2$.

The acceptance probability is unity because the FK measure is sampled exactly.
Autocorrelation times scale as $\tau \sim L^z$ with $z \approx 0$. The code
computes magnetisation $M$ and Binder cumulants $U = 1 - \langle M^4 \rangle /
(3 \langle M^2 \rangle^2)$.

## 3. Wolff single-cluster algorithm (`Wolff.F90`)

Wolff’s method grows one cluster per sweep:

1. Pick a random seed spin.
2. Attempt to add each aligned neighbour to the cluster with probability $p_{ij}$.
3. Continue breadth-first until no more spins are added.
4. Flip the cluster.

The acceptance probability is again unity. Observables are measured after each
cluster update; to compare with Sweeps the effective time increment is
$\Delta t = \lvert \mathcal{C} \rvert / N$ (cluster size over lattice size).

## 4. XY model clusters (`wolff_xy.F90`, `xy.F90`)

For the XY Hamiltonian
$H = -J \sum_{\langle ij \rangle} \cos(\theta_i - \theta_j)$ the Wolff variant
chooses a random reflection axis $\hat{r}$ and builds clusters using the bond
probability

$$
p_{ij} = 1 - \exp\left[ -2\beta J
  \max\left(0, (\hat{r} \cdot \mathbf{s}_i)
                  (\hat{r} \cdot \mathbf{s}_j)\right)
\right],
$$

then reflects all spins in the cluster:
$\mathbf{s}_i \rightarrow \mathbf{s}_i - 2 (\hat{r} \cdot \mathbf{s}_i) \hat{r}$.
`xy.F90` keeps a baseline Metropolis sampler with proposal
$\theta_i \rightarrow \theta_i + \delta$, $\delta \in [-\Delta, \Delta]$.

## 5. Cluster labelling (`BackTrack.F90`, `hosh_kop.f90`)

`BackTrack` demonstrates recursive cluster growth while
`hosh_kop` implements Hoshen–Kopelman labelling:

- Maintain equivalence classes via union–find.
- As the lattice is scanned row by row, assign the smallest neighbour label and
  record equivalences.
- After the pass, compress labels to obtain contiguous cluster IDs.

The Python port will reuse this union–find for SW and observables such as the
percolation strength $P_\infty = \max_\mathcal{C} \lvert \mathcal{C} \rvert / N$.

## Python roadmap

- `ising_cluster.py` – shared FK bond builder + Hoshen–Kopelman implementation.
- `wolff.py` – Ising and XY single-cluster updates (random axis reflection).
- `sw.py` – multi-cluster Swendsen–Wang sweeps with parallel labelling.
- `observables.py` – magnetisation, susceptibility, Binder cumulant, vortex
  density (for XY).

All modules will expose hooks to accumulate autocorrelation data so the LazyLearn
docs can reproduce the textbook plots (Binder cumulants vs temperature,
cluster-size distributions, etc.) exactly.
