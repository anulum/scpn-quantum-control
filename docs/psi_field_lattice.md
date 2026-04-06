# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Ψ-field Lattice Gauge Theory

# Ψ-field Lattice Gauge Simulator

## 1. Mathematical Formalism

### U(1) Compact Lattice Gauge Theory

The Ψ-field is modelled as a compact U(1) gauge theory on the SCPN graph.
Link variables $U_{ij} = e^{iA_{ij}}$ live on edges, where $A_{ij} \in [-\pi, \pi)$
is the gauge potential.

#### Wilson Plaquette Action

On general graphs, plaquettes are minimal cycles. For the SCPN complete graph
($K_{nm} > 0$ for all pairs), the minimal cycles are triangles $(i,j,k)$:

$$S = -\beta \sum_{\triangle(i,j,k)} \text{Re}(U_{ij} U_{jk} U_{ki}^\dagger) = -\beta \sum_\triangle \cos(A_{ij} + A_{jk} - A_{ik})$$

where $\beta = 1/g^2$ is the inverse gauge coupling. The 16-node SCPN
graph has $\binom{16}{3} = 560$ triangle plaquettes.

#### Hybrid Monte Carlo (HMC)

The gauge field is sampled via HMC with the following steps:

1. **Momentum refresh:** $\pi_e \sim \mathcal{N}(0, 1)$ for each edge $e$
2. **Leapfrog integration** of $(A, \pi)$ along $H = \frac{1}{2}\sum_e \pi_e^2 + S(A)$:
   - Half-step: $\pi \leftarrow \pi - \frac{\epsilon}{2}\frac{\partial S}{\partial A}$
   - Full-step: $A \leftarrow A + \epsilon\pi$, wrap to $[-\pi, \pi)$
   - Full-step: $\pi \leftarrow \pi - \epsilon\frac{\partial S}{\partial A}$
   - ... repeat $N_\text{LF}$ times
   - Half-step: $\pi \leftarrow \pi - \frac{\epsilon}{2}\frac{\partial S}{\partial A}$
3. **Metropolis accept/reject:** accept with probability $\min(1, e^{-\Delta H})$

The force is:

$$\frac{\partial S}{\partial A_e} = \beta \sum_{\triangle \ni e} s_e \sin(\text{phase}_\triangle)$$

where $s_e = \pm 1$ is the orientation of edge $e$ in the plaquette.

#### HMC Algorithm Detail

The full HMC algorithm with momentum refresh, leapfrog, and Metropolis:

```
Algorithm: HMC for U(1) Lattice Gauge

Input: gauge field A, coupling β, step_size ε, n_leapfrog N_LF
Output: updated gauge field A' (accepted or rejected)

1. Store A_old ← A
2. Sample momenta: π_e ~ N(0, 1) for each edge e
3. Compute H_old = ½Σπ² + S(A)

4. Leapfrog integration:
   a. π ← π − (ε/2) × ∂S/∂A              [half-step momentum]
   b. For step = 1 to N_LF − 1:
      i.   A ← A + ε × π                  [full-step position]
      ii.  A ← wrap(A, [−π, π))            [compact U(1)]
      iii. π ← π − ε × ∂S/∂A              [full-step momentum]
   c. A ← A + ε × π                        [final position step]
   d. A ← wrap(A, [−π, π))
   e. π ← π − (ε/2) × ∂S/∂A              [final half-step momentum]

5. Compute H_new = ½Σπ² + S(A)
6. ΔH ← H_new − H_old
7. Accept with probability min(1, exp(−ΔH)):
   - If accepted: A' ← A
   - If rejected: A' ← A_old
```

The key property: HMC is **exact** — the Metropolis step corrects any
discretisation error from the leapfrog integrator. In the limit $\epsilon \to 0$,
$\Delta H \to 0$ and acceptance rate → 100%. For finite $\epsilon$, the
acceptance rate depends on the number of plaquettes (more plaquettes =
larger force = more sensitivity to step size).

For the 16-layer SCPN (560 plaquettes), step sizes of 0.01–0.05 are
needed for reasonable acceptance. For 4-layer (4 plaquettes), 0.1–0.5
works well.

#### Force Computation

The force $\partial S / \partial A_e$ for edge $e$ sums contributions from
all plaquettes containing $e$:

$$\frac{\partial S}{\partial A_e} = \beta \sum_{\triangle \ni e} s_e^\triangle \sin(\phi_\triangle)$$

where $s_e^\triangle = \pm 1$ is the orientation of edge $e$ in plaquette
$\triangle$, and $\phi_\triangle$ is the plaquette phase. Each edge in
the complete 16-node graph appears in $(16 - 2) = 14$ triangles (one
for each third vertex), so the force computation is $O(n_\text{edges} \times n_\text{triangles per edge})$.

The Rust implementation (`gauge_force_batch`) iterates over triangles
and accumulates force contributions — same asymptotic complexity but
without Python loop overhead.

#### Infoton Scalar QED

The SCPN infoton is a complex scalar field $\phi_i$ at lattice sites,
coupled to the gauge via the covariant derivative:

$$S_\text{matter} = \sum_{\langle ij \rangle} \left(|\phi_i|^2 + |\phi_j|^2 - 2\text{Re}(\phi_i^* U_{ij} \phi_j)\right) + \sum_i \left(m^2|\phi_i|^2 + \lambda|\phi_i|^4\right)$$

This is gauge-invariant under:
$$\phi_i \to e^{i\alpha_i}\phi_i, \quad A_{ij} \to A_{ij} + (\alpha_i - \alpha_j)/g$$

**Proof of gauge invariance** (verified in tests):
$$\phi_i'^* U_{ij}' \phi_j' = \phi_i^* e^{-i\alpha_i} \cdot e^{i\alpha_i} U_{ij} e^{-i\alpha_j} \cdot e^{i\alpha_j} \phi_j = \phi_i^* U_{ij} \phi_j \quad \checkmark$$

### Observables

#### Polyakov Loop

$$P(\mathcal{C}) = \prod_{(i,j) \in \mathcal{C}} U_{ij} = e^{i\sum_\mathcal{C} A_{ij}}$$

Along a path through SCPN layers, this measures the total phase accumulated
across the hierarchy. $|P| = 1$ for any path (compact U(1)).

#### Topological Charge

$$Q = \frac{1}{2\pi} \sum_\triangle \text{wrap}(\text{phase}_\triangle)$$

where $\text{wrap}(\theta) = ((\theta + \pi) \mod 2\pi) - \pi$ maps to $[-\pi, \pi)$.
For smooth configurations, $Q \approx 0$. Non-zero $Q$ indicates vortex
defects in the Ψ-field.

#### String Tension

$$\sigma = -\ln\langle\text{Re}(U_\triangle)\rangle$$

Lowest-order estimate from plaquette expectation. Positive $\sigma$
indicates confinement; $\sigma \leq 0$ indicates the deconfined/ordered phase.

## 2. Theoretical Context

### Why a Lattice Gauge Simulator?

The SCPN defines the Ψ-field as a U(1) gauge field with the infoton as
its boson. The existing `gauge/` package computes Wilson loops and
confinement observables from quantum statevectors (exact diagonalisation).
The `psi_field/` package complements this with classical Monte Carlo
simulation — enabling large-scale thermodynamic studies that are
impractical with exact diagonalisation.

Key difference:
- `gauge/` — quantum observables on small systems ($n \leq 16$ qubits)
- `psi_field/` — classical lattice gauge on arbitrary-size graphs

### Phase Structure on the SCPN Graph

The U(1) lattice gauge theory has two phases:

- **Ordered (large $\beta$):** $\langle\text{Re}(U_\triangle)\rangle \to 1$,
  links are nearly aligned, vortex density is low. This corresponds to
  the synchronised phase of the Kuramoto model.
- **Disordered (small $\beta$):** $\langle\text{Re}(U_\triangle)\rangle \to 0$,
  links are random, vortex density is high. This corresponds to the
  desynchronised phase.

The transition between phases is the lattice analogue of the BKT
transition. On the SCPN complete graph (unlike the square lattice),
the transition temperature depends on the graph connectivity and the
K_nm coupling weights. This has not been mapped — it is an open question
whether the SCPN graph produces a sharp transition or a crossover.

### Vortex Excitations

In compact U(1), the gauge field admits vortex excitations — plaquettes
where the phase winds by $\pm 2\pi$. The topological charge $Q$ counts
the net number of vortices. In 2D U(1) pure gauge theory, vortices are
always present (the theory is confining). On the SCPN graph (which is
not 2D), the vortex dynamics are more complex.

The `topological_charge()` function measures $Q$ by summing wrapped
plaquette phases. For smooth configurations (all links near zero),
$Q \approx 0$. For random configurations, $Q$ fluctuates with standard
deviation $\sim \sqrt{N_\triangle} / (2\pi)$.

### Non-Hypercubic Lattice

Standard lattice gauge theory operates on hypercubic lattices ($\mathbb{Z}^d$).
The SCPN, however, is a complete graph with exponential-decay coupling.
This module supports arbitrary graph topologies by precomputing triangle
plaquettes from the adjacency structure.

For the 16-layer SCPN: 120 edges, 560 triangles. This is qualitatively
different from a $4 \times 4$ square lattice (32 edges, 16 plaquettes).

### What This Module Does NOT Claim

- The classical lattice simulation is NOT quantum — it samples the
  Boltzmann distribution $e^{-\beta S}$, not the quantum partition function
- The infoton field is treated classically (no path integral over $\phi$)
- HMC acceptance rates depend strongly on step size and lattice size —
  the module provides tools, not guaranteed convergence
- The mapping SCPN → lattice gauge is a model, not a derivation

## 3. Pipeline Position

```
bridge/knm_hamiltonian.py → K_nm (adjacency matrix)
        ↓
psi_field/scpn_mapping.py → scpn_to_lattice() → SCPNLattice
  ├─ psi_field/lattice.py → U1LatticGauge (gauge dynamics, HMC)
  ├─ psi_field/infoton.py → InfitonField (scalar QED)
  └─ psi_field/observables.py → Polyakov, Q, σ
        ↓
gauge/ (Wilson loops, confinement — complementary quantum observables)
```

**Inputs:** K_nm coupling matrix (graph adjacency), $\beta$, mass, couplings.

**Outputs:** `SCPNLattice` with gauge field, infoton, and observable
measurement functions.

## 4. Features

- **Arbitrary graph topology:** not restricted to hypercubic lattices
- **Compact U(1) gauge:** link variables in $[-\pi, \pi)$, vortex excitations
- **HMC dynamics:** exact sampling with leapfrog integrator and Metropolis step
- **Triangle plaquettes:** auto-detected from adjacency matrix
- **Infoton scalar QED:** complex scalar field with gauge-covariant kinetic energy
- **Gauge invariance:** verified in test suite under local U(1) transformation
- **Observables:** Polyakov loop, topological charge, string tension, average link
- **SCPN mapping:** direct construction from `build_knm_paper27()`
- **Rust acceleration:** plaquette action, force, kinetic energy, topological charge
- **Deterministic seeding:** reproducible simulations via `numpy.random.default_rng`

### Comparison with gauge/ Package

| Feature | `gauge/` | `psi_field/` |
|---------|----------|-------------|
| Regime | Quantum (statevector) | Classical (Boltzmann) |
| System size | $\leq 16$ qubits (exact diag) | Arbitrary graph size |
| Wilson loops | Operator expectation $\langle W \rangle$ | Classical phase circulation |
| Confinement | From $\langle W \rangle$ area/perimeter law | From plaquette mean |
| Dynamics | Hamiltonian evolution | HMC Monte Carlo |
| Matter field | No | Infoton scalar QED |
| Hardware | IBM Quantum (optional) | CPU (Rust-accelerated) |

The two packages are complementary: `gauge/` provides exact quantum
answers for small systems, while `psi_field/` provides statistical
mechanics answers for arbitrary-size systems.

## 5. Usage Examples

### Build SCPN Lattice

```python
from scpn_quantum_control.psi_field.scpn_mapping import scpn_to_lattice

lattice = scpn_to_lattice(beta=2.0, seed=42)
print(f"Layers: {lattice.n_layers}")
print(f"Edges: {lattice.gauge.n_edges}")
print(f"Plaquettes: {len(lattice.gauge.plaquettes)}")
```

### HMC Thermalisation

```python
from scpn_quantum_control.psi_field.lattice import hmc_update

n_accepted = 0
for step in range(100):
    accepted, dH = hmc_update(lattice.gauge, n_leapfrog=10, step_size=0.02)
    if accepted:
        n_accepted += 1

print(f"Acceptance rate: {n_accepted}%")
plaq = lattice.gauge.measure_plaquettes()
print(f"Mean plaquette: {plaq.mean_plaquette:.4f}")
```

### Thermalisation and Measurement

```python
# Full thermalisation + measurement cycle
from scpn_quantum_control.psi_field.scpn_mapping import scpn_to_lattice
from scpn_quantum_control.psi_field.lattice import hmc_update
from scpn_quantum_control.psi_field.observables import topological_charge
import numpy as np

lattice = scpn_to_lattice(beta=5.0, seed=42)

# Thermalise
for _ in range(200):
    hmc_update(lattice.gauge, n_leapfrog=10, step_size=0.01)

# Measure plaquette and Q over 100 configurations
plaq_values = []
q_values = []
for _ in range(100):
    hmc_update(lattice.gauge, n_leapfrog=10, step_size=0.01)
    plaq = lattice.gauge.measure_plaquettes()
    plaq_values.append(plaq.mean_plaquette)
    q_values.append(topological_charge(lattice.gauge))

print(f"<plaquette> = {np.mean(plaq_values):.4f} ± {np.std(plaq_values):.4f}")
print(f"<Q> = {np.mean(q_values):.4f} ± {np.std(q_values):.4f}")
```

### Custom Graph Topology

```python
# Use a ring graph instead of SCPN complete graph
n = 8
adj = np.zeros((n, n))
for i in range(n):
    adj[i, (i + 1) % n] = 1.0
    adj[(i + 1) % n, i] = 1.0

from scpn_quantum_control.psi_field.lattice import U1LatticGauge, hmc_update

g = U1LatticGauge(adj, beta=3.0, seed=42)
print(f"Ring graph: {g.n_edges} edges, {len(g.plaquettes)} plaquettes")
# Ring has 0 triangles → no plaquettes → free field
```

### Measure Observables

```python
from scpn_quantum_control.psi_field.observables import (
    polyakov_loop, topological_charge, string_tension_from_wilson,
)

# Polyakov loop through first 4 layers
p = polyakov_loop(lattice.gauge, [0, 1, 2, 3])
print(f"|P| = {abs(p):.4f}, arg(P) = {np.angle(p):.4f}")

# Topological charge
Q = topological_charge(lattice.gauge)
print(f"Q = {Q:.4f}")

# String tension
sigma = string_tension_from_wilson(lattice.gauge)
print(f"σ = {sigma}" if sigma else "σ: disordered phase")
```

### Infoton Matter Action

```python
from scpn_quantum_control.psi_field.infoton import (
    gauge_covariant_kinetic, matter_action,
)

T = gauge_covariant_kinetic(lattice.infoton, lattice.gauge)
V = lattice.infoton.potential_energy()
S = matter_action(lattice.infoton, lattice.gauge)
print(f"T = {T:.4f}, V = {V:.4f}, S_matter = {S:.4f}")
```

## 6. Technical Reference

### Classes

#### `U1LatticGauge`

| Attribute | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of lattice sites |
| `beta` | `float` | Inverse coupling $1/g^2$ |
| `edges` | `list[tuple[int, int]]` | Edge list ($i < j$) |
| `links` | `ndarray` | Gauge potentials $A_e \in [-\pi, \pi)$ |
| `plaquettes` | `list[list[tuple]]` | Triangle plaquettes |
| `n_edges` | `int` | Number of edges |

Key methods: `total_action()`, `measure_plaquettes()`, `force()`.

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adjacency` | `ndarray` | required | $n \times n$ coupling matrix |
| `beta` | `float` | `1.0` | Inverse gauge coupling $1/g^2$ |
| `seed` | `int \| None` | `None` | RNG seed |

#### `InfitonField`

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `ndarray` (complex) | $\phi_i$ at each site |
| `mass_sq` | `float` | $m^2$ |
| `coupling` | `float` | $\lambda$ (quartic) |
| `gauge_coupling` | `float` | $g$ |

Key methods: `density()`, `total_charge()`, `potential_energy()`.

Constructor via `create_infoton()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_sites` | `int` | required | Number of lattice sites |
| `mass_sq` | `float` | `1.0` | Scalar mass² ($m^2 > 0$: symmetric phase) |
| `coupling` | `float` | `0.1` | Quartic self-coupling $\lambda$ |
| `gauge_coupling` | `float` | `1.0` | Gauge-matter coupling $g$ |
| `amplitude` | `float` | `0.1` | Initial random amplitude |
| `seed` | `int \| None` | `None` | RNG seed |

#### `SCPNLattice`

| Attribute | Type | Description |
|-----------|------|-------------|
| `gauge` | `U1LatticGauge` | Gauge field |
| `infoton` | `InfitonField` | Scalar field |
| `K` | `ndarray` | K_nm coupling |
| `omega` | `ndarray` | Natural frequencies |
| `n_layers` | `int` | Number of SCPN layers |

### Functions

#### `hmc_update(gauge, n_leapfrog, step_size)`

Single HMC step. Returns `(accepted: bool, delta_H: float)`.

#### `scpn_to_lattice(K, omega, beta, mass_sq, quartic_coupling, gauge_coupling, seed)`

Build `SCPNLattice` from K_nm coupling. Default: Paper 27 16-layer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K` | `ndarray \| None` | Paper 27 | Coupling matrix |
| `omega` | `ndarray \| None` | `OMEGA_N_16` | Natural frequencies |
| `beta` | `float` | `1.0` | Inverse gauge coupling |
| `mass_sq` | `float` | `1.0` | Infoton mass² |
| `quartic_coupling` | `float` | `0.1` | Infoton $\lambda$ |
| `gauge_coupling` | `float` | `1.0` | Gauge-matter $g$ |
| `seed` | `int \| None` | `None` | RNG seed |

#### `polyakov_loop(gauge, path)`

Polyakov loop along ordered path. Returns `complex`.

#### `topological_charge(gauge)`

Topological charge $Q$. Rust-accelerated.

#### `gauge_covariant_kinetic(field, gauge)`

Gauge-covariant kinetic energy. Rust-accelerated.

## 7. Performance Benchmarks

Measured on Intel i5-11600K, Python 3.12, 16-layer SCPN (120 edges,
560 plaquettes).

| Function | Time | Engine |
|----------|------|--------|
| `total_action` | 62 μs | Rust |
| `force` | 91 μs | Rust |
| `topological_charge` | 53 μs | Rust |
| `hmc_update` (10 leapfrog) | 1.24 ms | Rust (inner) |
| `gauge_covariant_kinetic` | 753 μs | Rust |
| `measure_plaquettes` | ~100 μs | Python |
| `scpn_to_lattice` | < 50 ms | Python |

### HMC Acceptance

For 4-layer graph ($\beta = 5$, step size 0.02, 10 leapfrog steps):
acceptance rate > 10% (verified in test). For 16-layer graph, acceptance
is lower due to 560 plaquettes — smaller step size needed.

Acceptance rate guidelines:

| $n$ | Edges | Plaquettes | Recommended $\epsilon$ | Expected acceptance |
|-----|-------|------------|----------------------|---------------------|
| 4 | 6 | 4 | 0.05–0.10 | 50–80% |
| 8 | 28 | 56 | 0.02–0.05 | 30–60% |
| 16 | 120 | 560 | 0.01–0.02 | 10–30% |

The optimal step size scales as $\epsilon \sim n_\text{plaq}^{-1/4}$
(Duane et al. 1987). With more plaquettes, the force grows and the
leapfrog integrator requires smaller steps to maintain reversibility.

### Reversibility

HMC with step size $\to 0$ produces $\Delta H \to 0$ (verified: step = 0.001,
$|\Delta H| < 0.1$). This confirms correct leapfrog implementation.

### Rust Engine API

Available via `import scpn_quantum_engine`:

#### `plaquette_action_batch(links, triangles, signs, n_triangles, beta)`

Compute mean plaquette and total action from flat triangle arrays.
Returns `(mean_plaquette: float, action: float)`.

#### `gauge_force_batch(links, triangles, signs, n_triangles, n_edges, beta)`

Compute force $\partial S / \partial A$ for all edges.
Returns `ndarray[f64]` of length `n_edges`.

#### `gauge_covariant_kinetic_rust(phi_re, phi_im, links, edges, g_coupling)`

Gauge-covariant kinetic energy from real/imag parts of $\phi$.
Returns `float`.

#### `topological_charge_rust(links, triangles, signs, n_triangles)`

Topological charge $Q$. Returns `float`.

### Flat Triangle Arrays

For Rust acceleration, the plaquette data is stored as flat arrays:
- `_tri_flat`: `ndarray[int64]` — edge indices, 3 per triangle
- `_tri_signs`: `ndarray[float64]` — orientation signs ($\pm 1$), 3 per triangle

These are precomputed in `_build_flat_triangles()` during `U1LatticGauge`
construction. The overhead is negligible ($< 1$ ms for 560 triangles).

### Test Coverage

22 tests across 6 dimensions:
- Empty/null: 4 tests (single edge, disconnected graph, zero infoton, shape)
- Error handling: 3 tests (single-site Polyakov, huge HMC step, disordered string tension)
- Negative cases: 3 tests (gauge invariance, smooth Q, plaquette bounds)
- Pipeline integration: 5 tests (SCPN creation, topology match, HMC thermalisation, imports, bridge)
- Roundtrip: 4 tests (HMC reversibility, closed Polyakov, T+V=S, determinism)
- Performance: 3 tests (plaquette, HMC step, topological charge budgets)

## 8. Citations

1. Wilson, K. G. "Confinement of quarks."
   *Phys. Rev. D* **10**, 2445 (1974).
   DOI: 10.1103/PhysRevD.10.2445

2. Creutz, M. "Quarks, Gluons and Lattices."
   Cambridge University Press (1983). ISBN: 0-521-31535-2

3. Rothe, H. J. "Lattice Gauge Theories: An Introduction."
   World Scientific, 4th ed. (2012). ISBN: 978-981-4365-87-1

4. Smit, J. "Introduction to Quantum Fields on a Lattice."
   Cambridge University Press (2002). ISBN: 0-521-89051-9

5. Duane, S. et al. "Hybrid Monte Carlo."
   *Phys. Lett. B* **195**, 216–222 (1987).
   DOI: 10.1016/0370-2693(87)91197-X

6. Kogut, J. B. "An introduction to lattice gauge theory and spin systems."
   *Rev. Mod. Phys.* **51**, 659 (1979).
   DOI: 10.1103/RevModPhys.51.659

7. Montvay, I. & Münster, G. "Quantum Fields on a Lattice."
   Cambridge University Press (1994). ISBN: 0-521-40432-0

8. Šotek, M. "God of the Math — The SCPN Master Publications."
   Zenodo (2025). DOI: 10.5281/zenodo.17419678

### Open Questions

1. **Phase transition on SCPN graph:** where is the critical $\beta_c$
   for the deconfinement transition on the K_nm complete graph?
   Standard 2D U(1) results do not apply directly.

2. **Infoton condensation:** for $m^2 < 0$ (negative mass²), the scalar
   field undergoes spontaneous symmetry breaking. What does this mean
   for the SCPN Ψ-field? The Higgs mechanism would give the gauge field
   a mass — screening the Ψ-field at long range.

3. **Quantum-classical correspondence:** how do the classical lattice MC
   observables compare with the quantum gauge/ observables for the same
   SCPN topology? This comparison has not been done.
