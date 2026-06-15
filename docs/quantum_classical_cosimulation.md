# Quantum/Classical Co-Simulation of K_nm Networks

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.cosimulation` simulates a large Kuramoto-XY coupling
network by evolving a small, strongly-coupled core as an exact quantum
statevector while the weakly-coupled remainder runs as a classical Kuramoto
bath. A network of `N` oscillators needs a `2**N` statevector for a full quantum
treatment; the co-simulation keeps the quantum cost at `2**N_core` for a fixed
core and a classical cost linear in the bath couplings.

This is a **local mean-field embedding**: the quantum/classical boundary
couplings are treated at mean-field level, with no cross-boundary entanglement.
It is not an exact treatment of the full network and not a hardware path. The
partition `cross_fraction` bounds the decoupling approximation.

## Partitioning

`partition_knm` splits a symmetric `K_nm` matrix into a quantum-strong core and
a classical-weak bath. It seeds the core with the highest-weighted-degree node
and greedily adds the node with the strongest coupling into the current core,
stopping at `max_quantum_nodes` (capped at 14, the statevector ceiling) or when
the best coupling falls below `coupling_threshold`. The result is deterministic.

```python
from scpn_quantum_control.cosimulation import partition_knm

part = partition_knm(K, omega, max_quantum_nodes=8)
print(part.quantum_indices, part.conservation.cross_fraction)
```

The split is **edge-exact**: every coupling lands in exactly one of the
quantum-internal, classical-internal, or cross buckets, and the
`ConservationReport` proves `total = quantum_internal + classical_internal +
cross` to floating-point rounding. `cross_fraction = cross / total` is the
quality signal — the smaller it is, the better the mean-field decoupling holds.

## Co-simulation

`cosimulate` interleaves the two subsystems:

* the **quantum core** evolves under its internal XY Hamiltonian
  `H = -Σ K_ij (X_iX_j + Y_iY_j) - Σ ω_i Z_i` with a second-order Trotter split
  — an exact internal propagator from an eigendecomposition, then an exact
  single-qubit rotation for the classical mean field
  `(b_x, b_y)_i = Σ_c K_ic (cos θ_c, sin θ_c)`;
* the **classical bath** takes one explicit-Euler Kuramoto step driven by the
  coherence-weighted quantum moments, `cos θ_c · Σ_q K_cq ⟨Y_q⟩ −
  sin θ_c · Σ_q K_cq ⟨X_q⟩`.

```python
from scpn_quantum_control.cosimulation import cosimulate

result = cosimulate(K, omega, dt=0.02, n_steps=200, max_quantum_nodes=8, seed=0)
result.global_order        # combined order parameter per step
result.baseline_deviation  # RMS deviation from the all-classical baseline
```

The result carries the classical-phase and quantum-moment trajectories, the
quantum/classical/global order parameters, and an all-classical baseline (the
same network evolved entirely classically). `baseline_deviation` measures the
quantum-core effect plus the decoupling error — a diagnostic, not a certified
error bound.

### Validation anchors

In the decoupled limit (zero cross coupling), the co-simulation reduces exactly
to its parts: the quantum core matches an independent exact statevector
evolution and the classical order parameter matches the isolated all-classical
baseline. Both are asserted in `tests/test_cosimulation.py`.

## Acceleration

The per-step classical Kuramoto update dispatches to a Rust kernel
(`cosim_classical_substep`, `scpn_quantum_engine/src/cosimulation.rs`) that skips
zero couplings, so a sparse bath costs far less than a dense NumPy sweep. The
internal quantum Hamiltonian reuses the existing `build_xy_hamiltonian_dense`
Rust kernel. Both fall back to NumPy references that agree to floating-point
rounding.

Measured (release build, `scripts/bench_cosimulation.py`,
`results/cosimulation_benchmark.json`, `functional_non_isolated`, sparse
nearest-neighbour bath):

| stage | time |
|---|---|
| classical substep, N=128 (NumPy) | 171.5 µs |
| classical substep, N=128 (Rust) | 13.4 µs (12.8×) |
| end-to-end co-simulation, N=128, 8-node core, 100 steps | 1.33 s |

The Rust advantage comes from sparsity skipping: the NumPy reference evaluates
the full dense `N×N` sine matrix, while the Rust kernel touches only the present
couplings.

## Consumers

The partition and co-simulation surface the strong-correlation core of a K_nm
network for the differentiable-programming and control lanes without paying the
full `2**N` statevector cost.
