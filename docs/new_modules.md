# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — New Modules Documentation (v0.9.5+)

# New Modules (March 2026)

11 modules added in the 2026-03-30 session, closing gaps identified by
competitive analysis against QuSpin, quimb, Mitiq, MISTIQS, Tequila, and NetKet.
78 tests, all passing, zero skips.

---

## Open-System Dynamics

### `phase/lindblad.py` — Lindblad Master Equation Solver

Solves $d\rho/dt = -i[H, \rho] + \sum_k (L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\})$
for the Kuramoto-XY Hamiltonian with configurable amplitude damping ($\gamma_\text{amp}$)
and dephasing ($\gamma_\text{deph}$) channels.

**Why it matters:** The quantum synchronisation community (Ameri et al. PRA 2015,
Giorgi et al. PRA 2012) primarily uses QuTiP Lindblad dynamics. Without this module,
our package was limited to unitary (closed-system) evolution.

```python
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
import numpy as np

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
omega = np.linspace(0.8, 1.2, n)

solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05, gamma_deph=0.02)
result = solver.run(t_max=2.0, dt=0.05)

print(f"R: {result['R'][0]:.3f} → {result['R'][-1]:.3f}")
print(f"Purity: {result['purity'][0]:.3f} → {result['purity'][-1]:.3f}")
```

**API:**

| Function | Returns |
|----------|---------|
| `LindbladKuramotoSolver(n, K, omega, gamma_amp, gamma_deph)` | Solver instance |
| `.run(t_max, dt)` | `{times, R, purity, rho_final}` |
| `.order_parameter(rho)` | Kuramoto R from density matrix |
| `.purity(rho)` | $\text{Tr}(\rho^2)$ |

**Tests:** 13 (purity preservation under unitary, purity decay under damping,
R bounded, strong damping kills sync, density matrix positivity, matches
unitary solver at zero dissipation)

---

### `phase/tensor_jump.py` — Monte Carlo Wave Function Method

Stochastic simulation of open Kuramoto-XY using quantum jumps. Each trajectory
evolves a pure state under the effective non-Hermitian Hamiltonian
$H_\text{eff} = H - \frac{i}{2}\sum_k L_k^\dagger L_k$, with random quantum
jumps at rate $dp = 1 - \langle\psi|\psi\rangle$.

Ensemble averaging over many trajectories recovers the density matrix.
Scales better than full Lindblad for larger systems (state vector vs density matrix).

Based on Dalibard et al., PRL 68, 580 (1992) and the Tensor Jump Method
(Causer et al., Nature Comms 2025).

```python
from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

result = mcwf_ensemble(K, omega, gamma_amp=0.1, t_max=1.0, dt=0.05,
                       n_trajectories=50, seed=42)
print(f"R_mean(T) = {result['R_mean'][-1]:.3f} ± {result['R_std'][-1]:.3f}")
print(f"Total jumps: {result['total_jumps']}")
```

**API:**

| Function | Returns |
|----------|---------|
| `mcwf_trajectory(K, omega, gamma_amp, ...)` | `{times, R, psi_final, n_jumps}` |
| `mcwf_ensemble(K, omega, ..., n_trajectories)` | `{times, R_mean, R_std, R_trajectories, total_jumps}` |

**Tests:** 5 (single trajectory, R bounded, ensemble shape, no-damping norm, output keys)

---

### `phase/ancilla_lindblad.py` — Single-Ancilla Open-System Circuit

Hardware-executable circuit for open-system dynamics using 1 ancilla qubit
with repeated reset. Each dissipation step:

1. Coherent Trotter evolution on system qubits
2. Controlled-Ry from each system qubit to ancilla ($\theta \propto \sqrt{\gamma \cdot dt}$)
3. Reset ancilla

This implements amplitude damping without density matrix representation.
Runs on IBM hardware with mid-circuit measurement and reset.

Based on Cattaneo et al., PRR 6, 043321 (2024).

```python
from scpn_quantum_control.phase.ancilla_lindblad import (
    build_ancilla_lindblad_circuit, ancilla_circuit_stats
)

qc = build_ancilla_lindblad_circuit(K, omega, t=0.5, gamma=0.05,
                                     n_dissipation_steps=5)
print(f"Qubits: {qc.num_qubits} ({qc.num_qubits-1} system + 1 ancilla)")

stats = ancilla_circuit_stats(K, omega)
print(f"CX gates: {stats['n_cx_gates']}, Resets: {stats['n_resets']}")
```

**Tests:** 4 (ancilla present, reset count, stats, measurements)

---

## Exact Diagonalisation

### `analysis/magnetisation_sectors.py` — U(1) Magnetisation Sectors

The XY interaction $X_iX_j + Y_iY_j = 2(\sigma^+_i\sigma^-_j + \sigma^-_i\sigma^+_j)$ is a
flip-flop: it swaps excitations but never creates or destroys them. Therefore the total
magnetisation $M = \sum_i Z_i$ is conserved. This decomposes the $2^N$ Hilbert space into
$N+1$ sectors labelled by $M$.

The largest sector ($M=0$) has dimension $\binom{N}{N/2}$:

| N | Full dim | Z₂ sector | U(1) largest | Reduction |
|:-:|:--------:|:---------:|:------------:|:---------:|
| 12 | 4,096 | 2,048 | 924 | 4.4× |
| 16 | 65,536 | 32,768 | 12,870 | 5.1× |
| 18 | 262,144 | 131,072 | 48,620 | 5.4× |
| 20 | 1,048,576 | 524,288 | 184,756 | 5.7× |

For N=16: full ED needs 32 GB, U(1) sector needs 2.5 GB. **13× memory reduction.**

```python
from scpn_quantum_control.analysis.magnetisation_sectors import (
    eigh_by_magnetisation, level_spacing_by_magnetisation, memory_estimate
)

# All sectors — exact spectrum
result = eigh_by_magnetisation(K, omega)
print(f"Ground: E={result['ground_energy']:.4f}, M={result['ground_sector']}")

# Level spacing within M=0 sector (avoids inter-sector artefact)
ls = level_spacing_by_magnetisation(K, omega, M=0)
print(f"r̄(M=0) = {ls['r_bar']:.3f} (Poisson=0.386, GOE=0.530)")

# Memory comparison
est = memory_estimate(16)
print(f"Full: {est['full_ed_mb']:.0f} MB, U(1): {est['u1_largest_mb']:.0f} MB")
```

**API:**

| Function | Returns |
|----------|---------|
| `basis_by_magnetisation(n)` | dict[M] → array of basis indices |
| `sector_dimensions(n)` | dict[M] → dimension |
| `eigh_by_magnetisation(K, omega, sectors)` | Full spectrum decomposed by M |
| `level_spacing_by_magnetisation(K, omega, M)` | r̄ within single sector |
| `memory_estimate(n)` | Comparison: full vs Z₂ vs U(1) |

**Tests:** 21 (partition, binomial match, eigenvalue exact match at n=4,6,8,
level spacing, memory estimates, N=16/20 dimensions)

---

### `analysis/symmetry_sectors.py` — Z₂ Parity Sector Decomposition

Exploits the Z₂ parity symmetry $P = Z_1 \otimes \cdots \otimes Z_N$ to halve
the Hilbert space for exact diagonalisation. At n=16, full ED needs 32 GB;
with sectors, each sector needs 8 GB.

Inspired by QuSpin's symmetry handling (Weinberg & Bukov, SciPost 2017).

```python
from scpn_quantum_control.analysis.symmetry_sectors import (
    eigh_by_sector, level_spacing_by_sector, memory_estimate_mb
)

result = eigh_by_sector(K, omega)
print(f"Ground energy: {result['ground_energy']:.4f}")
print(f"Ground parity: {'even' if result['ground_parity'] == 0 else 'odd'}")
print(f"Memory (full): {memory_estimate_mb(16, False):.0f} MB")
print(f"Memory (sector): {memory_estimate_mb(16, True):.0f} MB")

# Sector-aware level spacing (avoids artefact of overlaying two spectra)
ls = level_spacing_by_sector(K, omega)
print(f"r̄ (even): {ls['r_bar_even']:.3f}, r̄ (odd): {ls['r_bar_odd']:.3f}")
```

**Tests:** 13 (partition correctness, sector dimensions, Hermiticity, eigenvalue
match with full ED, level spacing bounded, memory estimates)

---

## Tensor Network

### `phase/mps_evolution.py` — MPS/DMRG via quimb

Matrix Product State backend for systems beyond ED limits (n=32-64).
DMRG for ground state, TEBD for time evolution.

Currently nearest-neighbour couplings only (quimb `SpinHam1D` limitation).
Longer-range couplings from $K_{nm}$ are dropped; for the exponential-decay
matrix, NN terms dominate.

Requires `pip install quimb`.

```python
from scpn_quantum_control.phase.mps_evolution import dmrg_ground_state, tebd_evolution

# Ground state
gs = dmrg_ground_state(K, omega, bond_dim=64, max_sweeps=20)
print(f"DMRG energy: {gs['energy']:.4f}, converged: {gs['converged']}")

# Time evolution
dyn = tebd_evolution(K, omega, t_max=1.0, dt=0.05, bond_dim=64)
print(f"R(0) = {dyn['R'][0]:.3f}, R(T) = {dyn['R'][-1]:.3f}")
```

**Tests:** 10 (DMRG energy, bond dims, TEBD R bounded, output keys)

---

## Error Mitigation

### `mitigation/mitiq_integration.py` — Mitiq ZNE + DDD

Production-quality error mitigation via Mitiq (LaRose et al., Quantum 6, 774, 2022).
Wraps Mitiq's ZNE (Richardson extrapolation) and DDD (digital dynamical decoupling)
around our circuits.

Requires `pip install mitiq`.

```python
from scpn_quantum_control.mitigation.mitiq_integration import zne_mitigated_expectation
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

mitigated = zne_mitigated_expectation(qc, scale_factors=[1.0, 3.0, 5.0])
print(f"ZNE-mitigated ⟨Z⟩ = {mitigated:.4f}")
```

**Tests:** 5 (availability, returns float, bounded, identity circuit, custom executor)

**Note:** Mitiq 1.0 has a bug where `from __future__ import annotations` breaks
executor introspection. This module avoids the `__future__` import as a workaround.

---

## Variational Methods

### `phase/param_shift.py` — Parameter-Shift Gradient Rule

Analytic gradient computation: $\partial\langle H\rangle/\partial\theta_k = \frac{1}{2}[\langle H\rangle(\theta_k + \pi/2) - \langle H\rangle(\theta_k - \pi/2)]$.

No finite-difference error. Works on real hardware (only needs 2 circuit evaluations
per parameter). Replaces the finite-difference gradients in `nqs_ansatz.py`.

Based on Mitarai et al., PRA 98, 032309 (2018).

```python
from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

result = vqe_with_param_shift(cost_fn, n_params=10, learning_rate=0.1,
                               n_iterations=100, seed=42)
print(f"Energy: {result['energy']:.4f}")
```

**Tests:** 4 (quadratic gradient, VQE convergence, output keys, zero at minimum)

---

### `phase/nqs_ansatz.py` — Neural Quantum State (RBM)

Restricted Boltzmann Machine wavefunction for variational ground state search.
$\log\psi(\sigma) = \sum_i a_i \sigma_i + \sum_j \log\cosh(\sum_i W_{ji}\sigma_i + b_j)$.

Pure numpy, no JAX/torch. Exact mode for n ≤ 12 (all $2^n$ configurations).
For production at larger scales, use NetKet.

Based on Carleo & Troyer, Science 355, 602 (2017).

```python
from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state

result = vmc_ground_state(K, omega, n_iterations=200, seed=42)
print(f"VMC energy: {result['energy']:.4f}, params: {result['n_params']}")
```

**Tests:** 10 (log_psi type, normalisation, n_params, reproducibility, VMC energy,
convergence, output keys, large-n rejection)

---

## Circuit Compilation

### `phase/xy_compiler.py` — XY-Optimised Gate Decomposition

Domain-specific compiler for the XX+YY interaction. Each coupling term
$e^{-iK_{ij}t(X_iX_j + Y_iY_j)}$ decomposes into 2 CNOT + 1 Rx, which is
more efficient than generic PauliEvolutionGate Trotter decomposition.

Inspired by MISTIQS domain-specific TFIM compiler.

```python
from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter, depth_comparison

qc = compile_xy_trotter(K, omega, t=0.1, reps=5, order=2)
print(f"Depth: {qc.depth()}, gates: {qc.size()}")

cmp = depth_comparison(K, omega, t=0.1, reps=5)
print(f"Generic: {cmp['generic_depth']}, Optimised: {cmp['optimised_depth']}")
print(f"Reduction: {cmp['reduction_pct']}%")
```

**Tests:** 4 (circuit creation, order 2 ≥ order 1, depth comparison, unitarity)

---

### `hardware/circuit_export.py` — Multi-Platform Export

Export Kuramoto-XY circuits to OpenQASM, Quil (Rigetti), and Cirq formats.

```python
from scpn_quantum_control.hardware.circuit_export import export_all

result = export_all(K, omega, t=0.1, reps=5)
print(f"QASM length: {len(result['qasm3'])} chars")
print(f"Quil length: {len(result['quil'])} chars")
print(f"Depth: {result['depth']}, gates: {result['gate_count']}")

# Save for Rigetti
with open("kuramoto.quil", "w") as f:
    f.write(result["quil"])
```

**Tests:** 4 (QASM string, Quil string, export_all keys, measurements present)

---

## Infrastructure

### `phase/backend_selector.py` — Automatic Backend Selection

Auto-selects the best simulation backend based on system size, available RAM,
installed packages, and whether open-system dynamics are needed.

Inspired by Maestro (Qoro, arXiv:2512.04216).

```python
from scpn_quantum_control.phase.backend_selector import recommend_backend, auto_solve

# Recommendation only
rec = recommend_backend(n=16, ram_gb=32.0)
print(f"Backend: {rec['backend']}, Memory: {rec['memory_mb']:.0f} MB")

# Auto-solve (selects backend and runs)
result = auto_solve(K, omega)
print(f"Used: {result['backend_used']}, E₀ = {result['result']['ground_energy']:.4f}")
```

| System size | Backend selected |
|:-----------:|-----------------|
| n ≤ 14 | `exact_diag` (numpy eigh) |
| n = 15-16 | `sector_ed` (Z₂ parity) |
| n = 17-64 | `mps_dmrg` (quimb, if installed) |
| n > 64 | `hardware` (IBM) |
| Open system, n ≤ 12 | `lindblad_scipy` |

**Tests:** 6 (small/medium/large/open/huge system selection, auto_solve runs)
