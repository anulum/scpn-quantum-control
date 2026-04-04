# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dynamic Coupling Benchmark

# Benchmark: Dynamic Coupling (Quantum Hebbian Learning)

**Module:** `scpn_quantum_control.qsnn.dynamic_coupling`
**Class:** `DynamicCouplingEngine`

## Overview

The Dynamic Coupling Engine implements quantum-classical co-evolution where a
classical coupling matrix $K_{nm}$ drives quantum XY Hamiltonian evolution, and
quantum correlation measurements feed back to update $K_{nm}$ via a Hebbian rule:

$$K_{nm}(t+1) = (1 - \gamma) K_{nm}(t) + \eta \langle X_n X_m + Y_n Y_m \rangle$$

Each co-evolution step requires: Hamiltonian construction, sparse time evolution,
correlation matrix measurement, and coupling update.

## Rust Acceleration

The correlation measurement step uses `correlation_matrix_xy` from
`scpn_quantum_engine` when available, providing **2.9x** speedup over the
pure NumPy/Qiskit path.

| Operation | Python (ms) | Rust (ms) | Speedup |
|-----------|------------|-----------|---------|
| `correlation_matrix_xy` (4q) | 0.29 | 0.10 | 2.9x |

## Co-evolution Step Timing

Each `.step(dt)` call performs one full loop: evolve + measure + update $K$.

| System Size | Step Time | Notes |
|-------------|-----------|-------|
| N=2 | <1 ms | Trivial 4x4 Hilbert space |
| N=4 | ~2 ms | Sparse evolution dominates |
| N=6 | ~8 ms | 64-dimensional Hilbert space |
| N=8 | ~35 ms | 256-dimensional, correlation matrix 8x8 |

## Multi-step Co-evolution

`run_coevolution(steps, dt)` runs multiple steps and returns the full trajectory.

| Steps | N=4 Time | N=6 Time | N=8 Time |
|-------|----------|----------|----------|
| 10 | ~20 ms | ~80 ms | ~350 ms |
| 50 | ~100 ms | ~400 ms | ~1.8 s |
| 100 | ~200 ms | ~800 ms | ~3.5 s |

## Physical Invariants (Verified by Tests)

- $K_{nm}$ remains symmetric at every step
- $K_{nm} \geq 0$ (non-negative coupling)
- Diagonal $K_{nn} = 0$ (no self-coupling)
- Statevector normalised ($\|\psi\| = 1$)
- Correlation matrix symmetric ($C = C^T$)
- Decay drives $K \to 0$ when learning rate is zero

## Test Coverage

9 tests in `tests/test_dynamic_coupling.py`:

- `test_dynamic_coupling_engine_step` — single step produces valid output
- `test_run_coevolution` — multi-step trajectory length and structure
- `test_k_symmetry_preserved` — symmetry invariant across steps
- `test_correlation_matrix_symmetric` — $C_{nm} = C_{mn}$
- `test_k_diagonal_stays_zero` — no self-coupling
- `test_k_non_negative` — coupling positivity
- `test_decay_drives_k_toward_zero` — thermodynamic limit
- `test_statevector_normalised` — quantum state normalisation
- `test_rust_python_correlation_parity` — Rust and Python paths agree

## Running Benchmarks

```bash
pytest tests/test_dynamic_coupling.py -v -s
pytest tests/test_rust_path_benchmarks.py -k correlation_matrix -v -s
```
