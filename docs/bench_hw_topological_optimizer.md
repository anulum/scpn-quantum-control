# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Topological Optimizer Benchmark

# Benchmark: Hardware Topological Optimizer

**Modules:**

- `scpn_quantum_control.control.topological_optimizer` — local simulation path
- `scpn_quantum_control.control.hardware_topological_optimizer` — hardware-in-the-loop

## Overview

The Topological Coupling Optimizer iteratively rewires the coupling matrix
$K_{nm}$ to minimise persistent 1-cycles ($p_{h1}$) in the quantum state's
correlation topology. It bridges Topological Data Analysis (TDA) and Quantum
Optimal Control: instead of minimising energy (VQE), it optimises for
macroscopic coherence and absence of phase vortices.

The hardware variant (`HardwareTopologicalOptimizer`) replaces local sparse
evolution with real IBM Quantum hardware execution.

## Optimisation Step Breakdown

Each `.step()` call performs:

1. **Hamiltonian evolution** — sparse expm on current $K$
2. **Correlation measurement** — $\langle X_n X_m + Y_n Y_m \rangle$
3. **TDA computation** — `ripser` persistent homology on correlation matrix
4. **Gradient estimation** — SPSA with 2 samples per step
5. **Coupling update** — gradient descent on $p_{h1}$

## Local Simulation Benchmarks

| System Size | Step Time | TDA (ripser) | Evolution | Notes |
|-------------|-----------|-------------|-----------|-------|
| N=4 | ~0.5 s | <10 ms | ~2 ms | Fast convergence |
| N=6 | ~2.4 s | <20 ms | ~8 ms | SPSA overhead dominates |
| N=8 | ~8 s | <50 ms | ~35 ms | 256-dim Hilbert space |
| N=16 | ~45 s | <50 ms | ~1.8 s | Sparse engine critical |

**TDA Efficiency:** `ripser` computes $H_1$ intervals for N=16 correlation
matrices in under 50 ms — negligible compared to quantum evolution.

## Hardware-in-the-Loop

When using `HardwareTopologicalOptimizer` with IBM Quantum:

| Component | Time | Notes |
|-----------|------|-------|
| Circuit construction | <5 ms | Trotter circuit + XYZ measurement bases |
| IBM queue wait | 10 s–10 min | Dominates total time |
| Hardware execution | ~2 s/shot-batch | 5000 shots default |
| TDA + gradient update | <50 ms | Same as local |

Total step time is dominated by IBM Quantum queue latency, not computation.

## TCBO Observables (Related)

The TCBO observer (`compute_tcbo_observables`) provides the observables that
the optimizer targets:

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_tcbo_observables(K, omega)` | 4 qubits | 4.6 ms | $p_{h1}$, TEE, string order |

$p_{h1} \in [0, 1]$, TEE finite, $|\text{string\_order}| \leq 1$,
$\beta_0 + \beta_1 \approx 1$.

## Physical Invariants (Verified by Tests)

- $K_{nm}$ remains symmetric at every step
- $K_{nm} \geq 0$ (non-negative coupling)
- $K_{nn} = 0$ (no self-coupling)
- Gradient norm is finite (no NaN/Inf)
- Optimisation loop converges ($p_{h1}$ decreases or stabilises)

## Test Coverage

9 tests across 2 files:

**`tests/test_topological_optimizer.py`** (6 tests):

- `test_topological_optimizer_step` — single step produces valid output
- `test_optimize_loop` — multi-step convergence
- `test_k_symmetry_preserved` — symmetry invariant
- `test_k_non_negative` — coupling positivity
- `test_diagonal_stays_zero` — no self-coupling
- `test_gradient_norm_finite` — numerical stability

**`tests/test_hardware_topological_optimizer.py`** (3 tests):

- `test_hardware_optimizer_step` — hardware path single step
- `test_hardware_optimizer_multi_step` — multi-step with mock runner
- `test_hardware_optimizer_k_non_negative` — coupling positivity on hardware path

## Running Benchmarks

```bash
pytest tests/test_topological_optimizer.py -v -s
pytest tests/test_hardware_topological_optimizer.py -v -s
```
