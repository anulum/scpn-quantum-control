# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — End-to-End Testing Documentation

# End-to-End Testing

`tests/test_e2e_new_modules.py`

End-to-end integration tests for the 20 new modules added in March 2026.
These tests verify that modules work together as integrated pipelines,
not just in isolation.

---

## Running the Tests

```bash
# All e2e tests
pytest tests/test_e2e_new_modules.py -v

# Specific pipeline
pytest tests/test_e2e_new_modules.py -v -k "TestOpenSystemPipeline"

# With timing
pytest tests/test_e2e_new_modules.py -v --tb=short --durations=10

# All tests (unit + e2e)
pytest tests/ -v -m "not slow and not hardware"
```

---

## Test Structure

| Test class | Pipeline | Modules exercised | Tests |
|------------|----------|-------------------|:-----:|
| `TestOpenSystemPipeline` | Lindblad ↔ MCWF agreement | `lindblad`, `tensor_jump`, `ancilla_lindblad` | 4 |
| `TestSymmetryPipeline` | Z₂ → U(1) → sparse | `symmetry_sectors`, `magnetisation_sectors`, `translation_symmetry`, `sparse_hamiltonian` | 8 |
| `TestMultiPlatformPipeline` | Compile → export | `xy_compiler`, `circuit_export`, `ancilla_lindblad` | 4 |
| `TestVariationalPipeline` | NQS + param-shift + VQE | `nqs_ansatz`, `param_shift`, `gpu_batch_vqe`, `sparse_hamiltonian`, `contraction_optimiser` | 5 |
| `TestHardwareReadyPipeline` | Ancilla → stats → export | `ancilla_lindblad`, `lindblad`, `tensor_jump` | 2 |
| `TestFullPipeline` | Recommend → solve → export | `backend_selector`, `circuit_export`, `ancilla_lindblad`, `magnetisation_sectors`, `sparse_hamiltonian` | 4 |
| `TestBackendDispatchIntegration` | Backend switching | `backend_dispatch`, `backend_selector` | 3 |
| `TestPluginRegistryIntegration` | Registry → runner | `plugin_registry` | 2 |
| `TestCrossModuleConsistency` | Shared computation checks | All ED + sparse + open-system modules | 6 |
| **Total** | | **20 modules** | **38** |

---

## Pipeline Descriptions

### Pipeline 1: Open-System Kuramoto (Lindblad ↔ MCWF)

**Modules:** `lindblad.py`, `tensor_jump.py`, `ancilla_lindblad.py`

**What is tested:**
- Lindblad density matrix R(T) and MCWF ensemble R(T) agree within
  statistical error for the same physical parameters
- Zero-dissipation case produces identical results (no jumps)
- Ancilla circuit compiles correctly for the same system
- All three open-system methods produce valid output

**Physical invariant:** Lindblad is exact; MCWF converges to Lindblad
in the limit of infinite trajectories. Agreement within $\pm 0.15$ for
200 trajectories at $n=2$ is expected.

### Pipeline 2: Scaling with Symmetry (Z₂ → U(1) → Sparse)

**Modules:** `symmetry_sectors.py`, `magnetisation_sectors.py`,
`translation_symmetry.py`, `sparse_hamiltonian.py`

**What is tested:**
- Z₂ sector eigenvalues reconstruct the full spectrum exactly
- U(1) sector eigenvalues reconstruct the full spectrum exactly
- Z₂ and U(1) find the same ground energy
- Sparse eigsh matches dense eigh
- Sparse within U(1) sector matches dense U(1) sector
- All four methods agree at $n = 4, 6, 8$
- Memory estimates are consistent (full > Z₂ > U(1))
- Translation symmetry ground within full spectrum

**Physical invariant:** All methods diagonalise the same Hamiltonian.
Ground energy must be identical (up to numerical precision). The only
difference is computational cost.

### Pipeline 3: Multi-Platform Execution (Compile → Export)

**Modules:** `xy_compiler.py`, `circuit_export.py`, `ancilla_lindblad.py`

**What is tested:**
- XY-compiled circuit exports to valid QASM
- XY compiler produces circuits with measurable depth
- All export formats (Qiskit, QASM, Quil) are consistent
- Ancilla circuit is QASM-exportable

**Physical invariant:** All formats represent the same unitary evolution.
Format-specific syntax is validated (OPENQASM header, DECLARE in Quil).

### Pipeline 4: Variational Ground State (NQS + Param-Shift + Batch VQE)

**Modules:** `nqs_ansatz.py`, `param_shift.py`, `gpu_batch_vqe.py`,
`sparse_hamiltonian.py`, `contraction_optimiser.py`

**What is tested:**
- RBM VMC energy within 50% of exact (conservative bound — VMC is not
  guaranteed to converge for all initialisations)
- Parameter-shift VQE reduces energy over iterations
- Batch VQE scan finds energy below mean random
- NQS energy is a variational upper bound (≥ exact)
- Contraction optimiser gives same results as `np.einsum`

**Physical invariant:** Variational principle — VMC/VQE energy $\geq$ exact
ground energy. The gap depends on ansatz expressibility and optimisation.

### Pipeline 5: Hardware-Ready Open-System Circuit

**Modules:** `ancilla_lindblad.py`, `lindblad.py`, `tensor_jump.py`

**What is tested:**
- Build circuit → check stats → export to QASM (full workflow)
- All three open-system methods produce valid output for same system

**Physical invariant:** Circuit should have $n+1$ qubits, non-zero resets,
and be exportable.

### Pipeline 6: Full Auto-Solve Pipeline

**Modules:** `backend_selector.py`, `circuit_export.py`,
`ancilla_lindblad.py`, `magnetisation_sectors.py`, `sparse_hamiltonian.py`

**What is tested:**
- `recommend_backend` → `auto_solve` consistency
- Solve → export for same system
- Solve open-system → build ancilla circuit
- U(1) sector → sparse eigsh → level spacing analysis

**Physical invariant:** `auto_solve` should use the backend recommended by
`recommend_backend`. Ground energy should be negative for the XY model.
Level-spacing ratio should be between 0.2 and 0.7 (Poisson to GOE range).

### Pipeline 7: Backend Dispatch

**Modules:** `backend_dispatch.py`, `backend_selector.py`

**What is tested:**
- Setting numpy backend then solving works
- All available backends are settable
- `to_numpy(from_numpy(x))` roundtrip preserves data

### Pipeline 8: Plugin Registry

**Modules:** `plugin_registry.py`

**What is tested:**
- Qiskit runner from registry produces valid runner
- Custom backend registration and invocation

---

## Cross-Module Consistency Checks

The `TestCrossModuleConsistency` class verifies shared computation
invariants across multiple modules:

| Test | Invariant |
|------|-----------|
| `test_sparse_hermiticity` | $H = H^\dagger$ |
| `test_sparse_vs_dense_matrix` | Sparse and dense Hamiltonians are identical |
| `test_sparsity_stats_consistent` | Stats match actual matrix properties |
| `test_lindblad_order_parameter_bounded` | $0 \leq R(t) \leq 1$ |
| `test_mcwf_std_decreases_with_trajectories` | More trajectories → less noise |
| `test_all_ed_methods_agree` | Full/Z₂/U(1)/sparse give same $E_0$ at $n = 4, 6, 8$ |

---

## Module Coverage Matrix

Each row is a test class; columns are the 20 modules. ✓ = directly tested.

| Module | Open | Symm | Multi | Var | HW | Full | Disp | Plug | Cross |
|--------|:----:|:----:|:-----:|:---:|:--:|:----:|:----:|:----:|:-----:|
| `lindblad` | ✓ | | | | ✓ | | | | ✓ |
| `tensor_jump` | ✓ | | | | ✓ | | | | ✓ |
| `ancilla_lindblad` | ✓ | | ✓ | | ✓ | ✓ | | | |
| `symmetry_sectors` | | ✓ | | | | | | | ✓ |
| `magnetisation_sectors` | | ✓ | | | | ✓ | | | ✓ |
| `translation_symmetry` | | ✓ | | | | | | | |
| `sparse_hamiltonian` | | ✓ | | ✓ | | ✓ | | | ✓ |
| `mps_evolution` | | | | | | | | | |
| `contraction_optimiser` | | | | ✓ | | | | | |
| `nqs_ansatz` | | | | ✓ | | | | | |
| `jax_nqs` | | | | | | | | | |
| `mitiq_integration` | | | | | | | | | |
| `param_shift` | | | | ✓ | | | | | |
| `xy_compiler` | | | ✓ | | | | | | |
| `circuit_export` | | | ✓ | | | ✓ | | | |
| `backend_selector` | | | | | | ✓ | ✓ | | |
| `backend_dispatch` | | | | | | | ✓ | | |
| `plugin_registry` | | | | | | | | ✓ | |
| `gpu_batch_vqe` | | | | ✓ | | | | | |

**Note:** `mps_evolution`, `jax_nqs`, and `mitiq_integration` require
optional dependencies (quimb, JAX, Mitiq) and are tested in their
respective unit test files (`test_mps_evolution.py`, `test_batch3_modules.py`,
`test_mitiq_integration.py`). The e2e tests avoid optional dependencies
to ensure they run on the CI matrix without extras.

---

## Adding New E2E Tests

When adding new cross-module functionality:

1. Identify the pipeline (which modules interact)
2. Write a test that exercises the full pipeline, not individual functions
3. Assert on physical invariants (energy bounds, conservation laws,
   consistency between methods) rather than implementation details
4. Use `_system(n)` helper for standard test systems
5. Keep systems small ($n \leq 8$) for speed — e2e tests should complete
   in seconds, not minutes

---

## See Also

- [Contributing Guide](contributing.md) — how to run the full test suite
- [Symmetry Sectors](symmetry.md) — theory behind Pipeline 2
- [Lindblad Solver](lindblad.md) — theory behind Pipeline 1
- [Backend Selector](backend_selector.md) — theory behind Pipeline 6
