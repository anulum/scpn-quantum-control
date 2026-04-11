# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Test Infrastructure Documentation

# Test Infrastructure

## Overview

scpn-quantum-control has **4,828 collected tests** across **248 test
files** with **97%+ branch coverage**. Every file contains at least 11
tests, the new modules from April 2026 each ship with 17–25 STRONG
tests across 6 dimensions (empty/null, error handling, negative cases,
pipeline integration, roundtrip, performance). The test suite verifies
correctness, pipeline wiring, Rust acceleration parity, and performance
benchmarks.

```
tests/
  conftest.py              # Shared fixtures: knm_Nq, coupling_variant, hypothesis strategies
  test_pipeline_wiring_performance.py  # ~120 tests — every __all__ export verified functional
                                       # (now includes TestGUESSPipeline, TestDynQPipeline,
                                       # TestPulseShapingPipeline)
  test_rust_path_benchmarks.py         # 51+ tests — all 37 Rust functions benchmarked
  test_symmetry_decay.py               # 20 STRONG tests — GUESS (April 2026)
  test_qubit_mapper.py                 # 17 STRONG tests — DynQ (April 2026)
  test_pulse_shaping.py                # 25 STRONG tests — ICI + (α,β)-hypergeometric (April 2026)
  test_*.py                            # 240+ module-specific test files (11+ tests each)
```

## Running Tests

```bash
# Full suite (skip slow 18-qubit tests)
pytest tests/ -v --tb=short -m "not slow"

# Single file
pytest tests/test_xy_kuramoto.py -v --tb=short -s

# With performance output
pytest tests/test_pipeline_wiring_performance.py -v -s

# Rust benchmarks
pytest tests/test_rust_path_benchmarks.py -v -s

# Coverage
pytest tests/ --cov=scpn_quantum_control --cov-report=html -m "not slow"
```

## Test Categories

### 1. Pipeline Wiring Tests (`test_pipeline_wiring_performance.py`)

113 tests that verify **every** symbol in `scpn_quantum_control.__all__` (77 symbols)
is importable, callable, and produces valid output when wired into a data pipeline.

Each test follows the pattern:
1. Build inputs from canonical SCPN parameters (build_knm_paper27, OMEGA_N_16)
2. Call the function/class under test
3. Assert output has correct type, shape, and physical bounds
4. Print wall-time performance

```python
class TestTopLevelExports:
    @pytest.mark.parametrize("name", sqc.__all__)
    def test_export_exists(self, name):
        obj = getattr(sqc, name, None)
        assert obj is not None, f"{name} is None — not wired"
```

15 subsystems tested end-to-end:
- Bridge (Knm→H, Knm→Ansatz)
- Phase Solvers (VQE, UPDE, Kuramoto)
- Hardware (Runner, Noise Model)
- Mitigation (ZNE, PEC)
- QEC (ControlQEC, SurfaceCodeUPDE)
- QSNN (Synapse, LIF, STDP)
- Identity (Attractor, Fingerprint)
- Crypto (Key Hierarchy, QKD)
- Analysis (FSS, H1, OTOC, XXZ)
- SSGF (Quantum Loop)
- Orchestrator (Adapter Roundtrip)
- Cutting (24-oscillator Partitioned Simulation)
- Control (VQLS, QAOA-MPC, QPetriNet)
- Benchmarks (Scaling)
- Applications (Reservoir, Kernel, Disruption)

### 2. Rust Path Benchmarks (`test_rust_path_benchmarks.py`)

51 tests covering all 18 functions in `scpn_quantum_engine` (PyO3 + rayon).

Each Rust function is tested for:
- **Correctness:** Output shape, dtype, physical bounds
- **Parity:** Exact match with Python fallback (where applicable)
- **Performance:** Wall-time measurement with `time.perf_counter()`

```python
class TestBuildKnm:
    def test_parity_with_python_paper27(self):
        K_rust = np.array(eng.build_knm(16, 0.45, 0.3))
        K_py = build_knm_paper27(L=16)
        np.testing.assert_allclose(K_rust, K_py, atol=1e-12)
```

Measured speedups:

| Function | Speedup |
|----------|---------|
| `build_knm` | 4.7× |
| `kuramoto_euler` | 33.1× |
| `pec_coefficients` | exact parity |
| `build_xy_hamiltonian_dense` | exact parity with Qiskit |

### 3. Module-Specific Tests (217 files)

Each test file covers one source module with multi-angle tests:

**Physical invariants** — Hermiticity, unitarity, trace, bounds:
```python
def test_hamiltonian_hermitian():
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)
```

**Pipeline wiring** — Data flows end-to-end, not decorative:
```python
def test_pipeline_knm_to_spectrum():
    K = build_knm_paper27(L=4)
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    assert eigvals[0] < 0
    print(f"\n  PIPELINE Knm→H→spectrum (4q): {dt:.1f} ms")
```

**Rust path parity** — Python and Rust produce identical results:
```python
def test_rust_dense_matrix_parity():
    H_py = knm_to_dense_matrix(K, omega)
    H_rust = np.array(eng.build_xy_hamiltonian_dense(K_flat, omega, n))
    np.testing.assert_allclose(H_rust, H_py, atol=1e-10)
```

**Edge cases** — Zero input, boundary values, degenerate systems:
```python
def test_zero_coupling_decoupled():
    K = np.zeros((n, n))
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    np.testing.assert_allclose(eigvals[0], -np.sum(np.abs(omega)), atol=1e-10)
```

**Performance benchmarks** — Wall-time printed for regression tracking:
```python
def test_pipeline_full_kuramoto_evolution():
    t0 = time.perf_counter()
    result = solver.run(t_max=0.3, dt=0.1, trotter_per_step=5)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  PIPELINE Knm→KuramotoSolver→R(t) (4q): {dt:.1f} ms")
```

### 4. Property-Based Tests (Hypothesis)

Several files use `hypothesis` for randomised property testing:

```python
@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=10, deadline=30000)
def test_hamiltonian_hermitian(n: int) -> None:
    K = rng.uniform(0, 0.5, (n, n))
    K = (K + K.T) / 2
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-12)
```

Files using hypothesis:
- `test_knm_properties.py` — Hermiticity, eigenvalues, ansatz params
- `test_qec_properties.py` — Zero-error syndrome, correction shapes
- `test_crypto_properties.py` — Key hierarchy determinism, HMAC roundtrip
- `test_new_module_properties.py` — Cross-module property tests

## Shared Fixtures (`conftest.py`)

```python
# System sizes
SMALL_SIZES = [2, 3, 4]
MEDIUM_SIZES = [2, 3, 4, 6]
ALL_SIZES = [2, 3, 4, 6, 8]

# Coupling matrices
@pytest.fixture
def knm_4q():
    return build_knm_paper27(L=4), OMEGA_N_16[:4]

# Coupling variants
@pytest.fixture(params=["paper27", "ring", "zero", "identity"])
def coupling_variant_4q(request): ...

# Hardware runner
@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path))
    runner.connect()
    return runner

# Hypothesis strategies
@st.composite
def st_coupling_matrix(draw, n=None): ...

@st.composite
def st_statevector(draw, n_qubits=None): ...
```

## Markers

```ini
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (18-qubit, OOM on CI)"]
```

The `@pytest.mark.slow` marker is applied to tests that:
- Create 18-qubit systems (2^18 = 262,144 Hilbert space dimension)
- Require > 7 GB RAM (GitHub Actions runner limit)
- Take > 60 seconds

CI skips slow tests: `pytest -m "not slow"`.

## CI Integration

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: pytest tests/ -v --tb=short -x -m "not slow" --ignore=tests/test_hardware_runner.py
```

Test matrix: Python 3.10, 3.11, 3.12, 3.13.
Coverage target: 95% (`--cov-fail-under=95`).

## Test Quality Standards

Every test file must have:
1. **11+ test functions** (no exceptions)
2. **Physical invariant checks** (Hermiticity, bounds, conservation laws)
3. **Pipeline wiring test** proving data flows end-to-end
4. **Performance benchmark** with wall-time output
5. **Rust path verification** where `scpn_quantum_engine` function exists
6. **Edge cases** (zero input, boundary, degenerate)
7. **SPDX 7-line header**

## File Naming Convention

```
tests/test_{module_name}.py           # Primary module test
tests/test_{module_name}_edge_cases.py # Edge case / error path focus
tests/test_{module_name}_properties.py # Property-based (hypothesis)
tests/test_coverage_100_{area}.py      # Coverage-push tests
tests/test_e2e_{pipeline}.py           # End-to-end integration
```

## Performance Regression

Pipeline benchmark values are printed to stdout during test execution.
To capture a performance baseline:

```bash
pytest tests/test_pipeline_wiring_performance.py -v -s 2>&1 | grep "PIPELINE" > benchmarks.txt
pytest tests/test_rust_path_benchmarks.py -v -s 2>&1 | grep "RUST" >> benchmarks.txt
```

Key regression indicators:
- Kuramoto solver (4q): should be < 50 ms
- VQE solve (2q): should be < 300 ms
- Rust kuramoto_euler: should show > 20× speedup
- Pipeline wiring: all 113 tests should pass in < 10 s total

## Dependencies

Test-only dependencies (in `[dev]` extras):
- `pytest >= 8.0`
- `pytest-cov >= 7.0`
- `hypothesis >= 6.100`
- `qiskit-aer >= 0.17`

Optional (for full test coverage):
- `scpn-quantum-engine >= 0.2.0` (Rust acceleration tests)
- `quimb >= 1.8` (MPS/DMRG tests — `importorskip`)
- `sc-neurocore >= 3.14` (ArcaneNeuron E2E tests — `skipif`)
- `pennylane >= 0.40` (PennyLane adapter tests — `skipif`)
