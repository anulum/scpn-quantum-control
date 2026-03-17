# Validation

## Test Suite

~553 unit, integration, property-based, and regression tests across ~69 test files. All pass on Python 3.9-3.12 with Qiskit 1.0+. 99%+ line coverage.

```bash
pytest tests/ -v
```

## Test Categories

### Unit Tests (~480 tests, ~59 files)

Cover individual modules: Hamiltonian construction, Trotter evolution, VQE, QAOA, QSNN neurons/synapses, crypto protocols, QEC decoder, error mitigation, identity continuity analysis. Each test runs in <1s on statevector simulator.

### Integration Tests (21 tests, 4 files)

End-to-end pipeline validation from K_nm coupling matrix to quantum observables:

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_integration.py` | 5 | Quantum Trotter vs classical exact evolution (N=2,3,4,6), ZNE on noiseless circuit, energy conservation under Trotter |
| `test_integration_pipeline.py` | 4 | Full pipeline: K_nm → Hamiltonian → VQE ground state (4q), K_nm → Trotter → energy (4q), 8q spectrum properties, 16q Hamiltonian construction |
| `test_cross_module.py` | 5 | Solver ↔ bridge Hamiltonian identity, classical_exact_diag vs numpy.eigvalsh, classical R ∈ [0,1], Z-parity conservation |
| `test_regression_baselines.py` | 7 | K_nm calibration anchors (Paper 27 Table 2 ±0.001), ω values, 4q ground energy E₀ = -6.303 ± 0.01, R range guards |

### Property-Based Tests (12 tests, 3 files)

Hypothesis-driven fuzzing of invariants:

| File | Tests | Properties |
|------|-------|-----------|
| `test_bridge_properties.py` | 5 | K_nm symmetry/positivity/diagonal, Hamiltonian Hermiticity (2-6 qubits), probability ↔ angle roundtrip |
| `test_crypto_properties.py` | 4 | CHSH S-parameter bound, key generation roundtrip, QKD sifting preserves key length |
| `test_qec_properties.py` | 3 | Syndrome length, decoder output shape, correction preserves code space |

### Identity Continuity Tests (43 tests, 4 files)

Validate the identity analysis subpackage:

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_identity_ground_state.py` | 11 | VQE attractor basin, robustness gap, binding spec input, error on mismatched dimensions, stronger coupling → larger gap |
| `test_identity_coherence_budget.py` | 15 | Fidelity monotonicity (depth, qubits), budget bounds, hardware param propagation, worse hardware → shorter budget |
| `test_identity_entanglement.py` | 13 | Bell state CHSH violation (S≈2√2), product state respects bound, GHZ tripartite (no bipartite), disposition labels, integration metric |
| `test_identity_key.py` | 9 | Spectral fingerprint, commitment format, challenge-response (correct/wrong K_nm), binding spec input |

### Hardware Smoke Tests (34 tests, 3 files)

Validate all 20 experiment circuits on AerSimulator (no IBM credentials needed):

| File | Tests | Coverage |
|------|-------|----------|
| `test_hardware_runner.py` | 22 | All 20 experiments produce valid result dicts with expected keys, R ∈ [0,1], job logging, timeout |
| `test_experiments_expanded.py` | 7 | Edge cases: 2q/6q scaling, custom Trotter orders, QAOA p=1/p=3 |
| `test_experiments_edge_cases.py` | 5 | Boundary inputs: single qubit, empty coupling, zero time |

## Physics Verification Gates

### 1. Quantum-Classical Parity

Each quantum module is verified against its classical counterpart:

| Module | Classical Reference | Parity Check |
|--------|-------------------|--------------|
| `qlif.py` | Bernoulli(sin^2(theta/2)) | Spike rate within 2-sigma of expectation |
| `xy_kuramoto.py` | `hardware/classical.py` `classical_kuramoto_reference()` | R(t) within 5% of ODE solution for K >> delta-omega |
| `trotter_upde.py` | `hardware/classical.py` `classical_exact_evolution()` | Per-layer phase evolution tracks classical trajectory at n={2,3,4,6} qubits |
| `phase_vqe.py` | `hardware/classical.py` `classical_exact_diag()` | Ground energy within 0.1% (simulator) |
| `qaoa_mpc.py` | `hardware/classical.py` `classical_brute_mpc()` | Finds optimal action for small horizons |
| `knm_hamiltonian.py` | SCPN canonical K_nm (Paper 27) | Hamiltonian eigenspectrum matches analytical bounds |

### 2. Circuit Validity

All generated QuantumCircuits must:
- Transpile on `AerSimulator` without error
- Satisfy qubit count = expected (no stray qubits)
- Produce measurement distributions that sum to 1.0 within floating-point tolerance

### 3. Hardware Validation (ibm_fez)

12-point decoherence curve (depth 5 to 770) validates:
- Readout noise floor: 0.1% at depth 5
- Linear decoherence regime: depth 85-400
- Coherence wall: depth 250-400

VQE hardware result: 0.05% error on 4-qubit subsystem.

### 4. Numerical Invariants

| Invariant | Where Checked |
|-----------|---------------|
| Hamiltonian Hermiticity | `test_knm_hamiltonian.py` |
| Synapse weight bounds [w_min, w_max] | `test_qsynapse.py` |
| Angle-probability roundtrip: p = sin^2(arcsin(sqrt(p))/1) | `test_sc_to_quantum.py` |
| Order parameter R in [0, 1] | `test_xy_kuramoto.py` |
| Surface code syndrome validity | `test_control_qec.py` |
| QPN marking conservation | `test_qpetri.py` |
| STDP weight update direction | `test_qstdp.py` |
| Pauli qubit ordering (Qiskit little-endian) | `test_classical.py` |
| Energy conservation under Trotter | `test_integration.py` |
| Trotter order-2 convergence | `test_trotter_error.py` |
| Knm calibration anchors (Paper 27 Table 2) | `test_regression_baselines.py` |
| 4q ground energy E₀ = -6.303 ± 0.01 | `test_regression_baselines.py` |
| Hamiltonian Z-parity conservation | `test_cross_module.py` |
| Solver ↔ bridge Hamiltonian identity | `test_cross_module.py` |
| Robustness gap >= 0 | `test_identity_ground_state.py` |
| Fidelity monotonically decreasing with depth | `test_identity_coherence_budget.py` |
| Bell state CHSH S > 2 | `test_identity_entanglement.py` |
| Product state CHSH S <= 2 | `test_identity_entanglement.py` |
| GHZ state: zero bipartite entanglement | `test_identity_entanglement.py` |
| Challenge-response: correct K_nm passes, wrong K_nm fails | `test_identity_key.py` |
| Knm symmetry, positivity, diagonal=K_base | `test_bridge_properties.py` (hypothesis) |
| Hamiltonian Hermiticity (fuzz 2-6 qubits) | `test_bridge_properties.py` (hypothesis) |
| Probability ↔ angle roundtrip (fuzz) | `test_bridge_properties.py` (hypothesis) |

### 5. Classical Benchmark Comparison

`examples/09_classical_vs_quantum_benchmark.py` compares wall-clock time
and accuracy for classical Euler ODE, classical exact evolution (matrix
exponential), and quantum Trotterized XY simulation at N=4, 8, 16.
At current NISQ scale, classical solvers are 10-1000x faster with exact
results. Quantum Trotter adds O(dt²) discretization error. This benchmark
establishes the baseline that quantum advantage requires N>>20 with error
correction.

## Coverage

99%+ line coverage (CI enforces `--cov-fail-under=99`).

```bash
pytest tests/ --cov=scpn_quantum_control --cov-report=term-missing
```

## Running Validation

```bash
# Full suite
pytest tests/ -v

# Integration + regression only
pytest tests/test_integration.py tests/test_integration_pipeline.py tests/test_cross_module.py tests/test_regression_baselines.py -v

# Hardware tests only (requires IBM credentials)
pytest tests/test_hardware_runner.py -v

# Classical vs quantum benchmark
python examples/09_classical_vs_quantum_benchmark.py

# Type check (42 source files, zero errors)
mypy

# Lint (zero errors)
ruff check src/ tests/

# Security scan
pip install bandit pip-audit && bandit -r src/ -ll -q && pip-audit --desc on
```
