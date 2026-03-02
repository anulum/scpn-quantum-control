# Validation

## Test Suite

456 unit, integration, property-based, and regression tests across 51 test files. All pass on Python 3.9-3.12 with Qiskit 1.0+.

```bash
pytest tests/ -v
```

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

VQE hardware result (0.05% error) provides publication-quality ground truth.

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
| Knm symmetry, positivity, diagonal=K_base | `test_bridge_properties.py` (hypothesis) |
| Hamiltonian Hermiticity (fuzz 2-6 qubits) | `test_bridge_properties.py` (hypothesis) |
| Probability ↔ angle roundtrip (fuzz) | `test_bridge_properties.py` (hypothesis) |

## Coverage

99%+ line coverage (CI enforces `--cov-fail-under=95`).

```bash
pytest tests/ --cov=scpn_quantum_control --cov-report=term-missing
```

## Running Validation

```bash
# Full suite
pytest tests/ -v

# Hardware tests only (requires IBM credentials)
pytest tests/test_hardware_runner.py -v

# Type check (32 source files, zero errors)
mypy

# Lint (zero errors)
ruff check src/ tests/
```
