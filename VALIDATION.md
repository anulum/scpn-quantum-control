# Validation

## Test Suite

627+ unit, integration, property-based, and regression tests across ~80 test files. All pass on Python 3.9–3.12 with Qiskit 1.0+. 100% line coverage.

```bash
pytest tests/ -v
```

## Test Categories

### Unit Tests (~540 tests, ~70 files)

Cover individual modules: Hamiltonian construction, Trotter evolution, VQE, QAOA, QSNN neurons/synapses, crypto protocols, QEC decoder, error mitigation (ZNE + PEC), trapped-ion backend, ITER disruption classifier, quantum advantage benchmark, SNN adapter, SSGF adapter, identity binding spec, QSNN training, fault-tolerant UPDE. Each test runs in <1s on statevector simulator.

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

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_identity_ground_state.py` | 11 | VQE attractor basin, robustness gap, binding spec input |
| `test_identity_coherence_budget.py` | 15 | Fidelity monotonicity, budget bounds, hardware param propagation |
| `test_identity_entanglement.py` | 13 | Bell state CHSH violation (S≈2√2), product state respects bound |
| `test_identity_key.py` | 9 | Spectral fingerprint, challenge-response verification |

### v1.0 Module Tests (74 tests, 9 files)

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_pec.py` | 9 | PEC quasi-probability coefficients, Monte Carlo sampling, overhead scaling |
| `test_trapped_ion.py` | 8 | MS gate noise model, transpilation to {cx,ry,rz,sx,x}, unitarity preservation |
| `test_q_disruption_iter.py` | 10 | ITER 11-feature normalization, synthetic data generation, classifier benchmark |
| `test_quantum_advantage.py` | 8 | Classical vs quantum timing, crossover extrapolation, memory guard at n>14 |
| `test_snn_adapter.py` | 8 | Spike-to-rotation conversion, measurement-to-current, bridge forward pass |
| `test_ssgf_adapter.py` | 8 | W→Hamiltonian, phase encoding/recovery roundtrip, state extraction |
| `test_binding_spec.py` | 7 | 6-layer 18-oscillator topology, K/omega compilation, VQE attractor |
| `test_qsnn_training.py` | 8 | Parameter-shift gradient, epoch training, loss decrease |
| `test_fault_tolerant.py` | 8 | Repetition code encoding, transversal RZZ, syndrome extraction, qubit count |

### Cross-Repo Wiring Tests (17 tests, 1 file)

| File | Tests | What It Validates |
|------|-------|-------------------|
| `test_cross_repo_wiring.py` | 17 | ArcaneNeuronBridge (6, skip without sc-neurocore), SSGFQuantumLoop (4), orchestrator mapping roundtrip (4), fusion-core shot adapter (3) |

### Hardware Smoke Tests (34 tests, 3 files)

All 20 experiment circuits validated on AerSimulator (no IBM credentials needed).

## Physics Verification Gates

### 1. Quantum-Classical Parity

| Module | Classical Reference | Parity Check |
|--------|-------------------|--------------|
| `qlif.py` | Bernoulli(sin²(θ/2)) | Spike rate within 2σ |
| `xy_kuramoto.py` | `classical_kuramoto_reference()` | R(t) within 5% for K >> Δω |
| `trotter_upde.py` | `classical_exact_evolution()` | Per-layer phase tracks classical at n={2,3,4,6} |
| `phase_vqe.py` | `classical_exact_diag()` | Ground energy within 0.1% (simulator) |
| `qaoa_mpc.py` | `classical_brute_mpc()` | Optimal action for small horizons |
| `pec.py` | Analytical quasi-prob coefficients | q_I + 3·q_XYZ = 1, overhead = γ^n_gates |
| `quantum_advantage.py` | Classical expm timing | Exponential fit crossover at n>>14 |
| `fault_tolerant.py` | Distance-d repetition code | Syndrome detects injected bit-flip |

### 2. Numerical Invariants

| Invariant | Where Checked |
|-----------|---------------|
| Hamiltonian Hermiticity | `test_knm_hamiltonian.py`, `test_bridge_properties.py` (hypothesis) |
| K_nm symmetry, positivity, diagonal | `test_bridge_properties.py` (hypothesis) |
| K_nm calibration anchors (Paper 27 Table 2) | `test_regression_baselines.py` |
| 4q ground energy E₀ = -6.303 ± 0.01 | `test_regression_baselines.py` |
| Order parameter R ∈ [0, 1] | `test_xy_kuramoto.py` |
| Synapse weight bounds [w_min, w_max] | `test_qsynapse.py` |
| Angle-probability roundtrip | `test_sc_to_quantum.py` |
| Energy conservation under Trotter | `test_integration.py` |
| Trotter order-2 convergence | `test_trotter_error.py` |
| Z-parity conservation | `test_cross_module.py` |
| Bell state CHSH S > 2 | `test_identity_entanglement.py` |
| Product state CHSH S ≤ 2 | `test_identity_entanglement.py` |
| Fidelity monotonically decreasing with depth | `test_identity_coherence_budget.py` |
| PEC overhead = (Σ|q_k|)^n_gates | `test_pec.py` |
| Repetition code: d qubits encode 1 logical | `test_fault_tolerant.py` |
| Orchestrator phase roundtrip (mod 2π) | `test_cross_repo_wiring.py` |
| Fusion-core features normalized to [0,1] | `test_cross_repo_wiring.py` |

### 3. Hardware Validation (ibm_fez)

12-point decoherence curve (depth 5 to 770):

- Readout noise floor: 0.1% at depth 5
- Linear decoherence regime: depth 85–400
- Coherence wall: depth 250–400
- VQE hardware result: 0.05% error on 4-qubit subsystem

## Coverage

100% line coverage locally. CI enforces `--cov-fail-under=95` (excludes hardware runner/experiments which require IBM credentials).

```bash
pytest tests/ --cov=scpn_quantum_control --cov-report=term-missing
```
