# Validation

## BL-54 DLA and topology-constrained differentiable control

The BL-54 validation lane exercises the public
`scpn_quantum_control.dla_topology_control` facade, notebook 51, and the real
evidence CLI. Focused gates cover:

- delegation to the existing DLA-parity projector and exact projector JVP/VJP;
- absolute and normalised leakage gradients against central differences;
- strict-decrease parity-projected synthetic optimisation;
- topology-ledger JVP finite-difference agreement and JVP/VJP adjoint identity;
- supported signed, nonnegative, fixed-sign, clipping, hardware-mask, frozen-
  edge, and inactive-budget branches;
- fail-closed sign/bound kinks, active weight rescaling, positive connectivity
  thresholds, and discrete topology or persistent-homology changes;
- composition with the existing hard projected SPSA optimiser;
- immutable array custody, deterministic JSON/Markdown evidence, and byte
  checks; and
- strict API documentation, typing, lint, security, and focused branch-
  coverage gates.

The committed evidence reaches zero final outside-sector mass, matches the
analytic objective gradient within `3.59233087721e-10`, matches the parity and
topology JVPs within `9.98454884353e-11` and `3.93055310521e-11`, and satisfies
the topology adjoint identity within `8.881784197e-16`. It does not validate a
full DLA, controllability, persistent-homology derivative, error correction,
hardware parity preservation, provider, QPU, deployment, or application
claim. Reproduce with `scripts/run_dla_topology_control_evidence.py --check`.

## BL-60 chimera and multiscale control

The BL-60 validation lane exercises the public
`scpn_quantum_control.chimera_control` facade and the real evidence CLI. Its
focused gates cover:

- complete, disjoint, fine-to-coarse nested hierarchy validation;
- deterministic two-population Kuramoto–Sakaguchi trajectories through the
  production force dispatcher;
- population and ensemble order parameters composed from existing Shanahan
  diagnostics;
- analytic hierarchy-target gradients against central finite differences;
- strict objective decrease for an unapplied backtracking proposal;
- before/after constraint violations through the existing
  `TopologyConstraintLedger`;
- read-only array custody and SHA-256 binding;
- JSON/Markdown write and byte-check failure on drift;
- execution of notebook 50 and the evidence runner through user-facing
  boundaries; and
- strict API, documentation, typing, lint, security, and focused branch-
  coverage gates.

The committed 64-per-population evidence separates the frozen chimera transient
from its synchronised control for the exact configuration and checks the
analytic gradient to `2.49120728928e-11` maximum absolute error. It does not
validate an infinite-population attractor, arbitrary network, biological or EEG
system, learned topology, stability, controllability, provider, QPU, hardware,
or deployment path. Reproduce with
`scripts/run_chimera_multiscale_control_evidence.py --check`.

## Test Suite

Unit, integration, property-based, regression, claim-guard, and workflow-contract
tests run as one CI-gated suite on Python 3.11–3.13 with Qiskit 2.2+ on every
push. Current file/module counts are volatile and live only in the generated
inventory: `docs/_generated/capability_snapshot.md` and the README capability
snapshot.

```bash
pytest tests/ -v
```

## Test Categories

> The category map below documents the founding core-physics test surface
> (recorded 2026-03, ≤ v1.0-module epoch) and is retained as orientation for
> the physics gates that still anchor the suite. The suite has since grown by
> roughly an order of magnitude; the generated capability inventory is the
> source of truth for current counts.

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

### Quantum-Classical Co-Design Tests

The co-design package has module-specific tests for immutable contracts, exact
phase-objective evaluation, optional bounded open-system composition, latency
and safety policies, all three loop directions, deterministic replay, existing
control-stack ports, active-sensing/identity/geometry observer mapping, and the
evidence CLI. The focused lane requires
100% line and branch coverage over `scpn_quantum_control.codesign` and the
repository evidence runner. The end-to-end tests use actual local simulator,
QAOA-MPC, realtime-feedback, and co-simulation surfaces; no provider job or QPU
execution occurs.

### Bounded L16 Director Tests

The bounded L16 director runs three frozen small-system scenarios through the real exact-simulator
indicator path, verifies deterministic replay, and tests the co-design mappings
`continue -> allow`, `adjust -> hold`, and `halt -> abort`. BL-67 policy tests
refuse both incomplete and otherwise ticketed hardware modes. Contract,
evidence-validation, route-matrix, CLI, and atomic-write tests provide 100%
line and branch coverage over the new director modules and touched co-design
and route-matrix surfaces. No provider, QPU, plant, or realtime-hardware
execution occurs.

### ENAQT Transport Tests

BL-87 validates a trace-preserving single-excitation Lindblad generator,
source-to-target sink efficiency, exact zero- and high-noise endpoints,
intermediate-optimum classification, deterministic evidence replay, memory
budgets, compatibility aliases, and malformed-input rejection. The frozen
suite contains one disordered-chain intermediate optimum plus coherent-chain
and disconnected-target negative controls. The focused lane requires 100%
statement and branch coverage over `analysis/enaqt.py` and
`analysis/enaqt_evidence.py`. It does not execute hardware or validate a
universal, biological, synchronisation, BKT, consciousness, advantage, or
physical noise-setpoint claim.

### Entangled Initial-State Coherence Tests

The entanglement-sync evidence lane validates visibility-aware local phase
order, bounded transverse-exchange coherence, all four state-preparation
families, population-matched dephased controls, and pure-state one-qubit linear
entropy. The frozen four-qubit suite
requires Bell/W coherence differences, a GHZ zero-difference negative control,
and a separable product attribution control. Deterministic replay, digest
validation, the real CLI, atomic writing, no-advantage language governance,
dense budgets, and malformed inputs are covered. The focused lane requires 100%
statement and branch coverage over `analysis/entanglement_sync_evidence.py`;
it does not validate an
entanglement-specific cause, shifted critical coupling, spontaneous
synchronisation, advantage, provider, or hardware claim.

### QNN, QGNN, and QSNN Convergence Tests

The focused ML convergence lane validates immutable task and evidence contracts, exact
certificate arithmetic, deterministic replay, one real convergence task for
each QNN/QGNN/QSNN family, all framework-status cells, digest drift, atomic
evidence writing, and the repository CLI. Installed QNN adapters execute real
JAX and PyTorch gradient agreement; unavailable TensorFlow is recorded
explicitly and becomes a failing gate when required. The lane requires 100%
statement and branch coverage over `scpn_quantum_control.ml_examples`.

```bash
PYTHONPATH=src:oscillatools/src python -m coverage run --branch -m pytest -q \
  tests/test_ml_convergence_contracts.py \
  tests/test_ml_convergence_qnn.py \
  tests/test_ml_convergence_qgnn.py \
  tests/test_ml_convergence_qsnn.py \
  tests/test_ml_convergence_evidence.py
python -m coverage report --fail-under=100 \
  --include='*/scpn_quantum_control/ml_examples/*.py'
```

This validation is local and synthetic. It does not execute a provider, QPU,
or neuromorphic device and does not test temporal-coding or production-scale
convergence.

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

CI collects both line and branch coverage on the aggregate suite. The tracked
policy audit preserves a **90% line-coverage gate** and requires non-empty
branch data; branch coverage is reported in observation mode until consecutive
remote main-branch runs establish a baseline for a separate threshold. The
`slow`, `hardware`, `internal_corpus`, and `performance` markers are deselected
in the coverage lane because hardware runner/experiment paths require explicit
resources or credentials. New modules ship with focused branch coverage; the
remaining coverage debt is tracked on the internal execution queue.

```bash
pytest tests/ --cov=scpn_quantum_control --cov-branch --cov-report=xml --cov-fail-under=0
python tools/audit_coverage_policy.py --coverage-xml coverage.xml
```
