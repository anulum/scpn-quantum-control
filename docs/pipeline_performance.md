# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Pipeline Performance Benchmarks

# Pipeline Performance Benchmarks

Every module in scpn-quantum-control is verified as **wired into the pipeline**
(not decorative) by `tests/test_pipeline_wiring_performance.py` (155 tests) and
per-module pipeline tests embedded in each test file. This page documents the
measured wall-time performance for every subsystem.

## Test infrastructure

```bash
pytest tests/test_pipeline_wiring_performance.py -v -s   # 155 tests, prints benchmarks
pytest tests/test_rust_path_benchmarks.py -v -s           # 68 tests, Rust parity + timing
```

**Hardware:** ML350 Gen8, 2× Xeon E5-2650v2, 128 GB RAM, Ubuntu 24.04.
**Python:** 3.12.3 with Qiskit 1.4.5, Aer 0.17.2.
**Rust:** scpn-quantum-engine 0.2.0 (PyO3 + rayon).

---

## 1. Bridge Layer

The bridge compiles SCPN coupling matrices (K_nm) into quantum objects.

### Knm to Hamiltonian

| System size | Compilation time | Output |
|-------------|-----------------|--------|
| L=2 (4×4 Hilbert) | <0.1 ms | SparsePauliOp, 2 qubits |
| L=4 (16×16 Hilbert) | <0.1 ms | SparsePauliOp, 4 qubits |
| L=8 (256×256 Hilbert) | <0.1 ms | SparsePauliOp, 8 qubits |
| L=16 (65536×65536 Hilbert) | ~6.7 ms | SparsePauliOp, 16 qubits, 256 Pauli terms |

**Rust path:** `build_xy_hamiltonian_dense` matches Qiskit `SparsePauliOp.to_matrix()`
to machine precision (atol=1e-10). Dense matrix construction in Rust takes 0.02 ms
for 3-qubit systems.

### Knm to Ansatz

| System size | Reps | Parameters | Time |
|-------------|------|-----------|------|
| L=2 | 2 | 8 | <0.1 ms |
| L=4 | 2 | 16 | <0.1 ms |
| L=8 | 2 | 32 | <0.1 ms |
| L=16 | 1 | 32 | <0.1 ms |

The Knm-informed ansatz uses coupling topology to determine entanglement gates,
producing fewer parameters than generic two_local (12 vs 18 for 3 qubits).
Benchmark: knm_informed E=-3.19 beats two_local E=-2.68 at equal iterations.

### Knm Construction (Rust)

| Function | Input | Time | Speedup |
|----------|-------|------|---------|
| `build_knm(16, 0.45, 0.3)` | 16×16 matrix | 0.02 ms | **4.7×** vs Python |

Rust `build_knm` includes paper27 overrides (L1-L2, L3-L5, L1-L16 boosted
couplings) — exact parity with Python `build_knm_paper27`.

---

## 2. Phase Solvers

### QuantumKuramotoSolver

The core solver maps Kuramoto dynamics to Trotterised XY Hamiltonian evolution.

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `run(t_max=0.3, dt=0.1)` | 4 qubits | 16.5 ms | R trajectory: 0.806 → 0.796 → 0.766 |
| `evolve(t=0.5, trotter_steps=3)` | 4 qubits | ~5 ms | QuantumCircuit, depth ~45 |
| `energy_expectation(sv)` | 4 qubits | <1 ms | float |

**Quantum-classical agreement:** R(quantum)=0.702 vs R(classical)=0.700 at t=0.2,
dt=0.1, trotter_per_step=5 — 0.3% deviation.

**Trotter convergence:** Error decreases as O(t²/n) for first-order, O(t³/n²) for
second-order Suzuki-Trotter. At 4 qubits, second-order produces strictly lower
Frobenius error than first-order at equal step count.

### PhaseVQE

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `solve(maxiter=20, seed=42)` | 2 qubits | ~150 ms | E=-3.94, exact=-3.94 |
| `solve(maxiter=30, seed=0)` | 3 qubits | ~200 ms | E, exact_energy, gap, params |

Variational principle verified: VQE energy >= exact ground energy (within 0.5
tolerance for short optimisation).

### VarQITE (Imaginary Time Evolution)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `varqite_ground_state(tau=0.5, n_steps=5)` | 3 qubits | 196.8 ms | E=-4.783 vs exact=-4.783 |

**0.0% error** — VarQITE achieves exact ITE convergence on 3-qubit system.
Energy trajectory: -4.753 → -4.783 (monotonic decrease).

### QuantumUPDESolver (Trotter UPDE)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `run(n_steps=5, dt=0.05)` | 4 qubits | 20.5 ms | R: 0.806 → 0.804 → 0.796 → 0.783 → 0.765 → 0.743 |
| `step(dt=0.1)` | 3 qubits | ~4 ms | R_global, theta |

Second-order Trotter (`trotter_order=2`) passes through correctly to underlying
solver. Reset reinitialises state exactly (first step after reset matches first
step from fresh solver).

### Adiabatic State Preparation

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `adiabatic_ramp(K_target=3.0, T=5.0, n_steps=15)` | 3 qubits | 54.4 ms | min_gap=0.0012 at K=2.80 |

Fidelity degrades through the BKT transition where the gap closes. The gap
minimum at K=2.80 confirms the transition location.

### Floquet-Kuramoto (Time Crystal)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `floquet_evolve(K=1.0, amp=0.5, freq=2.0)` | 2 qubits | 0.6 ms | R(t), subharmonic ratio |
| `scan_drive_amplitude(5 amplitudes)` | 2 qubits | ~3 ms | subharmonic ratio per amplitude |

DTC candidate detection via subharmonic_ratio > threshold. Drive signal oscillates
between K_base*(1-amp) and K_base*(1+amp) as expected.

---

## 3. Hardware Layer

### HardwareRunner (Simulator)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `connect()` | AerSimulator | ~50 ms | Backend ready |
| `transpile(GHZ circuit)` | 4 qubits | ~10 ms | ISA circuit, depth ~10 |
| `run_sampler(shots=1000)` | 4 qubits | ~100 ms | counts dict |
| `circuit_stats()` | — | <1 ms | depth, n_qubits, ECR count |

**Fractional gates:** With `use_fractional_gates=True`, Kuramoto circuit depth
reduces from ~80 to ~60 (25% reduction) for 4 qubits, 2 Trotter steps.
RZZ gates remain native instead of decomposing to ECR+RZ.

### Noise Model (Heron r2)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `heron_r2_noise_model()` | — | <1 ms | NoiseModel |
| Noisy Bell pair (10k shots) | 2 qubits | ~150 ms | non-ideal counts |
| Noisy Kuramoto R comparison | 3 qubits | 1349 ms | R_clean=0.734, R_noisy=0.734 |

At default Heron r2 parameters (CZ error=0.005), noise degradation is minimal
(R_clean ≈ R_noisy). Higher CZ error (0.1) produces measurable non-ideal counts.

### Trapped-Ion Backend

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `transpile_for_trapped_ion()` | 4 qubits | ~5 ms | All-to-all connectivity, no SWAPs |

Kuramoto circuits transpile without SWAP gates (ion trap all-to-all).
Unitarity preserved (Operator equivalence verified).

### Circuit Depth Regression

| System | Trotter reps | Transpiled depth | Gate count |
|--------|-------------|-----------------|------------|
| 2q, 1 rep | 1 | <50 | <100 |
| 4q, 1 rep | 1 | <100 | <300 |
| 4q, 3 reps | 3 | 134 | ~200 |
| 8q, 1 rep | 1 | <300 | <600 |
| 16q, 1 rep | 1 | <1000 | ~1500 |

Depth scales sub-linearly with reps (3 reps < 4× depth of 1 rep due to gate
cancellation in transpilation).

### QASM Export

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `export_trotter_qasm(K, omega, t=0.5, reps=3)` | 4 qubits | 3.4 ms | 1903 chars, 48 gates |

Exports OpenQASM 3.0 with qubit declarations and gate definitions.

---

## 4. Error Mitigation

### Zero-Noise Extrapolation (ZNE)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| Fold at scales [1,3,5] + extrapolate | 3 qubits | 34.4 ms | R_ZNE estimate |

Folded circuits preserve unitarity (norm=1.0 at all odd scales). Fit residual >= 0.
On noiseless simulator, all scale values are identical (folding is identity).

### Probabilistic Error Cancellation (PEC)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `pauli_twirl_decompose(0.05)` | 1 qubit | <0.01 ms | 4 coefficients |
| `pec_sample(circuit, p=0.05, n=200)` | 1 qubit | 160.9 ms | mitigated <Z>=-1.07 (ideal -1.0) |

**Rust path:** `pec_coefficients(p)` matches Python `pauli_twirl_decompose(p)` to
machine precision (atol=1e-10). Rust `pec_sample_parallel(100k samples)` takes
49-91 ms using rayon parallelism.

Quasi-probability invariant: identity coefficient > 1, error coefficients < 0,
sum = 1.0 (trace preservation).

---

## 5. Quantum Error Correction

### ControlQEC (Surface Code)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `ControlQEC(distance=3)` | 18 data qubits | <0.1 ms | Decoder ready |
| `get_syndrome()` + `decode()` | d=3 | 0.6 ms | correction vector |

Below-threshold correction: >80% success at p=0.01. Above-threshold: significant
failure at p=0.3. Zero-error syndrome is all-zero (verified).

### FaultTolerantUPDE (Repetition Code)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `build_step_circuit(dt=0.1)` | 2 osc, d=3 | <0.1 ms | 10-qubit circuit |
| `step_with_qec(dt=0.1)` | 3 osc, d=3 | 0.3 ms | syndromes, errors_detected |

Qubit layout: n_osc × (2d-1) physical qubits. Contains RZZ (transversal coupling),
CX (encoding + syndrome), RZ (field terms).

### SurfaceCodeUPDE

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `SurfaceCodeUPDE(n_osc=4, code_distance=3)` | 4 oscillators | <1 ms | Resource model |

Total physical qubits = n_osc × (2d²-1). For d=3: 4 × 17 = 68 physical qubits.

---

## 6. QSNN (Quantum Spiking Neural Network)

### QuantumSynapse

| Operation | Time | Output |
|-----------|------|--------|
| `apply(circuit, pre, post)` | <0.01 ms | CRy gate appended |

theta = pi × (w - w_min) / (w_max - w_min). Effective weight = sin²(theta/2).
Pre=|1> → post rotates; pre=|0> → post stays |0> (controlled rotation).

### QuantumLIFNeuron

| Operation | Time | Output |
|-----------|------|--------|
| `step(input_current=1.5)` | ~1 ms | spike ∈ {0, 1} |

Membrane equation: v(t+1) = v(t) - (dt/tau)(v(t) - v_rest) + R*I*dt.
Quantum mapping: P(spike) = sin²(theta/2) where theta encodes membrane potential.

### QuantumSTDP

| Operation | Time | Output |
|-----------|------|--------|
| `update(syn, pre=1, post=1)` | <0.01 ms | weight updated |

Hebbian LTP: pre+post fire → weight increases. LTD: pre fires, post doesn't →
weight decreases. No pre spike → no change (verified).

### QSNNTrainer

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `train(X, y, epochs=3)` | 2×2 layer | 47.6 ms | loss history |

Parameter-shift gradient: g = (L(+pi/2) - L(-pi/2)) / 2. Gradient sign flips
for opposite targets (antisymmetry). Zero learning rate → zero weight change
(verified to 1e-14). Forward probabilities bounded [0,1].

### SNNQuantumBridge

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `forward(spike_history)` | 4→3 neurons | 2.2 ms | output currents |

Spike-to-rotation: firing_rate × pi ∈ [0, pi]. Higher rate → larger angle
(monotonic). Measurement-to-current: P(|1>) × scale.

---

## 7. Identity Layer (Arcane Sapience)

### IdentityAttractor

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `solve(maxiter=30, seed=42)` | 3 qubits | 108.4 ms | E_0=-4.749, gap=1.383 |

Robustness gap = E_1 - E_0. Gap=1.383 provides strong identity protection.
Eigenvalues sorted ascending. Variational bound: E_vqe >= E_exact.
Stronger coupling → larger gap (verified).

### Identity Fingerprint

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `identity_fingerprint(K, omega)` | 4 qubits | ~150 ms | commitment (SHA-256 hex) |

Returns dict with commitment, spectral data (fiedler, eigenvalues), ground_energy,
n_parameters. Different K → different commitment. Spectral data deterministic.

### Challenge-Response Protocol

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `prove_identity(K, challenge)` | 3 qubits | <1 ms | response bytes |
| `verify_identity(K, challenge, response)` | 3 qubits | <1 ms | True/False |

Wrong K produces wrong response → verification fails. Different challenges →
different responses (no replay).

### Robustness Certificate

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_robustness_certificate(K, omega)` | 3 qubits | 0.9 ms | gap=1.383, P_transition=5.2e-5 |

P_transition = 5.2×10⁻⁵ — probability of identity confusion under noise.
Fidelity at depth: deeper circuits → lower fidelity (decoherence monotonicity).

---

## 8. Cryptographic Layer

### Key Hierarchy

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `key_hierarchy(K, phases, R, nonce)` | 4 layers | 0.11 ms | master (32 bytes) + 4 layer keys |

All layer keys unique. Master key differs from all layer keys. Same inputs →
same keys (deterministic). Different R or nonce → different keys.
`verify_key_chain()` detects tampered master, tampered layer keys, wrong nonce.

### Topology Commitment

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `topology_commitment(K)` | 4×4 matrix | <0.1 ms | 32-byte SHA-256 |

Deterministic hash of coupling topology. Combined pipeline (hierarchy +
fingerprint + commitment): 0.46 ms.

### SCPN-QKD Protocol

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `scpn_qkd_protocol(K, omega, alice, bob)` | 4 qubits | 692 ms | QBER, raw keys, Bell |

QBER ∈ [0, 1]. Ground energy < 0. Raw key shapes match qubit allocation.
Secure key length >= 0.

### Evolving Key Phases

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `evolve_key_phases(K, omega, theta_0, t=0.5)` | 4 layers | ~1 ms | (n_layers, n_samples) trajectory |

Kuramoto ODE integration via `solve_ivp(RK45)`. Initial condition preserved at t=0.
All values finite. ODE failure → RuntimeError with message.

---

## 9. Analysis Layer

### Finite-Size Scaling

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `finite_size_scaling(sizes=[2,3,4])` | 3 sizes | 0.8 ms | K_c per size + extrapolation |

K_c values finite. gap_min > 0. Extrapolation via BKT or power-law fit.

### H1 Persistence

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `scan_h1_persistence(omega, n_points=10)` | 4 osc | 14.9 ms | K_critical, p_h1 |

K_critical > 0. p_h1 ∈ [0, 1]. Vortex densities bounded. K values sorted.

### OTOC Synchronisation Probe

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `otoc_sync_scan(K, omega, n_K=6, n_t=8)` | 3 qubits | 7.6 ms | Lyapunov, R_classical |

R_classical bounded [0, 1]. Lyapunov values finite. OTOC detects transition: True.

### Berry Phase

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `berry_phase_scan(omega, T, k_range)` | 3 qubits | 6.6 ms | curvature peak at K=0.75 |

Fidelity ∈ [0, 1]. Spectral gap > 0. Curvature finite. Fidelity susceptibility
chi_F peaks near BKT transition (max chi_F = 0.005).

### Loschmidt Echo / DQPT

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `loschmidt_quench(K_i=0.5, K_f=3.0)` | 3 qubits | 0.8 ms | 3 cusps detected |

|G(0)| = 1 exactly. Rate function r(0) = 0. Times monotonic. No-quench: |G(t)| = 1
for all t. Large quench: amplitude oscillations.

### Krylov Complexity

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `krylov_complexity(H, Z0, t_max=5.0)` | 3 qubits | 155 ms | peak K(t) = 3.031 |

K(0) = 0 (operator starts in first basis element). K(t) >= 0. K(t) <= d²
(bounded by Hilbert space dimension). Lanczos b_n decay for finite dimension.

**Rust path:** `lanczos_b_coefficients` produces same coefficients as Python
(verified to atol=1e-6 on first few b_n).

### Entanglement Entropy

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `entanglement_at_coupling(omega, T, K=2.0)` | 4 qubits | 0.3 ms | S=0.928, gap=0.224 |

S ∈ [0, log₂(d)] where d = 2^(n/2). Schmidt gap ∈ [0, 1]. Weak coupling →
S ≈ 0 (product state). Strong coupling → S > 0. Schmidt gap closes near BKT.

### QFI Criticality

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `qfi_vs_coupling(omega, T, k_range)` | 3 qubits | 8.5 ms | peak QFI=0.225 at K=3.07 |

QFI >= 0. Total QFI >= max single-generator QFI. Peak at K=3.07 confirms
criticality-enhanced quantum correlations.

### Quantum Speed Limit

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_qsl(K, omega, t=1.0)` | 3 qubits | 10.4 ms | tau_MT, tau_ML bounds |

Mandelstam-Tamm bound tau_MT >= 0. Margolus-Levitin bound tau_ML >= 0.
Actual time tau_actual >= both bounds (QSL is a lower bound).

### Spectral Form Factor

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_sff(K, omega, n_times=20)` | 4 qubits | 1.2 ms | r_bar=0.488, gap=1.132 |

K(t=0) = 1 exactly (trace identity). SFF ∈ [0, 1]. Times monotonic. Level
spacing ratio r_bar = 0.488 (near GOE Wigner-Dyson 0.536 — quantum chaotic).

### Magic (Non-stabilizerness)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `magic_vs_coupling(omega, T, k_range)` | 3 qubits | ~5 ms | SRE peak |

SRE (stabiliser Renyi entropy) M₂ >= 0. Weak coupling → M₂ ≈ 0 (stabiliser
ground state). Strong coupling → M₂ > 0 (magic resource). Berry curvature
F_μν is antisymmetric (traceless).

### Lindblad NESS

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_ness(omega, T, K=2.0, gamma=0.1)` | 2 qubits | ~1 ms | R_ness, purity |

Purity ∈ [1/d, 1]. R_ness ∈ [0, 1]. gamma=0 → NESS = ground state (R_ness ≈
R_ideal). Purity decreases with noise (generally).

### Hamiltonian Learning

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `measure_correlators` + `learn_hamiltonian` | 3 qubits | 34.6 ms | loss=0, corr_error=0 |

Correlator matrix symmetric, zero diagonal, bounded [-2, 2]. Learned K symmetric,
non-negative. Perfect recovery for 3-qubit system (loss=0). Self-consistent: true
K as init → near-zero error.

### Hamiltonian Self-Consistency

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `self_consistency_from_exact(K, omega)` | 2 qubits | 10.9 ms | Frobenius=1.81, loss=0 |

2-qubit inverse problem is degenerate: loss=0 but Frobenius error=1.81 because
different K values produce identical correlators.

### XXZ Phase Diagram

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `anisotropy_phase_diagram(3δ × 6K)` | 3 qubits | 36.1 ms | K_c(Δ=0)=0.5, K_c(Δ=0.5)=1.2 |

XY (Δ=0) and Heisenberg (Δ=1) produce different gap structure. All gaps > 0.

### QRC Phase Detector

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `qrc_phase_detection(8 train, 2 test)` | 3 qubits | 39.3 ms | accuracy=100%, 36 features |

Self-probing: reservoir features from ground state observables. Linear readout
achieves perfect phase classification on well-separated data.

---

## 10. Application Layer

### Quantum Reservoir Computing

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `reservoir_ridge_regression(12 samples)` | 3 qubits | 33.9 ms | MSE=0.022 |

Feature matrix has non-trivial rank (expressive reservoir). Higher weight →
more features. Ridge regression produces actionable predictions.

### Quantum Kernel

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_kernel_matrix(5 samples)` | 3 qubits | 16.1 ms | PSD Gram matrix |

Mercer conditions verified: symmetric, PSD (min eigenvalue=0.028 > 0), diagonal=1.
K(x,x) = 1. Close inputs → high overlap (>0.95). Different Knm → different kernel.

### Disruption Classifier

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `run_disruption_benchmark(10+5)` | 3 qubits | 297 ms | accuracy=80% |

Kernel Gram matrix symmetric + PSD. Binary predictions. Accuracy bounded [0, 1].

### Quantum Disruption (ITER)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `predict(features)` | 5 qubits | 4.6 ms | risk=0.495 |
| `DisruptionBenchmark(20+10, 2 epochs)` | 5 qubits | 11.9 s | accuracy=70% |

Feature normalisation clamps to [0, 1]. Prediction deterministic for same params.
Circuit depth > 0. Training updates parameters.

### FMO Photosynthetic Benchmark

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `fmo_benchmark(K, omega)` | 7 sites | 1.4 ms | topology ρ=0.304 |

SCPN vs FMO topology correlation ρ=0.304 (weak positive). FMO self-comparison:
ρ=1.0. FMO coupling: symmetric, non-negative, zero diagonal, 7×7.

### Quantum Advantage Scaling

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `run_scaling_benchmark(sizes=[3,4])` | 3-4 qubits | 101 ms | timing comparison |

n=3: classical=23 ms, quantum=11 ms (quantum wins). n=4: classical=26 ms,
quantum=34 ms (classical wins). Crossover near n=4.

---

## 11. Bridge Adapters

### SSGF Adapter

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| W→H→encode→decode | 4 oscillators | 1.5 ms | R_global=0.767 |
| SSGFQuantumLoop.quantum_step | 4 oscillators | ~9 ms | theta updated, R returned |

Encoding: 2 gates per oscillator (Ry + Rz). Normalisation preserved. Uniform
phases → R ≈ 1. Opposite phases → R ≈ 0.

### SSGF Spectral Bridge

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `spectral_bridge_analysis(K, omega)` | 4 oscillators | 0.2 ms | fiedler=0.872, QPE=7 bits |

Fiedler > 0 for connected graph. Eigenvalues non-negative (Laplacian PSD).
Disconnected graph → fiedler=0. QPE bits estimate for spectral resolution.

### SSGF W Adapter

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `adapt_w_from_quantum(K, theta, lr=0.1)` | 4 oscillators | 4.9 ms | max_update=0.027 |

W_updated symmetric, non-negative, zero diagonal. Correlators symmetric.
lr=0 → no change. W changes with non-zero lr.

### Orchestrator Adapter

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `from_orchestrator_state` → `to_scpn_control_telemetry` | 3 layers | 0.07 ms | regime, R, stability |

Handles both dataclass and dict payloads. Legacy field names (locks, cross_alignment,
stability, regime) resolved automatically.

### Orchestrator Feedback

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_orchestrator_feedback(K, omega)` | 4 qubits | ~0.5 ms | action, confidence, R_global |

Actions: advance, hold, rollback. Confidence ∈ [0, 1]. R_global ∈ [0, 1].
Custom thresholds supported.

---

## 12. PGBO (Parameter-space Geometry Bridge)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_pgbo_tensor(K, omega)` | 4 qubits | 6.7 ms | metric (6×6), curvature (6×6) |

Quantum Fisher metric: symmetric, PSD (det >= 0). Berry curvature: antisymmetric
(traceless). Parameter count: C(n,2) upper-triangle couplings.

---

## 13. TCBO Observer

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compute_tcbo_observables(K, omega)` | 4 qubits | 4.6 ms | p_h1, TEE, string_order |

p_h1 ∈ [0, 1]. TEE finite. |string_order| <= 1. beta_0 + beta_1 ≈ 1
(connected components + loops = 1). Different coupling → different observables.

---

## 14. Trotter Error Analysis

### Commutator Bounds

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `commutator_norm_bound` + `optimal_dt` | 4 qubits | <0.1 ms | gamma=5.344, dt*=0.004, n_steps=268 |

Equal frequencies → gamma=0 (no Trotter error). Heterogeneous frequencies →
larger gamma. Second-order bound < first-order. Optimal dt respects epsilon target.

### Trotter Error Sweep

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `trotter_error_sweep(3t × 3reps)` | 3 qubits | 483 ms | 2D error map |

Error at t=0: < 1e-10. Error decreases with reps. Error increases with time.
Quadratic scaling: doubling t roughly quadruples error.

---

## 15. Experiment Registry

| Operation | Time | Output |
|-----------|------|--------|
| List all experiments | 0.18 ms | 20 registered experiments |

Every experiment has: runner as first param, docstring > 10 chars, lowercase
underscore name, no private experiments. At least half accept shots parameter.

---

## 16. Cutting Runner (Large-Scale)

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `run_cutting_simulation(n=16, max=8)` | 16 oscillators | 39.3 ms | 2 partitions, R=1.0 |
| `run_cutting_simulation(n=24, max=8)` | 24 oscillators | ~53 ms | 3 partitions |
| `run_cutting_simulation(n=32, max=8)` | 32 oscillators | ~60 ms | 4 partitions |

Partitions: ceil(n/max_partition_size). R per partition bounded [0, 1].
Combined R bounded [0, 1]. Energy estimate finite.

---

## 17. GUESS Symmetry-Decay ZNE (April 2026)

`mitigation/symmetry_decay.py` (Rust path: `fit_symmetry_decay`,
`guess_extrapolate_batch`).

| Operation | Input | Time | Notes |
|-----------|-------|------|-------|
| `learn_symmetry_decay` | 5 noise scales (Rust) | < 1 µs | least-squares α fit |
| `learn_symmetry_decay` | 5 noise scales (Python fallback) | < 1 µs | numpy polyfit |
| `guess_extrapolate` | single observable | < 0.1 µs | analytic correction |
| `guess_extrapolate_batch` | 1,000 observables (Rust) | < 50 µs | rayon-parallel |
| `guess_extrapolate_batch` | 100,000 observables (Rust) | < 2 ms | scales linearly |

Pipeline test: `tests/test_pipeline_wiring_performance.py::TestGUESSPipeline`.

## 18. DynQ Topology-Agnostic Qubit Mapper (April 2026)

`hardware/qubit_mapper.py` (Rust path: `score_regions_batch`).

| Operation | Input | Time |
|-----------|-------|------|
| `build_calibration_graph` | 156-qubit heavy-hex | < 5 ms |
| `detect_execution_regions` | 156 qubits | < 50 ms |
| `dynq_initial_layout` (full pipeline) | 156 qubits → 5-qubit layout | < 100 ms |
| `score_regions_batch` (Rust) | 100 regions × 50 qubits | < 5 ms |

Pipeline test: `tests/test_pipeline_wiring_performance.py::TestDynQPipeline`.

## 19. Pulse Shaping — ICI + (α,β)-Hypergeometric (April 2026)

`phase/pulse_shaping.py` (Rust paths: `hypergeometric_envelope_batch`,
`ici_mixing_angle_batch`, `ici_three_level_evolution_batch`).

| Operation | Input | Python | Rust | Speedup |
|-----------|-------|------:|-----:|--------:|
| `hypergeometric_envelope` | 200 points | 0.4 ms | 0.1 ms | 4× |
| `hypergeometric_envelope` | 10,000 points | 114.5 ms | 2.6 ms | **44×** |
| `ici_mixing_angle` | 1,000 points | 0.05 ms | 0.04 ms | parity-checked |
| `ici_three_level_evolution` | 2,000 points | 68.30 ms | 0.04 ms | **1,665×** |
| `build_trotter_pulse_schedule` | 4 qubits, K all-to-all | 1.2 ms | n/a | full schedule |

Verified parity (Rust vs Python): max abs diff $4.97 \times 10^{-14}$
for $n_\text{points} = 500$ on `ici_three_level_evolution`.

Pipeline test: `tests/test_pipeline_wiring_performance.py::TestPulseShapingPipeline`.

## 20. Phase 1 IBM Hardware Campaign (April 2026)

Real-world QPU runs on `ibm_kingston` (Heron r2, 156 q):

| Sub-phase | Circuits | Wall (queue + exec) | QPU rate |
|-----------|---------:|--------------------:|---------:|
| Pipe cleaner | 2 | ~0.1 s | 1.0 s/circuit |
| Phase 1 (A/B/C) | 42 | 44.1 s | 0.65 s/circuit |
| Phase 1.5 (D/E) | 72 | 56.7 s | 0.55 s/circuit |
| Phase 2 exhaust (F/G/H/I) | 138 | 97.5 s | 0.55 s/circuit |
| Phase 2.5 final burn (J) | 90 | 65.1 s | 0.55 s/circuit |
| **Total** | **344** | **~264 s** | **~0.55 s/circuit** |

Reproducible from `data/phase1_dla_parity/*.json` via
`scripts/analyse_phase1_dla_parity.py` (no QPU needed for the analysis).

---

## 21. Measured Rust speedups vs. Python baseline

Re-run from `tests/test_rust_path_benchmarks.py` on 2026-04-17
(ML350 Gen8, scpn-quantum-engine 0.2.0, PyO3 0.25 + rayon 1.10).
These are the only cross-language acceleration numbers we publish;
they are measured, not estimated.

| Function | Python | Rust | Speedup |
|----------|-------:|-----:|--------:|
| `build_knm` (16×16) | 0.1 ms | 0.01 ms | **18.4×** |
| `kuramoto_euler` (8 osc, 1 000 steps) | 2.3 ms | 0.25 ms | **9.3×** |
| `correlation_matrix_xy` (n=3) | 0.7 ms | 0.04 ms | **19.5×** |
| `lindblad_jump_ops_coo` (n=3) | 0.0 ms | 0.0 ms | **9.2×** |
| `lindblad_anti_hermitian_diag` (n=3) | 0.0 ms | 0.0 ms | **4.7×** |

Standalone Rust paths (no Python parity comparison; absolute wall time):

| Function | Configuration | Wall time |
|----------|---------------|----------:|
| `kuramoto_trajectory` | 16 osc, 10 000 steps | 11.79 ms |
| `expectation_pauli_fast` | 10 qubits, single-Z | 0.02 ms |
| `pec_sample_parallel` | 100 000 samples, 5 gates | 14.04 ms |
| `mc_xy_simulate` | 8 osc, 5k therm + 2k meas | 1.39 ms |
| `brute_mpc` | dim=3, horizon=4 | 0.27 ms |
| `parity_filter_mask` | 10 000 × 20-bit | 0.14 ms |

### Cross-language outlook

These measured numbers are the verification baseline against which
any future Mojo / Julia / Lean 4 investment is judged. As of
2026-04-17 we have **not** benchmarked Mojo or Julia in this
codebase; speedup claims of "5–12× (Mojo)" and "10–40× (Julia)" that
appear in vendor literature are not reproducible here without a
matching local prototype, and we do not publish unmeasured ranges.

Decision criteria for adopting a new acceleration backend:

1. Identify a specific compute-hot Python module that does **not**
   already have a Rust path. (Currently every module flagged in the
   internal Rust audit already has one — see `docs/rust_engine.md`.)
2. Build a minimal port and re-run the relevant section of
   `tests/test_rust_path_benchmarks.py` shape.
3. Publish the measured number in this table. Vague "X–Y faster"
   ranges are explicitly out of scope.
