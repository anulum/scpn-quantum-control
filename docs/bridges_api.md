# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Bridges API Reference

# Bridges API Reference

The `bridge` package is the central nervous system of scpn-quantum-control.
Every module in this package translates between classical SCPN state
representations and quantum operator formats. Without the bridge layer,
coupling matrices are numbers on paper; with it, they become executable
Hamiltonians, circuits, and feedback signals.

12 modules, 24 public symbols, 5 cross-repo integration points.

## Architecture

```
Classical World                    Bridge                     Quantum World
─────────────────                 ─────────                  ──────────────
K_nm (Paper 27)         ──→  knm_hamiltonian      ──→  SparsePauliOp (XY/XXZ)
K_nm (Paper 27)         ──→  sparse_hamiltonian    ──→  scipy.sparse.csc_matrix
K_nm (Paper 27)         ──→  knm_hamiltonian       ──→  QuantumCircuit (ansatz)
Plasma config           ──→  control_plasma_knm    ──→  K_nm → SparsePauliOp
Orchestrator state      ──→  orchestrator_adapter  ──→  UPDEPhaseArtifact
Quantum observables     ──→  orchestrator_feedback ──→  advance/hold/rollback
SC bitstreams           ←→   sc_to_quantum         ←→   Ry angles / statevectors
SNN spike trains        ──→  snn_adapter           ──→  QuantumDenseLayer output
Loss gradient           ←──  snn_backward          ←──  Parameter-shift rule
SPN weight matrices     ──→  spn_to_qcircuit       ──→  QuantumCircuit (CRy/anti-CRy)
SSGF W + theta          ←→   ssgf_adapter          ←→   Trotter evolution
SSGF W adaptation       ←──  ssgf_w_adapter        ←──  Quantum correlators
```

## Module Reference

### 1. `knm_hamiltonian` — Core Hamiltonian Compiler

The foundational module. Compiles the K_nm coupling matrix and natural
frequencies omega into Qiskit `SparsePauliOp` for quantum simulation.

**Kuramoto-XY mapping** (Paper 27, Section 3):

```
K[i,j] * sin(theta_j - theta_i)  ↔  -K[i,j] * (X_i X_j + Y_i Y_j)
omega_i                           ↔  -omega_i * Z_i
```

#### `OMEGA_N_16`

Canonical natural frequencies from Paper 27, Table 1. 16 values in rad/s,
experimentally calibrated. Used as the default frequency vector throughout
the entire package.

```python
OMEGA_N_16 = np.array([1.329, 2.610, 0.844, 1.520, 0.710, 3.780, 1.055, 0.625,
                        2.210, 1.740, 0.480, 3.210, 0.915, 1.410, 2.830, 0.991])
```

#### `build_knm_paper27(L=16, K_base=0.45, K_alpha=0.3)`

Builds the canonical K_nm matrix. Three-layer construction:

1. **Base kernel**: `K[i,j] = K_base * exp(-K_alpha * |i-j|)` (Eq. 3)
2. **Calibration anchors**: Table 2 overrides for (0,1), (1,2), (2,3), (3,4)
3. **Cross-hierarchy boosts**: S4.3 long-range couplings L1-L16, L5-L7

Returns symmetric (L, L) float64 array. Always positive, zero diagonal implicit
from the exponential decay (K[i,i] = K_base, but never used as self-coupling
in the Hamiltonian — the Z_i term handles on-site energy).

```python
K = build_knm_paper27(L=4)        # 4x4 submatrix
K_full = build_knm_paper27()       # 16x16 canonical
```

**Rust acceleration**: `scpn_quantum_engine.build_knm(L, K_base, K_alpha)` produces
identical output at 4.7x speedup. Parity verified to 1e-12 atol in
`test_rust_path_benchmarks.py`.

#### `build_kuramoto_ring(n, coupling=1.0, omega=None, rng_seed=None)`

Nearest-neighbour ring topology. Useful for BKT transition studies
and finite-size scaling where Paper 27's specific topology is not needed.

Returns `(K, omega)` tuple. If omega is None, draws from N(0,1).

#### `knm_to_hamiltonian(K, omega)`

The primary compiler. Converts K_nm + omega to `SparsePauliOp` using
Qiskit's little-endian qubit ordering.

```
H = -sum_{i<j} K[i,j] * (X_i X_j + Y_i Y_j) - sum_i omega_i * Z_i
```

Sparsity filtering: terms with |K[i,j]| < `COUPLING_SPARSITY_EPS` (1e-15)
are dropped. This keeps the Pauli list compact for large sparse K matrices
(Paper 27's K_nm has ~60% of off-diagonal entries below 0.01).

Equivalent to `knm_to_xxz_hamiltonian(K, omega, delta=0.0)`.

#### `knm_to_xxz_hamiltonian(K, omega, delta=0.0)`

Extended compiler with ZZ anisotropy parameter delta:

```
H = -sum_{i<j} K[i,j] * (X_iX_j + Y_iY_j + delta * Z_iZ_j) - sum_i omega_i * Z_i
```

| delta | Model | Physics |
|-------|-------|---------|
| 0.0 | XY | Standard Kuramoto mapping, in-plane S^2 dynamics |
| 1.0 | Heisenberg | Full S^2 dynamics (Kouchekian-Teodorescu 2025, arXiv:2601.00113) |
| -1.0..1.0 | XXZ | Interpolation, BKT/Ising transitions |

At delta=1, perturbations around equilibria connect to the semiclassical
Gaudin model and Richardson pairing mechanism.

#### `knm_to_dense_matrix(K, omega)`

Returns the full complex (2^n, 2^n) dense matrix. Two-path implementation:

1. **Rust fast path**: `scpn_quantum_engine.build_xy_hamiltonian_dense()` — exact
   parity with Qiskit verified to 1e-10 atol
2. **Qiskit fallback**: `knm_to_hamiltonian(K, omega).to_matrix()`

Used primarily for exact diagonalisation of small systems (n <= 14) and
as the ground truth reference for sparse eigensolvers.

#### `knm_to_ansatz(K, reps=2, threshold=0.01)`

Physics-informed variational ansatz. CZ entanglement gates placed only
between K_nm-connected pairs (|K[i,j]| >= threshold). Structure:

```
for each repetition:
    Ry(p[2k])   on each qubit k
    Rz(p[2k+1]) on each qubit k
    CZ(i, j)    for each connected pair
```

Returns parameterised `QuantumCircuit` with `n * 2 * reps` parameters.
The CZ topology mirrors the K_nm graph, encoding the coupling structure
into the ansatz architecture.

---

### 2. `sparse_hamiltonian` — Large-N Sparse Construction

For n >= 12, dense matrix construction is impractical (n=16: 32 GB).
This module builds the XY Hamiltonian directly as `scipy.sparse.csc_matrix`.

**Memory comparison:**

| n | Dense | Sparse | Reduction |
|---|-------|--------|-----------|
| 8 | 0.5 MB | 0.06 MB | 8x |
| 12 | 512 MB | 6 MB | 85x |
| 16 | 32 GB | 200 MB | 160x |
| 18 | 512 GB | 800 MB | 640x |
| 20 | 8 TB | 3 GB | 2700x |

#### `build_sparse_hamiltonian(K, omega)`

Constructs the full-space sparse Hamiltonian. Matrix elements:

- **Diagonal**: `H[k,k] = -sum_i omega_i * (1 - 2*b_i(k))` where b_i(k) is bit i of basis state k
- **Off-diagonal**: `H[k, k XOR mask_ij] = -2*K[i,j]` when bits i and j differ in state k

Rust fast path via `scpn_quantum_engine.build_sparse_xy_hamiltonian()` at 80x speedup.

#### `build_sparse_sector_hamiltonian(K, omega, M)`

Combines sparse construction with U(1) magnetisation symmetry. The XY model
conserves total magnetisation M = sum_i Z_i. Working within a single sector
reduces the Hilbert space dimension from 2^n to C(n, (n+M)/2).

For n=16, M=0 sector: dim = C(16,8) = 12,870 vs full 65,536 (5x reduction).
Combined with sparse storage: ~40 MB vs 32 GB dense full-space.

Returns `(H_sparse, sector_indices)`.

#### `sparse_eigsh(K, omega, k=10, which="SA", M=None)`

ARPACK eigensolver wrapper. Computes k smallest (or largest) eigenvalues.
Automatic fallback to dense `numpy.linalg.eigh` when the matrix is too small
for iterative methods (dim < k+2).

Returns dict: `{eigvals, eigvecs, nnz, dim, method, M, sector_dim}`.

#### `sparsity_stats(n, K)`

Estimates memory usage without constructing the matrix. Useful for
pre-flight checks before committing to large computations.

---

### 3. `control_plasma_knm` — Tokamak Plasma Bridge

Compatibility bridge to `scpn_control.phase.plasma_knm` for plasma-native
K_nm construction from tokamak parameters.

#### `build_knm_plasma(mode, L, K_base, zeta_uniform, ...)`

Delegates to `scpn-control` for plasma-specific coupling matrices. Returns
(L, L) float64 array.

#### `build_knm_plasma_from_config(R0, a, B0, Ip, n_e, ...)`

Constructs K_nm directly from tokamak machine parameters:
- `R0`: major radius (m)
- `a`: minor radius (m)
- `B0`: toroidal field (T)
- `Ip`: plasma current (MA)
- `n_e`: electron density (1e19/m^3)

#### `plasma_omega(L=8)`

Returns plasma natural frequencies from scpn-control.

All functions require `scpn-control` on sys.path. The bridge handles
`sys.path` insertion/cleanup for local development with `repo_src=` parameter.

---

### 4. `orchestrator_adapter` — Phase Orchestrator Integration

Bidirectional translation between scpn-phase-orchestrator state payloads
and the quantum bridge's `UPDEPhaseArtifact` schema.

#### `PhaseOrchestratorAdapter`

Static methods — no state, no side effects:

| Method | Direction | Description |
|--------|-----------|-------------|
| `from_orchestrator_state(state)` | Orch → Quantum | Parse orchestrator payload into `UPDEPhaseArtifact` |
| `to_orchestrator_payload(artifact)` | Quantum → Orch | Emit canonical orchestrator field names |
| `to_scpn_control_telemetry(artifact)` | Quantum → Control | Emit scpn-control compatible telemetry |
| `build_knm_from_binding_spec(spec)` | Orch → K_nm | Extract coupling matrix from BindingSpec |
| `build_omega_from_binding_spec(spec)` | Orch → omega | Extract per-oscillator frequencies |

The adapter uses duck-typed field resolution (`_read_field`) that accepts
both dict and object attributes, making it compatible with Pydantic models,
dataclasses, and plain dicts.

**Roundtrip guarantee**: `to_orchestrator_payload(from_orchestrator_state(x))`
produces a dict structurally equivalent to the input. Verified in
`test_pipeline_wiring_performance.py`.

---

### 5. `orchestrator_feedback` — Quantum Decision Loop

Closes the cybernetic feedback loop: quantum observables drive orchestrator
phase lifecycle decisions.

#### `OrchestratorFeedback`

Dataclass with fields: `action`, `r_global`, `stability_score`, `l16_action`,
`confidence`, `reason`.

#### `compute_orchestrator_feedback(K, omega, r_advance=0.8, r_hold=0.5)`

Computes quantum-informed feedback using `compute_l16_lyapunov` from the
L16 quantum director module.

Decision logic:

```
R >= 0.8 AND stable  →  "advance"   (proceed to next phase)
R >= 0.5             →  "hold"      (continue monitoring)
R <  0.5             →  "rollback"  (return to previous phase)
```

Confidence is computed as a normalised score within the active regime:
- advance: `min(R, stability)`
- hold: `(R - r_hold) / (r_advance - r_hold)`
- rollback: `1.0 - R / r_hold`

---

### 6. `phase_artifact` — Interoperability Schema

Three frozen dataclasses defining the canonical state exchange format
between classical and quantum subsystems.

#### `LockSignatureArtifact`

Pairwise phase-locking metrics (Lachaux et al., HBM 1999):
- `source_layer`, `target_layer`: int indices
- `plv`: Phase Locking Value in [0, 1]
- `mean_lag`: mean phase difference at PLV maximum (radians)

Validation: finite floats, non-negative layer indices.

#### `LayerStateArtifact`

Per-layer coherence:
- `R`: Kuramoto order parameter |z| in [0, 1]
- `psi`: mean phase angle arg(z) in radians
- `lock_signatures`: dict of `LockSignatureArtifact`

Validation: R in [0, 1], finite floats, string keys.

#### `UPDEPhaseArtifact`

Top-level artifact containing:
- `layers`: list of `LayerStateArtifact`
- `cross_layer_alignment`: (n_layers, n_layers) float64 matrix
- `stability_proxy`: scalar stability measure
- `regime_id`: non-empty string identifier
- `metadata`: arbitrary dict

Full serialisation support: `to_dict()`, `to_json()`, `from_dict()`, `from_json()`.
Validation ensures alignment matrix shape matches layer count and all
values are finite.

---

### 7. `sc_to_quantum` — Stochastic Computing Bridge

Maps between stochastic computing probabilities and qubit rotation angles.

**Core identity**: For `Ry(theta)|0>`, the probability of measuring |1> is
`P(|1>) = sin^2(theta/2)`.

| Function | Direction | Formula |
|----------|-----------|---------|
| `probability_to_angle(p)` | SC → Quantum | `theta = 2 * arcsin(sqrt(p))` |
| `angle_to_probability(theta)` | Quantum → SC | `P = sin^2(theta/2)` |
| `bitstream_to_statevector(bits)` | SC → Quantum | Mean probability → single-qubit `[alpha, beta]` |
| `measurement_to_bitstream(counts, length)` | Quantum → SC | Shot counts → Bernoulli bitstream |

These functions enable the SPN/SNN layers of SCPN to exchange state
with quantum circuits through the bitstream probability interface.

---

### 8. `snn_adapter` — Spiking Neural Network Bridge

Bidirectional bridge between spike trains and quantum circuits.

#### Standalone Functions

```python
spike_train_to_rotations(spikes, window=10) -> np.ndarray
```
Converts (timesteps, n_neurons) binary spike array to Ry rotation angles.
Uses firing rate within the last `window` timesteps: `angle = rate * pi`.
Output in [0, pi].

```python
quantum_measurement_to_current(values, scale=1.0) -> np.ndarray
```
Converts quantum P(|1>) probabilities to SNN input currents. Linear scaling.

#### `SNNQuantumBridge`

Pure-numpy bridge (no sc-neurocore dependency):

```python
bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
output = bridge.forward(spike_history)  # (T, n_inputs) → (n_neurons,)
```

Internal pipeline: spike rates → Ry angles → `QuantumDenseLayer` → P(|1>) → currents.

#### `ArcaneNeuronBridge`

Full sc-neurocore integration with `ArcaneNeuron`:

```python
bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
result = bridge.step(np.array([1.0, 0.5, 0.0]))
# result["spikes"]: binary spike vector
# result["output_currents"]: quantum layer output
# result["v_deep"]: identity state (persists across reset)
# result["confidence"]: neuron confidence
```

Key property: `v_deep` persists through `reset()`, implementing the identity
binding mechanism from the Arcane Sapience specification.

---

### 9. `snn_backward` — Parameter-Shift Gradient

Enables end-to-end training of the SNN-quantum hybrid via the parameter-shift
rule.

#### Gradient Chain

```
SNN forward → spike rates → theta (Ry angles) → quantum evolution → y
                                                                    ↓
Loss L(y, target) ← dL/dy ← dy/dtheta (param-shift) ← dtheta/d(rates) = pi
```

#### `parameter_shift_gradient(layer, input_values, target, shift=0.25)`

Computes `dL/d(input)` for MSE loss:

```
dL/dtheta_k = [L(theta_k + pi/4) - L(theta_k - pi/4)] / (actual_shift)
dL/d(spike_rate) = dL/dtheta * pi
```

Cost: 2 quantum forward passes per parameter (2n total for n inputs).

Returns `BackwardResult`: `grad_params`, `grad_spikes`, `loss`, `n_evaluations`.

Boundary handling: shift is clamped to [0, 1] value range to prevent
invalid rotation angles. If both bounds are hit, gradient is zero.

---

### 10. `spn_to_qcircuit` — Petri Net Circuit Compiler

Compiles Stochastic Petri Net (SPN) topology into quantum circuits.

**Mapping**:
- Places → qubits (amplitude encodes token density)
- Transitions → controlled-Ry gates (arc weights → rotation angles)
- Inhibitor arcs → anti-control pattern: X-CRy-X

#### `spn_to_circuit(W_in, W_out, thresholds)`

Args:
- `W_in`: (n_transitions, n_places) — input arc weights. Negative = inhibitor.
- `W_out`: (n_places, n_transitions) — output arc weights.
- `thresholds`: (n_transitions,) — firing thresholds.

For each transition t:
1. Normal input arcs: `Ry(-theta * threshold[t])` on input places
2. Output arcs: `Ry(theta)` or anti-controlled `Ry(theta)` if inhibitors present

#### `inhibitor_anti_control(circuit, inhibitor_qubits, target, theta)`

Anti-control pattern for inhibitor arcs. An inhibitor arc requires the
source place to be empty (|0>) for the transition to fire. Implementation:

```
X on each inhibitor qubit  (flip control sense)
CRy(theta) controlled by inhibitor qubits, target on output qubit
X on each inhibitor qubit  (restore)
```

For multi-qubit inhibition: uses `RYGate.control(n)` for n > 1 controls.

---

### 11. `ssgf_adapter` — SSGF Quantum Loop

Bidirectional bridge between the Self-Sustaining Geometry Field (SSGF)
engine and quantum evolution.

#### Standalone Functions

```python
ssgf_w_to_hamiltonian(W, omega) -> SparsePauliOp
```
W has the same structure as K_nm (symmetric, non-negative). Delegates
directly to `knm_to_hamiltonian`.

```python
ssgf_state_to_quantum({"theta": [...]}) -> QuantumCircuit
```
Encodes oscillator phases as `Ry(pi/2) Rz(theta_i)` per qubit,
producing `(|0> + e^{i*theta}|1>) / sqrt(2)`. This preserves phase
information in `<X> = cos(theta)`, `<Y> = sin(theta)`.

```python
quantum_to_ssgf_state(statevector, n_osc) -> {"theta": [...], "R_global": float}
```
Extracts phases via `theta_i = atan2(<Y_i>, <X_i>)` and computes
`R_global = |mean(exp(i*theta))|`.

#### `SSGFQuantumLoop`

Quantum-in-the-loop wrapper for SSGFEngine:

```python
loop = SSGFQuantumLoop(engine, dt=0.1, trotter_reps=3)
result = loop.quantum_step()
```

Each step:
1. Read W matrix and theta phases from SSGFEngine
2. Compile W → SparsePauliOp via `knm_to_hamiltonian`
3. Encode theta → quantum circuit (`Ry(pi/2) Rz(theta)`)
4. Trotter evolve via `PauliEvolutionGate(LieTrotter)`
5. Extract updated theta and R_global from evolved statevector
6. Write theta back to SSGFEngine (in-place mutation)

Omega is set to zeros because SSGF handles natural frequencies internally;
only the coupling structure W enters the quantum Hamiltonian.

---

### 12. `ssgf_w_adapter` — Geometry Adaptation

Closes the SSGF feedback loop: quantum correlators modify the geometry
matrix W, not just the phases theta.

#### Update Rule

```
W_new[i,j] = W_old[i,j] + eta * (-delta_R) * C[i,j]
```

Where:
- `eta`: learning rate
- `delta_R = R_quantum - R_target`: synchronisation error
- `C[i,j] = <X_i X_j + Y_i Y_j>`: quantum XY correlator

Positive correlator with R below target → strengthen coupling.
Negative correlator → weaken coupling (anti-correlated oscillators).

#### `adapt_w_from_quantum(W, theta, r_target=0.9, learning_rate=0.01, ...)`

One adaptation step. Returns `WAdaptResult`:
- `W_updated`: new geometry matrix (non-negative, zero diagonal enforced)
- `r_global`: measured quantum synchronisation
- `delta_r`: gap to target
- `correlators`: (n, n) XY correlator matrix
- `max_update`: largest absolute element change

The correlator measurement requires `O(n^2)` Pauli expectation values,
each computed from the full statevector. For n=8 this is 28 pairs.

---

## Cross-Repository Dependencies

| Module | External Package | Required? |
|--------|-----------------|-----------|
| `control_plasma_knm` | scpn-control | Optional (ImportError with instructions) |
| `snn_adapter.ArcaneNeuronBridge` | sc-neurocore >= 3.14 | Optional (ImportError with instructions) |
| `ssgf_adapter.SSGFQuantumLoop` | SCPN-CODEBASE (SSGFEngine) | Optional (runtime only) |
| `orchestrator_adapter` | scpn-phase-orchestrator | No (works with any dict/object) |
| `orchestrator_feedback` | l16.quantum_director | Internal dependency |

All optional dependencies fail gracefully with `ImportError` and clear
installation instructions. The core modules (`knm_hamiltonian`, `sparse_hamiltonian`,
`sc_to_quantum`, `spn_to_qcircuit`) have zero optional dependencies beyond
Qiskit, NumPy, and SciPy.

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Pipeline | System Size | Wall Time |
|----------|------------|-----------|
| `build_knm_paper27` → `knm_to_hamiltonian` | 4 qubits | 0.3 ms |
| `build_knm_paper27` → `knm_to_dense_matrix` (Rust) | 4 qubits | 0.1 ms |
| `build_knm_paper27` → `knm_to_ansatz` | 4 qubits, 2 reps | 0.5 ms |
| `build_sparse_hamiltonian` → `sparse_eigsh` | 8 qubits | 12 ms |
| `build_sparse_hamiltonian` → `sparse_eigsh` | 12 qubits | 340 ms |
| `SNNQuantumBridge.forward` | 3 inputs, 2 neurons | 4 ms |
| `spn_to_circuit` | 3 places, 2 transitions | 0.4 ms |
| `ssgf_w_to_hamiltonian` → Trotter evolve | 4 oscillators | 8 ms |
| `PhaseOrchestratorAdapter` roundtrip | 3 layers | 0.2 ms |
| `compute_orchestrator_feedback` | 4 qubits | 15 ms |

## Rust Acceleration Summary

| Function | Bridge Module | Speedup | Parity |
|----------|--------------|---------|--------|
| `build_knm` | `knm_hamiltonian` | 4.7x | 1e-12 atol |
| `build_xy_hamiltonian_dense` | `knm_hamiltonian` | exact | 1e-10 atol |
| `build_sparse_xy_hamiltonian` | `sparse_hamiltonian` | 80x | exact index match |

All Rust paths are optional. Python fallbacks produce identical results.
Rust availability is detected at call time via `try/except ImportError`.

## Testing

59 tests across 6 test files covering the bridge package:

- `test_knm_hamiltonian.py` — Hermiticity, eigenvalues, sparsity, XXZ, ansatz
- `test_sparse_hamiltonian.py` — Sparse vs dense parity, sector Hamiltonian, eigsh
- `test_orchestrator_adapter.py` — Roundtrip, field resolution, telemetry format
- `test_snn_adapter.py` — Spike rotations, bridge forward, ArcaneNeuron integration
- `test_spn_to_qcircuit.py` — Circuit depth, inhibitor pattern, weight encoding
- `test_ssgf_adapter.py` — W→H compilation, phase encoding/extraction, quantum loop

Every test file includes physical invariant checks, pipeline wiring verification,
and performance benchmarks. The bridge package has zero skipped tests when all
optional dependencies are installed.
