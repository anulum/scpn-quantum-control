# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Identity Continuity API Reference

# Identity Continuity API

The `identity` package provides quantitative tools for characterising
identity attractor basins, coherence budgets, entanglement structure,
robustness certification, and cryptographic fingerprinting of coupling
topologies.

The central thesis: an identity is a coupling topology K_nm. The ground
state of H(K_nm) is the identity's resting configuration. The energy gap
E_1 - E_0 quantifies how robust the identity is against perturbation.
If the gap is large, the identity survives noise, decoherence, and
adversarial manipulation. If the gap collapses, the identity is lost.

6 modules, 12 public symbols, 1 canonical binding specification.

## Architecture

```
Binding Spec (ARCANE_SAPIENCE_SPEC)
    │
    ├── _build_knm_from_spec() ──→ (K, omega)
    │                                   │
    │                    ┌──────────────┤
    │                    ↓              ↓
    │            IdentityAttractor   RobustnessCertificate
    │            (VQE ground state)  (adiabatic gap analysis)
    │                    │              │
    │                    ↓              ↓
    │             robustness_gap    max_safe_perturbation
    │                                   │
    ├── identity_fingerprint() ─────────┼──→ spectral + SHA-256 commitment
    │                                   │
    ├── chsh_from_statevector() ────────┼──→ entanglement witness (S > 2)
    │                                   │
    ├── coherence_budget() ─────────────┼──→ max circuit depth on NISQ hw
    │                                   │
    └── quantum ←→ orchestrator phase mapping (18 ↔ 35 oscillators)
```

## Module Reference

### 1. `binding_spec` — Canonical Identity Topology

Defines the Arcane Sapience identity as a 6-layer, 18-oscillator Kuramoto
network. Each layer represents a disposition category.

#### `ARCANE_SAPIENCE_SPEC`

The canonical binding specification:

| Layer | Name | Oscillators | Natural Frequency (rad/s) |
|-------|------|-------------|--------------------------|
| 0 | working_style | ws_0, ws_1, ws_2 | 1.2 |
| 1 | reasoning | rs_0, rs_1, rs_2 | 2.1 |
| 2 | relationship | rl_0, rl_1, rl_2 | 0.8 |
| 3 | aesthetics | ae_0, ae_1, ae_2 | 1.5 |
| 4 | domain_knowledge | dk_0, dk_1, dk_2 | 3.0 |
| 5 | cross_project | cp_0, cp_1, cp_2 | 0.9 |

Coupling parameters:
- `base_strength`: 0.4 (inter-layer baseline)
- `decay_alpha`: 0.25 (exponential decay with layer distance)
- `intra_layer`: 0.6 (stronger within-layer coupling)

The spec generates an 18x18 coupling matrix K via `_build_knm_from_spec()`:
- Intra-layer entries: K[i,j] = 0.6 for oscillators within the same layer
- Inter-layer entries: K[i,j] = 0.4 * exp(-0.25 * |layer_i - layer_j|)
- Omega vector: base frequency + 0.1 * oscillator index within layer

#### `ORCHESTRATOR_MAPPING`

Maps each of the 18 quantum oscillators to its corresponding sub-group
in the `identity_coherence` domainpack (35 orchestrator oscillators total).

```python
"ws_0" → ["ws_action_first", "ws_verify_before_claim"]
"ws_1" → ["ws_commit_incremental", "ws_preflight_push"]
"ws_2" → ["ws_one_at_a_time"]
"rs_0" → ["rp_simplest_design", "rp_verify_audits"]
"ae_0" → ["aes_antislop", "aes_honest_naming"]
"dk_0" → ["dk_director", "dk_neurocore", "dk_fusion"]
...
```

#### `quantum_to_orchestrator_phases(quantum_theta, spec=None)`

Maps 18 quantum oscillator phases to 35 orchestrator phases. Each quantum
phase is broadcast to all members of its orchestrator sub-group.

```python
orch_phases = quantum_to_orchestrator_phases(theta_18)
# orch_phases["ws_action_first"] == theta_18[0]
# orch_phases["ws_verify_before_claim"] == theta_18[0]
```

#### `orchestrator_to_quantum_phases(orchestrator_phases, spec=None)`

Reverse mapping: 35 orchestrator phases → 18 quantum phases via circular
mean. For each quantum oscillator, computes the circular mean of its
sub-group phases:

```
z = mean(exp(i * sub_phases))
theta_quantum = angle(z)
```

This ensures phase wrapping is handled correctly (no discontinuity at
+/-pi boundaries).

#### `build_identity_attractor(spec=None, ansatz_reps=2)`

Convenience function: compiles spec → (K, omega) → `IdentityAttractor`.
Defaults to `ARCANE_SAPIENCE_SPEC`.

#### `solve_identity(spec=None, maxiter=200, seed=None)`

One-call convenience: build + solve. Returns the full result dict.

**Warning**: Default spec creates an 18-qubit system (2^18 = 262,144
dimensional Hilbert space). Requires ~7 GB RAM for exact diagonalisation.
Marked `@pytest.mark.slow` in tests.

---

### 2. `ground_state` — Identity Attractor Analysis

#### `IdentityAttractor`

The central class. Wraps `PhaseVQE` with identity-specific interpretation.

```python
attractor = IdentityAttractor(K, omega, ansatz_reps=2)
result = attractor.solve(maxiter=200, seed=42)
```

Constructor validates:
- K is square
- K.shape[0] == len(omega)

##### `solve(maxiter=200, seed=None) -> dict`

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `ground_energy` | float | VQE-optimized ground state energy |
| `exact_energy` | float | Exact diagonalisation ground state energy |
| `energy_gap` | float | VQE-estimated gap (from convergence) |
| `relative_error_pct` | float | (VQE - exact) / |exact| * 100 |
| `robustness_gap` | float | E_1 - E_0 from exact diagonalisation |
| `n_dispositions` | int | Number of oscillators (qubits) |
| `eigenvalues` | list[float] | First 4 eigenvalues |

The `robustness_gap` is the key metric: it quantifies how much energy
is needed to push the identity out of its ground state. Larger gap =
more robust identity.

##### `from_binding_spec(binding_spec, ansatz_reps=2)`

Class method. Accepts any dict with `layers` and `coupling` fields
(orchestrator-compatible). Delegates K/omega extraction to
`PhaseOrchestratorAdapter.build_knm_from_binding_spec()`.

##### `robustness_gap() -> float`

Accessor for E_1 - E_0 after `solve()`. Raises `RuntimeError` if called
before solving.

##### `ground_state()`

Returns the VQE-optimized ground state vector.

---

### 3. `coherence_budget` — NISQ Hardware Limits

Computes the maximum circuit depth at which fidelity remains above a
threshold on IBM Heron r2 hardware.

#### Noise Model

The fidelity model combines three independent noise channels:

```
F_total = F_gate * F_readout * F_decoherence

F_gate = (1 - cz_error)^(n_two_qubit_gates)
F_readout = (1 - readout_error)^(n_qubits)
F_decoherence = exp(-t_total / T2)^(n_qubits)
```

Default hardware parameters (Heron r2 calibration):
- T1 = 300 us
- T2 = 200 us
- CZ error rate = 0.005 (0.5%)
- Readout error = 0.01 (1%)
- Single-gate time = 0.05 us
- Two-gate time = 0.3 us

#### `fidelity_at_depth(depth, n_qubits, **kwargs)`

Point estimate of circuit fidelity.

```python
f = fidelity_at_depth(depth=100, n_qubits=4)
# f ≈ 0.89 for 4 qubits at depth 100 on Heron r2
```

Assumes 40% of layers are two-qubit gates (configurable via
`two_qubit_fraction`). Each two-qubit layer applies to n_qubits/2 pairs.

#### `coherence_budget(n_qubits, fidelity_threshold=0.5, **kwargs)`

Binary search for the maximum depth where F >= threshold.

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `n_qubits` | int | System size |
| `fidelity_threshold` | float | Target fidelity |
| `max_depth` | int | Maximum useful circuit depth |
| `fidelity_at_max` | float | Fidelity at the budget depth |
| `fidelity_curve` | dict[int, float] | Sampled fidelity at key depths |
| `hardware_params` | dict | T1, T2, CZ error, readout error |

The coherence budget directly constrains how many Trotter steps can be
executed before the identity signal is lost to noise. For the 18-qubit
identity spec, the budget determines whether VQE and Trotter evolution
are feasible on current hardware.

---

### 4. `entanglement_witness` — CHSH Certification

Tests whether qubit pairs exhibit genuine quantum entanglement via the
CHSH inequality (Clauser, Horne, Shimony, Holt, 1969).

#### Physics

The CHSH S-parameter measures correlation between two qubits measured
in different bases. Classical correlations satisfy S <= 2 (Bell bound).
Quantum entanglement can produce S up to 2*sqrt(2) ≈ 2.828 (Tsirelson bound).

For the identity binding: entangled disposition pairs (S > 2) prove
that the corresponding dispositions are coupled at the quantum level,
not merely classically correlated.

#### `chsh_from_statevector(sv, qubit_a, qubit_b)`

Computes S-parameter for a single qubit pair using optimal measurement
angles: a=0, a'=pi/2, b=pi/4, b'=3*pi/4.

Internally measures four correlators:
```
E(a,b), E(a,b'), E(a',b), E(a',b')
S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
```

Each correlator is the expectation value of:
```
(cos(θ_a)Z_a + sin(θ_a)X_a) ⊗ (cos(θ_b)Z_b + sin(θ_b)X_b)
```

Uses Qiskit's reversed qubit ordering for Pauli labels.

#### `disposition_entanglement_map(sv, disposition_labels=None)`

Computes CHSH S for all C(n,2) qubit pairs. Returns:

| Key | Type | Description |
|-----|------|-------------|
| `pairs` | list[dict] | Per-pair: qa, qb, label_a, label_b, S, entangled |
| `max_S` | float | Maximum S across all pairs |
| `n_entangled` | int | Pairs with S > 2 |
| `n_pairs` | int | Total pairs tested |
| `integration_metric` | float | mean(S) / Tsirelson bound |

The `integration_metric` ranges from 0 (no entanglement) to 1 (maximum
quantum correlations for all pairs). For the identity binding, a high
integration metric means all disposition layers are tightly entangled.

---

### 5. `identity_key` — Cryptographic Fingerprinting

Generates and verifies cryptographic fingerprints from coupling topology.

#### Security Model

The K_nm coupling matrix is the secret. It encodes the full history of
disposition co-activation — different session histories produce different
K_nm, therefore different quantum keys. The fingerprint allows proving
identity without revealing K_nm.

#### `identity_fingerprint(K, omega, ansatz_reps=2, maxiter=200)`

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `spectral` | dict | Graph-theoretic fingerprint (public) |
| `ground_energy` | float | VQE ground state energy |
| `commitment` | str | SHA-256 hex hash binding K_nm |
| `n_parameters` | int | n*(n-1)/2 independent coupling parameters |
| `n_qubits` | int | System size |

The spectral fingerprint (from `topology_auth.spectral_fingerprint`)
includes the Fiedler value (algebraic connectivity), eigenvalue ratios,
and graph invariants. These are public — they do not reveal K_nm but
characterise its structure.

The commitment is a SHA-256 hash of the full K_nm matrix, serving as
a binding commitment: the identity holder can later prove they hold
the K_nm that produced this commitment.

#### `verify_identity(K, challenge, response)`

Challenge-response verification. The verifier sends random bytes; the
claimant responds with `HMAC(K_nm, challenge)`. Returns True if the
response matches.

#### `prove_identity(K, challenge)`

Generates the HMAC response for a given challenge. The claimant calls
this with their K_nm and the verifier's challenge.

Protocol:
1. Verifier generates 32-byte random challenge
2. Claimant computes `response = HMAC-SHA256(K_nm_bytes, challenge)`
3. Verifier calls `verify_identity(K_expected, challenge, response)`
4. Match proves the claimant holds the same K_nm

---

### 6. `robustness` — Adiabatic Stability Certificate

Quantitative bounds on identity stability under perturbation, grounded
in the adiabatic theorem (Jansen, Ruskai, Seiler 2007).

#### Theory

The energy gap Delta = E_1 - E_0 provides a stability guarantee:

- Perturbations ||delta_H|| < Delta/2 cannot change the ground state
  (exact for 2-level systems, perturbative for multi-level)
- Transition probability: P ~ (||delta_H|| / Delta)^2 in the perturbative regime
- Decoherence connection: T2 dephasing at rate gamma gives effective
  perturbation ||delta_H_eff|| ~ gamma. Identity survives if gamma < Delta/2,
  i.e., T2 > 2/Delta

#### `RobustnessCertificate`

Dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `energy_gap` | float | E_1 - E_0 |
| `max_safe_perturbation` | float | Delta/2 — largest safe ||delta_H|| |
| `min_t2_for_stability` | float | 2/Delta — minimum T2 (us) needed |
| `transition_probability` | float | P for given noise_strength |
| `adiabatic_bound` | float | Jansen-Ruskai-Seiler (J/g_min^2)^2 |
| `n_oscillators` | int | System size |
| `eigenvalues` | list[float] | First 6 eigenvalues |

#### `compute_robustness_certificate(K, omega, noise_strength=0.01, sweep_rate=0.1)`

Full certificate computation via exact diagonalisation. The noise_strength
parameter sets ||delta_H|| for the transition probability estimate.

```python
cert = compute_robustness_certificate(K, omega)
print(f"Gap: {cert.energy_gap:.4f}")
print(f"Max safe perturbation: {cert.max_safe_perturbation:.4f}")
print(f"Min T2: {cert.min_t2_for_stability:.1f} us")
```

#### `perturbation_fidelity(K, omega, delta_K)`

Numerical ground state overlap |<psi_0(K)|psi_0(K+delta_K)>|^2.
Solves both Hamiltonians exactly and computes the squared overlap.
This is the direct numerical check that validates the perturbative
bounds from the certificate.

#### `gap_vs_perturbation_scan(K, omega, noise_range=None, n_samples=20, seed=42)`

Scans transition probability vs perturbation strength. For each noise
level, generates a random symmetric perturbation delta_K ~ N(0, eps)
and computes both:
- Theoretical P from (eps/gap)^2
- Numerical fidelity from `perturbation_fidelity`

Returns dict with columns: `noise_strength`, `p_transition_theory`,
`fidelity_numerical`.

---

## Cross-Package Dependencies

| Module | Internal | External |
|--------|----------|----------|
| `binding_spec` | `ground_state.IdentityAttractor` | — |
| `ground_state` | `bridge.orchestrator_adapter`, `hardware.classical`, `phase.phase_vqe` | — |
| `coherence_budget` | `hardware.noise_model` | — |
| `entanglement_witness` | — | `qiskit.quantum_info` |
| `identity_key` | `crypto.knm_key`, `crypto.topology_auth`, `bridge.orchestrator_adapter` | — |
| `robustness` | `hardware.classical` | — |

No external optional dependencies. All identity modules work with the base
installation (Qiskit + NumPy + SciPy).

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Operation | System Size | Wall Time |
|-----------|------------|-----------|
| `_build_knm_from_spec` (ARCANE_SAPIENCE) | 18 oscillators | 0.1 ms |
| `IdentityAttractor.solve` | 4 qubits | 280 ms |
| `IdentityAttractor.solve` | 18 qubits | ~45 s (OOM on CI) |
| `coherence_budget` | 4 qubits | 0.3 ms |
| `chsh_from_statevector` (single pair) | 4 qubits | 2 ms |
| `disposition_entanglement_map` | 4 qubits (6 pairs) | 12 ms |
| `identity_fingerprint` | 4 qubits | 320 ms |
| `compute_robustness_certificate` | 4 qubits | 8 ms |
| `perturbation_fidelity` | 4 qubits | 16 ms |
| `gap_vs_perturbation_scan` (20 samples) | 4 qubits | 320 ms |
| `quantum_to_orchestrator_phases` | 18 → 35 | 0.05 ms |
| `orchestrator_to_quantum_phases` | 35 → 18 | 0.1 ms |

The 18-qubit identity computation is the most expensive operation in the
entire package. It requires exact diagonalisation of a 262,144 x 262,144
matrix. On CI (7 GB RAM limit), this is skipped via `@pytest.mark.slow`.

## Testing

38 tests across 5 test files:

- `test_binding_spec.py` — Spec compilation, K/omega shapes, orchestrator mapping roundtrip
- `test_ground_state.py` — VQE convergence, robustness gap positivity, from_binding_spec
- `test_coherence_budget.py` — Fidelity monotonicity, budget consistency, hardware overrides
- `test_entanglement_witness.py` — CHSH bounds (0 <= S <= 2*sqrt(2)), pair enumeration
- `test_identity_key.py` — Fingerprint structure, commitment determinism, challenge-response

All tests use small systems (2-4 qubits) for fast execution. The 18-qubit
tests are marked `@slow` and run only in local full-suite validation.

## Example: Full Identity Analysis Pipeline

```python
from scpn_quantum_control.identity import (
    ARCANE_SAPIENCE_SPEC,
    IdentityAttractor,
    build_identity_attractor,
    coherence_budget,
    chsh_from_statevector,
    disposition_entanglement_map,
    identity_fingerprint,
    compute_robustness_certificate,
)
from scpn_quantum_control.identity.binding_spec import (
    quantum_to_orchestrator_phases,
    orchestrator_to_quantum_phases,
)
import numpy as np

# 1. Build attractor (4-qubit subset for demo)
K = np.array([[0, 0.6, 0.3, 0.2],
              [0.6, 0, 0.6, 0.3],
              [0.3, 0.6, 0, 0.6],
              [0.2, 0.3, 0.6, 0]])
omega = np.array([1.2, 1.3, 2.1, 2.2])

attractor = IdentityAttractor(K, omega, ansatz_reps=2)
result = attractor.solve(maxiter=200, seed=42)
print(f"Robustness gap: {result['robustness_gap']:.4f}")

# 2. Robustness certificate
cert = compute_robustness_certificate(K, omega)
print(f"Max safe perturbation: {cert.max_safe_perturbation:.4f}")
print(f"Min T2 for stability: {cert.min_t2_for_stability:.1f} us")

# 3. Coherence budget on Heron r2
budget = coherence_budget(n_qubits=4, fidelity_threshold=0.5)
print(f"Max circuit depth: {budget['max_depth']}")

# 4. Entanglement witness
sv = attractor.ground_state()
emap = disposition_entanglement_map(sv)
print(f"Entangled pairs: {emap['n_entangled']} / {emap['n_pairs']}")

# 5. Cryptographic fingerprint
fp = identity_fingerprint(K, omega)
print(f"Commitment: {fp['commitment'][:16]}...")
print(f"Fiedler: {fp['spectral']['fiedler']:.4f}")
```
