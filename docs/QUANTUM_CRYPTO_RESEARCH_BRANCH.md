# Quantum Cryptography Research Branch

## Thesis

The SCPN coupling matrix K_nm encodes oscillator topology as quantum
entanglement structure under the Kuramoto-XY isomorphism. Parties who
share K_nm can generate correlated measurement statistics from the
Hamiltonian's ground state — an eavesdropper without K_nm cannot
reconstruct these correlations. This gives a topology-authenticated
QKD protocol where the shared secret is not a bit string but a
physical coupling matrix.

## Literature Grounding

| Paper | Key Result | K_nm Connection |
|-------|-----------|-----------------|
| Frequency-bin QKD (npj Quantum Inf. 2025) | Coupled ring resonators implement BBM92 via coupling topology | K_nm maps to photonic coupling graph |
| Huygens quantum sync (Nature Comms. 2025) | Coupled oscillators under shared noise co-generate phase sync + entanglement | Kuramoto sync = entanglement generation |
| Quantum network science (AAAI 2025) | Fidelity-weighted graph structure determines key rate | K_nm spectral properties set QKD channel capacity |
| Entanglement percolation (arXiv 2026) | Mixed-state threshold lower than pure-state | Weak K_nm entries still contribute above threshold |
| Hierarchical group QKD (Sci. Reports 2025) | Multi-party protocol with sub-group key derivation | 16-layer hierarchy = natural key tree |

## Module Architecture

```
scpn_quantum_control/
└── crypto/                    # Topology-authenticated quantum cryptography
    ├── __init__.py
    ├── knm_key.py             # K_nm → key material pipeline
    ├── topology_auth.py       # Spectral fingerprint authentication
    ├── entanglement_qkd.py    # Entanglement-based key distribution
    ├── percolation.py         # K_nm entanglement percolation analysis
    ├── hierarchical_keys.py   # Multi-layer key derivation from SCPN
    └── noise_analysis.py      # Devetak-Winter key rates, noise channels

tests/
├── test_knm_key.py
├── test_topology_auth.py
├── test_entanglement_qkd.py
├── test_percolation.py
├── test_hierarchical_keys.py
└── test_noise_analysis.py
```

## Module Specifications

### 1. `knm_key.py` — Coupling Matrix to Key Material

**Core primitive**: Given shared K_nm and agreed Hamiltonian H(K_nm),
prepare the ground state |ψ₀⟩, measure in agreed basis, extract
correlated bit string.

```
Pipeline:
  K_nm → knm_to_hamiltonian(K) → VQE ground state → measurement → sift → key
```

**Functions:**
- `prepare_key_state(K, omega, ansatz_reps)` → QuantumCircuit
  Builds VQE-optimized circuit encoding K_nm's ground state.
- `extract_raw_key(counts, basis, n_qubits)` → BitArray
  Sifts measurement results into raw key bits.
- `estimate_qber(alice_bits, bob_bits)` → float
  Quantum bit error rate from shared subset.
- `privacy_amplification(raw_key, qber)` → SecureKey
  Universal₂ hash family compression.

**Security argument**: K_nm has 16×15/2 = 120 independent off-diagonal
entries (symmetric, zero diagonal). Each entry is a continuous real
value. An eavesdropper must reconstruct all 120 values to reproduce
the ground state — the search space is R^120. Measurement statistics
from a wrong K_nm' produce statistically distinguishable correlations
(detectable via QBER > threshold).

### 2. `topology_auth.py` — Spectral Fingerprint Authentication

**Core idea**: The Laplacian spectrum of K_nm (already computed by
SSGF's `spectral.py`) provides a public authentication token.
The Fiedler value λ₁ and spectral gap λ₁/λ₂ uniquely characterize
the coupling topology without revealing K_nm itself.

**Functions:**
- `spectral_fingerprint(K)` → dict
  Returns {fiedler, gap_ratio, spectral_entropy, n_components}.
- `verify_fingerprint(K, fingerprint, tol)` → bool
  Checks K against a claimed fingerprint.
- `topology_distance(fp1, fp2)` → float
  Metric between two fingerprints for drift detection.

**Why it works**: Spectral properties are graph invariants — many
different K_nm matrices share the same spectrum (co-spectral graphs).
Publishing the spectrum doesn't reveal K_nm, but any party with the
true K_nm can verify consistency.

### 3. `entanglement_qkd.py` — Topology-Authenticated QKD

**Protocol (SCPN-QKD):**
1. Alice and Bob share K_nm (pre-distributed secret).
2. Both construct H(K_nm) and prepare ground state |ψ₀⟩.
3. Alice measures qubits {0,...,7} in random {X, Z} basis.
4. Bob measures qubits {8,...,15} in random {X, Z} basis.
5. Public channel: announce basis choices, keep matching.
6. Sift → estimate QBER → privacy amplify → secure key.

**Functions:**
- `scpn_qkd_protocol(K, omega, alice_qubits, bob_qubits, shots)` → QKDResult
  Full protocol execution on simulator.
- `correlator_matrix(counts, alice_qubits, bob_qubits)` → ndarray
  Cross-correlation matrix between Alice and Bob measurements.
- `bell_inequality_test(correlator)` → dict
  CHSH violation test to certify entanglement.

**Key rate bound**: From the Devetak-Winter formula,
r ≥ 1 - h(QBER) - h(QBER) where h is binary entropy.
The topology-dependent entanglement structure means different
qubit pairs have different key rates — strongly coupled pairs
(K_nm > 0.2) yield higher rates.

### 4. `percolation.py` — Entanglement Percolation on K_nm

**Core question**: Which (n,m) pairs in K_nm are entangled
above threshold and usable as QKD channels?

**Functions:**
- `concurrence_map(K, omega)` → ndarray
  Compute pairwise concurrence from ground state reduced density matrices.
- `percolation_threshold(K)` → float
  Minimum K_nm value for end-to-end entanglement.
- `active_channel_graph(K, threshold)` → nx.Graph
  Graph of above-threshold entangled pairs.
- `key_rate_per_channel(concurrence_map)` → ndarray
  Devetak-Winter key rate for each link.

**Connection to SSGF**: The Fiedler value λ₁ from SSGF's spectral
bridge is the algebraic connectivity — when λ₁ > 0, the graph
is connected and end-to-end entanglement percolates.

### 5. `hierarchical_keys.py` — SCPN Layer Key Derivation

**Core idea**: The 16-layer SCPN hierarchy maps to a key tree.

```
Master key: hash(K_nm_full ‖ R_global)
  ├── L1 subkey: hash(K_nm[0,:] ‖ θ₁(t))
  ├── L2 subkey: hash(K_nm[1,:] ‖ θ₂(t))
  ├── ...
  └── L16 subkey: hash(K_nm[15,:] ‖ θ₁₆(t))
```

**Functions:**
- `derive_master_key(K, R_global, nonce)` → bytes
  Master key from full coupling matrix + order parameter.
- `derive_layer_key(K, layer_idx, phase_sequence, nonce)` → bytes
  Layer-specific subkey.
- `key_hierarchy(K, phases, n_layers)` → dict[int, bytes]
  Full hierarchy derivation.
- `verify_key_chain(master, layer_keys, K)` → bool
  Verify layer keys are consistent with master.

**Time-varying keys**: The Kuramoto phase sequence θ_n(t) adds
temporal entropy. Different time windows produce different keys
from the same K_nm — natural key rotation without re-keying.

## Hardware Experiments

| Experiment | Qubits | Status | Description |
|-----------|--------|--------|-------------|
| `bell_test_4q` | 4 | **Implemented (v0.6.4)** | CHSH violation with K_nm ground state on hardware |
| `correlator_4q` | 4 | **Implemented (v0.6.4)** | ZZ cross-correlation validates K_ij topology |
| `qkd_qber_4q` | 4 | **Implemented (v0.6.4)** | QBER from hardware vs BB84 threshold (< 0.11) |
| `correlator_8q` | 8 | Planned | Cross-correlation matrix on ibm_fez |
| `percolation_16q` | 16 | Planned | Full K_nm entanglement map on hardware |

## Dependencies on Existing Modules

| Existing Module | Used By | Purpose |
|----------------|---------|---------|
| `bridge.knm_hamiltonian` | knm_key, entanglement_qkd | K_nm → H conversion |
| `phase.phase_vqe` | knm_key | Ground state preparation |
| `qec.control_qec` | entanglement_qkd | Error correction on key circuits |
| `mitigation.zne` | entanglement_qkd | Error mitigation for key extraction |
| `hardware.runner` | all experiments | IBM hardware execution |

## Research Timeline

**Phase 1 — Complete**: All 6 crypto modules implemented with full test coverage.
`knm_key`, `topology_auth`, `entanglement_qkd`, `percolation`, `hierarchical_keys`, `noise_analysis`.

**Phase 2 — Complete**: Full SCPN-QKD protocol on simulator. Bell inequality
verification, QBER estimation, Devetak-Winter key rates under noise.

**Phase 3 — Complete**: Entanglement percolation on K_nm graph, hierarchical
key derivation from 16-layer SCPN structure.

**Phase 4 — In progress**: 3 hardware experiment wrappers implemented (v0.6.4).
Awaiting March QPU budget for ibm_fez execution.
- `bell_test_4q`: CHSH S-value from hardware counts
- `correlator_4q`: 4x4 connected correlation matrix
- `qkd_qber_4q`: Z-basis and X-basis QBER

**Phase 5 — Planned**: 8-qubit correlator, 16-qubit percolation on hardware.
