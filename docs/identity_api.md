SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li

# Identity Continuity API

Quantitative tools for characterizing identity attractor basins, coherence budgets, entanglement structure, and cryptographic fingerprinting of coupling topologies.

## IdentityAttractor

Characterizes the attractor basin of an identity coupling topology via VQE ground state analysis.

```python
from scpn_quantum_control.identity import IdentityAttractor

attractor = IdentityAttractor(K, omega, ansatz_reps=2)
result = attractor.solve(maxiter=200, seed=42)

print(result["robustness_gap"])   # Energy gap E_1 - E_0
print(result["ground_energy"])    # VQE ground state energy
print(result["eigenvalues"])      # First 4 eigenvalues
```

### From Binding Spec

```python
attractor = IdentityAttractor.from_binding_spec(binding_spec)
```

Accepts any scpn-phase-orchestrator binding spec dict with `layers`, `coupling.base_strength`, and `coupling.decay_alpha`.

## Coherence Budget

Computes the maximum circuit depth before fidelity drops below a threshold on Heron r2 hardware.

```python
from scpn_quantum_control.identity import coherence_budget, fidelity_at_depth

# Single fidelity estimate
f = fidelity_at_depth(depth=100, n_qubits=4)

# Full budget analysis
budget = coherence_budget(n_qubits=4, fidelity_threshold=0.5)
print(budget["max_depth"])        # Maximum useful depth
print(budget["fidelity_curve"])   # Fidelity at key depths
```

### Hardware Parameters

All functions accept `t1_us`, `t2_us`, `cz_error`, `readout_error` kwargs to override Heron r2 defaults.

## Entanglement Witness

CHSH S-parameter measurement for qubit pairs. S > 2 certifies non-classical correlation.

```python
from scpn_quantum_control.identity import chsh_from_statevector, disposition_entanglement_map

# Single pair
S = chsh_from_statevector(statevector, qubit_a=0, qubit_b=1)

# All pairs with labels
result = disposition_entanglement_map(
    statevector,
    disposition_labels=["verify", "honest_naming", "antislop"],
)
print(result["n_entangled"])       # Pairs with S > 2
print(result["integration_metric"])  # Mean S / Tsirelson bound
```

## Identity Key

Quantum fingerprint generation and verification via K\_nm coupling topology.

```python
from scpn_quantum_control.identity import identity_fingerprint, verify_identity, prove_identity

# Generate fingerprint
fp = identity_fingerprint(K, omega)
print(fp["spectral"]["fiedler"])   # Algebraic connectivity
print(fp["commitment"])            # SHA-256 hex commitment

# Challenge-response verification
challenge = os.urandom(32)
response = prove_identity(K, challenge)
assert verify_identity(K, challenge, response)
```

## Binding Spec & Orchestrator Mapping

```python
from scpn_quantum_control.identity.binding_spec import (
    ARCANE_SAPIENCE_SPEC,
    ORCHESTRATOR_MAPPING,
    quantum_to_orchestrator_phases,
    orchestrator_to_quantum_phases,
    build_identity_attractor,
    solve_identity,
)

# 18 quantum oscillators -> 35 orchestrator domainpack oscillators
orch_phases = quantum_to_orchestrator_phases(theta_18)

# 35 orchestrator -> 18 quantum (circular mean roundtrip)
theta_back = orchestrator_to_quantum_phases(orch_phases)

# Build + solve attractor in one call
result = solve_identity()  # uses ARCANE_SAPIENCE_SPEC by default
```

`ORCHESTRATOR_MAPPING` maps each of the 18 quantum oscillators to its
corresponding sub-group in the `identity_coherence` domainpack (35 total).

## Demo

See `examples/10_identity_continuity_demo.py` for end-to-end usage of all identity modules.
