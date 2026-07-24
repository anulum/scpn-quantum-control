# PennyLane + Qiskit migration guides product (BL-41)

Versioned **adoption-path product** mapping PL/Qiskit concepts to SCPN APIs
with materialised local round-trips and honest Runtime boundaries.

Module: `scpn_quantum_control.migration_guides_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `migration_guides_product.v1` |
| Default concept | `pl_parameter_shift_to_phase_qnode` |
| Live Runtime / QPU | Refused |
| Full feature parity | Refused |
| Blank/unknown concept | Fail closed |
| Local PL/Qiskit subset | Materialised demos |

Claim boundary:

> PennyLane + Qiskit migration guides product surface only; catalogues
> concept-map rows and materialised local round-trips on supported subsets
> over ambient phase.pennylane_import and phase.qiskit_gradients; refuses
> invent-green full Runtime feature parity and live QPU Runtime; does not
> claim full framework API coverage, companion notebooks, or version-skew CI
> (S41.5–S41.7 residual)

## Public API

```python
from scpn_quantum_control.migration_guides_product import (
    assert_migration_guides_product_integrity,
    build_migration_guides_product_registry,
    decide_migration_path,
    list_migration_concept_ids,
    materialise_demo_pennylane_round_trip,
    materialise_demo_qiskit_local_gradient,
)

assert "pl_parameter_shift_to_phase_qnode" in list_migration_concept_ids()
reg = assert_migration_guides_product_integrity(
    build_migration_guides_product_registry()
)

allowed = decide_migration_path()
assert allowed.allowed is True
refused = decide_migration_path(request_live_runtime=True)
assert refused.allowed is False

# PennyLane RX(θ)→⟨Z⟩ import round-trip vs Phase-QNode
pl = materialise_demo_pennylane_round_trip(theta=0.4)
assert pl.value_match and pl.gradient_match

# Qiskit local Statevector parameter-shift (analytic cos/sin)
qk = materialise_demo_qiskit_local_gradient(theta=0.4)
assert abs(qk.value - __import__("math").cos(0.4)) < 1e-9
```

## Catalogue (S41.0)

| ID | Framework |
|---|---|
| `pl_parameter_shift_to_phase_qnode` | pennylane |
| `pl_qnode_import_boundary` | pennylane |
| `qk_statevector_parameter_shift` | qiskit |
| `qk_runtime_boundary` | qiskit |
| `refuse_full_runtime_parity` | boundary |
| `guide_docs` | boundary |

## Worked demos

### PennyLane (S41.1)

`RX(θ)` on wire 0, measure `⟨Z⟩`, import into Phase-QNode; ambient
`check_pennylane_phase_qnode_import_round_trip` agrees value and gradient.

### Qiskit (S41.3)

Same circuit via Qiskit Statevector parameter-shift; analytic
`value=cos(θ)`, `grad=-sin(θ)`.

## Bounded product status

Shipped: S41.0 concept inventory · S41.1 PL round-trip · S41.3 Qiskit local
gradient · S41.4 Runtime boundary refuse · product docs.

Open residual: S41.2 full MkDocs nav polish · S41.5 BL-12 matrix sync ·
S41.6 companion notebooks · S41.7 version-skew CI.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
