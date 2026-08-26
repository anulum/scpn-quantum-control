# PennyLane + Qiskit migration guides product

Versioned **adoption-path product** mapping PL/Qiskit concepts to SCPN APIs
with materialised local round-trips and honest Runtime boundaries.

Module: `scpn_quantum_control.migration_guides_product`

This page documents a bounded adoption facade, not a claim of drop-in
framework compatibility. The catalogue and demonstrations identify supported
local subsets while preserving explicit boundaries around live Runtime, QPU
submission, full API parity, notebooks, and version-skew assurance.

## Contract discovery

| Function | Contract |
|---|---|
| `list_migration_concept_ids()` | Returns every stable concept id in catalogue order. |
| `get_migration_concept(concept_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_migration_concepts(...)` | Filters deterministically by source framework and/or support posture. |
| `map_migration_guides_public_surfaces()` | Groups concepts by their ambient SCPN implementation module. |

The support postures distinguish a locally materialised subset from guide-only,
boundary-only, and refuse-only rows. Discovery is static and local: it imports
no optional framework and contacts no provider.

## Public value objects

- `MigrationConceptRow` maps one PennyLane, Qiskit, or boundary concept to its
  SCPN API, ambient owner, support posture, and non-promotional flags.
- `PathEligibilityDecision` records a structured allowed/refused outcome,
  human-readable reason, and ordered blockers.
- `MaterialisedPennyLaneRoundTrip` records source/Phase-QNode values, value and
  gradient residuals, match flags, parameter count, and demo provenance.
- `MaterialisedQiskitLocalGradient` records the value/gradient, analytic
  references, residuals, ambient method, and demo provenance.

All records are immutable slot-backed dataclasses with validated construction
and JSON-ready `to_dict()` mappings. Neither materialised record is hardware
evidence or proof of general framework equivalence.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `migration_guides_product.v2` |
| Default concept | `pl_parameter_shift_to_phase_qnode` |
| Live Runtime / QPU | Refused |
| Full feature parity | Refused |
| Blank/unknown concept | Fail closed |
| Local PL/Qiskit subset | Materialised demos |

## Eligibility decisions

`decide_migration_path()` permits only a declared local supported subset. It
returns a structured refusal when any of these conditions holds:

- `request_live_runtime=True` asks for live Runtime or QPU behavior;
- `request_full_parity=True` asks for full PennyLane/Qiskit feature parity; or
- `local_supported_subset=False` provides no bounded migration target.

Multiple blockers are de-duplicated in first-seen order. An allowed decision
does not authorise provider use; it only permits the local product demos.

## PennyLane round trip

`materialise_demo_pennylane_round_trip()` compares the SCPN Phase-QNode
`RX(theta)` / `Z` expectation and parameter-shift gradient with the matching
PennyLane subset. When the optional PennyLane importer is healthy it uses the
ambient tape round-trip; otherwise it falls back to the same analytic
`cos(theta)` / `-sin(theta)` reference without claiming importer coverage.

The angle must be finite and both tolerances finite and non-negative. The
returned `value_match` and `gradient_match` are explicit tolerance comparisons,
not blanket compatibility badges. A blocked eligibility decision fails before
any materialisation.

## Qiskit local gradient

`materialise_demo_qiskit_local_gradient()` compares the same Phase-QNode target
with an analytic reference. When the optional Qiskit statevector surface is
healthy it uses `execute_qiskit_statevector_parameter_shift`; otherwise the
bounded Phase-QNode path remains available and is labelled accordingly.

The function rejects a non-finite angle and an empty ambient gradient. It does
not instantiate IBM Runtime, select a backend, submit a circuit, or spend shots.

Claim boundary:

> This migration-guide product maps supported PennyLane and Qiskit concepts to
> local SCPN APIs and materialises bounded local round trips through
> phase.pennylane_import and phase.qiskit_gradients. It refuses full Runtime
> feature parity and live QPU Runtime claims. Full framework API coverage,
> companion notebooks, and version-skew CI remain outside the current product.

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

## Concept catalogue

| ID | Framework |
|---|---|
| `pl_parameter_shift_to_phase_qnode` | pennylane |
| `pl_qnode_import_boundary` | pennylane |
| `qk_statevector_parameter_shift` | qiskit |
| `qk_runtime_boundary` | qiskit |
| `refuse_full_runtime_parity` | boundary |
| `guide_docs` | boundary |

## Registry integrity

`build_migration_guides_product_registry()` emits schema
`migration_guides_product.v2`, the full concept catalogue, ambient surface map,
default id, counts, policy note, and claim boundary. Every row keeps
`allows_live_runtime=False` and `allows_full_parity_claim=False`.

Always validate transported or stored payloads through
`assert_migration_guides_product_integrity()`. It rejects:

- missing, empty, or non-list concept collections;
- non-mapping, blank, duplicate, missing, or extra rows;
- unknown framework kinds or missing symbol names;
- any live-Runtime or full-parity flag relaxation;
- loss of the default mapping or explicit refuse row; and
- schema, claim-boundary, policy, canonical-row, public-surface, default-id, or
  count drift.

## Worked demos

### PennyLane import round trip

`RX(θ)` on wire 0, measure `⟨Z⟩`, import into Phase-QNode; ambient
`check_pennylane_phase_qnode_import_round_trip` agrees value and gradient.

### Qiskit local gradient

Same circuit via Qiskit Statevector parameter-shift; analytic
`value=cos(θ)`, `grad=-sin(θ)`.

## Failure handling and operational non-effects

Treat `ValueError` as a caller-contract, numerical-validation, or transported
registry failure. Treat `RuntimeError` from catalogue construction as
repository corruption. Never reinterpret an optional-framework fallback as
proof that the optional importer executed.

This product performs no credential lookup, network access, provider discovery,
live Runtime call, QPU submission, hardware execution, catalogue mutation,
notebook generation, migration rewrite, or evidence promotion. The worked
examples remain local one-parameter demonstrations over existing ambient APIs.

## Bounded product status

Shipped: concept inventory · PennyLane round trip · Qiskit local gradient ·
Runtime boundary refusal · product documentation.

Open product work: full MkDocs navigation polish · support-matrix
synchronisation · companion notebooks · version-skew CI.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
