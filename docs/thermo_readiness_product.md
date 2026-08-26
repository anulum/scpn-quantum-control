# Thermodynamics readiness product

Honest **quantum-thermodynamics readiness** as a productised boundary, plus an
**FEP research-only inventory**. No thermodynamic peak claim, no hardware
submission, no invent-green FEP product promotion.

Module: `scpn_quantum_control.thermo_readiness_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `thermo_readiness_product.v1` |
| Ambient readiness | `thermodynamics.readiness` (CLAIM_BOUNDARY machine-checked) |
| Peak claim | Refuse invent-green |
| Hardware submit | Refuse invent-green (no-submit readiness) |
| FEP status | `research_only` only (research-lane pointer) |
| Unknown capability / module | Fail closed |

## Readiness capabilities

| capability_id | ambient |
|---|---|
| `k_sweep_protocol` | `run_k_sweep_protocol` |
| `entropy_production` | `entropy_production_rate` |
| `work_identity` | `calibrated_work_identity` |
| `heat_dissipation` | `heat_dissipation_rate` |
| `claim_boundary_gate` | `CLAIM_BOUNDARY` |

## FEP inventory (research-only)

| module_id | path |
|---|---|
| `predictive_coding` | `scpn_quantum_control.fep.predictive_coding` |
| `variational_free_energy` | `scpn_quantum_control.fep.variational_free_energy` |

## Quick start

```python
from scpn_quantum_control.thermo_readiness_product import (
    assert_thermo_readiness_product_integrity,
    build_thermo_readiness_product_registry,
    decide_readiness_path,
    materialise_demo_k_sweep_probe,
)

reg = assert_thermo_readiness_product_integrity(
    build_thermo_readiness_product_registry()
)
assert reg["thermodynamic_peak_claim_allowed_policy"] is False
assert reg["fep_product_promotion_allowed_policy"] is False

assert decide_readiness_path(
    "k_sweep_protocol", invent_green_peak_claim=True
).allowed is False

probe = materialise_demo_k_sweep_probe()
assert probe.hardware_submission_allowed is False
assert probe.row_count >= 3
# peak_k is a calibrated candidate only — not a peak claim
assert probe.thermodynamic_peak_claim_allowed is False
```

## Public API contracts

### Discovery and inventory

| API | Contract |
|---|---|
| `verify_ambient_claim_boundary()` | Validate the ambient readiness claim boundary and reject missing honesty clauses. |
| `list_readiness_capability_ids()` | Return readiness capability identifiers in stable catalogue order. |
| `list_fep_module_ids()` | Return the bounded FEP research-only inventory identifiers. |
| `get_readiness_capability(capability_id)` | Resolve one capability and reject blank or unknown identifiers. |
| `get_fep_inventory_row(module_id)` | Resolve one FEP row and reject blank or unknown identifiers. |
| `iter_readiness_capabilities(kind=...)` | Return all capabilities or an immutable kind-filtered view. |
| `iter_fep_inventory(status=...)` | Return all FEP rows or a status-filtered view. |

### Eligibility and materialisation

`decide_readiness_path()` is the required fail-closed decision point. It binds
the ambient claim boundary and refuses thermodynamic peak claims, hardware
submission, unsupported capabilities, and FEP product promotion. A refused
decision carries explicit blockers.

`materialise_k_sweep_probe()` accepts only an allowed `k_sweep_protocol`
decision and returns a no-submit evidence probe. `materialise_demo_k_sweep_probe()`
uses the deterministic local fixture. `materialise_quantum_thermo_payload_probe()`
validates the ambient payload and preserves the same non-promotion boundary.

### Registry, surfaces, and provenance

| API | Contract |
|---|---|
| `map_thermo_readiness_public_surfaces()` | Emit deterministic public-surface descriptors and roles. |
| `build_thermo_readiness_product_registry()` | Build the schema-tagged capability/FEP catalogue and policy payload. |
| `assert_thermo_readiness_product_integrity(payload=None)` | Reject missing, duplicate, blank, count-drifted, or invent-green registry state. |
| `compute_k_sweep_request_digest(...)` | Hash the normalised capability and sweep request for reproducible provenance. |

The data records expose `to_dict()` JSON-ready payloads and validate their
required identifiers, status/kind enums, blockers, counts, no-submit flags,
and bounded claim text at construction time.

## Research boundaries

- An optional future FEP sync-control hook is design-only; implementation
  requires separate owner approval.
- FEP remains a research-only inventory. Promotion belongs to the research-lane
  registry and theory-hook promotion policy, not this product.

## Related

- Ambient: `scpn_quantum_control.thermodynamics.readiness`
- FEP ambient: `scpn_quantum_control.fep.*`
- Hardware-safe execution · research-lane registry · theory-hook promotion

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
