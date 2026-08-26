# KYMA / KYMA v2 mechanism benchmark product

**Public, preregistered, mechanism-only** sync-learning honesty surface. Design
constants come from **teacher dynamics only** (prereg §5) — never student
held-out accuracy. Refuse post-hoc retuning and invent-green advantage without
protocol id.

Module: `scpn_quantum_control.kyma_mechanism_benchmark_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `kyma_mechanism_benchmark_product.v2` |
| Design freeze | Ambient `kyma_v2.design` grids + targets + digest |
| Student held-out design | Refuse |
| Post-hoc retune | Refuse |
| Advantage invent-green | Refuse (requires protocol id and advantage-language governance) |
| Unknown suite | Fail closed |

## Suites

| suite_id | role |
|---|---|
| `kyma_v1` | Honest NEGATIVE baseline |
| `kyma_v2` | Corrected mechanism-only design |

Protocol: `KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21`

## Quick start

```python
from scpn_quantum_control.kyma_mechanism_benchmark_product import (
    assert_kyma_mechanism_benchmark_product_integrity,
    build_kyma_mechanism_benchmark_product_registry,
    decide_kyma_path,
    get_frozen_design_constants,
    materialise_demo_mechanism_certificate_probe,
)

reg = assert_kyma_mechanism_benchmark_product_integrity(
    build_kyma_mechanism_benchmark_product_registry()
)
assert reg["invent_green_advantage_policy"] is False
assert len(get_frozen_design_constants().content_digest) == 64

assert decide_kyma_path("kyma_v2", invent_green_advantage=True).allowed is False

probe = materialise_demo_mechanism_certificate_probe()
assert probe.meets_realise_target is True
assert probe.design_from_student_held_out is False
```

## Public API contracts

### Suite discovery and frozen design

| API | Contract |
|---|---|
| `list_kyma_suite_ids()` | Return `kyma_v1` and `kyma_v2` in stable catalogue order. |
| `get_kyma_suite(suite_id)` | Resolve one immutable suite row; blank and unknown identifiers raise `ValueError`. |
| `iter_kyma_suites(kind=None)` | Return every suite or a stable kind-filtered tuple. |
| `load_frozen_design_constants(verify_ambient=False)` | Load product-mirrored preregistration grids, targets, and their canonical SHA-256 digest; optional ambient verification fails on drift. |
| `get_frozen_design_constants()` | Return the validated process-local frozen constants record. |

`KymaSuiteRow` validates suite identity, mechanism-only policy, support posture,
protocol provenance, and the non-promotional claim boundary. The
`FrozenDesignConstants` record validates non-empty grids, bounded targets, and
a 64-character content digest. Both serialize through `to_dict()`.

### Eligibility and certificate materialisation

| API | Contract |
|---|---|
| `decide_kyma_path(suite_id, ...)` | Allow an honest mechanism-only route; return explicit blockers for invented advantage, missing protocol provenance, post-hoc retuning, or student held-out design. |
| `materialise_mechanism_certificate_probe(seed=0, config=None)` | Run the ambient teacher-dynamics certificate path when available; reject negative seeds and fail closed for unavailable custom configurations. |
| `materialise_demo_mechanism_certificate_probe()` | Run the deterministic seed-zero path, using the honest frozen-design fallback when optional JAX is absent. |

The returned `PathEligibilityDecision` keeps outcome, boolean permission, reason,
and blockers mutually consistent. `MaterialisedMechanismCertificateProbe`
validates bounded realisability/non-separability rates, digest/protocol
provenance, non-empty labels, and both invent-green flags as false.

### Registry evidence and integrity

| API | Contract |
|---|---|
| `map_kyma_mechanism_benchmark_public_surfaces()` | Emit deterministic product, teacher-design, trial-builder, and v1-baseline descriptors. |
| `build_kyma_mechanism_benchmark_product_registry()` | Build schema-tagged suites, frozen constants, surfaces, policy flags, counts, protocol, and residual-work evidence. |
| `assert_kyma_mechanism_benchmark_product_integrity(payload=None)` | Reject empty, malformed, blank, invalid-kind, duplicate, count-drifted, digest-drifted, protocol-drifted, or permissive policy state. |

The demo fallback is product-local evidence derived only from preregistered
floors. It is not an isolated performance benchmark, hardware result, student
accuracy claim, or quantum-advantage promotion.

## Open evidence and integration depth

- Full classical ML baseline-harness depth and multi-seed evidence
- Optional sync-challenge family registration
- Hermetic reproduction-kit entry

## Related

- Ambient: `benchmarks.kyma`, `benchmarks.kyma_v2.design`, `task`, `teacher`
- Campaigns: `docs/campaigns/kyma_v2_composition_probe_2026-07-21.md`

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
