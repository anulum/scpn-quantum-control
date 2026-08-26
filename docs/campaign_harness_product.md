# Campaign harness productisation

Reusable **hardware-campaign harness templates** with preregistration digests
and a no-submit default. Dry-run probes only. Refuse unsupported live QPU
submission and unattested claim promotion.

Module: `scpn_quantum_control.campaign_harness_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `campaign_harness_product.v1` |
| Default mode | dry_run / no_submit |
| Live submit | Owner ticket residual; would_live refused |
| Unattested claims | Refuse until attestation integration exists |
| Prereg mutation after freeze | Refuse |
| Unknown harness | Fail closed |

## Harnesses

| harness_id | ambient |
|---|---|
| `appqsim_protocol` | `benchmarks.appqsim_protocol` |
| `iqm_layout_transfer` | `benchmarks.iqm_layout_transfer_benchmark` |
| `closed_loop_publication` | `benchmarks.closed_loop_publication_run` |
| `benchmark_harness_registry` | `benchmark_harness.registry` |

## Quick start

```python
from scpn_quantum_control.campaign_harness_product import (
    assert_campaign_harness_product_integrity,
    build_campaign_harness_product_registry,
    decide_campaign_path,
    materialise_demo_campaign_probe,
)

reg = assert_campaign_harness_product_integrity(
    build_campaign_harness_product_registry()
)
assert reg["no_submit_default_policy"] is True

assert decide_campaign_path(
    "appqsim_protocol", invent_green_live_submit=True
).allowed is False

probe = materialise_demo_campaign_probe()
assert probe.invent_green_live_submit is False
assert probe.attestation_slot_present is False
```

## Public API contracts

### Harness discovery and eligibility

| API | Contract |
|---|---|
| `list_campaign_harness_ids()` | Return stable product harness identifiers in catalogue order. |
| `get_campaign_harness(harness_id)` | Resolve one immutable harness row; blank and unknown identifiers raise `ValueError`. |
| `iter_campaign_harnesses(kind=None)` | Return every harness or a stable kind-filtered tuple. |
| `list_ambient_benchmark_family_ids()` | Return the bounded ambient benchmark families referenced by the product. |
| `decide_campaign_path(harness_id, ...)` | Allow dry-run/no-submit use and return explicit blockers for live submission, preregistration mutation, unattested promotion, or invalid execution mode. |

`CampaignHarnessRow` validates identity, ambient pointers, preregistration
digest, supported execution modes, no-submit policy, attestation requirement,
and claim boundary. `PathEligibilityDecision` keeps outcome, permission,
reason, and blockers mutually consistent. Both records serialize through
`to_dict()` for evidence pipelines.

### Dry-run probe materialisation

| API | Contract |
|---|---|
| `materialise_appqsim_probe(seed=0)` | Run the ambient appqsim protocol in deterministic no-submit mode and bind its result digest. |
| `materialise_iqm_layout_probe(seed=0)` | Build the bounded IQM layout-transfer dry-run evidence without provider submission. |
| `materialise_closed_loop_probe(seed=0)` | Materialise closed-loop publication metadata while preserving no-submit and unattested status. |
| `materialise_demo_campaign_probe()` | Return the deterministic appqsim seed-zero demonstration. |

Every `MaterialisedCampaignProbe` validates non-negative seed, non-empty
harness/protocol/digest metadata, dry-run execution, no-submit status, finite
metrics, and false invent-green/attestation flags. Ambient import or payload
failures are surfaced explicitly; they are never converted into a successful
campaign claim.

### Registry evidence and integrity

| API | Contract |
|---|---|
| `map_campaign_harness_public_surfaces()` | Emit deterministic product and ambient module descriptors with harness and claim metadata. |
| `build_campaign_harness_product_registry()` | Build schema-tagged harnesses, surfaces, policy flags, counts, and residual-work evidence. |
| `assert_campaign_harness_product_integrity(payload=None)` | Reject empty, malformed, blank, invalid-kind, duplicate, count-drifted, digest-drifted, or permissive submit/promotion state. |

These APIs package reusable dry-run campaign evidence only. They do not submit
to a QPU, spend provider credits, seal an attestation, promote a scientific
claim, or substitute for ticketed multi-size campaign execution.

## Residuals

- Hermetic reproduction-kit integration and attestation sealing slots
- Full multi-size IQM campaign execution remains ticketed residual

## Related

- Ambient: appqsim, iqm_layout_transfer, closed_loop_publication, benchmark_harness
- No-submit hardware safety · advantage-language policy · control-stack composition

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
