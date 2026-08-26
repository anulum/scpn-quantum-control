# Multi-HAL provider federation product

**Capability-true federation matrix** over ambient `hardware/hal_*` adapters and
backend descriptors. Default **hardware-safe no-submit dry-run**. Never invent-green
live submit without an owner ticket; never invent online queue depth.

Module: `scpn_quantum_control.multi_hal_federation_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `multi_hal_federation_product.v2` |
| Inventory source | Ambient `list_hal_backend_descriptors` + `built_in_backend_profiles` |
| Unknown backend | Fail closed |
| Dry-run | Allowed, no network |
| Ticketed prep | Allowed only with owner ticket |
| Would-live auto-submit | Refused on product surface |
| Invent-green live submit | Refused |

## Capability fields (per HAL row)

`backend_id`, `provider`, `broker`, `adapter_module`, `modality`,
`supports_shots`, `supports_mid_circuit_measurement`, `supports_pulse`,
`supports_statevector`, `submit_requires_approval`, `can_submit`, `is_cloud`,
`ir_formats`, `max_qubits` (None = unknown, not invent-green),
`no_submit_default=True`.

`list_hal_backend_ids()` and `list_hal_providers()` return deterministic
catalogue-order identifiers. `get_hal_capability(backend_id)` performs exact
lookup after trimming outer whitespace and raises `ValueError` for blank or
unknown ids. `iter_hal_capabilities(...)` filters independently by provider,
support posture, and pulse capability; no matches return an empty tuple.

`HalCapabilityRecord` is immutable and validates provider, broker, adapter,
modality, IR formats, optional qubit bounds, and the no-submit policy.
`build_federation_matrix()` serialises every record without probing a network.

## Route decisions

`decide_federation_route()` returns a `PathEligibilityDecision`:

| Mode | Result |
|---|---|
| `dry_run` | allowed only with network access disabled |
| `ticketed_prep` | allowed only when an owner ticket is present |
| `would_live` | always refused on this product surface |

Unknown modes, invent-green live-submit requests, networked dry-runs, missing
tickets, and non-submitting backends produce typed blockers. An allowed
ticketed preparation is not submission authority and does not contact a HAL.

## Offline capability probes

`materialise_federation_dry_run_probe()` converts catalogue metadata into an
offline `ProviderCapabilitySnapshot`, then applies the ambient capability
assessor. Optional IR and minimum-qubit requirements can block the result.
Unknown qubit capacity uses a schema-only floor of one and is never presented
as measured hardware capacity. Queue depth, calibration time, shots, circuits,
and online status remain unknown/offline.

`materialise_demo_federation_dry_run_probe()` selects the first canonical row.
Both probes return immutable records with backend/provider, status, blockers,
warnings, `no_submit=True`, and `invent_green_live_submit=False`.

## Registry and integrity

`map_multi_hal_federation_public_surfaces()` identifies the product, descriptor
inventory, backend profiles, and offline assessor with their roles and policy
scope. `build_multi_hal_federation_product_registry()` combines those surfaces
with schemas, counts, providers, matrix rows, and no-submit policy.

`assert_multi_hal_federation_product_integrity(payload=None)` rejects missing
or malformed matrices, blank/duplicate ids, missing provider/adapter/IR data,
row-level submission permission, canonical-set drift, inconsistent counts, or
unsafe global policies. It returns a shallow validated mapping.

## Exported contracts and boundaries

The module exports schema/claim constants, `FederationRouteMode`,
`SupportPosture`, `PathDecisionOutcome`, and the three immutable record types.
Importing, listing, routing, probing, or building the registry does not open a
network connection, query a live queue, submit a workload, consume provider
credentials, execute feedback control, mutate HAL profiles, or promote
hardware readiness. Those actions remain separately ticketed and evidenced.

## Quick start

```python
from scpn_quantum_control.multi_hal_federation_product import (
    assert_multi_hal_federation_product_integrity,
    build_multi_hal_federation_product_registry,
    decide_federation_route,
    list_hal_backend_ids,
    materialise_demo_federation_dry_run_probe,
)

reg = assert_multi_hal_federation_product_integrity(
    build_multi_hal_federation_product_registry()
)
assert reg["no_submit_default_policy"] is True
assert reg["backend_count"] == len(list_hal_backend_ids())

backend = list_hal_backend_ids()[0]
assert decide_federation_route(backend, mode="dry_run").allowed is True
assert decide_federation_route(
    backend, mode="would_live", owner_ticket_present=True
).allowed is False

probe = materialise_demo_federation_dry_run_probe()
assert probe.invent_green_live_submit is False
assert probe.no_submit is True
```

## Open capabilities

- Complete feedback-adapter integration under closed-loop and hardware-safe policies
- Automated competitive-baseline version monitoring
- Live ticketed submit remains residual (product refuses auto would_live)

## Related

- Ambient: `hardware.backends`, `hardware.hal`, `provider_capability_core`
- Hardware-safe execution, governed route matrix, and control-stack composition

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
