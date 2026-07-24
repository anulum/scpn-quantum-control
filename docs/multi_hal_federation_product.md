# Multi-HAL provider federation product (BL-75)

**Capability-true federation matrix** over ambient `hardware/hal_*` adapters and
backend descriptors. Default **no-submit dry-run** (BL-47). Never invent-green
live submit without an owner ticket; never invent online queue depth.

Module: `scpn_quantum_control.multi_hal_federation_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `multi_hal_federation_product.v1` |
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

## Residuals (honest)

- **S75.4** — full feedback_* wire under BL-67/47
- **S75.5** — BL-61 competitor-watch version automation
- Live ticketed submit remains residual (product refuses auto would_live)

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl75_multi_hal_provider_federation.md`
- Ambient: `hardware.backends`, `hardware.hal`, `provider_capability_core`
- BL-47 hardware-safe execution; BL-52 route matrix; BL-67 control compose

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
