# Phase-QNode product surface (BL-90 / P1)

Versioned **primary quantum programming product** map for Phase-QNode: public
journeys, support badges, and dry-run posture. Ambient `phase/qnode_*` workbench
modules remain experimental under BL-97 honesty (not a frozen SemVer mega-contract).

Module: `scpn_quantum_control.phase_qnode_product`

## Rules

| Rule | Behaviour |
|---|---|
| Default journey | `build_differentiate_dry_run` |
| Dry-run | Structured allowed plan; no QPU submission |
| Hardware request | Refused (BL-47/BL-95 no-submit posture) |
| Stability | `experimental_workbench` (BL-97) |
| Blank/unknown journey | Fail closed |

Claim boundary:

> Phase-QNode product surface only; catalogues public journeys and support badges;
> ambient phase/qnode_* workbench is not a frozen SemVer mega-contract (BL-97);
> dry-run journeys refuse invent-green hardware/QPU spend (BL-47/BL-95); does not
> replace full circuit engines

## Public API

```python
from scpn_quantum_control.phase_qnode_product import (
    assert_phase_qnode_product_integrity,
    build_phase_qnode_product_registry,
    dry_run_phase_qnode_journey,
    list_phase_qnode_journey_ids,
    map_phase_qnode_public_surfaces,
)

assert "build_differentiate_dry_run" in list_phase_qnode_journey_ids()
reg = assert_phase_qnode_product_integrity(build_phase_qnode_product_registry())
d = dry_run_phase_qnode_journey("build_differentiate_dry_run")
assert d.allowed is True
assert d.steps_completed

refused = dry_run_phase_qnode_journey(
    "build_differentiate_dry_run",
    request_hardware=True,
)
assert refused.allowed is False
```

## Bounded product status

Shipped: S90.0 public API map · S90.1 journey catalogue · S90.2 badge fields
(product catalogue) · S90.4 BL-97 stability pointers · docs.

Open: S90.2 full badge CI job · S90.3 full ≤15 min notebook curriculum (BL-40) ·
S90.4 mass deprecation application across all qnode exports.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
