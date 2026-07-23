# Whole-program AD product surface (BL-91 / P1)

Versioned **frontend → IR → adjoint/replay product** map for whole-program AD:
public journeys, layered architecture map, support badges, and dry-run posture.
Ambient `whole_program_*` / `program_ad_*` workbench modules remain experimental
under BL-97 honesty (not a frozen SemVer mega-contract).

Module: `scpn_quantum_control.whole_program_ad_product`

## Rules

| Rule | Behaviour |
|---|---|
| Default journey | `frontend_compile_dry_run` |
| Dry-run | Structured allowed plan; no QPU submission |
| Hardware request | Refused (BL-95 no invent-green compute) |
| Unsupported frontend execute | Refused → BL-53 pointer |
| Polyglot cert invent-green | Refused (BL-49 residual) |
| Edge/WASM invent-green | Refused (BL-74 residual) |
| Stability | `experimental_workbench` (BL-97) |
| Blank/unknown journey | Fail closed |

Claim boundary:

> Whole-program AD product surface only; catalogues public journeys and layered
> architecture map; ambient whole_program_*/program_ad_* workbench is not a frozen
> SemVer mega-contract (BL-97); unsupported frontend cases fail closed toward BL-53;
> polyglot parity certs (BL-49) and edge/WASM (BL-74) remain residual; dry-run
> journeys refuse invent-green hardware and unsupported execution (BL-95); does not
> replace full IR/adjoint engines

## Public API

```python
from scpn_quantum_control.whole_program_ad_product import (
    assert_whole_program_ad_product_integrity,
    build_whole_program_ad_product_registry,
    dry_run_whole_program_ad_journey,
    list_whole_program_ad_journey_ids,
    map_whole_program_ad_architecture_layers,
    map_whole_program_ad_public_surfaces,
)

assert "frontend_compile_dry_run" in list_whole_program_ad_journey_ids()
reg = assert_whole_program_ad_product_integrity(build_whole_program_ad_product_registry())
d = dry_run_whole_program_ad_journey("frontend_compile_dry_run")
assert d.allowed is True
assert d.steps_completed

refused = dry_run_whole_program_ad_journey(
    "frontend_compile_dry_run",
    request_hardware=True,
)
assert refused.allowed is False

layers = map_whole_program_ad_architecture_layers()
assert any(row["layer"] == "frontend" for row in layers)
```

## Architecture layers (S91.0)

| Layer | Role |
|---|---|
| frontend | `compile_whole_program_frontend`, contracts, semantics |
| ir | result records, effect IR, primitive registry |
| adjoint | adjoint generation / replay dry-run |
| product | `whole_program_value_and_grad` product entry |
| residual | BL-49 polyglot certs, BL-74 edge/WASM boundaries |

## Bounded product status

Shipped: S91.0 layered architecture map · S91.1 public entrypoints catalogue ·
S91.2 unsupported frontend → BL-53 fail-closed product path · docs · BL-97
stability pointers.

Open: S91.3 polyglot parity certificate subset (BL-49) · S91.4 edge/WASM routing
(BL-74) · mass call-site migration of ambient workbench exports.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
