# qpu_compute product surface (BL-95 / substrate)

Fail-closed typed **compute plan** product between algorithms and HALs. Default
posture is **dry-run / no-submit**; would-live and hardware_enabled plans are
refused without inventing QPU spend. Composes `qpu_compute_types` kernels and
BL-47 `hardware_safe_execution` audit posture.

Module: `scpn_quantum_control.qpu_compute_product`

## Rules

| Rule | Behaviour |
|---|---|
| Default kind | `dry_run_simulator` (`simulator_statevector`, hardware off) |
| Would-live / hardware_enabled | Refused on product surface |
| Ticketed prep | Requires non-empty ticket; plan-only (no live submit) |
| Kernels | `sync_witness`, `dla_parity`, `sync_dla` from `qpu_compute_types` |
| Blank/unknown plan kind | Fail closed |
| Audit | Secret-free; composes BL-47 audit fields |

Claim boundary:

> qpu_compute product only; default posture is dry-run / no-submit; would_live
> and hardware_enabled plans are refused without owner gate; composes
> qpu_compute_types kernels and BL-47 hardware-safe audit posture; never
> executes QPU jobs or invents hardware results

## Public API

```python
from scpn_quantum_control.qpu_compute_product import (
    assert_qpu_compute_product_integrity,
    audit_compute_plan_decision,
    build_qpu_compute_product_registry,
    dry_run_compute_plan,
    list_plan_kind_ids,
)

assert "dry_run_simulator" in list_plan_kind_ids()
reg = assert_qpu_compute_product_integrity(build_qpu_compute_product_registry())
d = dry_run_compute_plan("dry_run_simulator", kernel="sync_dla", shots=64)
assert d.allowed is True
assert d.outcome == "allowed_plan"

live = dry_run_compute_plan("live_would_submit")
assert live.allowed is False
audit = audit_compute_plan_decision(live)
assert audit["contains_secrets"] is False
```

## Bounded product status

Shipped: S95.0 type inventory · S95.1 public plan kinds + validation ·
S95.2 dry-run path · S95.3 BL-47 audit compose · S95.4 docs.

Open: mass algorithm call-site migration onto this layer · live HAL wiring ·
full runtime simulator integration in the product façade (existing
`qpu_compute_runtime` remains the low-level simulator path).

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
