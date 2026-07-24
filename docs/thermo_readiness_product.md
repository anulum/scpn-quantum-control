# Thermodynamics readiness product (BL-100)

Honest **S9 quantum-thermodynamics readiness** as a productised boundary, plus
an **FEP research-only inventory** (BL-84 / tier C). No thermodynamic peak
claim, no hardware submission, no invent-green FEP product promotion.

Module: `scpn_quantum_control.thermo_readiness_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `thermo_readiness_product.v1` |
| Ambient readiness | `thermodynamics.readiness` (CLAIM_BOUNDARY machine-checked) |
| Peak claim | Refuse invent-green |
| Hardware submit | Refuse invent-green (no-submit S9) |
| FEP status | `research_only` only (BL-84 pointer) |
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

## Residuals (honest)

- **S100.3** — optional future FEP sync-control hook design only (no
  implementation without owner approval)
- FEP remains tier **C** inventory; promotion path is BL-84 / BL-98, not this product

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl100_thermodynamics_readiness_and_fep_inventory.md`
- Ambient: `scpn_quantum_control.thermodynamics.readiness`
- FEP ambient: `scpn_quantum_control.fep.*`
- BL-47 hardware-safe · BL-84 research-lane registry · BL-98 theory-hook promotion

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
