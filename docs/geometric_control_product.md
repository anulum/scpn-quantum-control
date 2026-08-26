# Geometric quantum control product

Geometry of quantum control for synchronisation: **QFI / McLachlan metric /
QNG** catalogue, local metric spectrum diagnostics, and regularised natural-
gradient direction probes. No experimental advantage claims at criticality.

Module: `scpn_quantum_control.geometric_control_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `geometric_control_product.v1` |
| Ambient metric | `phase.variational_metric.mclachlan_metric` |
| Ambient QNG | `phase.natural_gradient.solve_natural_gradient_direction` (BL-13) |
| Experimental advantage at criticality | Refuse invent-green |
| Live QPU geometry | Refuse invent-green (BL-47) |
| Indefinite metric silent repair | Refuse invent-green |

## Glossary (S50.0)

`QFI` · `Fubini_Study_McLachlan` · `QNG` · `criticality`

## Quick start

```python
from scpn_quantum_control.geometric_control_product import (
    assert_geometric_control_product_integrity,
    build_geometric_control_product_registry,
    decide_geometry_path,
    materialise_demo_metric_diagnostics_probe,
    materialise_qng_direction_probe,
)

reg = assert_geometric_control_product_integrity(
    build_geometric_control_product_registry()
)
assert reg["experimental_advantage_criticality_policy"] is False

assert decide_geometry_path(
    "criticality_diagnostics", invent_green_advantage=True
).allowed is False

probe = materialise_demo_metric_diagnostics_probe()
assert probe.metric_rank + probe.metric_nullity == probe.n_parameters

qng = materialise_qng_direction_probe()
assert qng.regularization_reason
```

## Residuals (honest)

- **S50.5** — BL-34 metric spectrum dashboard panel depth
- **S50.6** — BL-40 notebook + fuller claim-bounded narrative docs

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl50_geometric_quantum_control_qfi.md`
- Ambient: `phase.variational_metric`, `phase.natural_gradient`, `analysis.qfi`
- BL-13 QNG regularisation · BL-47 hardware-safe · BL-71 PGBO (compose later)

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
