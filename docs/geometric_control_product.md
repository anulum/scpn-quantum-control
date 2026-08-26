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
| Ambient QNG | `phase.natural_gradient.solve_natural_gradient_direction` with regularisation |
| Experimental advantage at criticality | Refuse invent-green |
| Live QPU geometry | Refused by the no-submit safety policy |
| Indefinite metric silent repair | Refuse invent-green |

## Glossary

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

## Residuals

- Metric-spectrum dashboard panel depth
- Notebook coverage and fuller claim-bounded narrative documentation

## Related

- Ambient: `phase.variational_metric`, `phase.natural_gradient`, `analysis.qfi`
- QNG regularisation · no-submit hardware safety · PGBO geometry composition

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
