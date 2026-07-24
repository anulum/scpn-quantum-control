# PGBO quantum geometric tensor product (BL-71)

**Metric + Berry curvature on coupling space** extracted from quantum ground
states via ambient `compute_pgbo_tensor`. Small-system simulation only; fail-
closed oscillator caps; no experimental geometry overclaim.

Module: `scpn_quantum_control.pgbo_qgt_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `pgbo_qgt_product.v1` |
| Ambient | `pgbo.quantum_bridge.compute_pgbo_tensor` |
| Max oscillators | `MAX_OSCILLATORS = 6` (fail closed) |
| Derivatives | Central FD on K with phase alignment (not exact AD) |
| Experimental geometry | Refuse invent-green |
| Live QPU | Refuse invent-green (BL-47) |

## Quick start

```python
from scpn_quantum_control.pgbo_qgt_product import (
    assert_pgbo_qgt_product_integrity,
    build_pgbo_qgt_product_registry,
    decide_qgt_path,
    materialise_demo_pgbo_tensor_probe,
)

reg = assert_pgbo_qgt_product_integrity(build_pgbo_qgt_product_registry())
assert reg["experimental_geometry_claim_policy"] is False

assert decide_qgt_path(
    "pgbo_tensor", invent_green_experimental_geometry=True
).allowed is False

probe = materialise_demo_pgbo_tensor_probe()
assert probe.n_parameters == 1
assert probe.invent_green_live_qpu is False
```

## Residuals (honest)

- **S71.3** — BL-50 dashboard panel depth
- **S71.4** — BL-46 metamorphic registration depth

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl71_pgbo_quantum_geometric_tensor.md`
- Ambient: `scpn_quantum_control.pgbo.quantum_bridge`
- BL-50 geometric control · BL-47 hardware-safe

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
