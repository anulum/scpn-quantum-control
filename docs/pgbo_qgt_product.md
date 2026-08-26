# PGBO quantum geometric tensor product

**Metric + Berry curvature on coupling space** extracted from quantum ground
states via ambient `compute_pgbo_tensor`. Small-system simulation only; fail-
closed oscillator caps; no experimental geometry overclaim.

Module: `scpn_quantum_control.pgbo_qgt_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `pgbo_qgt_product.v2` |
| Ambient | `pgbo.quantum_bridge.compute_pgbo_tensor` |
| Max oscillators | `MAX_OSCILLATORS = 6` (fail closed) |
| Derivatives | Central FD on K with phase alignment (not exact AD) |
| Experimental geometry | Refuse invent-green |
| Live QPU | Refuse under the hardware-safe no-submit policy |

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

## Registry integrity

Schema v2 accepts only the canonical registry keys, claim boundary, public
surface map, policy note, capability rows, boundary rows, size cap, and default
finite-difference epsilon. Schema v1 and drifted serialized content fail closed;
no planning-code compatibility aliases are retained. The product schema remains
part of each deterministic probe digest, while the QGT observables and numerical
method are unchanged.

## Residuals (honest)

- Dashboard integration depth remains open
- Metamorphic registration depth remains open

## Related

- Ambient: `scpn_quantum_control.pgbo.quantum_bridge`
- Geometric-control catalogue · hardware-safe no-submit policy

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
