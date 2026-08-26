# Differentiable error-mitigation product

Versioned **mitigator taxonomy** with differentiability classes, local ZNE and
readout probes, and hard gaps for invent-green ideal-gradient restoration or
live QPU mitigation.

Module: `scpn_quantum_control.error_mitigation_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `error_mitigation_product.v1` |
| Ambient ZNE | `mitigation.zne.zne_extrapolate` / `zne_uncertainty` |
| Ambient readout | `mitigation.readout_matrix` |
| Studio composition | `executive_mitigate` claim boundary |
| Ideal gradient restore | Refuse invent-green |
| Live QPU mitigation | Refused by the no-submit safety policy |
| mitiq | `optional_extra` only — not a hard dependency |

## Differentiability classes

| Class | Meaning |
|---|---|
| `analytic_fd` | Linear map / FD-friendly (e.g. readout invert) |
| `fd_only` | Extrapolation arithmetic on scalars (ZNE) |
| `non_diff` | Sampling / discrete post-selection (PEC, symmetry) |
| `optional_extra` | Optional dependency path (mitiq) |

## Quick start

```python
from scpn_quantum_control.error_mitigation_product import (
    assert_error_mitigation_product_integrity,
    build_error_mitigation_product_registry,
    decide_mitigation_path,
    materialise_demo_zne_probe,
    materialise_readout_probe,
)

reg = assert_error_mitigation_product_integrity(
    build_error_mitigation_product_registry()
)
assert reg["ideal_gradient_restore_policy"] is False

assert decide_mitigation_path(
    "zne_richardson", invent_green_ideal_gradient_restore=True
).allowed is False

probe = materialise_demo_zne_probe()
assert probe.invent_green_live_qpu is False

readout = materialise_readout_probe()
assert readout.n_qubits == 1
```

## Residuals (honest)

- Deeper open-system objective integration
- Metamorphic registration for mitigated estimators

## Related

- Ambient: `scpn_quantum_control.mitigation.*`
- No-submit hardware safety · Studio executive mitigation

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
