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
| Studio compose | BL-62 `executive_mitigate` claim boundary |
| Ideal gradient restore | Refuse invent-green |
| Live QPU mitigation | Refuse invent-green (BL-47) |
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

- **S59.5** — deeper open-system objective compose (BL-51) depth
- **S59.6** — metamorphic registration (BL-46) for mitigated estimators depth

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl59_differentiable_error_mitigation.md`
- Ambient: `scpn_quantum_control.mitigation.*`
- BL-47 hardware-safe · BL-62 Studio executive

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
