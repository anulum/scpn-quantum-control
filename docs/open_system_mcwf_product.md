# Open-system MCWF completeness product

Documented, fail-closed **open-system dynamics completeness**: Lindblad and
MCWF surfaces, seeded ensemble variance certificates, simulation noise-model
I/O, and hard gaps for non-CP / non-Markovian / adjoint / hardware noise.

Module: `scpn_quantum_control.open_system_mcwf_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `open_system_mcwf_product.v1` |
| Ambient MCWF | `phase.tensor_jump.mcwf_trajectory` / `mcwf_ensemble` |
| Ambient certificates | `phase.open_system_objectives.certify_mcwf_reproducibility` |
| Hardware noise fidelity | Refuse invent-green |
| Adjoint Lindblad | Refuse invent-green (FD scales only) |
| Non-Markovian / process tensor | Refuse invent-green (out of v1) |
| Unseeded variance claim | Refuse |

## Surfaces

| surface_id | role |
|---|---|
| `lindblad_density` | Scipy density-matrix path composed with the bounded objective suite |
| `mcwf_trajectory` | Single sparse MCWF trajectory |
| `mcwf_ensemble` | Seeded ensemble mean/std |
| `noise_model_io` | Sim-only rate schema import/export |
| `gradient_boundary` | FD expectation boundary catalogue |

## Hard-gap boundaries

`non_cp_map` · `non_markovian_dynamics` · `adjoint_lindblad_gradient` ·
`hardware_noise_fidelity` · `process_tensor_ad`

## Quick start

```python
from scpn_quantum_control.open_system_mcwf_product import (
    assert_open_system_mcwf_product_integrity,
    build_open_system_mcwf_product_registry,
    decide_open_system_path,
    materialise_demo_mcwf_ensemble_probe,
    materialise_reproducibility_probe,
)

reg = assert_open_system_mcwf_product_integrity(
    build_open_system_mcwf_product_registry()
)
assert reg["hardware_noise_fidelity_claim_policy"] is False

assert decide_open_system_path(
    "mcwf_ensemble", invent_green_hardware_noise=True
).allowed is False

probe = materialise_demo_mcwf_ensemble_probe()
assert probe.invent_green_hardware_noise is False

repro = materialise_reproducibility_probe()
assert repro.certificate["passed"] is True
```

## Residuals (honest)

- Fuller committed evidence breadth beyond the current objective suite
- Comparative closed, open, and hardware-noisy gradient documentation

## Related

- Ambient: `scpn_quantum_control.phase.tensor_jump`, `open_system_objectives`
- Bounded open-system objectives · no-submit hardware-safety policy

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
