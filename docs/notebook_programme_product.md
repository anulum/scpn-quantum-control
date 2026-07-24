# Differentiable notebook programme product (BL-40)

Versioned **core-six onboarding curriculum** under `notebooks/differentiable/`
with mandatory `hardware_execution: false` honesty.

Module: `scpn_quantum_control.notebook_programme_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `notebook_programme_product.v1` |
| Default notebook | `01_parameter_shift_kuramoto_xy` |
| Hardware execution | Always false |
| Live QPU notebooks | Refused |
| Full archive conversion | Refused |
| Blank/unknown id | Fail closed |

Claim boundary:

> Differentiable notebook programme product surface only; catalogues the
> core-six onboarding curriculum under notebooks/differentiable with
> hardware_execution=false honesty; materialised manifest/probe only;
> refuses invent-green live QPU notebooks and full archive conversion;
> does not claim full nbclient CI matrix green (S40.8 residual)

## Public API

```python
from scpn_quantum_control.notebook_programme_product import (
    assert_notebook_programme_product_integrity,
    build_notebook_programme_registry,
    decide_notebook_programme_path,
    list_curriculum_notebook_ids,
    materialise_curriculum_probe,
)

assert len(list_curriculum_notebook_ids()) == 6
reg = assert_notebook_programme_product_integrity(
    build_notebook_programme_registry()
)
probe = materialise_curriculum_probe()
assert probe.hardware_execution_any is False
assert probe.missing_path_count == 0  # when run from repo root

refused = decide_notebook_programme_path(request_hardware_execution=True)
assert refused.allowed is False
```

## Core six (S40.0)

| Order | ID |
|---:|---|
| 1 | `01_parameter_shift_kuramoto_xy` |
| 2 | `02_gradient_tape_simulator` |
| 3 | `03_jax_batched_quantum_gradients` |
| 4 | `04_pytorch_quantum_layer` |
| 5 | `05_fail_closed_boundaries` |
| 6 | `06_witnesses_challenge_fixture` |

## Bounded product status

Shipped: S40.0 curriculum map · S40.1 directory + manifest product · S40.9
product docs / API map · refuse QPU and full-archive invent-green.

Open residual: S40.2–S40.7 long-form notebook body rewrites · S40.8 CI-light
nbclient matrix · S40.10 stretch PL/Qiskit companions.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
