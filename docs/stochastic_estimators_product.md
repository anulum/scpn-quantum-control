# Stochastic estimators & policies product (BL-93 / P1)

Versioned **finite-shot / stochastic gradient product** over ambient SPSA,
score-function, parameter-shift shot allocation, and confidence-policy
primitives. Materialised uncertainty only; composes BL-47 no-submit honesty.

Module: `scpn_quantum_control.stochastic_estimators_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `stochastic_estimators_product.v1` |
| Default estimator | `spsa_gradient` |
| Hardware shots | Always refused (BL-47) |
| Blank/unknown estimator | Fail closed |
| Live QPU execution | Never claimed by this product |
| Variance/bias campaign | Residual S93.2 |

Claim boundary:

> Stochastic estimators product surface only; catalogues SPSA, score-function,
> and shot-allocation helpers with confidence-policy contracts; materialised
> finite-shot uncertainty only; composes BL-47 no-submit / shot-budget honesty;
> does not invent-green live QPU shot runs or full variance/bias experiment
> campaigns (S93.2 residual)

## Public API

```python
from scpn_quantum_control.stochastic_estimators_product import (
    assert_stochastic_estimators_product_integrity,
    build_product_failure_policy,
    build_stochastic_estimators_product_registry,
    dry_run_stochastic_estimator,
    list_stochastic_estimator_ids,
    materialise_demo_spsa_probe,
)

assert "spsa_gradient" in list_stochastic_estimator_ids()
reg = assert_stochastic_estimators_product_integrity(
    build_stochastic_estimators_product_registry()
)

d = dry_run_stochastic_estimator("spsa_gradient", planned_shots=100)
assert d.allowed is True

refused = dry_run_stochastic_estimator(
    "spsa_gradient",
    request_hardware_shots=True,
)
assert refused.allowed is False

probe = materialise_demo_spsa_probe(seed=0)
assert probe.gradient
assert probe.max_abs_gradient >= 0.0

policy = build_product_failure_policy(max_standard_error=0.05)
assert policy.max_standard_error == 0.05
```

## Catalogue (S93.0)

| ID | Kind |
|---|---|
| `spsa_gradient` | SPSA |
| `score_function_gradient` | score-function |
| `parameter_shift_shot_allocation` | shot allocation |
| `gradient_failure_policy` | confidence policy |

## Bounded product status

Shipped: S93.0 estimator catalogue · S93.1 contracts + tests (incl. materialised
SPSA demo probe) · S93.3 policy objects composing BL-47 · S93.4 product docs /
API map rows.

Open: S93.2 full variance/bias documentation + experimental campaigns.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
