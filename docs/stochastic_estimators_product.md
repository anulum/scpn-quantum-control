# Stochastic estimators & policies product

Versioned **finite-shot / stochastic gradient product** over ambient SPSA,
score-function, parameter-shift shot allocation, and confidence-policy
primitives. Materialised uncertainty only; composes the hardware-safe
no-submit and shot-budget policy.

Module: `scpn_quantum_control.stochastic_estimators_product`

This page documents a bounded product facade over existing local estimator and
policy primitives. It does not claim calibrated hardware uncertainty, execute
shots, or substitute a dry-run plan for experimental evidence.

## Contract discovery

| Function | Contract |
|---|---|
| `list_stochastic_estimator_ids()` | Returns every stable estimator id in catalogue order. |
| `get_stochastic_estimator(estimator_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_stochastic_estimators(...)` | Filters deterministically by kind and/or support posture. |
| `map_stochastic_estimators_public_surfaces()` | Groups estimator ids by their ambient implementation owner. |

Discovery is static and local. It performs no sampling, provider lookup,
credential access, hardware submission, or shot allocation.

## Public value objects

- `StochasticEstimatorRow` maps a stable id to its kind, ambient owner, symbol,
  support posture, hardware-safety pointer, and no-hardware boundary.
- `EstimatorDryRunDecision` records the selected estimator, allowed/refused
  outcome, reason, ordered blockers, and acknowledged planned shots.
- `MaterialisedSPSAProbe` records the gradient, seed, repetition count, shot
  mode, maximum absolute component, and shared claim boundary.

All records are immutable slot-backed dataclasses with validated construction
and JSON-ready `to_dict()` mappings. A positive dry-run decision authorises
only local planning; it never means that QPU shots ran.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `stochastic_estimators_product.v2` |
| Default estimator | `spsa_gradient` |
| Hardware shots | Always refused by the hardware-safe no-submit policy |
| Blank/unknown estimator | Fail closed |
| Live QPU execution | Never claimed by this product |
| Variance/bias campaign | Open capability; not claimed by this product |

## Confidence and failure policy

`build_product_failure_policy()` constructs the ambient
`GradientFailurePolicy` with optional positive standard-error and confidence-
radius thresholds plus the trainability requirement. Validation remains owned
by that ambient policy; this facade does not weaken or duplicate it.

## Dry-run decisions

`dry_run_stochastic_estimator()` validates the exact catalogue id and returns a
structured local plan for a positive integer shot budget. A hardware-shot
request is refused before budget validation and records zero planned shots,
preserving the hardware-safe no-submit boundary.

The default `planned_shots=100` is planning metadata, not spend authority,
provider availability, a queue reservation, or a completed experiment.

Claim boundary:

> Stochastic estimators product surface only; catalogues SPSA, score-function,
> and shot-allocation helpers with confidence-policy contracts; materialised
> finite-shot uncertainty only; composes hardware-safe no-submit / shot-budget honesty;
> does not invent-green live QPU shot runs or full variance/bias experiment
> campaigns

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

## Local SPSA probe

`materialise_demo_spsa_probe()` calls the ambient
`spsa_gradient_estimate()` on the deterministic local quadratic objective
`f(x) = sum(x_i**2)`. The default parameter vector is `[0.5, -0.25]`; callers
may set the seed, repetition count, perturbation radius, and values.

The probe uses `shots=None`, flattens the returned gradient into immutable
floats, and fails closed on an empty gradient. Its result exercises a local
contract and deterministic seed path; it is not a full estimator-bias or
variance campaign.

## Estimator catalogue

| ID | Kind |
|---|---|
| `spsa_gradient` | SPSA |
| `score_function_gradient` | score-function |
| `parameter_shift_shot_allocation` | shot allocation |
| `gradient_failure_policy` | confidence policy |

## Registry integrity

`build_stochastic_estimators_product_registry()` emits schema
`stochastic_estimators_product.v2`, the complete catalogue, ambient surface
map, default id, counts, policy note, and shared claim boundary.

Always validate transported or stored payloads through
`assert_stochastic_estimators_product_integrity()`. It rejects:

- missing, empty, non-list, non-mapping, blank, duplicate, missing, or extra rows;
- unknown estimator kinds or missing symbol names;
- any `allows_hardware_shots=True` relaxation;
- loss of the default `spsa_gradient` row; and
- `blank_entry_count` or `estimator_count` drift.

## Failure handling and operational non-effects

Treat `ValueError` as a caller-contract, ambient estimator, or transported
registry failure. Treat `RuntimeError` from catalogue construction as
repository corruption.

This product performs no credential lookup, network access, provider or QPU
discovery, hardware execution, shot submission, queue reservation, spend,
result retrieval, feedback, benchmark promotion, or evidence mutation. The
score-function and shot-allocation entries remain ambient catalogue contracts;
the shipped demo materialises SPSA only.

## Bounded product status

Shipped: estimator catalogue, contracts and tests including the materialised
SPSA demo probe, confidence-policy objects composing the hardware-safe
no-submit boundary, product documentation, and API map rows.

Open: full variance/bias documentation and experimental campaigns.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
