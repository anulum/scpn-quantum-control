# Wirtinger + implicit differentiation product

Versioned **complex Wirtinger + real implicit sensitivity product** over ambient
`wirtinger_calculus` and `differentiable_implicit_sensitivity`. Materialised
local scalar demos only; composes the complex-without-Wirtinger unsuitable
scenario and anti-silent metamorphic law.

Module: `scpn_quantum_control.wirtinger_implicit_product`

This is a bounded product facade over existing numerical implementations. It
does not install a complex-gradient convention into arbitrary callers, prove
holomorphicity beyond the reported local probe, or produce hardware evidence.

## Contract discovery

| Function | Contract |
|---|---|
| `list_wirtinger_implicit_surface_ids()` | Returns all stable surface ids in catalogue order. |
| `get_wirtinger_implicit_surface(surface_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_wirtinger_implicit_surfaces(...)` | Filters deterministically by typed surface kind and/or support posture. |
| `map_wirtinger_implicit_public_surfaces()` | Groups catalogue rows by their ambient module owner. |

The support postures distinguish locally materialised demonstrations,
policy-only contracts, and refuse-only guardrails. Discovery reads static local
metadata only; it does not execute a numerical probe.

## Public value objects

- `WirtingerImplicitSurfaceRow` binds a stable id to its kind, ambient module
  and symbol, support posture, governance pointers, and claim boundary.
- `MaterialisedWirtingerProbe` carries the complex evaluation point,
  `df/dz`, `df/dconj_z`, local residual, and thresholded holomorphic flag.
- `MaterialisedImplicitProbe` carries a flattened real sensitivity matrix,
  its shape, condition number, method, and demo label.
- `ComplexContractDecision` records whether the caller declared an explicit
  Wirtinger contract, together with blockers and governance pointers.

All four are immutable, slot-backed dataclasses with validated construction and
JSON-ready `to_dict()` mappings. A serialised probe remains local numerical
evidence; it is not a general theorem or provider result.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `wirtinger_implicit_product.v1` |
| Default surface | `wirtinger_partials` |
| Complex without Wirtinger | Refused by unsuitable-scenario and metamorphic-law policy |
| Blank/unknown surface | Fail closed |
| Full holomorphic QFT AD | Not claimed (out of scope) |
| Planner matrix rows | Open product boundary |

## Complex-objective decision

`decide_complex_objective_contract()` is intentionally fail closed. A caller
that declares no Wirtinger contract receives `allowed=False`, both the
unsuitable-scenario id and anti-silent-law id, and non-empty blockers.
The facade never silently substitutes an ordinary real gradient for a complex
objective.

An explicit declaration produces `allowed=True` with no blockers, but it proves
only that the caller selected a contract. The caller still owns objective
semantics, derivative validation, numerical tolerances, and backend support.

## Local Wirtinger probes

`materialise_demo_wirtinger_probe()` supports exactly two scalar objectives:

| Demo | Objective | Expected local result |
|---|---|---|
| `holomorphic_square` | `f(z) = z²` | `df/dz = 2z`; `df/dconj_z` is approximately zero. |
| `modulus_squared` | `f(z) = abs(z)²` | Non-zero `df/dconj_z`; classified non-holomorphic at generic points. |

The ambient `wirtinger_partials()` implementation uses the requested central
difference step. `is_holomorphic` is only the comparison
`holomorphic_residual <= tolerance` for this point and step. Unknown demo labels
or invalid ambient parameters raise `ValueError`.

## Local implicit probe

`materialise_demo_implicit_stationary_probe()` constructs the one-dimensional
stationary system `H = [[hessian_scale]]` and `B = [[cross_scale]]`, then calls
the ambient implicit-sensitivity implementation. The expected sensitivity is
`-cross_scale / hessian_scale`. The Hessian scale must be positive and finite;
the cross scale must be finite; and an empty ambient result fails closed.

The returned shape and row-major flattened sensitivity are validated together.
The reported condition number must also be finite and non-negative.

Claim boundary:

> Wirtinger + implicit differentiation product surface only; catalogues
> Wirtinger partials / holomorphic / CR real-objective gradients and implicit
> stationary/fixed-point sensitivity; materialised local scalar demos only;
> composes the complex-without-Wirtinger unsuitable scenario and anti-silent
> metamorphic law; does not invent-green full holomorphic QFT AD, planner
> matrix rows, or hardware gradients

## Public API

```python
from scpn_quantum_control.wirtinger_implicit_product import (
    assert_wirtinger_implicit_product_integrity,
    build_wirtinger_implicit_product_registry,
    decide_complex_objective_contract,
    list_wirtinger_implicit_surface_ids,
    materialise_demo_implicit_stationary_probe,
    materialise_demo_wirtinger_probe,
)

assert "wirtinger_partials" in list_wirtinger_implicit_surface_ids()
reg = assert_wirtinger_implicit_product_integrity(
    build_wirtinger_implicit_product_registry()
)

refused = decide_complex_objective_contract(has_wirtinger_contract=False)
assert refused.allowed is False

allowed = decide_complex_objective_contract(has_wirtinger_contract=True)
assert allowed.allowed is True

# Worked scalar example: f(z)=z^2 is holomorphic, df/dz = 2z
w = materialise_demo_wirtinger_probe(demo="holomorphic_square", z0=1.0 + 0.5j)
assert w.is_holomorphic
assert abs(w.df_dz[0][0] - 2.0) < 1e-4

# Non-holomorphic |z|^2 has positive CR residual
nh = materialise_demo_wirtinger_probe(demo="modulus_squared")
assert not nh.is_holomorphic

# Stationary demo: H=2, B=1 => dx*/dalpha = -0.5
imp = materialise_demo_implicit_stationary_probe(hessian_scale=2.0, cross_scale=1.0)
assert abs(imp.sensitivity[0] + 0.5) < 1e-9
```

## Surface catalogue

| ID | Kind |
|---|---|
| `wirtinger_partials` | Wirtinger partials |
| `holomorphic_gradient` | holomorphic df/dz |
| `real_objective_cr_gradient` | CR real-loss gradient |
| `implicit_stationary_sensitivity` | stationary dx*/dalpha |
| `implicit_fixed_point_sensitivity` | fixed-point sensitivity |
| `complex_without_wirtinger_refuse` | unsuitable-scenario and metamorphic-law refusal |

## Registry integrity

`build_wirtinger_implicit_product_registry()` emits schema
`wirtinger_implicit_product.v1`, the complete catalogue, ambient public-surface
map, exact governance pointers, default id, counts, and claim boundary.

Always validate transported or stored payloads with
`assert_wirtinger_implicit_product_integrity()`. It rejects:

- an absent, empty, or non-list `surfaces` value;
- non-mapping, blank, duplicate, missing, or extra surface rows;
- unknown kinds or missing symbol and governance pointers;
- loss of either the default partials row or the explicit refuse row; and
- drift in `blank_entry_count` or `surface_count`.

## Worked scalar examples

### Holomorphic square

For ``f(z) = z^2`` at ``z = 1 + 0.5i``:

- ``df/dz = 2z = 2 + i``
- ``df/dconj_z = 0`` (Cauchy-Riemann residual ~ 0)

### Modulus squared

For ``f(z) = |z|^2`` the function is real-valued and non-holomorphic; residual
``max|df/dconj_z|`` is strictly positive at generic points.

### Implicit stationary 1-D

With ``H = [[2]]`` and ``B = [[1]]``:

- ``dx*/dalpha = -H^{-1} B = -0.5``

## Failure handling and non-effects

Treat `ValueError` as a caller-contract or transported-registry failure. Treat
`RuntimeError` from internal catalogue construction as repository corruption.
Do not turn a refused decision into a fallback real gradient, and do not infer
global holomorphicity from one locally thresholded probe.

This module performs no network access, credential lookup, provider discovery,
QPU submission, hardware execution, registry mutation, evidence promotion, or
planner/support-matrix rewrite. Its demonstrations run locally through the
existing NumPy-based ambient implementations.

## Bounded product status

Shipped: surface catalogue · Wirtinger contracts and tests · implicit stationary
demo and tests · complex-objective refusal composition · worked scalar docs and
API map rows.

Open: full metamorphic expansion beyond the registered anti-silent law pointer ·
planner and governed support-matrix rows.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
