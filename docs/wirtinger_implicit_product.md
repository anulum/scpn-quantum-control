# Wirtinger + implicit differentiation product (BL-64)

Versioned **complex Wirtinger + real implicit sensitivity product** over ambient
`wirtinger_calculus` and `differentiable_implicit_sensitivity`. Materialised
local scalar demos only; composes BL-53 complex-without-Wirtinger refuse and
BL-46 anti-silent metamorphic law.

Module: `scpn_quantum_control.wirtinger_implicit_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `wirtinger_implicit_product.v1` |
| Default surface | `wirtinger_partials` |
| Complex without Wirtinger | Refused (BL-53 / BL-46) |
| Blank/unknown surface | Fail closed |
| Full holomorphic QFT AD | Not claimed (out of scope) |
| Planner matrix rows | Residual S64.5 |

Claim boundary:

> Wirtinger + implicit differentiation product surface only; catalogues
> Wirtinger partials / holomorphic / CR real-objective gradients and implicit
> stationary/fixed-point sensitivity; materialised local scalar demos only;
> composes BL-53 complex-without-Wirtinger refuse and BL-46 anti-silent
> metamorphic law; does not invent-green full holomorphic QFT AD, planner
> matrix rows, or hardware gradients (S64.4/S64.5 residual)

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

## Catalogue (S64.0)

| ID | Kind |
|---|---|
| `wirtinger_partials` | Wirtinger partials |
| `holomorphic_gradient` | holomorphic df/dz |
| `real_objective_cr_gradient` | CR real-loss gradient |
| `implicit_stationary_sensitivity` | stationary dx*/dalpha |
| `implicit_fixed_point_sensitivity` | fixed-point sensitivity |
| `complex_without_wirtinger_refuse` | BL-53/BL-46 refuse |

## Worked scalar examples (S64.6)

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

## Bounded product status

Shipped: S64.0 surface catalogue · S64.1 Wirtinger contracts + tests · S64.2
implicit stationary demo + tests · S64.3 BL-53 complex refuse compose · S64.6
worked scalar docs / API map rows.

Open: S64.4 full BL-46 metamorphic expansion beyond the registered anti-silent
law pointer · S64.5 planner / support-matrix rows (BL-52).

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
