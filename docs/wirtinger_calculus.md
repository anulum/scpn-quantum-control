# Wirtinger (CR) Calculus

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.wirtinger_calculus` provides the complex-derivative
surface that the registered Phase-QNode engine deliberately leaves
fail-closed. The Phase-QNode differentiates **real** rotation angles, so its
complex-derivative contract rejects complex parameters; this module supplies the
complementary Wirtinger (Cauchy-Riemann) calculus for an arbitrary complex
callable `f: C^n -> C`.

## Wirtinger partials

Writing `z = x + i y`, the Wirtinger partials are

```
df/dz      = 1/2 (df/dx - i df/dy)
df/dconj_z = 1/2 (df/dx + i df/dy)
```

`wirtinger_partials` evaluates both by central differences in the independent
real directions `x` and `y`, which is exact (to rounding) for holomorphic and
non-holomorphic functions alike.

```python
import numpy as np
from scpn_quantum_control.wirtinger_calculus import wirtinger_partials

d = wirtinger_partials(lambda z: z[0] ** 2, np.array([1.3 - 0.7j]))
d.df_dz            # 2 z
d.df_dconj_z       # 0  (holomorphic)
d.holomorphic_residual
```

Textbook checks (all asserted in `tests/test_wirtinger_calculus.py`): for
`z**2`, `df/dz = 2z` and `df/dconj_z = 0`; for `|z|**2 = z conj(z)`,
`df/dz = conj(z)` and `df/dconj_z = z`; for `conj(z)`, `df/dz = 0` and
`df/dconj_z = 1`.

## Holomorphicity and the complex derivative

A function is holomorphic exactly when `df/dconj_z = 0` (the Cauchy-Riemann
equations). `is_holomorphic` tests that residual, and `holomorphic_gradient`
returns the ordinary complex derivative `df/dz`, failing closed when the function
is not holomorphic at the point.

```python
from scpn_quantum_control.wirtinger_calculus import holomorphic_gradient

holomorphic_gradient(lambda z: z[0] ** 3, np.array([1.0 + 1.0j]))   # 3 z**2
```

## Optimising complex parameters

For a real-valued loss `L: C^n -> R`, the steepest-descent direction is
`dL/dconj_z` (equal to `conj(dL/dz)`). `real_objective_gradient` returns it and
`minimise_real_objective` runs CR steepest descent.

```python
from scpn_quantum_control.wirtinger_calculus import minimise_real_objective

target = np.array([0.8 - 0.3j, -0.4 + 0.6j])
result = minimise_real_objective(
    lambda z: float(np.sum(np.abs(z - target) ** 2)),
    np.zeros(2, dtype=complex),
    learning_rate=0.3,
    steps=80,
)
result.parameters   # -> target
```

## Acceleration

The partials are a central-difference harness over a caller-supplied complex
callable: the numerical work lives inside `f`, not in a fixed kernel, so there is
no polyglot Rust path to accelerate (as with the other finite-difference
diagnostics). Holomorphic primitives with closed-form derivatives can be
differentiated exactly by composing analytic rules outside this module.

## Claim boundary

This is a general complex-analysis utility for complex-valued objectives. It is
independent of the registered Phase-QNode statevector engine, which remains a
real-parameter route. `minimise_real_objective` reports observed local loss
decrease only and makes no global-convergence, provider, or hardware claim.
