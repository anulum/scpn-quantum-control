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
real directions `x` and `y`, for holomorphic and non-holomorphic functions
alike. Coordinate `k` moves by `step * max(1, |z_k|)` (default `step=1e-6`),
and each difference is divided by the step actually realised in floating
point. The relative accuracy therefore holds far from the origin too. The
result carries the central difference's `O(h**2)` truncation error and
rounding of order `eps * |f| / h`, so it is not exact. A step that rounds away
at `z` is refused.

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
is not holomorphic at the point. Their `tolerance` (default `1e-6`) is
relative: the residual may be at most `tolerance * max|df/dz|`. `z**2` at
`1e8(1 + i)` therefore counts as holomorphic and `1e-12 * conj(z)` does not.
The test applies per coordinate, `|df/dconj_z_k| <= tolerance * |df/dz_k| +
noise_k`, so `1e8 * z0 + conj(z1)` is not holomorphic. Here `noise_k`,
`4 eps max|f| / h_k`, is the rounding floor of that coordinate's differences,
reported as `difference_noise`. At a critical point such as `z**2 - 2z` at
`z = 1`, rounding noise is therefore not mistaken for a conjugate
dependence.

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

## Real-block Jacobian and framework conventions

The authoritative convention for any complex derivative here is the real-block
Jacobian. It treats `f: C^n -> C^m` as a map of the real and imaginary parts:

```
J = [[dRe f/dx, dRe f/dy],
     [dIm f/dx, dIm f/dy]]      shape (2m, 2n)
```

It exists for every real-differentiable `f`, holomorphic or not, and
`real_block_jacobian` returns it. For `conj(z)` it is `diag(1, -1)`. For
`|z|**2` it is `[[2x, 2y], [0, 0]]`. For a real loss, `real_block_gradient`
returns the first row, `(dL/dx, dL/dy)`; for `|z|**2` that is `(2x, 2y)`.
A real loss computed in complex arithmetic, such as `np.vdot(v, H @ v)`,
carries an imaginary rounding residue. Only the real part is differentiated.
The residue is admitted up to `1e-12 * scale`, and a larger one is refused as a
complex loss. By default `scale` is the largest `|L|` over every point
evaluated, so a tiny loss with a comparable imaginary part is refused in any
units. A real loss that passes near zero while summing large terms, such as
`<psi|H|psi>` between eigenvalues of opposite sign, is refused by that
default. Pass the size of its terms, for example `scale=np.linalg.norm(H, 2)`.

Frameworks report the gradient of a real loss of complex parameters in two
different complex forms, and the two are conjugates of each other.
`framework_complex_gradient` converts to either form explicitly:

| `convention` | Complex gradient | Equals | Matches |
| --- | --- | --- | --- |
| `"torch"` | `dL/dx + i dL/dy` | `2 dL/dconj_z` | PyTorch autograd `z.grad` |
| `"jax"` | `dL/dx - i dL/dy` | `2 dL/dz` | `jax.grad` |

Both rows are checked against the real frameworks in
`tests/test_complex_spectral_conformance.py`. Mixing them silently flips the
sign of the imaginary part of every update.

```python
import numpy as np
from scpn_quantum_control.wirtinger_calculus import (
    framework_complex_gradient,
    real_block_jacobian,
)

z = np.array([0.3 - 0.7j])
assert np.allclose(real_block_jacobian(np.conj, z), np.diag([1.0, -1.0]))
loss = lambda v: float(np.sum(np.abs(v) ** 2))
assert np.allclose(framework_complex_gradient(loss, z, convention="torch"), 2 * z)
assert np.allclose(framework_complex_gradient(loss, z, convention="jax"), 2 * np.conj(z))
```

## Spectral derivatives

`hermitian_eigenpair_derivative(A, dA, index)` differentiates one eigenpair of a
Hermitian (or real symmetric) matrix:

- `dlambda = v^H dA v`;
- `dv = sum_j v_j (v_j^H dA v) / (lambda - lambda_j)`, stated in the gauge
  `<v, dv> = 0`.

When another eigenvalue lies within the gap tolerance, it raises
`SpectralDegeneracyError`. The tolerance defaults to `1e-10 * max|lambda|`
over the whole spectrum, with no unit floor, so a matrix in small physical
units is judged at its own scale. The Hermitian checks on the matrix and on
the tangent are relative to each one's own largest entry in the same way.
At a repeated eigenvalue the individual eigenvector derivative is undefined, so
no denominator is clipped or regularised. Just above the tolerance the
derivative is returned at its true size, which grows as `1 / gap`.

Its accuracy also degrades as the gap closes. Rounding in the
eigendecomposition perturbs `dv` by up to about
`eps * max|lambda| * (|dv| / gap + 2 * |dA|_2 / gap**2)` in norm. The result
reports twice this as `rounding_error_estimate`; it is a first-order estimate
with a safety factor, not a proven bound. The `1 / gap**2` term
dominates when the tangent splits a near-degenerate pair without mixing it.
For eigenvalues of order one, a gap of `1e-8` then leaves `dv` wrong at order
one, a hundred times above the default tolerance. Compare the estimate with
`|dv|` before relying on the result. Against 50-digit references the
measured error stayed below the estimate. The worst ratio an independent
check found was about 0.65, for a spectrum offset by `1e3`; it was 1.29 before
the safety factor was added.

For a repeated eigenvalue, use the invariant subspace instead.
`spectral_projector_derivative(A, dA, indices)` returns the projector `P` onto
the chosen eigenvalues and its derivative:

```
dP = sum_{i in S, j not in S} (v_j v_j^H dA v_i v_i^H + h.c.) / (lambda_i - lambda_j)
```

Both are independent of every eigenvector phase and of the basis chosen inside
a degenerate cluster. The function needs only a gap between the chosen and the
remaining eigenvalues. A subspace that splits a repeated eigenvalue is refused.

```python
from scpn_quantum_control.wirtinger_calculus import (
    SpectralDegeneracyError,
    hermitian_eigenpair_derivative,
    spectral_projector_derivative,
)

a = np.diag([1.0, 1.0, 3.0])
da = np.array([[0.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.5, 0.1, 0.0]])
try:
    hermitian_eigenpair_derivative(a, da, 0)
except SpectralDegeneracyError:
    pass  # repeated eigenvalue: refused, not clipped
cluster = spectral_projector_derivative(a, da, (0, 1))
assert np.allclose(cluster.projector, np.diag([1.0, 1.0, 0.0]))
```

## Acceleration

The partials are a central-difference harness over a caller-supplied complex
callable: the numerical work lives inside `f`, not in a fixed kernel, so there is
no polyglot Rust path to accelerate (as with the other finite-difference
diagnostics). The spectral derivatives are closed-form expressions over one
LAPACK `eigh` call and a few matrix products. Holomorphic primitives with closed-form derivatives can be
differentiated exactly by composing analytic rules outside this module.

## Claim boundary

This is a general complex-analysis utility for complex-valued objectives. It is
independent of the registered Phase-QNode statevector engine, which remains a
real-parameter route. `minimise_real_objective` reports observed local loss
decrease only and makes no global-convergence, provider, or hardware claim.
