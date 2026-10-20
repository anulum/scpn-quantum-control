# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX native transform composition qualification
"""Qualify actual JAX compositions without host callbacks or global dtype changes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .differentiable_transform_algebra import (
    TransformAlgebraCase,
    _framework_quantum_bridge_boundary_case,
    _framework_transform_case,
)


def run_jax_transform_algebra_cases(
    matrix: NDArray[np.float64], *, batch_axis: int = 0
) -> tuple[TransformAlgebraCase, ...]:
    """Execute native CPU JAX derivatives and structured/custom compositions.

    Parameters
    ----------
    matrix:
        Validated finite dimensionless float64 samples, ordered sample first.
    batch_axis:
        Native ``vmap`` input axis, zero or one.

    Returns
    -------
    tuple[TransformAlgebraCase, ...]
        Numeric oracle comparisons and the distinct generic quantum-bridge
        refusal. The selected runtime is mandatory and import errors propagate.

    """
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        return _execute_cases(jax, jnp, matrix, batch_axis)


def _execute_cases(
    jax: Any, jnp: Any, matrix: NDArray[np.float64], batch_axis: int
) -> tuple[TransformAlgebraCase, ...]:
    """Evaluate actual transforms while the CPU and float64 contexts are active."""
    x = jnp.asarray(matrix[0], dtype=jnp.float64)
    batch = jnp.asarray(matrix if batch_axis == 0 else matrix.T, dtype=jnp.float64)
    tangent = jnp.linspace(0.2, 0.7, matrix.shape[1], dtype=jnp.float64)

    def loss(params: Any) -> Any:
        return jnp.sum(params**4)

    derivative = jax.grad(loss)
    curvature = np.diag(12.0 * matrix[0] ** 2)
    expected_gradient = 4.0 * matrix[0] ** 3
    mapped = jax.vmap(derivative, in_axes=batch_axis, out_axes=0)(batch)

    def aggregate(params: Any) -> Any:
        return jnp.sum(jax.vmap(loss, in_axes=batch_axis)(params))

    aggregate_gradient = jax.grad(aggregate)(batch)
    if batch_axis == 1:
        aggregate_gradient = aggregate_gradient.T
    _, pushed = jax.jvp(derivative, (x,), (tangent,))
    _, pullback = jax.vjp(derivative, x)
    (pulled,) = pullback(tangent)

    structured = {"weights": x, "bias": jnp.asarray(0.3, dtype=jnp.float64)}

    def structured_loss(params: Any) -> Any:
        return jnp.sum((params["weights"] + params["bias"]) ** 4)

    structured_gradient = jax.grad(structured_loss)(structured)
    structured_hessian = jax.jacfwd(jax.grad(structured_loss))(structured)
    shifted = matrix[0] + 0.3
    diagonal = 12.0 * shifted**2
    structured_actual = np.concatenate(
        [
            np.ravel(structured_gradient["weights"]),
            np.ravel(structured_gradient["bias"]),
            np.ravel(structured_hessian["weights"]["weights"]),
            np.ravel(structured_hessian["weights"]["bias"]),
            np.ravel(structured_hessian["bias"]["weights"]),
            np.ravel(structured_hessian["bias"]["bias"]),
        ]
    )
    structured_expected = np.concatenate(
        [
            4.0 * shifted**3,
            [np.sum(4.0 * shifted**3)],
            np.diag(diagonal).ravel(),
            diagonal,
            diagonal,
            [np.sum(diagonal)],
        ]
    )

    def custom_forward_value(params: Any) -> Any:
        return loss(params)

    custom_forward = jax.custom_jvp(custom_forward_value)

    def custom_forward_rule(primals: Any, tangents: Any) -> tuple[Any, Any]:
        (params,), (directions,) = primals, tangents
        return loss(params), jnp.sum(4.0 * params**3 * directions)

    custom_forward.defjvp(custom_forward_rule)

    def custom_reverse_value(params: Any) -> Any:
        return loss(params)

    custom_reverse = jax.custom_vjp(custom_reverse_value)

    def forward_rule(params: Any) -> tuple[Any, Any]:
        return loss(params), params

    def backward_rule(params: Any, cotangent: Any) -> tuple[Any]:
        return (cotangent * 4.0 * params**3,)

    custom_reverse.defvjp(forward_rule, backward_rule)

    def forward_composed(params: Any) -> Any:
        return jnp.sin(custom_forward(params))

    def reverse_composed(params: Any) -> Any:
        return jnp.sin(custom_reverse(params))

    primal = np.sum(matrix[0] ** 4)
    custom_curvature = np.cos(primal) * curvature - np.sin(primal) * np.outer(
        expected_gradient, expected_gradient
    )
    custom_actual = np.concatenate(
        [
            np.ravel(forward_composed(x)),
            np.ravel(reverse_composed(x)),
            np.ravel(jax.grad(forward_composed)(x)),
            np.ravel(jax.grad(reverse_composed)(x)),
            np.ravel(jax.hessian(forward_composed)(x)),
            np.ravel(jax.jacrev(jax.grad(reverse_composed))(x)),
        ]
    )
    custom_expected = np.concatenate(
        [
            [np.sin(primal)],
            [np.sin(primal)],
            np.cos(primal) * expected_gradient,
            np.cos(primal) * expected_gradient,
            custom_curvature.ravel(),
            custom_curvature.ravel(),
        ]
    )

    structured_batch = {"weights": batch, "bias": structured["bias"]}
    structured_mapped = jax.vmap(
        jax.grad(structured_loss), in_axes=({"weights": batch_axis, "bias": None},)
    )(structured_batch)
    batched_custom = jax.vmap(jax.grad(forward_composed), in_axes=batch_axis)(batch)
    batched_custom_reverse = jax.vmap(jax.grad(reverse_composed), in_axes=batch_axis)(batch)
    structured_batch_actual = np.concatenate(
        [np.ravel(structured_mapped["weights"]), np.ravel(structured_mapped["bias"])]
    )
    custom_batch_actual = np.concatenate(
        [np.ravel(batched_custom), np.ravel(batched_custom_reverse)]
    )
    custom_batch_expected = np.tile(
        (np.cos(np.sum(matrix**4, axis=1))[:, None] * 4.0 * matrix**3).ravel(), 2
    )

    shifted_batch_gradient = 4.0 * (matrix + 0.3) ** 3
    structured_batch_expected = np.concatenate(
        [shifted_batch_gradient.ravel(), np.sum(shifted_batch_gradient, axis=1)]
    )

    def case(category: str, actual: Any, expected: Any, *transforms: str) -> TransformAlgebraCase:
        return _framework_transform_case(
            "jax", str(jax.__version__), category, actual, expected, transforms
        )

    return (
        case("framework_grad", derivative(x), expected_gradient, "jax.grad"),
        case("framework_hessian", jax.hessian(loss)(x), curvature, "jax.hessian"),
        case(
            "framework_vmap_grad",
            mapped,
            4.0 * matrix**3,
            "jax.vmap(jax.grad)",
            f"in_axes={batch_axis}",
        ),
        case(
            "framework_grad_vmap",
            aggregate_gradient,
            4.0 * matrix**3,
            "jax.grad(sum(jax.vmap))",
            f"in_axes={batch_axis}",
        ),
        case(
            "framework_jacfwd_jacrev",
            np.concatenate(
                [np.ravel(jax.jacfwd(derivative)(x)), np.ravel(jax.jacrev(derivative)(x))]
            ),
            np.tile(curvature.ravel(), 2),
            "jax.jacfwd(jax.grad)",
            "jax.jacrev(jax.grad)",
        ),
        case(
            "framework_jvp_vjp",
            np.concatenate([np.ravel(pushed), np.ravel(pulled)]),
            np.concatenate([curvature @ np.asarray(tangent), curvature.T @ np.asarray(tangent)]),
            "jax.jvp(jax.grad)",
            "jax.vjp(jax.grad)",
        ),
        case(
            "framework_structured",
            structured_actual,
            structured_expected,
            "dict",
            "jax.grad",
            "jax.jacfwd(jax.grad)",
        ),
        case(
            "framework_custom_rules",
            custom_actual,
            custom_expected,
            "jax.custom_jvp",
            "jax.custom_vjp",
            "grad(sin(custom))",
            "hessian(sin(custom_jvp))",
            "jacrev(grad(sin(custom_vjp)))",
        ),
        case(
            "framework_structured_vmap",
            structured_batch_actual,
            structured_batch_expected,
            "jax.vmap(grad(dict),in_axes={weights:axis,bias:None})",
        ),
        case(
            "framework_custom_vmap",
            custom_batch_actual,
            custom_batch_expected,
            "jax.vmap(grad(sin(custom_jvp)))",
            "jax.vmap(grad(sin(custom_vjp)))",
        ),
        _framework_quantum_bridge_boundary_case("jax"),
    )
