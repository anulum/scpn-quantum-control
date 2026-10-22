# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow native transform composition qualification
"""Qualify native CPU TensorFlow tapes, forward accumulators and vectorized maps."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .differentiable_structured_transform_execution import (
    tensorflow_structured_gradient,
    tensorflow_structured_hessian,
)
from .differentiable_transform_algebra import (
    TransformAlgebraCase,
    _framework_quantum_bridge_boundary_case,
    _framework_transform_case,
)


def run_tensorflow_transform_algebra_cases(
    matrix: NDArray[np.float64], *, batch_axis: int = 0
) -> tuple[TransformAlgebraCase, ...]:
    """Execute native CPU TensorFlow derivatives and custom higher-order rules.

    Parameters
    ----------
    matrix:
        Validated finite dimensionless float64 samples, ordered sample first.
    batch_axis:
        Original batch axis. TensorFlow explicitly transposes axis one to its
        required leading vectorized-map axis; it never substitutes a Python loop.

    Returns
    -------
    tuple[TransformAlgebraCase, ...]
        Actual native oracle comparisons and the separate generic quantum
        bridge refusal. Missing TensorFlow fails this required qualification.

    """
    import tensorflow as tf

    with tf.device("/CPU:0"):
        return _execute_cases(tf, matrix, batch_axis)


def _execute_cases(
    tf: Any, matrix: NDArray[np.float64], batch_axis: int
) -> tuple[TransformAlgebraCase, ...]:
    """Evaluate actual tapes and forward accumulators inside the CPU context."""
    x = tf.convert_to_tensor(matrix[0], dtype=tf.float64)
    raw_batch = tf.convert_to_tensor(matrix if batch_axis == 0 else matrix.T, dtype=tf.float64)
    batch = raw_batch if batch_axis == 0 else tf.transpose(raw_batch)
    tangent = tf.constant(np.linspace(0.2, 0.7, matrix.shape[1]), dtype=tf.float64)

    def loss(params: Any) -> Any:
        return tf.reduce_sum(params**4)

    def derivative(params: Any) -> Any:
        with tf.GradientTape() as tape:
            tape.watch(params)
            value = loss(params)
        return tape.gradient(value, params)

    def hessian(params: Any) -> Any:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(params)
            gradient = derivative(params)
        return tape.jacobian(gradient, params, experimental_use_pfor=False)

    curvature = np.diag(12.0 * matrix[0] ** 2)
    expected_gradient = 4.0 * matrix[0] ** 3
    mapped = tf.vectorized_map(derivative, batch, fallback_to_while_loop=False)
    with tf.GradientTape() as aggregate_tape:
        aggregate_tape.watch(batch)
        aggregate_loss = tf.reduce_sum(
            tf.vectorized_map(loss, batch, fallback_to_while_loop=False)
        )
    aggregate_gradient = aggregate_tape.gradient(aggregate_loss, batch)

    def forward_gradient(direction: Any) -> Any:
        with tf.autodiff.ForwardAccumulator(x, direction) as accumulator:
            gradient = derivative(x)
        return accumulator.jvp(gradient)

    pushed = forward_gradient(tangent)
    with tf.GradientTape() as pullback:
        pullback.watch(x)
        gradient = derivative(x)
    pulled = pullback.gradient(gradient, x, output_gradients=tangent)
    forward_hessian = tf.transpose(
        tf.vectorized_map(
            forward_gradient,
            tf.eye(matrix.shape[1], dtype=tf.float64),
            fallback_to_while_loop=False,
        )
    )

    structured = {"weights": x, "bias": tf.constant(0.3, dtype=tf.float64)}
    with tf.GradientTape(persistent=True) as outer:
        outer.watch(structured)
        with tf.GradientTape() as inner:
            inner.watch(structured)
            structured_loss = tf.reduce_sum((structured["weights"] + structured["bias"]) ** 4)
        structured_gradient, flat_gradient = tensorflow_structured_gradient(
            tf, inner, structured_loss, structured
        )
    weights_hessian, bias_hessian = tensorflow_structured_hessian(outer, flat_gradient, structured)
    shifted = matrix[0] + 0.3
    diagonal = 12.0 * shifted**2
    structured_actual = np.concatenate(
        [
            np.ravel(structured_gradient["weights"]),
            np.ravel(structured_gradient["bias"]),
            np.ravel(weights_hessian[:-1]),
            np.ravel(bias_hessian[:-1]),
            np.ravel(weights_hessian[-1]),
            np.ravel(bias_hessian[-1]),
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

    def registered_value(params: Any) -> tuple[Any, Any]:
        def backward(cotangent: Any) -> Any:
            return cotangent * 4.0 * params**3

        return loss(params), backward

    registered = tf.custom_gradient(registered_value)

    def custom_composed(params: Any) -> Any:
        return tf.sin(registered(params))

    with tf.GradientTape(persistent=True) as custom_outer:
        custom_outer.watch(x)
        with tf.GradientTape() as custom_inner:
            custom_inner.watch(x)
            custom_value = custom_composed(x)
        custom_gradient = custom_inner.gradient(custom_value, x)
    custom_hessian = custom_outer.jacobian(custom_gradient, x, experimental_use_pfor=False)
    primal = np.sum(matrix[0] ** 4)
    custom_curvature = np.cos(primal) * curvature - np.sin(primal) * np.outer(
        expected_gradient, expected_gradient
    )

    def structured_batch_gradient(weights: Any) -> tuple[Any, Any]:
        bias = structured["bias"]
        with tf.GradientTape() as tape:
            tape.watch((weights, bias))
            value = tf.reduce_sum((weights + bias) ** 4)
        weights_gradient, bias_gradient = tape.gradient(value, (weights, bias))
        return weights_gradient, bias_gradient

    structured_weights, structured_bias = tf.vectorized_map(
        structured_batch_gradient, batch, fallback_to_while_loop=False
    )

    def custom_batch_gradient(params: Any) -> Any:
        with tf.GradientTape() as tape:
            tape.watch(params)
            value = custom_composed(params)
        return tape.gradient(value, params)

    batched_custom = tf.vectorized_map(custom_batch_gradient, batch, fallback_to_while_loop=False)
    structured_batch_actual = np.concatenate(
        [np.ravel(structured_weights), np.ravel(structured_bias)]
    )
    custom_batch_actual = batched_custom
    custom_batch_expected = np.cos(np.sum(matrix**4, axis=1))[:, None] * 4.0 * matrix**3

    shifted_batch_gradient = 4.0 * (matrix + 0.3) ** 3
    structured_batch_expected = np.concatenate(
        [shifted_batch_gradient.ravel(), np.sum(shifted_batch_gradient, axis=1)]
    )

    def case(category: str, actual: Any, expected: Any, *transforms: str) -> TransformAlgebraCase:
        numeric = tf.convert_to_tensor(actual, dtype=tf.float64).numpy()
        return _framework_transform_case(
            "tensorflow", str(tf.__version__), category, numeric, expected, transforms
        )

    return (
        case("framework_grad", derivative(x), expected_gradient, "tf.GradientTape.gradient"),
        case("framework_hessian", hessian(x), curvature, "nested GradientTape.jacobian"),
        case(
            "framework_vmap_grad",
            mapped,
            4.0 * matrix**3,
            "tf.vectorized_map(GradientTape.gradient)",
            "fallback_to_while_loop=False",
            f"input_batch_axis={batch_axis}",
        ),
        case(
            "framework_grad_vmap",
            aggregate_gradient,
            4.0 * matrix**3,
            "GradientTape.gradient(sum(tf.vectorized_map))",
        ),
        case(
            "framework_jacfwd_jacrev",
            np.concatenate([np.ravel(forward_hessian), np.ravel(hessian(x))]),
            np.tile(curvature.ravel(), 2),
            "ForwardAccumulator(GradientTape.gradient)",
            "GradientTape.jacobian(GradientTape.gradient)",
        ),
        case(
            "framework_jvp_vjp",
            np.concatenate([np.ravel(pushed), np.ravel(pulled)]),
            np.concatenate([curvature @ np.asarray(tangent), curvature.T @ np.asarray(tangent)]),
            "ForwardAccumulator(gradient)",
            "GradientTape.gradient(gradient,output_gradients)",
        ),
        case(
            "framework_structured",
            structured_actual,
            structured_expected,
            "dict",
            "GradientTape.gradient",
            "GradientTape.jacobian(gradient)",
        ),
        case(
            "framework_custom_rules",
            np.concatenate([np.ravel(custom_gradient), np.ravel(custom_hessian)]),
            np.concatenate([np.cos(primal) * expected_gradient, custom_curvature.ravel()]),
            "tf.custom_gradient",
            "gradient(sin(custom))",
            "jacobian(gradient(sin(custom)))",
        ),
        case(
            "framework_structured_vmap",
            structured_batch_actual,
            structured_batch_expected,
            "tf.vectorized_map(GradientTape.gradient(dict,bias_shared))",
        ),
        case(
            "framework_custom_vmap",
            custom_batch_actual,
            custom_batch_expected,
            "tf.vectorized_map(gradient(sin(custom_gradient)))",
        ),
        _framework_quantum_bridge_boundary_case("tensorflow"),
    )
