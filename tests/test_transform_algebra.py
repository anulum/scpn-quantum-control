# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- transform algebra contracts
"""Contracts for composing canonical differentiable-programming transforms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    WholeProgramADResult,
    custom_jacobian,
    finite_difference_vjp,
    grad,
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jvp,
    value_and_finite_difference_jvp,
    value_and_grad,
    value_and_jacfwd,
    value_and_jacrev,
    value_and_jvp,
    value_and_vjp,
    vjp,
    vmap,
    whole_program_grad,
)

FloatArray = NDArray[np.float64]


def _float_array(value: object) -> FloatArray:
    """Convert dynamic transform outputs into typed float arrays for assertions."""

    return np.asarray(value, dtype=np.float64)


def test_transform_algebra_grad_of_vmap_and_vmap_of_grad_are_consistent() -> None:
    """grad(vmap(f)) and vmap(grad(f)) should agree for separable objectives."""

    def sample_loss(row: FloatArray) -> float:
        return float(row[0] ** 2 + np.sin(row[1]))

    values = np.array([[1.5, 0.25], [-0.5, -0.75]], dtype=np.float64)
    per_sample_grad = vmap(
        lambda row: grad(sample_loss, row, method="finite_difference"),
    )(values)
    aggregate_grad = grad(
        lambda flat: float(np.sum(_float_array(vmap(sample_loss)(flat.reshape(values.shape))))),
        values.reshape(-1),
        method="finite_difference",
    ).reshape(values.shape)

    np.testing.assert_allclose(
        _float_array(per_sample_grad), aggregate_grad, rtol=1.0e-6, atol=1.0e-6
    )


def test_transform_algebra_jacfwd_jacrev_jvp_vjp_and_hessian_contracts() -> None:
    """Canonical Jacobian, directional, adjoint, and Hessian transforms should compose."""

    def vector_objective(values: FloatArray) -> FloatArray:
        return np.array(
            [values[0] ** 2, np.sin(values[1]), values[0] * values[1]],
            dtype=np.float64,
        )

    values = np.array([1.25, -0.4], dtype=np.float64)
    expected_jacobian = np.array(
        [[2.5, 0.0], [0.0, math.cos(-0.4)], [-0.4, 1.25]],
        dtype=np.float64,
    )
    forward = jacfwd(vector_objective, values)
    reverse = jacrev(vector_objective, values)
    canonical = jacobian(vector_objective, values)

    np.testing.assert_allclose(forward, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(reverse, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(canonical, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    tangent = np.array([0.5, -2.0], dtype=np.float64)
    jvp_result = value_and_finite_difference_jvp(vector_objective, values, tangent)
    np.testing.assert_allclose(
        jvp_result.jvp, expected_jacobian @ tangent, rtol=1.0e-6, atol=1.0e-6
    )
    cotangent = np.array([1.0, -0.5, 2.0], dtype=np.float64)
    vjp_result = finite_difference_vjp(vector_objective, values, cotangent)
    np.testing.assert_allclose(
        vjp_result.vjp, expected_jacobian.T @ cotangent, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        hessian(lambda row: float(row[0] ** 2 + row[0] * row[1] + np.sin(row[1])), values),
        [[2.0, 1.0], [1.0, -math.sin(-0.4)]],
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_transform_algebra_nested_batch_jacobian_and_adjoint_contracts() -> None:
    """Nested grad, vmap, Jacobian, Hessian, JVP, and VJP transforms should agree."""

    def sample_loss(row: FloatArray) -> float:
        return float(row[0] ** 3 + row[0] * row[1] + np.sin(row[1]))

    def sample_gradient(row: FloatArray) -> FloatArray:
        return np.array([3.0 * row[0] ** 2 + row[1], row[0] + math.cos(row[1])])

    def sample_hessian(row: FloatArray) -> FloatArray:
        return np.array([[6.0 * row[0], 1.0], [1.0, -math.sin(row[1])]], dtype=np.float64)

    values = np.array([[0.7, -0.2], [-1.1, 0.4]], dtype=np.float64)
    expected_gradients = np.vstack([sample_gradient(row) for row in values])

    per_sample_gradients = vmap(
        lambda row: grad(sample_loss, row, method="finite_difference", step=1.0e-6)
    )(values)
    aggregate_gradient = grad(
        lambda flat: float(np.sum(_float_array(vmap(sample_loss)(flat.reshape(values.shape))))),
        values.reshape(-1),
        method="finite_difference",
        step=1.0e-6,
    ).reshape(values.shape)
    np.testing.assert_allclose(
        _float_array(per_sample_gradients), expected_gradients, rtol=1.0e-5, atol=1.0e-5
    )
    np.testing.assert_allclose(aggregate_gradient, expected_gradients, rtol=1.0e-5, atol=1.0e-5)

    def batched_vector_objective(flat: FloatArray) -> FloatArray:
        return _float_array(vmap(sample_loss)(flat.reshape(values.shape)))

    expected_jacobian = np.zeros((2, 4), dtype=np.float64)
    expected_jacobian[0, 0:2] = expected_gradients[0]
    expected_jacobian[1, 2:4] = expected_gradients[1]
    flat_values = values.reshape(-1)
    forward_jacobian = jacfwd(batched_vector_objective, flat_values, step=1.0e-6)
    reverse_jacobian = jacrev(batched_vector_objective, flat_values, step=1.0e-6)
    np.testing.assert_allclose(forward_jacobian, expected_jacobian, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(reverse_jacobian, expected_jacobian, rtol=1.0e-5, atol=1.0e-5)

    tangent = np.array([0.5, -0.25, 1.25, -0.75], dtype=np.float64)
    jvp_result = value_and_finite_difference_jvp(
        batched_vector_objective,
        flat_values,
        tangent,
        step=1.0e-6,
    )
    np.testing.assert_allclose(
        jvp_result.jvp, expected_jacobian @ tangent, rtol=1.0e-5, atol=1.0e-5
    )

    cotangent = np.array([2.0, -0.5], dtype=np.float64)
    vjp_result = finite_difference_vjp(
        batched_vector_objective,
        flat_values,
        cotangent,
        step=1.0e-6,
    )
    np.testing.assert_allclose(
        vjp_result.vjp, expected_jacobian.T @ cotangent, rtol=1.0e-5, atol=1.0e-5
    )
    np.testing.assert_allclose(
        hessian(sample_loss, values[0], step=1.0e-4),
        sample_hessian(values[0]),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_transform_algebra_batched_hessian_blocks_match_nested_jacobians() -> None:
    """vmap(hessian(f)) and jacfwd/jacrev(vmap(grad(f))) should agree blockwise."""

    def sample_loss(row: FloatArray) -> float:
        return float(row[0] ** 3 + row[0] * row[1] + 0.5 * row[1] ** 2)

    def sample_gradient(row: FloatArray) -> FloatArray:
        return np.array([3.0 * row[0] ** 2 + row[1], row[0] + row[1]], dtype=np.float64)

    def sample_hessian(row: FloatArray) -> FloatArray:
        return np.array([[6.0 * row[0], 1.0], [1.0, 1.0]], dtype=np.float64)

    values = np.array([[0.6, -0.4], [-1.3, 0.8], [1.1, 0.25]], dtype=np.float64)
    flat_values = values.reshape(-1)
    expected_hessians = np.stack([sample_hessian(row) for row in values])
    expected_block_hessian = np.zeros((flat_values.size, flat_values.size), dtype=np.float64)
    expected_gradient_jacobian = np.zeros((flat_values.size, flat_values.size), dtype=np.float64)

    for row_index, row in enumerate(values):
        block = slice(2 * row_index, 2 * row_index + 2)
        expected_block_hessian[block, block] = sample_hessian(row)
        expected_gradient_jacobian[block, block] = sample_hessian(row)

    per_sample_hessians = _float_array(
        vmap(lambda row: hessian(sample_loss, row, step=1.0e-4))(values)
    )

    def batched_gradient(flat: FloatArray) -> FloatArray:
        rows = flat.reshape(values.shape)
        return _float_array(
            vmap(lambda row: grad(sample_loss, row, method="finite_difference", step=1.0e-6))(rows)
        ).reshape(-1)

    forward_nested = jacfwd(batched_gradient, flat_values, step=1.0e-5)
    reverse_nested = jacrev(batched_gradient, flat_values, step=1.0e-5)

    np.testing.assert_allclose(per_sample_hessians, expected_hessians, rtol=1.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(
        forward_nested, expected_gradient_jacobian, rtol=5.0e-4, atol=5.0e-4
    )
    np.testing.assert_allclose(
        reverse_nested, expected_gradient_jacobian, rtol=5.0e-4, atol=5.0e-4
    )
    np.testing.assert_allclose(
        forward_nested,
        expected_block_hessian,
        rtol=5.0e-4,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(reverse_nested, forward_nested, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(
        batched_gradient(flat_values),
        np.vstack([sample_gradient(row) for row in values]).reshape(-1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_canonical_jvp_vjp_transforms_compose_with_vmap_and_jacobians() -> None:
    """Canonical JVP/VJP names should compose with vmap, jacfwd, jacrev, and hessian."""

    def sample_loss(row: FloatArray) -> float:
        return float(row[0] ** 2 + row[0] * row[1] + np.sin(row[1]))

    values = np.array([[0.4, -0.3], [1.2, 0.5]], dtype=np.float64)
    flat_values = values.reshape(-1)
    tangent = np.array([0.5, -1.0, 1.5, -0.25], dtype=np.float64)
    cotangent = np.array([2.0, -0.75], dtype=np.float64)

    def batched_loss(flat: FloatArray) -> FloatArray:
        return _float_array(vmap(sample_loss)(flat.reshape(values.shape)))

    forward_jacobian = jacfwd(batched_loss, flat_values, step=1.0e-6)
    reverse_jacobian = jacrev(batched_loss, flat_values, step=1.0e-6)
    jvp_result = value_and_jvp(batched_loss, flat_values, tangent, step=1.0e-6)
    vjp_result = value_and_vjp(batched_loss, flat_values, cotangent, step=1.0e-6)

    np.testing.assert_allclose(forward_jacobian, reverse_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(jvp_result.value, batched_loss(flat_values), atol=1.0e-12)
    np.testing.assert_allclose(vjp_result.value, batched_loss(flat_values), atol=1.0e-12)
    np.testing.assert_allclose(
        jvp(batched_loss, flat_values, tangent, step=1.0e-6),
        forward_jacobian @ tangent,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        vjp(batched_loss, flat_values, cotangent, step=1.0e-6),
        reverse_jacobian.T @ cotangent,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        hessian(sample_loss, values[0], step=1.0e-4),
        jacfwd(lambda row: grad(sample_loss, row, method="finite_difference"), values[0]),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_transform_algebra_custom_rules_and_whole_program_ad_compose_with_vmap() -> None:
    """Custom rules and whole-program AD should compose under vmap."""

    rule = CustomDerivativeRule(
        name="affine_sine_rule",
        value_fn=lambda values: np.array([values[0] + np.sin(values[1])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + math.cos(values[1]) * tangent[1]], dtype=np.float64
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0], math.cos(values[1]) * cotangent[0]], dtype=np.float64
        ),
        parameter_names=("offset", "phase"),
        trainable=(True, True),
    )
    values = np.array([[1.0, 0.25], [-2.0, -0.5]], dtype=np.float64)
    custom_jacobians = _float_array(vmap(lambda row: custom_jacobian(rule, row))(values))
    trace_gradients = _float_array(
        vmap(
            lambda row: whole_program_grad(
                lambda trace_values: trace_values[0] + np.sin(trace_values[1]),
                row,
                trace=False,
            )
        )(values)
    )

    expected = np.array(
        [[[1.0, math.cos(0.25)]], [[1.0, math.cos(-0.5)]]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(custom_jacobians, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(trace_gradients, expected[:, 0, :], rtol=1.0e-12, atol=1.0e-12)


def test_canonical_grad_dispatches_whole_program_method_under_vmap() -> None:
    """Canonical value_and_grad/grad should expose whole-program AD to transform algebra."""

    def loss(values: FloatArray) -> object:
        return values[0] ** 2 + np.sin(values[1])

    values = np.array([[0.5, 0.25], [-1.2, 0.75], [2.0, -0.4]], dtype=np.float64)
    single = value_and_grad(loss, values[0], method="whole_program")

    assert isinstance(single, WholeProgramADResult)
    assert single.method == "whole_program_ad"
    assert single.semantics_report is not None
    assert "whole-program operator-intercepted AD" in single.claim_boundary

    batched_gradients = _float_array(
        vmap(lambda row: grad(loss, row, method="whole_program"))(values)
    )
    expected = np.column_stack((2.0 * values[:, 0], np.cos(values[:, 1])))

    np.testing.assert_allclose(single.gradient, expected[0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(batched_gradients, expected, rtol=1.0e-12, atol=1.0e-12)


def test_whole_program_grad_of_vmap_composes_with_higher_order_transforms() -> None:
    """Whole-program grad(vmap(f)) should compose with JVP/VJP/Hessian transforms."""

    def row_loss(row: FloatArray) -> object:
        return row[0] ** 2 + np.sin(row[1])

    def batched_loss(flat_values: FloatArray) -> object:
        rows = flat_values.reshape((2, 2))
        return np.sum(cast(FloatArray, vmap(row_loss)(rows)))

    values = np.array([0.5, 0.25, -1.2, 0.75], dtype=np.float64)
    tangent = np.array([0.25, -0.5, 1.5, -0.75], dtype=np.float64)
    cotangent = np.array([1.0, -2.0, 0.5, 1.25], dtype=np.float64)
    expected_gradient = np.array(
        [2.0 * values[0], math.cos(values[1]), 2.0 * values[2], math.cos(values[3])],
        dtype=np.float64,
    )
    expected_hessian = np.diag([2.0, -math.sin(values[1]), 2.0, -math.sin(values[3])])

    program_gradient = grad(batched_loss, values, method="whole_program")
    gradient_jacobian = jacfwd(
        lambda candidate: grad(batched_loss, candidate, method="whole_program"),
        values,
        step=1.0e-6,
    )
    jvp_result = jvp(
        lambda candidate: grad(batched_loss, candidate, method="whole_program"),
        values,
        tangent,
        step=1.0e-6,
    )
    vjp_result = vjp(
        lambda candidate: grad(batched_loss, candidate, method="whole_program"),
        values,
        cotangent,
        step=1.0e-6,
    )

    np.testing.assert_allclose(program_gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(gradient_jacobian, expected_hessian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        hessian(cast(Callable[[FloatArray], float], batched_loss), values, step=1.0e-4),
        expected_hessian,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(jvp_result, expected_hessian @ tangent, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        vjp_result, expected_hessian.T @ cotangent, rtol=1.0e-6, atol=1.0e-6
    )


def test_transform_algebra_aliases_are_exported_from_package_root() -> None:
    """Transform algebra aliases should be stable package-root APIs."""

    assert scpn.jacfwd is jacfwd
    assert scpn.jacrev is jacrev
    assert scpn.jvp is jvp
    assert scpn.vjp is vjp
    assert scpn.value_and_jacfwd is value_and_jacfwd
    assert scpn.value_and_jacrev is value_and_jacrev
    assert scpn.value_and_jvp is value_and_jvp
    assert scpn.value_and_vjp is value_and_vjp
