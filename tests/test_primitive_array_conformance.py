# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public primitive and array derivative conformance
"""Independent array oracles exercised through registered and whole-program APIs."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    Parameter,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    program_ad_array_getitem_derivative_rule,
    program_ad_assembly_broadcast_to_derivative_rule,
    program_ad_linalg_solve_derivative_rule,
    program_adjoint_gradient,
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
    whole_program_value_and_grad,
)

Transform = Literal["jvp", "vjp", "jacobian", "batch_jvp", "batch_vjp", "batch_jacobian"]


def _broadcast_product(values: NDArray[np.float64]) -> object:
    """Evaluate the scalar broadcast product through the original trace frontend."""
    column = np.reshape(values[:2], (2, 1))
    row = np.reshape(values[2:], (1, 3))
    return np.sum(column * row)


def test_broadcast_product_reduction_matches_analytic_gradient_and_adjoint() -> None:
    """Preserve every broadcast contribution in a real executed NumPy program."""
    values = np.array([2.0, -3.0, 0.5, -1.5, 4.0])
    original = values.copy()
    result = whole_program_value_and_grad(_broadcast_product, values)
    expected = np.array([3.0, 3.0, -1.0, -1.0, -1.0])
    assert result.value == pytest.approx(-3.0, abs=1e-12)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), expected, atol=1e-12)
    np.testing.assert_array_equal(values, original)


def test_duplicate_gather_scatter_accumulates_and_satisfies_duality() -> None:
    """Repeated static indices accumulate adjoints instead of overwriting them."""
    registry = CustomDerivativeRegistry()
    rule = program_ad_array_getitem_derivative_rule((3,), [0, 0, 2, 0])
    identity = "scpn.conformance:duplicate_gather@1"
    registry.register(identity, rule)
    values = np.array([2.0, -7.0, 5.0])
    tangent = np.array([0.25, 9.0, -0.5])
    cotangent = np.array([1.0, -2.0, 3.0, 4.0])
    before = [array.copy() for array in (values, tangent, cotangent)]

    push = registered_custom_jvp(identity, values, tangent, registry=registry)
    pull = registered_custom_vjp(identity, values, cotangent, registry=registry)
    jacobian = registered_custom_jacobian(identity, values, registry=registry)
    independent = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(pull.value, [2.0, 2.0, 5.0, 2.0])
    np.testing.assert_array_equal(push, [0.25, 0.25, -0.5, 0.25])
    np.testing.assert_array_equal(pull.vjp, [3.0, 0.0, 3.0])
    np.testing.assert_array_equal(jacobian.jacobian, independent)
    assert np.dot(push, cotangent) == pytest.approx(np.dot(tangent, pull.vjp), abs=1e-12)
    for original, actual in zip(before, (values, tangent, cotangent), strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6], ids=["unit", "thousand", "million"])
def test_linear_solve_directional_and_adjoint_oracles(scale: float) -> None:
    """Match a closed-form triangular solve and its real inner-product adjoint.

    Parameters
    ----------
    scale : float
        Prespecified positive diagonal scale in the conditioning sweep.

    """
    matrix = np.array([[2.0, 0.5], [0.0, scale]])
    rhs = np.array([1.5, -2.0])
    values = np.concatenate((matrix.ravel(), rhs))
    direction = np.array([0.1, -0.2, 0.3, -0.4, 0.7, -0.6])
    cotangent = np.array([-0.25, 1.5])
    # Independent 2x2 inverse arithmetic; the production owner uses solve.
    inverse = np.array([[0.5, -0.25 / scale], [0.0, 1.0 / scale]])
    expected_value = inverse @ rhs
    differential = direction[4:] - direction[:4].reshape(2, 2) @ expected_value
    expected_jvp = inverse @ differential
    rhs_adjoint = inverse.T @ cotangent
    expected_vjp = np.concatenate((-np.outer(rhs_adjoint, expected_value).ravel(), rhs_adjoint))
    registry = CustomDerivativeRegistry()
    identity = "scpn.conformance:triangular_solve@1"
    registry.register(identity, program_ad_linalg_solve_derivative_rule((2, 2), (2,)))

    push = registered_custom_jvp(identity, values, direction, registry=registry)
    pull = registered_custom_vjp(identity, values, cotangent, registry=registry)
    np.testing.assert_allclose(pull.value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(push, expected_jvp, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pull.vjp, expected_vjp, rtol=1e-10, atol=1e-12)
    assert np.dot(push, cotangent) == pytest.approx(np.dot(direction, pull.vjp), abs=1e-12)
    errors = []
    for step in (1e-2, 1e-3, 1e-4):
        plus = values + step * direction
        minus = values - step * direction

        # Explicit adjugate arithmetic supplies the independent primal refinement.
        def analytic_solution(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate the two-dimensional inverse formula."""
            a, b, c, d, x, y = map(float, vector)
            determinant = a * d - b * c
            return np.asarray(
                [(d * x - b * y) / determinant, (a * y - c * x) / determinant],
                dtype=np.float64,
            )

        difference = (analytic_solution(plus) - analytic_solution(minus)) / (2 * step)
        errors.append(float(np.max(np.abs(difference - push))))
    assert errors[-1] <= 1e-8
    assert errors[-1] < errors[0]


@pytest.mark.parametrize("transform", ["jvp", "vjp", "jacobian"])
def test_unnamed_rule_preserves_frozen_parameter_mask(transform: str) -> None:
    """A real broadcast rule retains its mask when default parameter names are used.

    Parameters
    ----------
    transform : str
        Public registered transform evaluated against the same masked linear map.

    """
    registry = CustomDerivativeRegistry()
    identity = "scpn.conformance:masked_broadcast@1"
    rule = replace(
        program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2)),
        trainable=(False, True),
    )
    registry.register(identity, rule)
    values = np.array([3.0, -2.0])
    if transform == "jvp":
        result = registered_custom_jvp(identity, values, [7.0, 5.0], registry=registry)
        np.testing.assert_array_equal(result, [0.0, 5.0, 0.0, 5.0])
    elif transform == "vjp":
        pull = registered_custom_vjp(identity, values, [1.0, 2.0, 3.0, 4.0], registry=registry)
        np.testing.assert_array_equal(pull.vjp, [0.0, 6.0])
        assert pull.parameter_names == ("theta_0", "theta_1")
        assert pull.trainable == (False, True)
    else:
        jacobian = registered_custom_jacobian(identity, values, registry=registry)
        np.testing.assert_array_equal(
            jacobian.jacobian, [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]
        )
        assert jacobian.trainable == (False, True)


def _evaluate_metadata_transform(rule: CustomDerivativeRule, transform: Transform) -> object:
    """Call the original public transform with fixed broadcast-shaped inputs."""
    registry = CustomDerivativeRegistry()
    identity = "scpn.conformance:metadata_admission@1"
    registry.register(identity, rule)
    if transform == "jvp":
        return registered_custom_jvp(identity, [1.0, 2.0], [3.0, 4.0], registry=registry)
    if transform == "vjp":
        return registered_custom_vjp(identity, [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], registry=registry)
    if transform == "jacobian":
        return registered_custom_jacobian(identity, [1.0, 2.0], registry=registry)
    if transform == "batch_jvp":
        return batch_value_and_custom_jvp(rule, [1.0, 2.0], [[3.0, 4.0]])
    if transform == "batch_vjp":
        return batch_value_and_custom_vjp(rule, [1.0, 2.0], [[1.0, 2.0, 3.0, 4.0]])
    return batch_value_and_custom_jacobian(rule, [[1.0, 2.0]])


@pytest.mark.parametrize(
    "transform", ["jvp", "vjp", "jacobian", "batch_jvp", "batch_vjp", "batch_jacobian"]
)
@pytest.mark.parametrize("metadata", ["short-mask", "duplicate-names"])
def test_invalid_rule_metadata_refuses_before_callback(
    transform: Transform,
    metadata: str,
) -> None:
    """Reject malformed rule metadata before the original numerical callback runs.

    Parameters
    ----------
    transform : Transform
        Public transform that must validate before numerical evaluation.
    metadata : str
        Demonstrated invalid shape or identity metadata.

    """
    original = program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2))
    calls: list[str] = []

    def recorded_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Count numerical evaluations without replacing their production implementation."""
        calls.append("value")
        return np.asarray(original.value_fn(values), dtype=np.float64)

    rule = replace(
        original,
        value_fn=recorded_value,
        parameter_names=("gain", "gain") if metadata == "duplicate-names" else (),
        trainable=(True, True) if metadata == "duplicate-names" else (False,),
    )
    with pytest.raises(ValueError, match="trainable mask length|parameter names must be unique"):
        _evaluate_metadata_transform(rule, transform)
    assert calls == []
    # A refusal must not poison a later original, correctly specified request.
    np.testing.assert_array_equal(
        _evaluate_metadata_transform(original, "jvp"),
        [3.0, 4.0, 3.0, 4.0],
    )


def test_explicit_parameters_preserve_original_override_semantics() -> None:
    """Explicit caller metadata still overrides the rule-local names and mask."""
    registry = CustomDerivativeRegistry()
    identity = "scpn.conformance:explicit_override@1"
    registry.register(
        identity,
        replace(
            program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2)),
            parameter_names=("original_a", "original_b"),
            trainable=(False, True),
        ),
    )
    parameters = (Parameter("caller_a", True), Parameter("caller_b", False))
    result = registered_custom_vjp(
        identity,
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        parameters=parameters,
        registry=registry,
    )
    np.testing.assert_array_equal(result.vjp, [4.0, 0.0])
    assert result.parameter_names == ("caller_a", "caller_b")
    assert result.trainable == (True, False)


@pytest.mark.parametrize("missing_rule", [False, True])
def test_batch_pullback_rejects_unsupported_rule_before_primal(missing_rule: bool) -> None:
    """Unsupported batch pullbacks refuse before any value callback executes.

    Parameters
    ----------
    missing_rule : bool
        Whether the invalid request omits the rule object or just its VJP.

    """
    original = program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2))
    calls: list[str] = []

    def recorded_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Count calls to the original broadcast implementation."""
        calls.append("value")
        return np.asarray(original.value_fn(values), dtype=np.float64)

    rule = (
        cast(CustomDerivativeRule, object())
        if missing_rule
        else replace(original, value_fn=recorded_value, vjp_rule=None)
    )
    with pytest.raises(ValueError, match="CustomDerivativeRule|does not define a VJP"):
        batch_value_and_custom_vjp(rule, [1.0, 2.0], [[1.0, 2.0, 3.0, 4.0]])
    assert calls == []


def test_batched_derivatives_retain_unnamed_mask_and_input_order() -> None:
    """All three batched public transforms keep the same frozen coordinate."""
    rule = replace(
        program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2)),
        trainable=(False, True),
    )
    pushes = batch_value_and_custom_jvp(rule, [1.0, 2.0], [[3.0, 4.0], [-5.0, -6.0]])
    pulls = batch_value_and_custom_vjp(
        rule, [1.0, 2.0], [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]
    )
    jacobians = batch_value_and_custom_jacobian(rule, [[1.0, 2.0], [-1.0, -2.0]])
    np.testing.assert_array_equal(pushes[0].jvp, [0.0, 4.0, 0.0, 4.0])
    np.testing.assert_array_equal(pushes[1].jvp, [0.0, -6.0, 0.0, -6.0])
    np.testing.assert_array_equal(pulls[0].vjp, [0.0, 6.0])
    np.testing.assert_array_equal(pulls[1].vjp, [0.0, -6.0])
    for result in jacobians:
        np.testing.assert_array_equal(
            result.jacobian, [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]
        )
        assert result.trainable == (False, True)


def test_missing_identity_and_missing_derivative_do_not_fabricate_a_result() -> None:
    """An unknown rule and an unsupported derivative cannot return identity or zero."""
    registry = CustomDerivativeRegistry()
    with pytest.raises(ValueError, match="no custom derivative rule"):
        registered_custom_jvp("scpn.conformance:absent@1", [1.0], [2.0], registry=registry)
    original = program_ad_assembly_broadcast_to_derivative_rule((2,), (2, 2))
    registry.register("scpn.conformance:jvp_only@1", replace(original, vjp_rule=None))
    with pytest.raises(ValueError, match="does not define a VJP"):
        registered_custom_vjp(
            "scpn.conformance:jvp_only@1", [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], registry=registry
        )


def test_singular_solve_refuses_and_regular_request_recovers() -> None:
    """No successful derivative is returned at a singular linear-system boundary."""
    registry = CustomDerivativeRegistry()
    identity = "scpn.conformance:solve_admission@1"
    registry.register(identity, program_ad_linalg_solve_derivative_rule((2, 2), (2,)))
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        registered_custom_jvp(
            identity, [1.0, 2.0, 2.0, 4.0, 1.0, 2.0], np.ones(6), registry=registry
        )
    np.testing.assert_allclose(
        registered_custom_jvp(
            identity,
            [1.0, 0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
            registry=registry,
        ),
        [3.0, 4.0],
        atol=1e-12,
    )
