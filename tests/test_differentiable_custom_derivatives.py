# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- custom differentiable rule tests
"""Tests for exact custom derivative rules and downstream custom solvers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import differentiable as differentiable_module
from scpn_quantum_control import differentiable_custom_derivatives as custom_derivative_module
from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    JacobianResult,
    JVPResult,
    LevenbergMarquardtStep,
    NaturalGradientResult,
    Parameter,
    VJPResult,
    batch_custom_jacobian,
    batch_custom_jvp,
    batch_custom_vjp,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    custom_gauss_newton_gradient,
    custom_jacobian,
    custom_jvp,
    custom_levenberg_marquardt_step,
    custom_vjp,
    value_and_custom_jacobian,
    value_and_custom_jvp,
    value_and_custom_vjp,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_facade_and_package_root_reuse_extracted_custom_derivative_helpers() -> None:
    """Facade and package-root exports should point at extracted custom helpers."""

    helper_names = (
        "batch_custom_jacobian",
        "batch_custom_jvp",
        "batch_custom_vjp",
        "batch_value_and_custom_jacobian",
        "batch_value_and_custom_jvp",
        "batch_value_and_custom_vjp",
        "custom_jacobian",
        "custom_jvp",
        "custom_vjp",
        "value_and_custom_jacobian",
        "value_and_custom_jvp",
        "value_and_custom_vjp",
    )

    for name in helper_names:
        helper = getattr(custom_derivative_module, name)
        assert getattr(differentiable_module, name) is helper
        assert getattr(scpn, name) is helper


def test_custom_derivative_rule_evaluates_exact_jvp_and_vjp() -> None:
    """Custom rules should expose exact primitive derivatives with provenance."""

    rule = CustomDerivativeRule(
        name="quadratic_coupler",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[1] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [
                cotangent[0] * values[1] + 2.0 * cotangent[1] * values[0],
                cotangent[0] * values[0],
            ]
        ),
        parameter_names=("theta", "phi"),
        trainable=(True, False),
    )

    jvp_result = value_and_custom_jvp(rule, [2.0, 3.0], [0.5, 7.0])
    vjp_result = value_and_custom_vjp(rule, [2.0, 3.0], [11.0, 13.0])

    _assert_allclose(custom_jvp(rule, [2.0, 3.0], [0.5, 7.0]), [1.5, 2.0])
    _assert_allclose(jvp_result.value, [6.0, 4.0])
    _assert_allclose(jvp_result.tangent, [0.5, 0.0])
    _assert_allclose(jvp_result.jvp, [1.5, 2.0])
    assert jvp_result.method == "custom_jvp:quadratic_coupler"
    assert jvp_result.step == pytest.approx(0.0)
    assert jvp_result.parameter_names == ("theta", "phi")
    assert jvp_result.trainable == (True, False)
    _assert_allclose(custom_vjp(rule, [2.0, 3.0], [11.0, 13.0]).vjp, [85.0, 0.0])
    _assert_allclose(vjp_result.value, [6.0, 4.0])
    _assert_allclose(vjp_result.vjp, [85.0, 0.0])
    assert vjp_result.method == "custom_vjp:quadratic_coupler"


def test_custom_derivative_rule_accepts_explicit_parameter_metadata() -> None:
    """Explicit parameters should override rule-local metadata for exact transforms."""

    rule = CustomDerivativeRule(
        name="identity_pair",
        value_fn=lambda values: values,
        jvp_rule=lambda values, tangent: tangent,
    )

    result = value_and_custom_jvp(
        rule,
        [1.0, 2.0],
        [3.0, 4.0],
        parameters=[Parameter("external_theta"), Parameter("external_phi", trainable=False)],
    )

    assert result.parameter_names == ("external_theta", "external_phi")
    assert result.trainable == (True, False)
    _assert_allclose(result.tangent, [3.0, 0.0])


def test_custom_derivative_rule_rejects_invalid_contracts() -> None:
    """Custom derivative rules must fail closed on bad exact-rule contracts."""

    with pytest.raises(ValueError, match="requires a JVP or VJP"):
        CustomDerivativeRule(name="bad", value_fn=lambda values: values)
    with pytest.raises(ValueError, match="lengths must match"):
        CustomDerivativeRule(
            name="bad_meta",
            value_fn=lambda values: values,
            jvp_rule=lambda values, tangent: tangent,
            parameter_names=("x", "y"),
            trainable=(True,),
        )

    rule = CustomDerivativeRule(
        name="bad_shapes",
        value_fn=lambda values: np.array([values[0]]),
        jvp_rule=lambda values, tangent: np.array([1.0, 2.0]),
        vjp_rule=lambda values, cotangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="JVP output shape"):
        value_and_custom_jvp(rule, [1.0], [1.0])
    with pytest.raises(ValueError, match="custom JVP requires a CustomDerivativeRule"):
        value_and_custom_jvp(cast(CustomDerivativeRule, object()), [1.0], [1.0])
    with pytest.raises(ValueError, match="custom JVP tangent length"):
        value_and_custom_jvp(rule, [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="cotangent shape"):
        value_and_custom_vjp(rule, [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="VJP output length"):
        value_and_custom_vjp(rule, [1.0], [1.0])
    with pytest.raises(ValueError, match="custom VJP requires a CustomDerivativeRule"):
        value_and_custom_vjp(cast(CustomDerivativeRule, object()), [1.0], [1.0])
    with pytest.raises(ValueError, match="does not define a JVP"):
        value_and_custom_jvp(
            CustomDerivativeRule(
                name="vjp_only",
                value_fn=lambda values: values,
                vjp_rule=lambda values, cotangent: cotangent,
            ),
            [1.0],
            [1.0],
        )
    with pytest.raises(ValueError, match="does not define a VJP"):
        value_and_custom_vjp(
            CustomDerivativeRule(
                name="jvp_only",
                value_fn=lambda values: values,
                jvp_rule=lambda values, tangent: tangent,
            ),
            [1.0],
            [1.0],
        )


def test_custom_derivative_rule_rejects_corrupted_parameter_metadata() -> None:
    """Exact transforms should still fail closed if frozen rule metadata is corrupted."""

    bad_names = CustomDerivativeRule(
        name="bad_names",
        value_fn=lambda values: values,
        jvp_rule=lambda values, tangent: tangent,
        parameter_names=("theta", "phi"),
    )
    with pytest.raises(ValueError, match="parameter_names length"):
        value_and_custom_jvp(bad_names, [1.0], [1.0])

    bad_trainable = CustomDerivativeRule(
        name="bad_trainable",
        value_fn=lambda values: values,
        jvp_rule=lambda values, tangent: tangent,
        parameter_names=("theta", "phi"),
    )
    object.__setattr__(bad_trainable, "trainable", (True,))
    with pytest.raises(ValueError, match="trainable mask length"):
        value_and_custom_jvp(bad_trainable, [1.0, 2.0], [1.0, 1.0])


def test_custom_jacobian_materializes_exact_jvp_columns() -> None:
    """Custom JVP rules should materialise exact dense Jacobian columns."""

    rule = CustomDerivativeRule(
        name="quadratic_vector",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                values[1] * tangent[0] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    result = value_and_custom_jacobian(rule, [2.0, 3.0])

    assert isinstance(result, JacobianResult)
    assert result.method == "custom_jacobian:quadratic_vector"
    assert result.step == pytest.approx(0.0)
    assert result.evaluations == 1
    assert result.parameter_names == ("theta", "frozen_phi")
    assert result.trainable == (True, False)
    _assert_allclose(result.value, [6.0, 4.0])
    _assert_allclose(result.jacobian, [[3.0, 0.0], [4.0, 0.0]])
    _assert_allclose(custom_jacobian(rule, [2.0, 3.0]), result.jacobian)


def test_custom_jacobian_materializes_exact_vjp_rows() -> None:
    """VJP-only rules should materialise exact dense Jacobian rows."""

    rule = CustomDerivativeRule(
        name="linear_readout",
        value_fn=lambda values: np.array([values[0] + 2.0 * values[1], -values[0]]),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0] - cotangent[1], 2.0 * cotangent[0]]
        ),
    )

    result = value_and_custom_jacobian(rule, [0.25, -0.5])

    _assert_allclose(result.value, [-0.75, -0.25])
    _assert_allclose(result.jacobian, [[1.0, 2.0], [-1.0, 0.0]])
    assert result.method == "custom_jacobian:linear_readout"


def test_custom_jacobian_rejects_invalid_exact_rule_shapes() -> None:
    """Custom Jacobian materialisation must reject malformed exact derivatives."""

    invalid_rule = cast(CustomDerivativeRule, object())
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        value_and_custom_jacobian(invalid_rule, [1.0])
    no_rule = CustomDerivativeRule(
        name="corrupted_empty",
        value_fn=lambda values: values,
        jvp_rule=lambda values, tangent: tangent,
    )
    object.__setattr__(no_rule, "jvp_rule", None)
    object.__setattr__(no_rule, "vjp_rule", None)
    with pytest.raises(ValueError, match="requires a JVP or VJP"):
        value_and_custom_jacobian(no_rule, [1.0])

    bad_jvp = CustomDerivativeRule(
        name="bad_jvp",
        value_fn=lambda values: np.array([values[0]]),
        jvp_rule=lambda values, tangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="JVP output shape"):
        value_and_custom_jacobian(bad_jvp, [1.0])

    bad_vjp = CustomDerivativeRule(
        name="bad_vjp",
        value_fn=lambda values: np.array([values[0]]),
        vjp_rule=lambda values, cotangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="VJP output length"):
        value_and_custom_jacobian(bad_vjp, [1.0])


def test_batched_custom_jvp_and_vjp_apply_exact_rules() -> None:
    """Batched custom transforms should preserve exact rule outputs and metadata."""

    rule = CustomDerivativeRule(
        name="affine_pair",
        value_fn=lambda values: np.array([values[0] + values[1], values[0] - values[1]]),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + tangent[1], tangent[0] - tangent[1]]
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0] + cotangent[1], cotangent[0] - cotangent[1]]
        ),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    jvp_results = cast(
        list[JVPResult],
        batch_value_and_custom_jvp(
            rule,
            [2.0, 5.0],
            [[1.0, 10.0], [-2.0, 30.0]],
        ),
    )
    vjp_results = cast(
        list[VJPResult],
        batch_value_and_custom_vjp(
            rule,
            [2.0, 5.0],
            [[3.0, 4.0], [-1.0, 2.0]],
        ),
    )

    assert len(jvp_results) == 2
    assert len(vjp_results) == 2
    _assert_allclose(
        batch_custom_jvp(rule, [2.0, 5.0], [[1.0, 10.0], [-2.0, 30.0]]),
        [[1.0, 1.0], [-2.0, -2.0]],
    )
    _assert_allclose(
        batch_custom_vjp(rule, [2.0, 5.0], [[3.0, 4.0], [-1.0, 2.0]]),
        [[7.0, 0.0], [1.0, 0.0]],
    )
    assert jvp_results[0].parameter_names == ("theta", "frozen_phi")
    assert vjp_results[0].trainable == (True, False)


def test_batched_custom_jacobian_materializes_parameter_batches() -> None:
    """Custom Jacobians should batch over parameter rows for benchmark workflows."""

    rule = CustomDerivativeRule(
        name="quadratic_vector",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                values[1] * tangent[0] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
    )

    results = cast(
        list[JacobianResult],
        batch_value_and_custom_jacobian(rule, [[2.0, 3.0], [-1.0, 4.0]]),
    )

    assert len(results) == 2
    _assert_allclose(
        batch_custom_jacobian(rule, [[2.0, 3.0], [-1.0, 4.0]]),
        [
            [[3.0, 2.0], [4.0, 0.0]],
            [[4.0, -1.0], [-2.0, 0.0]],
        ],
    )
    _assert_allclose(results[1].value, [-4.0, 1.0])
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_value_and_custom_jacobian(rule, [1.0, 2.0])
    with pytest.raises(ValueError, match="finite values"):
        batch_value_and_custom_jacobian(rule, [[1.0, np.nan]])


def test_custom_gauss_newton_gradient_uses_exact_custom_jacobian() -> None:
    """Exact custom residual Jacobians should feed Gauss-Newton directly."""

    rule = CustomDerivativeRule(
        name="scaled_residual",
        value_fn=lambda values: np.array([2.0 * values[0] - 1.0, values[1] + 3.0]),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0], tangent[1]]),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    result = custom_gauss_newton_gradient(
        rule,
        [2.0, -1.0],
        weights=[1.0, 4.0],
        damping=1.0,
    )

    assert isinstance(result, NaturalGradientResult)
    assert result.base_gradient.method == "gauss_newton:custom_jacobian:scaled_residual"
    assert result.base_gradient.parameter_names == ("theta", "frozen_phi")
    assert result.base_gradient.trainable == (True, False)
    assert result.base_gradient.value == pytest.approx(12.5)
    _assert_allclose(result.base_gradient.gradient, [6.0, 0.0])
    _assert_allclose(result.metric, [[5.0, 0.0], [0.0, 1.0]])
    _assert_allclose(result.natural_gradient, [1.2, 0.0])


def test_custom_levenberg_marquardt_step_uses_exact_custom_jacobian() -> None:
    """Exact custom residual Jacobians should feed bounded LM candidates."""

    rule = CustomDerivativeRule(
        name="identity_residual",
        value_fn=lambda values: np.array([values[0], 2.0 * values[1]]),
        jvp_rule=lambda values, tangent: np.array([tangent[0], 2.0 * tangent[1]]),
    )

    result = custom_levenberg_marquardt_step(
        rule,
        [2.0, -1.0],
        damping=1.0,
        max_step_norm=1.0,
    )

    assert isinstance(result, LevenbergMarquardtStep)
    assert (
        result.gauss_newton.base_gradient.method
        == "gauss_newton:custom_jacobian:identity_residual"
    )
    _assert_allclose(result.gauss_newton.base_gradient.gradient, [2.0, -4.0])
    assert np.linalg.norm(result.step) == pytest.approx(1.0)
    _assert_allclose(result.candidate_values, np.array([2.0, -1.0]) + result.step)
    assert result.predicted_reduction > 0.0
