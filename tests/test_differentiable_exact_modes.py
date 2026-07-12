# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable exact modes tests
# scpn-quantum-control -- exact scalar AD mode tests
"""Tests for extracted exact forward- and reverse-mode gradient wrappers."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import (
    DualNumber,
    GradientResult,
    Parameter,
    ReverseNode,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    grad,
    reverse_cos,
    reverse_exp,
    reverse_log,
    reverse_sin,
)
from scpn_quantum_control.differentiable_exact_modes import (
    forward_mode_gradient,
    reverse_mode_gradient,
    value_and_forward_mode_grad,
    value_and_reverse_mode_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-14, atol: float = 1.0e-14
) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_facade_and_package_root_reuse_extracted_exact_mode_helpers() -> None:
    """Facade and package-root exports should point at the extracted wrappers."""

    assert differentiable.value_and_forward_mode_grad is value_and_forward_mode_grad
    assert differentiable.forward_mode_gradient is forward_mode_gradient
    assert differentiable.value_and_reverse_mode_grad is value_and_reverse_mode_grad
    assert differentiable.reverse_mode_gradient is reverse_mode_gradient
    assert scpn.value_and_forward_mode_grad is value_and_forward_mode_grad
    assert scpn.forward_mode_gradient is forward_mode_gradient
    assert scpn.value_and_reverse_mode_grad is value_and_reverse_mode_grad
    assert scpn.reverse_mode_gradient is reverse_mode_gradient


def test_forward_mode_dual_gradient_matches_analytic_derivative() -> None:
    """Forward-mode dual wrappers should propagate exact first-order tangents."""

    def objective(values: tuple[DualNumber, ...]) -> DualNumber:
        return dual_sin(values[0]) + values[0] * values[1] + values[1] ** 2

    result = value_and_forward_mode_grad(
        objective,
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("bias")],
    )

    assert isinstance(result, GradientResult)
    assert result.method == "forward_mode_dual"
    assert result.evaluations == 3
    assert result.parameter_names == ("theta", "bias")
    _assert_allclose(result.gradient, [math.cos(0.25) - 0.5, 0.25 - 1.0])
    _assert_allclose(
        forward_mode_gradient(lambda values: dual_exp(values[0]) + dual_log(values[0]), [2.0]),
        [math.exp(2.0) + 0.5],
    )
    _assert_allclose(
        grad(lambda values: dual_cos(values[0]), [0.25], method="forward_mode"),
        [-math.sin(0.25)],
    )


def test_forward_mode_dual_gradient_respects_frozen_parameters() -> None:
    """Forward-mode gradients should keep frozen tangent lanes zeroed."""

    result = value_and_forward_mode_grad(
        lambda values: values[0] ** 2 + values[0] * values[1],
        [3.0, 5.0],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.trainable == (True, False)
    assert result.evaluations == 2
    _assert_allclose(result.gradient, [11.0, 0.0])


def test_forward_mode_dual_rejects_invalid_contracts() -> None:
    """Forward-mode wrappers should fail closed on invalid scalar/domain contracts."""

    def non_scalar_objective(_values: tuple[DualNumber, ...]) -> Any:
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="forward-mode objective must return a scalar"):
        forward_mode_gradient(non_scalar_objective, [1.0])
    with pytest.raises(ValueError, match="dual log input must be positive"):
        forward_mode_gradient(lambda values: dual_log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="dual division denominator"):
        forward_mode_gradient(lambda values: values[0] / (values[0] - 1.0), [1.0])
    with pytest.raises(ValueError, match="dual variable exponent"):
        forward_mode_gradient(lambda values: (-1.0) ** values[0], [2.0])


def test_reverse_mode_tape_gradient_matches_analytic_derivative() -> None:
    """Reverse-mode tape wrappers should backpropagate exact adjoints."""

    def objective(values: tuple[ReverseNode, ...]) -> ReverseNode:
        return reverse_sin(values[0]) + values[0] * values[1] + values[1] ** 2

    result = value_and_reverse_mode_grad(
        objective,
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("bias")],
    )

    assert isinstance(result, GradientResult)
    assert result.method == "reverse_mode_tape"
    assert result.evaluations == 1
    assert result.parameter_names == ("theta", "bias")
    _assert_allclose(result.gradient, [math.cos(0.25) - 0.5, 0.25 - 1.0])
    _assert_allclose(
        reverse_mode_gradient(
            lambda values: reverse_exp(values[0]) + reverse_log(values[0]), [2.0]
        ),
        [math.exp(2.0) + 0.5],
    )
    _assert_allclose(
        grad(lambda values: reverse_cos(values[0]), [0.25], method="reverse_mode"),
        [-math.sin(0.25)],
    )


def test_reverse_mode_tape_gradient_respects_frozen_parameters() -> None:
    """Reverse-mode gradients should backpropagate once and mask frozen outputs."""

    result = value_and_reverse_mode_grad(
        lambda values: values[0] ** 2 + values[0] * values[1],
        [3.0, 5.0],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.trainable == (True, False)
    assert result.evaluations == 1
    _assert_allclose(result.gradient, [11.0, 0.0])


def test_reverse_mode_tape_rejects_invalid_contracts() -> None:
    """Reverse-mode wrappers should fail closed on invalid scalar/domain contracts."""

    def non_scalar_objective(_values: tuple[ReverseNode, ...]) -> Any:
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="reverse-mode objective must return a scalar"):
        reverse_mode_gradient(non_scalar_objective, [1.0])
    with pytest.raises(ValueError, match="reverse log input must be positive"):
        reverse_mode_gradient(lambda values: reverse_log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="reverse division denominator"):
        reverse_mode_gradient(lambda values: values[0] / (values[0] - 1.0), [1.0])
    with pytest.raises(ValueError, match="reverse variable exponent"):
        reverse_mode_gradient(lambda values: (-1.0) ** values[0], [2.0])
