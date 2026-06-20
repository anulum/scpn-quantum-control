# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- scalar differentiable kernel tests
"""Tests for scalar forward- and reverse-mode differentiable kernels."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import differentiable
from scpn_quantum_control.differentiable_scalar_kernels import (
    DualNumber,
    ReverseNode,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    reverse_cos,
    reverse_exp,
    reverse_log,
    reverse_sin,
)


def _assert_dual(actual: DualNumber, expected_primal: float, expected_tangent: float) -> None:
    """Assert a forward-mode scalar value and tangent."""

    assert actual.primal == pytest.approx(expected_primal)
    assert actual.tangent == pytest.approx(expected_tangent)


def _assert_parent(
    node: ReverseNode,
    index: int,
    parent: ReverseNode,
    coefficient: float,
) -> None:
    """Assert one reverse-mode parent edge."""

    actual_parent, actual_coefficient = node.parents[index]
    assert actual_parent is parent
    assert actual_coefficient == pytest.approx(coefficient)


def test_scalar_kernels_keep_facade_and_package_root_identities() -> None:
    """Direct, facade, and package-root imports should resolve to one object."""

    assert differentiable.DualNumber is DualNumber
    assert differentiable.ReverseNode is ReverseNode
    assert differentiable.dual_sin is dual_sin
    assert differentiable.dual_cos is dual_cos
    assert differentiable.dual_exp is dual_exp
    assert differentiable.dual_log is dual_log
    assert differentiable.reverse_sin is reverse_sin
    assert differentiable.reverse_cos is reverse_cos
    assert differentiable.reverse_exp is reverse_exp
    assert differentiable.reverse_log is reverse_log
    assert scpn.DualNumber is DualNumber
    assert scpn.ReverseNode is ReverseNode
    assert scpn.dual_sin is dual_sin
    assert scpn.reverse_sin is reverse_sin
    assert "DualNumber" in differentiable.__all__
    assert "ReverseNode" in differentiable.__all__


def test_dual_number_arithmetic_and_elementary_primitives() -> None:
    """Forward-mode kernels should propagate scalar tangent rules exactly."""

    x = DualNumber(2.0, 3.0)
    y = DualNumber(4.0, -1.0)

    assert DualNumber.coerce(x) is x
    _assert_dual(DualNumber.coerce(5), 5.0, 0.0)
    _assert_dual(x + y, 6.0, 2.0)
    _assert_dual(10.0 + x, 12.0, 3.0)
    _assert_dual(x - y, -2.0, 4.0)
    _assert_dual(10.0 - x, 8.0, -3.0)
    _assert_dual(x * y, 8.0, 10.0)
    _assert_dual(2.0 * x, 4.0, 6.0)
    _assert_dual(x / y, 0.5, 0.875)
    _assert_dual(8.0 / x, 4.0, -6.0)
    _assert_dual(-x, -2.0, -3.0)
    _assert_dual(x**3.0, 8.0, 36.0)

    variable_power = DualNumber(2.0, 3.0) ** DualNumber(4.0, 0.5)
    _assert_dual(variable_power, 16.0, 16.0 * (0.5 * math.log(2.0) + 6.0))
    reflected_power = 2.0 ** DualNumber(3.0, 0.5)
    _assert_dual(reflected_power, 8.0, 8.0 * 0.5 * math.log(2.0))

    _assert_dual(dual_sin(x), math.sin(2.0), math.cos(2.0) * 3.0)
    _assert_dual(dual_cos(x), math.cos(2.0), -math.sin(2.0) * 3.0)
    _assert_dual(dual_exp(x), math.exp(2.0), math.exp(2.0) * 3.0)
    _assert_dual(dual_log(x), math.log(2.0), 1.5)


def test_dual_number_fail_closed_boundaries() -> None:
    """Forward-mode kernels should reject non-real and singular inputs."""

    with pytest.raises(ValueError, match="dual primal"):
        DualNumber(True)
    with pytest.raises(ValueError, match="dual tangent"):
        DualNumber(1.0, cast(Any, "bad"))
    with pytest.raises(ValueError, match="dual operand"):
        DualNumber.coerce(complex(1.0, 1.0))
    with pytest.raises(ValueError, match="denominator"):
        DualNumber(1.0, 1.0) / 0.0
    with pytest.raises(ValueError, match="positive base"):
        DualNumber(-2.0, 1.0) ** DualNumber(3.0, 1.0)
    with pytest.raises(ValueError, match="dual log"):
        dual_log(DualNumber(0.0, 1.0))


def test_reverse_node_arithmetic_and_elementary_pullbacks() -> None:
    """Reverse-mode kernels should expose local pullback coefficients."""

    x = ReverseNode(2.0)
    y = ReverseNode(4.0)

    assert ReverseNode.coerce(x) is x
    constant = ReverseNode.coerce(5)
    assert constant.primal == pytest.approx(5.0)
    assert constant.parents == ()
    assert constant.adjoint == pytest.approx(0.0)

    addition = x + y
    assert addition.primal == pytest.approx(6.0)
    _assert_parent(addition, 0, x, 1.0)
    _assert_parent(addition, 1, y, 1.0)
    reflected_addition = 10.0 + x
    assert reflected_addition.primal == pytest.approx(12.0)
    _assert_parent(reflected_addition, 0, x, 1.0)

    subtraction = x - y
    assert subtraction.primal == pytest.approx(-2.0)
    _assert_parent(subtraction, 0, x, 1.0)
    _assert_parent(subtraction, 1, y, -1.0)
    reflected_subtraction = 10.0 - x
    assert reflected_subtraction.primal == pytest.approx(8.0)
    _assert_parent(reflected_subtraction, 1, x, -1.0)

    product = x * y
    assert product.primal == pytest.approx(8.0)
    _assert_parent(product, 0, x, 4.0)
    _assert_parent(product, 1, y, 2.0)
    reflected_product = 3.0 * x
    assert reflected_product.primal == pytest.approx(6.0)
    _assert_parent(reflected_product, 0, x, 3.0)

    quotient = x / y
    assert quotient.primal == pytest.approx(0.5)
    _assert_parent(quotient, 0, x, 0.25)
    _assert_parent(quotient, 1, y, -0.125)
    reflected_quotient = 8.0 / x
    assert reflected_quotient.primal == pytest.approx(4.0)
    _assert_parent(reflected_quotient, 1, x, -2.0)

    negated = -x
    assert negated.primal == pytest.approx(-2.0)
    _assert_parent(negated, 0, x, -1.0)

    constant_power = x**3.0
    assert constant_power.primal == pytest.approx(8.0)
    _assert_parent(constant_power, 0, x, 12.0)

    exponent = reverse_exp(ReverseNode(1.0))
    variable_power = x**exponent
    assert variable_power.primal == pytest.approx(2.0**math.e)
    _assert_parent(variable_power, 0, x, math.e * 2.0 ** (math.e - 1.0))
    _assert_parent(variable_power, 1, exponent, (2.0**math.e) * math.log(2.0))

    reflected_power = 2.0**x
    assert reflected_power.primal == pytest.approx(4.0)
    _assert_parent(reflected_power, 1, x, 4.0 * math.log(2.0))

    sine = reverse_sin(x)
    assert sine.primal == pytest.approx(math.sin(2.0))
    _assert_parent(sine, 0, x, math.cos(2.0))
    cosine = reverse_cos(x)
    assert cosine.primal == pytest.approx(math.cos(2.0))
    _assert_parent(cosine, 0, x, -math.sin(2.0))
    exponential = reverse_exp(x)
    assert exponential.primal == pytest.approx(math.exp(2.0))
    _assert_parent(exponential, 0, x, math.exp(2.0))
    logarithm = reverse_log(x)
    assert logarithm.primal == pytest.approx(math.log(2.0))
    _assert_parent(logarithm, 0, x, 0.5)


def test_reverse_node_fail_closed_boundaries() -> None:
    """Reverse-mode kernels should reject non-real and singular inputs."""

    with pytest.raises(ValueError, match="reverse primal"):
        ReverseNode(True)
    with pytest.raises(ValueError, match="reverse operand"):
        ReverseNode.coerce(complex(1.0, 1.0))
    with pytest.raises(ValueError, match="denominator"):
        ReverseNode(1.0) / 0.0
    with pytest.raises(ValueError, match="positive base"):
        ReverseNode(-2.0) ** ReverseNode(3.0)
    with pytest.raises(ValueError, match="reverse log"):
        reverse_log(ReverseNode(0.0))
