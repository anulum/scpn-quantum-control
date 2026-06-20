# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD unary ufunc tests
"""Tests for Program AD unary ufunc adjoints and fail-closed boundaries."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_reciprocal_ufunc_matches_exact_adjoint() -> None:
    """Program AD reciprocal should preserve exact inverse derivatives and adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        reciprocal = np.reciprocal(values)
        return np.sum(reciprocal * np.array([1.0, -2.0, 3.0, -4.0])) + matrix[0, 1] ** -1

    values = np.array([2.0, -4.0, 0.5, -0.25], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("x2"), Parameter("x3")),
    )
    expected = np.array([-0.25, 0.0625, -12.0, 64.0], dtype=np.float64)

    assert result.value == pytest.approx(22.75)
    assert any(node.op == "reciprocal" for node in result.ir_nodes)
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_reciprocal_ufunc_fails_closed_at_zero() -> None:
    """Program AD reciprocal should reject singular inverse boundaries."""

    with pytest.raises(ValueError, match="reciprocal input must be non-zero"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reciprocal(values)),
            np.array([1.0, 0.0], dtype=np.float64),
        )


def test_program_ad_log1p_and_expm1_ufuncs_match_exact_adjoint() -> None:
    """Program AD should preserve stable log1p/expm1 derivatives and adjoints."""

    def objective(values: Any) -> object:
        transformed = np.log1p(values[:2]) + np.expm1(values[1:3])
        return np.sum(transformed * np.array([2.0, -3.0])) + np.log1p(values[2])

    values = np.array([0.5, -0.2, 1.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("x2")),
    )
    expected = np.array(
        [
            2.0 / (1.0 + values[0]),
            2.0 * math.exp(values[1]) - 3.0 / (1.0 + values[1]),
            -3.0 * math.exp(values[2]) + 1.0 / (1.0 + values[2]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "log1p" for node in result.ir_nodes)
    assert any(node.op == "expm1" for node in result.ir_nodes)
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_log1p_ufunc_fails_closed_at_domain_boundary() -> None:
    """Program AD log1p should reject inputs where the derivative is singular."""

    with pytest.raises(ValueError, match="log1p input must be greater than -1"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.log1p(values)),
            np.array([0.0, -1.0], dtype=np.float64),
        )


def test_program_ad_tan_ufunc_matches_exact_adjoint() -> None:
    """Program AD tangent should preserve exact trigonometric derivatives and adjoints."""

    def objective(values: Any) -> object:
        angles = values[:2]
        return np.sum(np.tan(angles) * np.array([2.0, -3.0])) + values[2] * np.tan(values[0])

    values = np.array([0.25, -0.4, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("theta0"), Parameter("theta1"), Parameter("gain")),
    )
    expected = np.array(
        [
            (2.0 + values[2]) / math.cos(values[0]) ** 2,
            -3.0 / math.cos(values[1]) ** 2,
            math.tan(values[0]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "tan" for node in result.ir_nodes)
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_tan_ufunc_fails_closed_at_singular_boundary() -> None:
    """Program AD tangent should reject singular cosine-zero boundaries."""

    with pytest.raises(ValueError, match="tan input must have non-zero cosine"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tan(values)),
            np.array([math.pi / 2.0], dtype=np.float64),
        )


def test_program_ad_arcsin_arccos_ufuncs_match_exact_adjoint() -> None:
    """Program AD inverse trig ufuncs should preserve exact branch-local adjoints."""

    def objective(values: Any) -> object:
        return (
            2.0 * np.arcsin(values[0])
            - 3.0 * np.arccos(values[1])
            + values[2] * np.arcsin(values[1])
        )

    values = np.array([0.25, -0.4, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("gain")),
    )
    expected = np.array(
        [
            2.0 / math.sqrt(1.0 - values[0] ** 2),
            (3.0 + values[2]) / math.sqrt(1.0 - values[1] ** 2),
            math.asin(values[1]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "arcsin" for node in result.ir_nodes)
    assert any(node.op == "arccos" for node in result.ir_nodes)
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_arcsin_arccos_fail_closed_at_domain_boundaries() -> None:
    """Program AD inverse trig ufuncs should reject singular or invalid domains."""

    def objective_for(ufunc: Any) -> Any:
        def objective(values: Any) -> object:
            return np.sum(ufunc(values))

        return objective

    for ufunc in (np.arcsin, np.arccos):
        with pytest.raises(ValueError, match="input must be strictly inside"):
            whole_program_value_and_grad(
                objective_for(ufunc),
                np.array([1.0], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="input must be strictly inside"):
            whole_program_value_and_grad(
                objective_for(ufunc),
                np.array([1.25], dtype=np.float64),
            )
