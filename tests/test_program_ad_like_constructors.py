# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD like constructors tests
# scpn-quantum-control -- Program AD like-constructor tests
"""Tests for Program AD like-constructor constant semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(actual: object, expected: object, *, atol: float = 0.0) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, atol=atol)


def test_program_ad_like_constant_constructors_have_zero_derivatives() -> None:
    """Program AD like-constructors should create derivative-zero constant arrays."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        constants = np.zeros_like(values) + np.ones_like(values) + np.full_like(values, 2.0)
        matrix_constants = np.full_like(matrix, -0.5)
        return np.sum(values * constants) + np.sum(matrix_constants * matrix)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
        ),
    )

    assert result.value == pytest.approx(25.0)
    _assert_allclose(result.gradient, [2.5, 2.5, 2.5, 2.5], atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_like_constant_constructors_reject_shape_overrides() -> None:
    """Program AD like-constructors should fail closed on unsupported shape overrides."""

    with pytest.raises(ValueError, match="shape overrides"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.zeros_like)(values, shape=(2, 2))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
