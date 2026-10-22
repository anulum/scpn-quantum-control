# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Supplied numerical API algorithm contracts
"""Exercise actual supplied AD providers through public numerical kernels."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    value_and_grad,
    value_and_hessian,
    value_and_jacobian,
)
from scpn_quantum_control.differentiable_api_contracts import UnifiedDifferentiableAPIResult
from scpn_quantum_control.differentiable_api_core_algorithms import (
    differentiable_gradient,
    differentiable_hessian,
    differentiable_jacobian,
    differentiable_value,
)
from scpn_quantum_control.differentiable_api_core_bindings import DifferentiableValueBindings


@pytest.mark.parametrize("operation", ["value", "gradient", "jacobian", "hessian"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_supplied_real_AD_providers_match_coupled_and_rectangular_references(
    operation: str, width: int
) -> None:
    """Qualify numerical kernels with real AD providers and independent derivatives.

    Parameters
    ----------
    operation
        Complete scalar or rectangular-vector operation under test.
    width
        Parameter dimension; the vector output has one extra component.

    """
    values = np.linspace(-0.4, 0.3, width, dtype=np.float64)
    original = values.copy()
    matrix = np.arange(1, (width + 1) * width + 1, dtype=np.float64).reshape(width + 1, width) / (
        (width + 1) * width
    )
    coefficients = np.linspace(0.1, 0.3, width + 1)

    def scalar(x: NDArray[np.float64]) -> float:
        """Evaluate coupled polynomial curvature through the real numerical provider."""
        return float(0.25 * np.sum(x**4) + 0.2 * np.sum(x**2) + 0.5 * np.sum(x) ** 2)

    def vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a rectangular vector objective with nonseparable derivatives."""
        return matrix @ np.sin(x) + coefficients * np.sum(x**2)

    bindings = DifferentiableValueBindings(
        read_UnifiedDifferentiableAPIResult=lambda: UnifiedDifferentiableAPIResult,
        read_value_and_grad=lambda: value_and_grad,
        read_value_and_hessian=lambda: value_and_hessian,
        read_value_and_jacobian=lambda: value_and_jacobian,
        read_float64=lambda: np.float64,
        read_float=lambda: float,
        read_list=lambda: list,
        read_error=lambda: ValueError,
    )
    if operation == "jacobian":
        result = differentiable_jacobian(vector, values, step=1e-6, _bindings=bindings)
        assert result.jacobian is not None
        np.testing.assert_allclose(
            result.jacobian,
            matrix * np.cos(values)[None, :] + np.outer(coefficients, 2.0 * values),
            atol=1e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            result.payload["objective_value"], vector(values), atol=1e-12, rtol=0
        )
    elif operation == "hessian":
        result = differentiable_hessian(scalar, values, step=3e-4, _bindings=bindings)
        assert result.hessian is not None
        np.testing.assert_allclose(
            result.hessian,
            np.ones((width, width)) + np.diag(3.0 * values**2 + 0.4),
            atol=3e-7,
            rtol=0,
        )
    elif operation == "gradient":
        result = differentiable_gradient(
            scalar, values, method="finite_difference", step=1e-6, _bindings=bindings
        )
        assert result.gradient is not None
        np.testing.assert_allclose(
            result.gradient, values**3 + 0.4 * values + np.sum(values), atol=1e-8, rtol=0
        )
    else:
        result = differentiable_value(
            scalar, values, method="finite_difference", _bindings=bindings
        )
        assert result.value == pytest.approx(scalar(values), abs=1e-12)
        assert result.gradient is None
    assert type(result) is UnifiedDifferentiableAPIResult
    assert len(result.payload["parameter_names"]) == width
    assert result.payload["trainable"] == [True] * width
    np.testing.assert_array_equal(values, original)
