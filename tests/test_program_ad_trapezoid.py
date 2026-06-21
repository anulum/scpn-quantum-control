# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD trapezoid tests
"""Tests for Program AD trapezoidal integration primitive semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_contract_for,
    program_ad_reduction_trapezoid_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _manual_trapezoid_axis1(
    values: NDArray[np.float64], x_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a manual axis-1 trapezoid integral for coverage-instrumented NumPy."""

    widths = np.diff(x_grid)
    return cast(
        NDArray[np.float64],
        np.sum(0.5 * widths * (values[:, :-1] + values[:, 1:]), axis=1),
    )


def test_program_ad_trapezoid_direct_rule_is_exposed_from_extracted_module() -> None:
    """The facade and extracted trapezoid module should expose the same factory."""

    from scpn_quantum_control.program_ad_trapezoid_primitives import (
        program_ad_reduction_trapezoid_derivative_rule as module_trapezoid_rule,
    )

    assert module_trapezoid_rule is program_ad_reduction_trapezoid_derivative_rule


def test_program_ad_trapezoid_reduction_contract_is_registry_gated() -> None:
    """Trapezoid integration should expose static-grid reduction registry contracts."""

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    x_grid = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    contract = primitive_contract_for("scpn.program_ad.reduction:trapezoid")

    assert contract.identity == PrimitiveIdentity("scpn.program_ad.reduction", "trapezoid", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.reduction.trapezoid"
    assert contract.lowering_metadata["static_derivative_factory"] == (
        "program_ad_reduction_trapezoid_derivative_rule"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((matrix, x_grid, 1.0, 1)) == (2,)
    assert contract.shape_rule((matrix.reshape(-1), None, 0.25, 0)) == ()
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((matrix, x_grid, 1.0, 1)) == "float64"
    assert contract.static_argument_rule is not None
    static_arguments = contract.static_argument_rule((matrix, x_grid, 1.0, 1))
    assert static_arguments == (("x", (3,), (0.0, 0.25, 1.0)), 1.0, 1)


def test_program_ad_trapezoid_static_derivative_factory_is_axis_aware() -> None:
    """Static trapezoid factories should expose exact axis-aware JVP and VJP rules."""

    matrix = np.array([[1.0, 2.0, 4.0], [0.5, -1.5, 3.0]], dtype=np.float64)
    tangent = np.array([[0.25, -0.5, 1.0], [1.5, -0.75, 0.5]], dtype=np.float64)
    x_grid = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    cotangent = np.array([1.5, -2.0], dtype=np.float64)
    rule = program_ad_reduction_trapezoid_derivative_rule((2, 3), x=x_grid, axis=1)

    assert rule.name == "program_ad_reduction_trapezoid_2x3_axis_1_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    _assert_allclose(
        rule.value_fn(matrix.reshape(-1)),
        _manual_trapezoid_axis1(matrix, x_grid),
    )
    _assert_allclose(
        jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        _manual_trapezoid_axis1(tangent, x_grid),
    )
    base_weights = np.array([0.125, 0.5, 0.375], dtype=np.float64)
    expected_vjp = np.vstack((cotangent[0] * base_weights, cotangent[1] * base_weights))
    _assert_allclose(vjp_rule(matrix.reshape(-1), cotangent), expected_vjp.reshape(-1))


def test_program_ad_trapezoid_extracted_module_covers_flat_and_grid_boundaries() -> None:
    """Extracted trapezoid helpers should cover flat rules and static-grid validation."""

    from scpn_quantum_control import program_ad_trapezoid_primitives as trapezoid

    values = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    tangent = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    cotangent = np.array([2.0], dtype=np.float64)
    _assert_allclose(trapezoid._program_ad_reduction_trapezoid_value(values), [4.5])
    _assert_allclose(trapezoid._program_ad_reduction_trapezoid_jvp(values, tangent), [0.25])
    _assert_allclose(
        trapezoid._program_ad_reduction_trapezoid_vjp(values, cotangent),
        [1.0, 2.0, 1.0],
    )

    matrix = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    full_grid = np.array([[0.0, 0.5], [0.0, 2.0]], dtype=np.float64)
    rule = program_ad_reduction_trapezoid_derivative_rule((2, 2), x=full_grid, axis=1)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    _assert_allclose(rule.value_fn(matrix.reshape(-1)), [0.75, 8.0])
    _assert_allclose(rule.jvp_rule(matrix.reshape(-1), np.ones(4, dtype=np.float64)), [0.5, 2.0])
    _assert_allclose(
        rule.vjp_rule(matrix.reshape(-1), np.array([2.0, -1.0])), [0.5, 0.5, -1.0, -1.0]
    )

    flat_rule = program_ad_reduction_trapezoid_derivative_rule((3,), dx=0.25, axis=0)
    assert flat_rule.jvp_rule is not None
    assert flat_rule.vjp_rule is not None
    _assert_allclose(flat_rule.value_fn(values), [1.125])
    _assert_allclose(flat_rule.jvp_rule(values, tangent), [0.0625])
    _assert_allclose(flat_rule.vjp_rule(values, cotangent), [0.25, 0.5, 0.25])

    with pytest.raises(ValueError, match="requires at least one value"):
        trapezoid._program_ad_reduction_trapezoid_value(np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="at least two values"):
        trapezoid._program_ad_reduction_trapezoid_flat_weights(1)
    with pytest.raises(ValueError, match="tangent shape must match"):
        trapezoid._program_ad_reduction_trapezoid_jvp(values, values[:2])
    with pytest.raises(ValueError, match="one scalar cotangent"):
        trapezoid._program_ad_reduction_trapezoid_vjp(values, values[:2])
    with pytest.raises(ValueError, match="non-negative dimensions"):
        program_ad_reduction_trapezoid_derivative_rule((2, -1))
    with pytest.raises(ValueError, match="at least one value"):
        program_ad_reduction_trapezoid_derivative_rule((0,))
    with pytest.raises(ValueError, match="axis must be a static integer"):
        program_ad_reduction_trapezoid_derivative_rule((2, 2), axis=True)
    with pytest.raises(ValueError, match="axis out of bounds"):
        program_ad_reduction_trapezoid_derivative_rule((2, 2), axis=2)
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        program_ad_reduction_trapezoid_derivative_rule((), axis=0)
    with pytest.raises(ValueError, match="requires values with 4 values"):
        rule.value_fn(values)
    with pytest.raises(ValueError, match="requires cotangent with 2 values"):
        rule.vjp_rule(matrix.reshape(-1), values)


def test_program_ad_trapezoid_static_grid_fail_closed_branches() -> None:
    """Static-grid helpers should reject traced, malformed, and ambiguous grids."""

    class TraceADArray:
        pass

    with pytest.raises(ValueError, match="at least two samples"):
        program_ad_reduction_trapezoid_derivative_rule((2, 1), axis=1)
    with pytest.raises(ValueError, match="grid x must be static"):
        program_ad_reduction_trapezoid_derivative_rule((2, 2), x=TraceADArray(), axis=1)
    with pytest.raises(ValueError, match="either x or dx"):
        program_ad_reduction_trapezoid_derivative_rule((2, 2), x=np.array([0.0, 1.0]), dx=0.25)
    with pytest.raises(ValueError, match="only finite values"):
        program_ad_reduction_trapezoid_derivative_rule((2, 2), x=np.array([0.0, np.inf]), axis=1)
    with pytest.raises(ValueError, match="match the integration axis"):
        program_ad_reduction_trapezoid_derivative_rule((2, 3), x=np.array([0.0, 1.0]), axis=1)
    with pytest.raises(ValueError, match="match the integration axis or full array shape"):
        program_ad_reduction_trapezoid_derivative_rule(
            (2, 2), x=np.zeros((2, 2, 1), dtype=np.float64), axis=1
        )


def test_program_ad_trapezoid_matches_static_grid_adjoint() -> None:
    """Program AD trapezoidal integration should apply exact static-grid adjoints."""

    x_grid = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    row_weights = np.array([2.0, -1.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        row_integrals = np.trapezoid(matrix, x=x_grid, axis=1)
        flat_integral = np.trapezoid(values, dx=0.25)
        return np.sum(row_integrals * row_weights) + 0.5 * flat_integral

    values = np.array([1.0, 2.0, 4.0, 0.5, -1.5, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    row_base_weights = np.array([0.125, 0.5, 0.375], dtype=np.float64)
    expected = np.zeros_like(values)
    expected[:3] += row_weights[0] * row_base_weights
    expected[3:] += row_weights[1] * row_base_weights
    expected += 0.5 * np.array([0.125, 0.25, 0.25, 0.25, 0.25, 0.125])

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_trapezoid_fails_closed_invalid_static_contracts() -> None:
    """Program AD trapezoidal integration should reject unsupported grid contracts."""

    with pytest.raises(ValueError, match="axis must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.trapezoid(np.reshape(values, (2, 2)), axis=True),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.trapezoid(np.reshape(values, (2, 2)), axis=2),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="grid x must be static"):
        whole_program_value_and_grad(
            lambda values: np.trapezoid(values, x=values),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="x must match the integration axis"):
        whole_program_value_and_grad(
            lambda values: np.trapezoid(values, x=np.array([0.0, 1.0])),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="dx must be finite"):
        whole_program_value_and_grad(
            lambda values: np.trapezoid(values, dx=np.inf),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
