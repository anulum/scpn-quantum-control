# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD linalg direct rules tests
# scpn-quantum-control -- Program AD linalg direct derivative rule tests
"""Tests for Program AD linear-algebra direct derivative rules."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.program_ad_linalg_primitives as linalg_primitives
from scpn_quantum_control.differentiable import (
    PrimitiveIdentity,
    custom_derivative_rule_for,
    program_ad_linalg_eig_derivative_rule,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_ad_linalg_solve_derivative_rule,
    program_ad_linalg_trace_derivative_rule,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_linalg_primitive_derivative_rules_are_direct_kernels() -> None:
    """Feasible linalg primitive contracts should expose direct derivative kernels."""

    assert (
        program_ad_linalg_matrix_power_derivative_rule
        is linalg_primitives.program_ad_linalg_matrix_power_derivative_rule
    )
    assert (
        program_ad_linalg_multi_dot_derivative_rule
        is linalg_primitives.program_ad_linalg_multi_dot_derivative_rule
    )
    assert (
        program_ad_linalg_solve_derivative_rule
        is linalg_primitives.program_ad_linalg_solve_derivative_rule
    )
    assert (
        program_ad_linalg_trace_derivative_rule
        is linalg_primitives.program_ad_linalg_trace_derivative_rule
    )
    assert (
        program_ad_linalg_eig_derivative_rule
        is linalg_primitives.program_ad_linalg_eig_derivative_rule
    )

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    tangent_matrix = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    rhs = np.array([0.25, -1.0], dtype=np.float64)
    tangent_rhs = np.array([0.5, -0.25], dtype=np.float64)

    det_rule = custom_derivative_rule_for(PrimitiveIdentity("scpn.program_ad.linalg", "det", "1"))
    inv_rule = custom_derivative_rule_for(PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1"))
    solve_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "solve", "1")
    )
    trace_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "trace", "1")
    )
    diag_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "diag", "1")
    )
    diagflat_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "diagflat", "1")
    )

    assert det_rule.name == "program_ad_linalg_det_direct_rule"
    assert inv_rule.name == "program_ad_linalg_inv_direct_rule"
    assert solve_rule.name == "program_ad_linalg_solve_direct_rule"
    assert trace_rule.name == "program_ad_linalg_trace_direct_rule"
    assert diag_rule.name == "program_ad_linalg_diag_trace_contract"
    assert diagflat_rule.name == "program_ad_linalg_diagflat_trace_contract"
    assert det_rule.jvp_rule is not None
    assert inv_rule.jvp_rule is not None
    assert solve_rule.jvp_rule is not None
    assert trace_rule.jvp_rule is not None
    assert det_rule.vjp_rule is not None
    assert inv_rule.vjp_rule is not None
    assert solve_rule.vjp_rule is not None
    assert trace_rule.vjp_rule is not None

    _assert_allclose(det_rule.value_fn(matrix.reshape(-1)), [np.linalg.det(matrix)])
    cofactor = np.array([[matrix[1, 1], -matrix[1, 0]], [-matrix[0, 1], matrix[0, 0]]])
    _assert_allclose(
        det_rule.jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        [np.sum(cofactor * tangent_matrix)],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        det_rule.vjp_rule(matrix.reshape(-1), np.array([1.75])),
        (1.75 * cofactor).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    inverse = np.linalg.inv(matrix)
    inverse_cotangent = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    _assert_allclose(inv_rule.value_fn(matrix.reshape(-1)), inverse.reshape(-1))
    _assert_allclose(
        inv_rule.jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        (-(inverse @ tangent_matrix @ inverse)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        inv_rule.vjp_rule(matrix.reshape(-1), inverse_cotangent.reshape(-1)),
        (-(inverse.T @ inverse_cotangent @ inverse.T)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    solve_values = np.concatenate((matrix.reshape(-1), rhs))
    solve_tangent = np.concatenate((tangent_matrix.reshape(-1), tangent_rhs))
    solution = np.linalg.solve(matrix, rhs)
    solve_cotangent = np.array([1.25, -0.5], dtype=np.float64)
    solve_adjoint_rhs = np.linalg.solve(matrix.T, solve_cotangent)
    _assert_allclose(solve_rule.value_fn(solve_values), solution)
    _assert_allclose(
        solve_rule.jvp_rule(solve_values, solve_tangent),
        np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        solve_rule.vjp_rule(solve_values, solve_cotangent),
        np.concatenate(((-np.outer(solve_adjoint_rhs, solution)).reshape(-1), solve_adjoint_rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    _assert_allclose(trace_rule.value_fn(matrix.reshape(-1)), [np.trace(matrix)])
    _assert_allclose(
        trace_rule.jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        [np.trace(tangent_matrix)],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        trace_rule.vjp_rule(matrix.reshape(-1), np.array([2.0])),
        (2.0 * np.eye(2, dtype=np.float64)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        diag_rule.value_fn(matrix.reshape(-1))
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        diagflat_rule.value_fn(matrix.reshape(-1))

    matrix_power_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
    )
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        matrix_power_rule.value_fn(matrix.reshape(-1))


def test_program_ad_linalg_static_derivative_factories_are_direct_kernels() -> None:
    """Static linalg primitive factories should expose direct value/JVP kernels."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    tangent_matrix = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    left = np.array([0.75, -1.5], dtype=np.float64)
    middle = matrix
    right = np.array([1.25, 0.5], dtype=np.float64)
    tangent_left = np.array([0.2, -0.1], dtype=np.float64)
    tangent_middle = tangent_matrix
    tangent_right = np.array([0.4, -0.3], dtype=np.float64)

    square_rule = program_ad_linalg_matrix_power_derivative_rule(2)
    inverse_rule = program_ad_linalg_matrix_power_derivative_rule(-1)
    zero_rule = program_ad_linalg_matrix_power_derivative_rule(0)
    matrix_cotangent = np.array([[1.5, -0.5], [0.75, 2.0]], dtype=np.float64)
    assert square_rule.vjp_rule is not None
    assert inverse_rule.vjp_rule is not None
    assert zero_rule.vjp_rule is not None

    _assert_allclose(square_rule.value_fn(matrix.reshape(-1)), (matrix @ matrix).reshape(-1))
    _assert_allclose(
        cast(Any, square_rule).jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        (tangent_matrix @ matrix + matrix @ tangent_matrix).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, square_rule).vjp_rule(matrix.reshape(-1), matrix_cotangent.reshape(-1)),
        (matrix_cotangent @ matrix.T + matrix.T @ matrix_cotangent).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    inverse = np.linalg.inv(matrix)
    _assert_allclose(inverse_rule.value_fn(matrix.reshape(-1)), inverse.reshape(-1))
    _assert_allclose(
        cast(Any, inverse_rule).jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        (-(inverse @ tangent_matrix @ inverse)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, inverse_rule).vjp_rule(matrix.reshape(-1), matrix_cotangent.reshape(-1)),
        (-(inverse.T @ matrix_cotangent @ inverse.T)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, zero_rule).jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        np.zeros(matrix.size, dtype=np.float64),
    )
    _assert_allclose(
        cast(Any, zero_rule).vjp_rule(matrix.reshape(-1), matrix_cotangent.reshape(-1)),
        np.zeros(matrix.size, dtype=np.float64),
    )

    multi_rule = program_ad_linalg_multi_dot_derivative_rule(((2,), (2, 2), (2,)))
    assert multi_rule.vjp_rule is not None
    values = np.concatenate((left, middle.reshape(-1), right))
    tangent = np.concatenate((tangent_left, tangent_middle.reshape(-1), tangent_right))
    _assert_allclose(
        multi_rule.value_fn(values),
        np.asarray(np.linalg.multi_dot((left, middle, right))).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected_jvp = (
        np.linalg.multi_dot((tangent_left, middle, right))
        + np.linalg.multi_dot((left, tangent_middle, right))
        + np.linalg.multi_dot((left, middle, tangent_right))
    )
    _assert_allclose(
        cast(Any, multi_rule).jvp_rule(values, tangent),
        np.asarray(expected_jvp).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, multi_rule).vjp_rule(values, np.array([2.5], dtype=np.float64)),
        np.concatenate(
            (
                (2.5 * (middle @ right)).reshape(-1),
                (2.5 * np.outer(left, right)).reshape(-1),
                (2.5 * (left @ middle)).reshape(-1),
            )
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="integer power"):
        program_ad_linalg_matrix_power_derivative_rule(cast(Any, 1.5))
    with pytest.raises(ValueError, match="at least two shapes"):
        program_ad_linalg_multi_dot_derivative_rule(((2,),))

    eig_rule = program_ad_linalg_eig_derivative_rule((2, 2))
    assert eig_rule.vjp_rule is not None
    eig_matrix = np.array([[2.0, 0.25], [0.5, 1.25]], dtype=np.float64)
    eig_tangent = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    eig_values, eig_vectors = np.linalg.eig(eig_matrix)
    eig_values = np.real(eig_values)
    eig_vectors = np.real(eig_vectors)
    left_rows = np.linalg.inv(eig_vectors)
    expected_eig_jvp: list[float] = []
    for index in range(eig_matrix.shape[0]):
        expected_eig_jvp.append(float(left_rows[index, :] @ eig_tangent @ eig_vectors[:, index]))
    expected_eig_vector_jvp = np.zeros_like(eig_vectors, dtype=np.float64)
    for column in range(eig_matrix.shape[0]):
        source = np.real(eig_vectors[:, column])
        raw_column = np.zeros(eig_matrix.shape[0], dtype=np.float64)
        for other in range(eig_matrix.shape[0]):
            if other == column:
                continue
            scale = float(left_rows[other, :] @ eig_tangent @ source) / float(
                eig_values[column] - eig_values[other]
            )
            raw_column += scale * np.real(eig_vectors[:, other])
        expected_eig_vector_jvp[:, column] = raw_column - source * float(source.T @ raw_column)
    _assert_allclose(
        eig_rule.value_fn(eig_matrix.reshape(-1)),
        np.concatenate((np.real(eig_values), np.real(eig_vectors).reshape(-1))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, eig_rule).jvp_rule(eig_matrix.reshape(-1), eig_tangent.reshape(-1)),
        np.concatenate((np.array(expected_eig_jvp), expected_eig_vector_jvp.reshape(-1))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    eig_cotangent = np.array([0.5, -1.25, 0.75, -0.25, 1.5, 0.25], dtype=np.float64)
    expected_eig_vjp = np.zeros_like(eig_matrix)
    for row in range(eig_matrix.shape[0]):
        for col in range(eig_matrix.shape[1]):
            basis = np.zeros_like(eig_matrix)
            basis[row, col] = 1.0
            expected_eig_vjp[row, col] = float(
                cast(Any, eig_rule).jvp_rule(eig_matrix.reshape(-1), basis.reshape(-1))
                @ eig_cotangent
            )
    _assert_allclose(
        cast(Any, eig_rule).vjp_rule(eig_matrix.reshape(-1), eig_cotangent),
        expected_eig_vjp.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    trace_rule = program_ad_linalg_trace_derivative_rule((2, 3), offset=1)
    assert trace_rule.name == "program_ad_linalg_trace_2x3_offset_1_direct_rule"
    assert trace_rule.jvp_rule is not None
    assert trace_rule.vjp_rule is not None
    rectangular = np.array([[2.0, -0.5, 1.0], [0.75, 1.5, -2.0]], dtype=np.float64)
    rectangular_tangent = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
    _assert_allclose(trace_rule.value_fn(rectangular.reshape(-1)), [-2.5])
    _assert_allclose(
        cast(Any, trace_rule).jvp_rule(rectangular.reshape(-1), rectangular_tangent.reshape(-1)),
        [0.4],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected_trace_vjp = np.zeros_like(rectangular)
    expected_trace_vjp[0, 1] = 1.25
    expected_trace_vjp[1, 2] = 1.25
    _assert_allclose(
        cast(Any, trace_rule).vjp_rule(
            rectangular.reshape(-1), np.array([1.25], dtype=np.float64)
        ),
        expected_trace_vjp.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_linalg_solve_static_derivative_factory_supports_matrix_rhs() -> None:
    """Static solve factories should expose exact matrix-RHS JVP and VJP rules."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    rhs = np.array([[0.25, -1.0, 0.5], [1.25, 0.75, -0.25]], dtype=np.float64)
    tangent_matrix = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    tangent_rhs = np.array([[0.5, -0.25, 0.75], [1.0, -0.5, 0.25]], dtype=np.float64)
    cotangent = np.array([[1.25, -0.5, 0.75], [-1.0, 0.25, 1.5]], dtype=np.float64)

    solve_rule = program_ad_linalg_solve_derivative_rule((2, 2), (2, 3))
    assert solve_rule.name == "program_ad_linalg_solve_2x2_rhs_2x3_direct_rule"
    assert solve_rule.jvp_rule is not None
    assert solve_rule.vjp_rule is not None

    values = np.concatenate((matrix.reshape(-1), rhs.reshape(-1)))
    tangent = np.concatenate((tangent_matrix.reshape(-1), tangent_rhs.reshape(-1)))
    solution = np.linalg.solve(matrix, rhs)
    rhs_adjoint = np.linalg.solve(matrix.T, cotangent)

    _assert_allclose(
        solve_rule.value_fn(values),
        solution.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, solve_rule).jvp_rule(values, tangent),
        np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _assert_allclose(
        cast(Any, solve_rule).vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate(((-(rhs_adjoint @ solution.T)).reshape(-1), rhs_adjoint.reshape(-1))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    vector_rule = program_ad_linalg_solve_derivative_rule((2, 2), (2,))
    assert vector_rule.name == "program_ad_linalg_solve_2x2_rhs_2_direct_rule"
    vector_values = np.concatenate((matrix.reshape(-1), rhs[:, 0]))
    _assert_allclose(vector_rule.value_fn(vector_values), np.linalg.solve(matrix, rhs[:, 0]))

    with pytest.raises(ValueError, match="square matrix"):
        program_ad_linalg_solve_derivative_rule((2, 3), (2,))
    with pytest.raises(ValueError, match="right-hand side rows"):
        program_ad_linalg_solve_derivative_rule((2, 2), (3,))
