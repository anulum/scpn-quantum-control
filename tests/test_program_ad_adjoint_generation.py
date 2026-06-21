# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD linalg adjoint contribution generation tests
"""Tests for linalg reverse-adjoint contribution generation.

Each linear-algebra primitive contribution helper is exercised directly with
hand-built whole-program AD IR nodes. Happy-path contributions are validated
against central finite differences of the corresponding scalar output element;
every shape/metadata validation branch is driven with a malformed operation
label or input set.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import scpn_quantum_control.program_ad_adjoint_generation as generation
from scpn_quantum_control.program_ad_adjoint_generation import (
    _program_adjoint_det_contributions,
    _program_adjoint_diag_contributions,
    _program_adjoint_diagflat_contributions,
    _program_adjoint_eig_eigenvalue_contributions,
    _program_adjoint_eig_eigenvector_contributions,
    _program_adjoint_eigh_eigenvalue_contributions,
    _program_adjoint_eigh_eigenvector_contributions,
    _program_adjoint_eigvals_contributions,
    _program_adjoint_eigvalsh_contributions,
    _program_adjoint_inv_contributions,
    _program_adjoint_matrix_power_contributions,
    _program_adjoint_multi_dot_contributions,
    _program_adjoint_multi_dot_output_metadata,
    _program_adjoint_parse_shape_label,
    _program_adjoint_pinv_contributions,
    _program_adjoint_solve_contributions,
    _program_adjoint_svdvals_contributions,
    _program_adjoint_trace_contributions,
)
from scpn_quantum_control.whole_program_ad_result import WholeProgramIRNode

Contribution = tuple[tuple[str, float], ...]


def _value_node(index: int, value: float) -> WholeProgramIRNode:
    """Build a leaf IR value node carrying one primal scalar."""

    return WholeProgramIRNode(
        index=index, op="parameter", inputs=(), value=float(value), tangent=np.zeros(1)
    )


def _flat_inputs(
    values: ArrayLike,
) -> tuple[tuple[str, ...], dict[str, WholeProgramIRNode]]:
    """Return SSA input tokens and a name table for a flattened value array."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    names = tuple(f"%{i}" for i in range(flat.size))
    table = {name: _value_node(i, float(flat[i])) for i, name in enumerate(names)}
    return names, table


def _node(op: str, inputs: Sequence[str]) -> WholeProgramIRNode:
    """Build an operation IR node with the given label and input tokens."""

    return WholeProgramIRNode(
        index=99, op=op, inputs=tuple(inputs), value=0.0, tangent=np.zeros(1)
    )


def _contribution_vector(
    contributions: Contribution, names: tuple[str, ...]
) -> NDArray[np.float64]:
    """Project named contributions back onto the input-token order."""

    accumulator = dict.fromkeys(names, 0.0)
    for name, partial in contributions:
        accumulator[name] += partial
    return np.array([accumulator[name] for name in names], dtype=np.float64)


def _fd_gradient(
    scalar: Callable[[NDArray[np.float64]], float],
    matrix: ArrayLike,
    *,
    eps: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Central finite-difference gradient of ``scalar`` over a flattened matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    shape = array.shape
    flat = array.reshape(-1)
    gradient = np.zeros_like(flat)
    for index in range(flat.size):
        up = flat.copy()
        down = flat.copy()
        up[index] += eps
        down[index] -= eps
        gradient[index] = (scalar(up.reshape(shape)) - scalar(down.reshape(shape))) / (2.0 * eps)
    return gradient


# --------------------------------------------------------------------------- det


def test_det_contributions_match_finite_difference() -> None:
    """Det contributions match finite difference."""
    matrix = np.array([[2.0, 1.0], [0.5, 3.0]])
    names, table = _flat_inputs(matrix)
    contributions = _program_adjoint_det_contributions(_node("linalg:det:2x2", names), table)
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.det(m)), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-6, atol=1e-8)


def test_det_empty_matrix_returns_no_contributions() -> None:
    """Det empty matrix returns no contributions."""
    assert _program_adjoint_det_contributions(_node("linalg:det:0x0", ()), {}) == ()


def test_det_rejects_unqualified_label() -> None:
    """Det rejects unqualified label."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape-qualified"):
        _program_adjoint_det_contributions(_node("linalg:det", names), table)


def test_det_rejects_malformed_shape() -> None:
    """Det rejects malformed shape."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_det_contributions(_node("linalg:det:2xz", names), table)


def test_det_rejects_non_square_inputs() -> None:
    """Det rejects non square inputs."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_det_contributions(_node("linalg:det:2x3", names), table)


# --------------------------------------------------------------------------- inv


def test_inv_contributions_match_finite_difference() -> None:
    """Inv contributions match finite difference."""
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    names, table = _flat_inputs(matrix)
    for out_row in range(2):
        for out_col in range(2):
            op = f"linalg:inv:2x2:{out_row}:{out_col}"
            contributions = _program_adjoint_inv_contributions(_node(op, names), table)
            analytic = _contribution_vector(contributions, names)

            def element(m: NDArray[np.float64], r: int = out_row, c: int = out_col) -> float:
                return float(np.linalg.inv(m)[r, c])

            expected = _fd_gradient(element, matrix)
            np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_inv_rejects_short_label() -> None:
    """Inv rejects short label."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="shape and output-index"):
        _program_adjoint_inv_contributions(_node("linalg:inv:2x2:0", names), table)


def test_inv_rejects_malformed_metadata() -> None:
    """Inv rejects malformed metadata."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_inv_contributions(_node("linalg:inv:2x2:0:z", names), table)


def test_inv_rejects_non_square_inputs() -> None:
    """Inv rejects non square inputs."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_inv_contributions(_node("linalg:inv:2x3:0:0", names), table)


def test_inv_rejects_output_index_out_of_range() -> None:
    """Inv rejects output index out of range."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="outside inverse shape"):
        _program_adjoint_inv_contributions(_node("linalg:inv:2x2:2:0", names), table)


def test_inv_rejects_singular_matrix() -> None:
    """Inv rejects singular matrix."""
    names, table = _flat_inputs(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(ValueError, match="nonsingular"):
        _program_adjoint_inv_contributions(_node("linalg:inv:2x2:0:0", names), table)


# ------------------------------------------------------------------------- solve


def test_solve_vector_contributions_match_finite_difference() -> None:
    """Solve vector contributions match finite difference."""
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([1.0, -2.0])
    names, table = _flat_inputs(np.concatenate([matrix.reshape(-1), rhs]))
    matrix_names = names[:4]
    op = "linalg:solve:2x2:rhs:2:1"
    contributions = _program_adjoint_solve_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)

    def scalar(flat: NDArray[np.float64]) -> float:
        m = flat[:4].reshape(2, 2)
        b = flat[4:]
        return float(np.linalg.solve(m, b)[1])

    expected = _fd_gradient(scalar, np.concatenate([matrix.reshape(-1), rhs]))
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)
    assert len(matrix_names) == 4


def test_solve_matrix_contributions_match_finite_difference() -> None:
    """Solve matrix contributions match finite difference."""
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([[1.0, 0.0], [0.0, 2.0]])
    flat = np.concatenate([matrix.reshape(-1), rhs.reshape(-1)])
    names, table = _flat_inputs(flat)
    op = "linalg:solve:2x2:rhs:2x2:0:1"
    contributions = _program_adjoint_solve_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)

    def scalar(values: NDArray[np.float64]) -> float:
        m = values[:4].reshape(2, 2)
        b = values[4:].reshape(2, 2)
        return float(np.linalg.solve(m, b)[0, 1])

    expected = _fd_gradient(scalar, flat)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_solve_rejects_label_without_rhs_marker() -> None:
    """Solve rejects label without rhs marker."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="shape, rhs, and output-index"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:lhs:2:0", names), table)


def test_solve_rejects_malformed_metadata() -> None:
    """Solve rejects malformed metadata."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:z:0", names), table)


def test_solve_rejects_non_square_matrix() -> None:
    """Solve rejects non square matrix."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="non-empty square matrix"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x3:rhs:2:0", names), table)


def test_solve_rejects_incompatible_vector_rhs() -> None:
    """Solve rejects incompatible vector rhs."""
    names, table = _flat_inputs(np.ones(7))
    with pytest.raises(ValueError, match="vector adjoint rhs shape"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:3:0", names), table)


def test_solve_rejects_vector_output_row_out_of_range() -> None:
    """Solve rejects vector output row out of range."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="output row is outside"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:2:5", names), table)


def test_solve_rejects_incompatible_matrix_rhs() -> None:
    """Solve rejects incompatible matrix rhs."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="matrix adjoint rhs shape"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:3x2:0:0", names), table)


def test_solve_rejects_matrix_output_index_out_of_range() -> None:
    """Solve rejects matrix output index out of range."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="output index is outside"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:2x2:0:5", names), table)


def test_solve_rejects_high_rank_rhs() -> None:
    """Solve rejects high rank rhs."""
    names, table = _flat_inputs(np.ones(12))
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:2x2x2:0:0", names), table)


def test_solve_rejects_input_count_mismatch() -> None:
    """Solve rejects input count mismatch."""
    names, table = _flat_inputs(np.ones(5))
    with pytest.raises(ValueError, match="matrix followed by rhs"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:2:0", names), table)


def test_solve_rejects_singular_matrix() -> None:
    """Solve rejects singular matrix."""
    flat = np.concatenate([np.array([1.0, 2.0, 2.0, 4.0]), np.array([1.0, 1.0])])
    names, table = _flat_inputs(flat)
    with pytest.raises(ValueError, match="nonsingular"):
        _program_adjoint_solve_contributions(_node("linalg:solve:2x2:rhs:2:0", names), table)


# ------------------------------------------------------------------------- trace


def test_trace_contributions_select_diagonal() -> None:
    """Trace contributions select diagonal."""
    names, table = _flat_inputs(np.arange(2.0))
    contributions = _program_adjoint_trace_contributions(_node("linalg:trace:2x2:offset:0", names))
    assert contributions == ((names[0], 1.0), (names[1], 1.0))


def test_trace_rejects_label_without_offset() -> None:
    """Trace rejects label without offset."""
    names, _ = _flat_inputs(np.arange(2.0))
    with pytest.raises(ValueError, match="shape and offset"):
        _program_adjoint_trace_contributions(_node("linalg:trace:2x2:0", names))


def test_trace_rejects_malformed_metadata() -> None:
    """Trace rejects malformed metadata."""
    names, _ = _flat_inputs(np.arange(2.0))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_trace_contributions(_node("linalg:trace:2xz:offset:0", names))


def test_trace_rejects_non_matrix_shape() -> None:
    """Trace rejects non matrix shape."""
    names, _ = _flat_inputs(np.arange(3.0))
    with pytest.raises(ValueError, match="rank-2 matrix"):
        _program_adjoint_trace_contributions(_node("linalg:trace:3:offset:0", names))


def test_trace_rejects_diagonal_length_mismatch() -> None:
    """Trace rejects diagonal length mismatch."""
    names, _ = _flat_inputs(np.arange(3.0))
    with pytest.raises(ValueError, match="selected diagonal"):
        _program_adjoint_trace_contributions(_node("linalg:trace:2x2:offset:0", names))


# -------------------------------------------------------------------------- diag


def test_diag_construct_returns_single_contribution() -> None:
    """Diag construct returns single contribution."""
    names, _ = _flat_inputs(np.arange(3.0))
    contributions = _program_adjoint_diag_contributions(
        _node("linalg:diag:3:offset:0:construct:1", (names[1],))
    )
    assert contributions == ((names[1], 1.0),)


def test_diag_extract_returns_single_contribution() -> None:
    """Diag extract returns single contribution."""
    contributions = _program_adjoint_diag_contributions(
        _node("linalg:diag:3x3:offset:0:extract:1", ("%4",))
    )
    assert contributions == (("%4", 1.0),)


def test_diag_rejects_malformed_structure() -> None:
    """Diag rejects malformed structure."""
    with pytest.raises(ValueError, match="shape, offset, mode, and index"):
        _program_adjoint_diag_contributions(_node("linalg:diag:3:offset:0:rotate:1", ("%1",)))


def test_diag_rejects_multiple_inputs() -> None:
    """Diag rejects multiple inputs."""
    with pytest.raises(ValueError, match="one source input"):
        _program_adjoint_diag_contributions(
            _node("linalg:diag:3:offset:0:construct:1", ("%0", "%1"))
        )


def test_diag_rejects_malformed_metadata() -> None:
    """Diag rejects malformed metadata."""
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_diag_contributions(_node("linalg:diag:3:offset:z:construct:1", ("%1",)))


def test_diag_construct_rejects_index_out_of_range() -> None:
    """Diag construct rejects index out of range."""
    with pytest.raises(ValueError, match="outside vector shape"):
        _program_adjoint_diag_contributions(_node("linalg:diag:3:offset:0:construct:5", ("%1",)))


def test_diag_extract_rejects_non_matrix_shape() -> None:
    """Diag extract rejects non matrix shape."""
    with pytest.raises(ValueError, match="rank-2 source"):
        _program_adjoint_diag_contributions(_node("linalg:diag:3:offset:0:extract:0", ("%1",)))


def test_diag_extract_rejects_index_out_of_range() -> None:
    """Diag extract rejects index out of range."""
    with pytest.raises(ValueError, match="outside diagonal shape"):
        _program_adjoint_diag_contributions(_node("linalg:diag:3x3:offset:0:extract:9", ("%1",)))


# ---------------------------------------------------------------------- diagflat


def test_diagflat_returns_single_contribution() -> None:
    """Diagflat returns single contribution."""
    contributions = _program_adjoint_diagflat_contributions(
        _node("linalg:diagflat:2x2:offset:0:construct:2", ("%2",))
    )
    assert contributions == (("%2", 1.0),)


def test_diagflat_rejects_malformed_structure() -> None:
    """Diagflat rejects malformed structure."""
    with pytest.raises(ValueError, match="shape, offset, construct, and index"):
        _program_adjoint_diagflat_contributions(
            _node("linalg:diagflat:2x2:offset:0:extract:2", ("%2",))
        )


def test_diagflat_rejects_multiple_inputs() -> None:
    """Diagflat rejects multiple inputs."""
    with pytest.raises(ValueError, match="one source input"):
        _program_adjoint_diagflat_contributions(
            _node("linalg:diagflat:2x2:offset:0:construct:0", ("%0", "%1"))
        )


def test_diagflat_rejects_malformed_metadata() -> None:
    """Diagflat rejects malformed metadata."""
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_diagflat_contributions(
            _node("linalg:diagflat:2x2:offset:z:construct:0", ("%0",))
        )


def test_diagflat_rejects_index_out_of_range() -> None:
    """Diagflat rejects index out of range."""
    with pytest.raises(ValueError, match="flattened source shape"):
        _program_adjoint_diagflat_contributions(
            _node("linalg:diagflat:2x2:offset:0:construct:9", ("%0",))
        )


# ------------------------------------------------------------- parse_shape_label


def test_parse_shape_label_reads_dimensions() -> None:
    """Parse shape label reads dimensions."""
    assert _program_adjoint_parse_shape_label("2x3x4") == (2, 3, 4)


def test_parse_shape_label_rejects_empty_label() -> None:
    """Parse shape label rejects empty label."""
    with pytest.raises(ValueError, match="must not be empty"):
        _program_adjoint_parse_shape_label("")


def test_parse_shape_label_rejects_negative_dimension() -> None:
    """Parse shape label rejects negative dimension."""
    with pytest.raises(ValueError, match="non-negative"):
        _program_adjoint_parse_shape_label("2x-1")


# ------------------------------------------------------------------ matrix_power


def test_matrix_power_contributions_match_finite_difference() -> None:
    """Matrix power contributions match finite difference."""
    matrix = np.array([[1.5, 0.4], [0.2, 1.1]])
    names, table = _flat_inputs(matrix)
    op = "linalg:matrix_power:2x2:power:3:0:1"
    contributions = _program_adjoint_matrix_power_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.matrix_power(m, 3)[0, 1]), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_matrix_power_negative_exponent_rejects_singular() -> None:
    """Matrix power negative exponent rejects singular."""
    names, table = _flat_inputs(np.array([[1.0, 2.0], [2.0, 4.0]]))
    op = "linalg:matrix_power:2x2:power:-1:0:0"
    with pytest.raises(ValueError, match="nonsingular"):
        _program_adjoint_matrix_power_contributions(_node(op, names), table)


def test_matrix_power_rejects_short_label() -> None:
    """Matrix power rejects short label."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="shape, power, and output-index"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x2:3:0", names), table
        )


def test_matrix_power_rejects_malformed_metadata() -> None:
    """Matrix power rejects malformed metadata."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x2:power:z:0:0", names), table
        )


def test_matrix_power_rejects_non_square_metadata() -> None:
    """Matrix power rejects non square metadata."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="non-empty square matrix"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x3:power:2:0:0", names), table
        )


def test_matrix_power_rejects_input_count_mismatch() -> None:
    """Matrix power rejects input count mismatch."""
    names, table = _flat_inputs(np.ones(3))
    with pytest.raises(ValueError, match="one flattened square matrix"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x2:power:2:0:0", names), table
        )


def test_matrix_power_rejects_output_index_out_of_range() -> None:
    """Matrix power rejects output index out of range."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="outside matrix shape"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x2:power:2:5:0", names), table
        )


# --------------------------------------------------------------------- multi_dot


def test_multi_dot_contributions_match_finite_difference() -> None:
    """Multi dot contributions match finite difference."""
    left = np.array([[1.0, 2.0], [0.5, 1.5]])
    right = np.array([[2.0, 0.0], [1.0, 3.0]])
    flat = np.concatenate([left.reshape(-1), right.reshape(-1)])
    names, table = _flat_inputs(flat)
    op = "linalg:multi_dot:2x2__2x2:out:2x2:1"
    contributions = _program_adjoint_multi_dot_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)

    def scalar(values: NDArray[np.float64]) -> float:
        a = values[:4].reshape(2, 2)
        b = values[4:].reshape(2, 2)
        return float(np.linalg.multi_dot([a, b]).reshape(-1)[1])

    expected = _fd_gradient(scalar, flat)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_multi_dot_scalar_output_contributions() -> None:
    """Multi dot scalar output contributions."""
    left = np.array([[1.0, 2.0]])
    right = np.array([[3.0], [4.0]])
    flat = np.concatenate([left.reshape(-1), right.reshape(-1)])
    names, table = _flat_inputs(flat)
    op = "linalg:multi_dot:1x2__2x1:out:scalar"
    contributions = _program_adjoint_multi_dot_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)

    def scalar(values: NDArray[np.float64]) -> float:
        a = values[:2].reshape(1, 2)
        b = values[2:].reshape(2, 1)
        return float(np.linalg.multi_dot([a, b]).reshape(-1)[0])

    expected = _fd_gradient(scalar, flat)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_multi_dot_rejects_label_without_out_marker() -> None:
    """Multi dot rejects label without out marker."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="operand-shape and output"):
        _program_adjoint_multi_dot_contributions(
            _node("linalg:multi_dot:2x2__2x2:res:2x2:0", names), table
        )


def test_multi_dot_rejects_malformed_metadata() -> None:
    """Multi dot rejects malformed metadata."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_multi_dot_contributions(
            _node("linalg:multi_dot:2x2__2xz:out:2x2:0", names), table
        )


def test_multi_dot_rejects_input_count_mismatch() -> None:
    """Multi dot rejects input count mismatch."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened operand shapes"):
        _program_adjoint_multi_dot_contributions(
            _node("linalg:multi_dot:2x2__2x2:out:2x2:0", names), table
        )


def test_multi_dot_rejects_output_index_out_of_range() -> None:
    """Multi dot rejects output index out of range."""
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="outside result shape"):
        _program_adjoint_multi_dot_contributions(
            _node("linalg:multi_dot:2x2__2x2:out:2x2:9", names), table
        )


def test_multi_dot_output_metadata_parses_scalar() -> None:
    """Multi dot output metadata parses scalar."""
    assert _program_adjoint_multi_dot_output_metadata(["scalar"]) == ((), 0)


def test_multi_dot_output_metadata_parses_shape_index() -> None:
    """Multi dot output metadata parses shape index."""
    assert _program_adjoint_multi_dot_output_metadata(["2x3", "4"]) == ((2, 3), 4)


def test_multi_dot_output_metadata_rejects_wrong_arity() -> None:
    """Multi dot output metadata rejects wrong arity."""
    with pytest.raises(ValueError, match="scalar or shape plus index"):
        _program_adjoint_multi_dot_output_metadata(["2x3", "4", "5"])


def test_multi_dot_output_metadata_rejects_empty_shape() -> None:
    """Multi dot output metadata rejects empty shape."""
    with pytest.raises(ValueError, match="must not be empty"):
        _program_adjoint_multi_dot_output_metadata(["", "0"])


# ---------------------------------------------------------------------- eigvalsh


def _symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return (matrix + matrix.T) / 2.0


def test_eigvalsh_contributions_match_finite_difference() -> None:
    """Eigvalsh contributions match finite difference."""
    matrix = _symmetric(np.array([[3.0, 1.0], [1.0, 2.0]]))
    names, table = _flat_inputs(matrix)
    contributions = _program_adjoint_eigvalsh_contributions(
        _node("linalg:eigvalsh:1", names), table
    )
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.eigvalsh(_symmetric(m))[1]), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_eigvalsh_rejects_non_integer_index() -> None:
    """Eigvalsh rejects non integer index."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="eigenvalue index"):
        _program_adjoint_eigvalsh_contributions(_node("linalg:eigvalsh:z", names), table)


def test_eigvalsh_rejects_non_square_inputs() -> None:
    """Eigvalsh rejects non square inputs."""
    names, table = _flat_inputs(np.ones(3))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_eigvalsh_contributions(_node("linalg:eigvalsh:0", names), table)


def test_eigvalsh_rejects_index_out_of_range() -> None:
    """Eigvalsh rejects index out of range."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="outside the spectrum"):
        _program_adjoint_eigvalsh_contributions(_node("linalg:eigvalsh:5", names), table)


# ----------------------------------------------------------------------- eigvals


def test_eigvals_contributions_match_finite_difference() -> None:
    """Eigvals contributions match finite difference."""
    matrix = np.array([[2.0, 0.3], [0.1, 3.0]])
    names, table = _flat_inputs(matrix)
    contributions = _program_adjoint_eigvals_contributions(
        _node("linalg:eigvals:2x2:0", names), table
    )
    analytic = _contribution_vector(contributions, names)

    def scalar(m: NDArray[np.float64]) -> float:
        return float(np.sort(np.linalg.eigvals(m).real)[0])

    expected = _fd_gradient(scalar, matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-6)


def test_eigvals_rejects_short_label() -> None:
    """Eigvals rejects short label."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="shape-qualified spectral"):
        _program_adjoint_eigvals_contributions(_node("linalg:eigvals:2x2", names), table)


def test_eigvals_rejects_malformed_metadata() -> None:
    """Eigvals rejects malformed metadata."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_eigvals_contributions(_node("linalg:eigvals:2x2:z", names), table)


def test_eigvals_rejects_non_square_inputs() -> None:
    """Eigvals rejects non square inputs."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_eigvals_contributions(_node("linalg:eigvals:2x3:0", names), table)


def test_eigvals_rejects_index_out_of_range() -> None:
    """Eigvals rejects index out of range."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="outside the spectrum"):
        _program_adjoint_eigvals_contributions(_node("linalg:eigvals:2x2:5", names), table)


# --------------------------------------------------------------------------- eig


def test_eig_eigenvalue_contributions_match_finite_difference() -> None:
    """Eig eigenvalue contributions match finite difference."""
    matrix = np.array([[2.0, 0.3], [0.1, 3.0]])
    names, table = _flat_inputs(matrix)
    op = "linalg:eig:eigenvalue:2x2:0"
    contributions = _program_adjoint_eig_eigenvalue_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)

    def scalar(m: NDArray[np.float64]) -> float:
        return float(np.sort(np.linalg.eigvals(m).real)[0])

    expected = _fd_gradient(scalar, matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-6)


def test_eig_eigenvector_contributions_have_input_shape() -> None:
    """Eig eigenvector contributions have input shape."""
    matrix = np.array([[2.0, 0.3], [0.1, 3.0]])
    names, table = _flat_inputs(matrix)
    op = "linalg:eig:eigenvector:2x2:0:0"
    contributions = _program_adjoint_eig_eigenvector_contributions(_node(op, names), table)
    assert tuple(name for name, _ in contributions) == names


def test_eig_metadata_rejects_short_label() -> None:
    """Eig metadata rejects short label."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="shape-qualified spectral"):
        _program_adjoint_eig_eigenvalue_contributions(
            _node("linalg:eig:eigenvalue:2x2", names), table
        )


def test_eig_metadata_rejects_malformed_metadata() -> None:
    """Eig metadata rejects malformed metadata."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_eig_eigenvalue_contributions(
            _node("linalg:eig:eigenvalue:2x2:z", names), table
        )


def test_eig_metadata_rejects_non_square_inputs() -> None:
    """Eig metadata rejects non square inputs."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_eig_eigenvalue_contributions(
            _node("linalg:eig:eigenvalue:2x3:0", names), table
        )


def test_eig_metadata_rejects_eigenvalue_index_out_of_range() -> None:
    """Eig metadata rejects eigenvalue index out of range."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="eigenvalue index is outside"):
        _program_adjoint_eig_eigenvalue_contributions(
            _node("linalg:eig:eigenvalue:2x2:5", names), table
        )


def test_eig_metadata_rejects_eigenvector_row_out_of_range() -> None:
    """Eig metadata rejects eigenvector row out of range."""
    names, table = _flat_inputs(np.array([[2.0, 0.3], [0.1, 3.0]]))
    with pytest.raises(ValueError, match="eigenvector row index is outside"):
        _program_adjoint_eig_eigenvector_contributions(
            _node("linalg:eig:eigenvector:2x2:0:5", names), table
        )


# -------------------------------------------------------------------------- eigh


def test_eigh_eigenvalue_contributions_match_finite_difference() -> None:
    """Eigh eigenvalue contributions match finite difference."""
    matrix = _symmetric(np.array([[3.0, 1.0], [1.0, 2.0]]))
    names, table = _flat_inputs(matrix)
    op = "linalg:eigh:eigenvalue:2x2:L:1"
    contributions = _program_adjoint_eigh_eigenvalue_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.eigvalsh(_symmetric(m))[1]), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_eigh_eigenvector_contributions_have_input_shape() -> None:
    """Eigh eigenvector contributions have input shape."""
    matrix = _symmetric(np.array([[3.0, 1.0], [1.0, 2.0]]))
    names, table = _flat_inputs(matrix)
    op = "linalg:eigh:eigenvector:2x2:L:0:0"
    contributions = _program_adjoint_eigh_eigenvector_contributions(_node(op, names), table)
    assert tuple(name for name, _ in contributions) == names


def test_eigh_metadata_rejects_short_label() -> None:
    """Eigh metadata rejects short label."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="shape-qualified spectral"):
        _program_adjoint_eigh_eigenvalue_contributions(
            _node("linalg:eigh:eigenvalue:2x2:L", names), table
        )


def test_eigh_metadata_rejects_malformed_metadata() -> None:
    """Eigh metadata rejects malformed metadata."""
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_eigh_eigenvalue_contributions(
            _node("linalg:eigh:eigenvalue:2x2:L:z", names), table
        )


def test_eigh_metadata_rejects_non_square_inputs() -> None:
    """Eigh metadata rejects non square inputs."""
    names, table = _flat_inputs(np.ones(6))
    with pytest.raises(ValueError, match="flattened square matrix"):
        _program_adjoint_eigh_eigenvalue_contributions(
            _node("linalg:eigh:eigenvalue:2x3:L:0", names), table
        )


def test_eigh_metadata_rejects_eigenvalue_index_out_of_range() -> None:
    """Eigh metadata rejects eigenvalue index out of range."""
    names, table = _flat_inputs(_symmetric(np.array([[3.0, 1.0], [1.0, 2.0]])))
    with pytest.raises(ValueError, match="eigenvalue index is outside"):
        _program_adjoint_eigh_eigenvalue_contributions(
            _node("linalg:eigh:eigenvalue:2x2:L:5", names), table
        )


def test_eigh_metadata_rejects_eigenvector_row_out_of_range() -> None:
    """Eigh metadata rejects eigenvector row out of range."""
    names, table = _flat_inputs(_symmetric(np.array([[3.0, 1.0], [1.0, 2.0]])))
    with pytest.raises(ValueError, match="eigenvector row index is outside"):
        _program_adjoint_eigh_eigenvector_contributions(
            _node("linalg:eigh:eigenvector:2x2:L:0:5", names), table
        )


# ----------------------------------------------------------------------- svdvals


def test_svdvals_contributions_match_finite_difference() -> None:
    """Svdvals contributions match finite difference."""
    matrix = np.array([[3.0, 1.0], [0.5, 2.0], [1.0, 0.2]])
    names, table = _flat_inputs(matrix)
    contributions = _program_adjoint_svdvals_contributions(
        _node("linalg:svdvals:3x2:0", names), table
    )
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.svd(m, compute_uv=False)[0]), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-5, atol=1e-7)


def test_svdvals_rejects_short_label() -> None:
    """Svdvals rejects short label."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="singular-value metadata"):
        _program_adjoint_svdvals_contributions(_node("linalg:svdvals:2x2", names), table)


def test_svdvals_rejects_malformed_metadata() -> None:
    """Svdvals rejects malformed metadata."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_svdvals_contributions(_node("linalg:svdvals:2x2:z", names), table)


def test_svdvals_rejects_input_count_mismatch() -> None:
    """Svdvals rejects input count mismatch."""
    names, table = _flat_inputs(np.ones(5))
    with pytest.raises(ValueError, match="flattened matrix inputs"):
        _program_adjoint_svdvals_contributions(_node("linalg:svdvals:2x2:0", names), table)


def test_svdvals_rejects_index_out_of_range() -> None:
    """Svdvals rejects index out of range."""
    names, table = _flat_inputs(np.array([[3.0, 1.0], [0.5, 2.0]]))
    with pytest.raises(ValueError, match="outside the spectrum"):
        _program_adjoint_svdvals_contributions(_node("linalg:svdvals:2x2:5", names), table)


# -------------------------------------------------------------------------- pinv


def test_pinv_contributions_match_finite_difference() -> None:
    """Pinv contributions match finite difference."""
    matrix = np.array([[3.0, 1.0], [0.5, 2.0], [1.0, 0.2]])
    names, table = _flat_inputs(matrix)
    op = "linalg:pinv:3x2:0.0:1:2"
    contributions = _program_adjoint_pinv_contributions(_node(op, names), table)
    analytic = _contribution_vector(contributions, names)
    expected = _fd_gradient(lambda m: float(np.linalg.pinv(m)[1, 2]), matrix)
    np.testing.assert_allclose(analytic, expected, rtol=1e-4, atol=1e-6)


def test_pinv_rejects_short_label() -> None:
    """Pinv rejects short label."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape, cutoff, and output-index"):
        _program_adjoint_pinv_contributions(_node("linalg:pinv:2x2:0.0:0", names), table)


def test_pinv_rejects_malformed_metadata() -> None:
    """Pinv rejects malformed metadata."""
    names, table = _flat_inputs(np.ones((2, 2)))
    with pytest.raises(ValueError, match="malformed"):
        _program_adjoint_pinv_contributions(_node("linalg:pinv:2x2:0.0:z:0", names), table)


def test_pinv_rejects_input_count_mismatch() -> None:
    """Pinv rejects input count mismatch."""
    names, table = _flat_inputs(np.ones(5))
    with pytest.raises(ValueError, match="flattened matrix inputs"):
        _program_adjoint_pinv_contributions(_node("linalg:pinv:2x2:0.0:0:0", names), table)


def test_pinv_rejects_output_index_out_of_range() -> None:
    """Pinv rejects output index out of range."""
    names, table = _flat_inputs(np.ones((3, 2)))
    with pytest.raises(ValueError, match="outside pseudoinverse shape"):
        _program_adjoint_pinv_contributions(_node("linalg:pinv:3x2:0.0:9:0", names), table)


# --------------------------------------------------- defensive VJP/shape guards


def test_matrix_power_requires_vjp_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrix power requires vjp rule."""
    monkeypatch.setattr(
        generation,
        "program_ad_linalg_matrix_power_derivative_rule",
        lambda exponent: SimpleNamespace(vjp_rule=None),
    )
    names, table = _flat_inputs(np.eye(2))
    with pytest.raises(ValueError, match="requires a VJP rule"):
        _program_adjoint_matrix_power_contributions(
            _node("linalg:matrix_power:2x2:power:2:0:0", names), table
        )


def test_multi_dot_requires_vjp_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi dot requires vjp rule."""
    monkeypatch.setattr(
        generation,
        "program_ad_linalg_multi_dot_derivative_rule",
        lambda operand_shapes: SimpleNamespace(vjp_rule=None),
    )
    names, table = _flat_inputs(np.ones(8))
    with pytest.raises(ValueError, match="requires a VJP rule"):
        _program_adjoint_multi_dot_contributions(
            _node("linalg:multi_dot:2x2__2x2:out:2x2:0", names), table
        )


def test_multi_dot_output_metadata_rejects_degenerate_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi dot output metadata rejects degenerate shape."""
    monkeypatch.setattr(generation, "_program_adjoint_parse_shape_label", lambda label: ())
    with pytest.raises(ValueError, match="must not be empty"):
        _program_adjoint_multi_dot_output_metadata(["2x3", "0"])
