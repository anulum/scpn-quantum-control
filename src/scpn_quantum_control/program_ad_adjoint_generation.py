# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD reverse-adjoint generation over stabilised IR
"""Reverse-mode adjoint generation over the stabilised whole-program AD IR.

This module owns the reverse pass of whole-program AD for supported scalar
programs:

- Per-node local pullback contribution rules: elementwise/unary and binary
  operations, selection primitives (``where``, ``clip``, ``choose``), and the
  linear-algebra primitives (determinant, inverse, linear solve, trace, diagonal
  construction/extraction, matrix power, multi-dot, ``eig``/``eigh`` eigenvalue
  and eigenvector elements, ``eigvals``/``eigvalsh`` spectra, ``svdvals``
  singular values, and ``pinv`` elements). Each returns the local cotangent
  contributions ``((input_name, partial), ...)``.
- The dispatcher :func:`_program_adjoint_node_contributions` routing one IR node
  to its contribution rule and failing closed on unsupported operations.
- The replay driver :func:`_program_adjoint_result_from_nodes` (and its
  :func:`_program_adjoint_steps_from_ir` helper) that propagates cotangents
  backward over the captured nodes, assembles the parameter gradient, and emits
  the per-step reverse-adjoint record over ``program_ad_effect_ir.v1`` metadata,
  including replayed runtime branch rows and blocked non-executed phi inputs.

Linear-algebra VJP rules are sourced from
:mod:`scpn_quantum_control.program_ad_linalg_primitives`; IR-node and result
records come from :mod:`scpn_quantum_control.program_ad_effect_ir` and
:mod:`scpn_quantum_control.program_ad_adjoint`. The public reverse-mode wrappers
(``program_adjoint_grad`` and friends) live in
:mod:`scpn_quantum_control.program_ad_adjoint`, while
:mod:`scpn_quantum_control.differentiable` keeps compatibility imports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .program_ad_adjoint import (
    ProgramADAdjointResult,
    ProgramADAdjointStep,
    _program_adjoint_input_value,
    _program_adjoint_is_ir_value,
)
from .program_ad_cumulative_primitives import (
    program_ad_cumulative_cumprod_derivative_rule,
    program_ad_cumulative_cumsum_derivative_rule,
    program_ad_cumulative_diff_derivative_rule,
)
from .program_ad_effect_ir import (
    ProgramADControlRegion,
    ProgramADEffectIR,
    ProgramADPhiNode,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_det_cofactor_matrix,
    _program_ad_linalg_eig_eigenvector_jvp_matrix,
    _program_ad_linalg_eigh_vjp_matrix,
    _program_ad_linalg_pinv_value_matrix,
    _program_ad_linalg_pinv_vjp_matrix,
    _program_ad_linalg_real_simple_eig_decomposition_from_matrix,
    _program_ad_linalg_require_distinct_eigenvalues,
    _program_ad_linalg_require_distinct_positive_singular_values,
    _program_ad_linalg_require_symmetric,
    _program_ad_linalg_uplo,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
)
from .whole_program_ad_result import WholeProgramIRNode


def _program_adjoint_det_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a determinant primitive node."""
    parts = node.op.split(":")
    if len(parts) != 3:
        raise ValueError("det adjoint requires shape-qualified determinant metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
    except ValueError as exc:
        raise ValueError("det adjoint metadata is malformed") from exc
    if rows != cols or rows < 0 or rows * cols != len(node.inputs):
        raise ValueError("det adjoint requires flattened square matrix inputs")
    if rows == 0:
        return ()
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(cofactors[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_inv_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one inverse-output primitive node."""
    parts = node.op.split(":")
    if len(parts) != 5:
        raise ValueError("inverse adjoint requires shape and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        output_row = int(parts[3])
        output_col = int(parts[4])
    except ValueError as exc:
        raise ValueError("inverse adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("inverse adjoint requires flattened square matrix inputs")
    if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= cols:
        raise ValueError("inverse adjoint output index is outside inverse shape")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("inverse adjoint requires a nonsingular matrix") from exc
    cotangent = np.zeros((rows, cols), dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    local_adjoint = -(inverse.T @ cotangent @ inverse.T)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_solve_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one linear-solve output node."""
    parts = node.op.split(":")
    if len(parts) not in {6, 7} or parts[3] != "rhs":
        raise ValueError("solve adjoint requires shape, rhs, and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        rhs_shape = tuple(int(part) for part in parts[4].split("x"))
        output_row = int(parts[5])
        output_col = int(parts[6]) if len(parts) == 7 else -1
    except ValueError as exc:
        raise ValueError("solve adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0:
        raise ValueError("solve adjoint requires a non-empty square matrix")
    if len(rhs_shape) == 1:
        if len(parts) != 6 or rhs_shape[0] != rows:
            raise ValueError("solve vector adjoint rhs shape is incompatible with matrix")
        if output_row < 0 or output_row >= rows:
            raise ValueError("solve adjoint output row is outside solution shape")
    elif len(rhs_shape) == 2:
        if len(parts) != 7 or rhs_shape[0] != rows or rhs_shape[1] <= 0:
            raise ValueError("solve matrix adjoint rhs shape is incompatible with matrix")
        if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= rhs_shape[1]:
            raise ValueError("solve adjoint output index is outside solution shape")
    else:
        raise ValueError("solve adjoint rhs shape must be rank-1 or rank-2")
    rhs_size = int(np.prod(rhs_shape, dtype=np.int64))
    matrix_size = rows * cols
    if len(node.inputs) != matrix_size + rhs_size:
        raise ValueError("solve adjoint inputs must contain matrix followed by rhs")
    matrix_input_names = node.inputs[:matrix_size]
    rhs_input_names = node.inputs[matrix_size:]
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in matrix_input_names],
        dtype=np.float64,
    ).reshape(rows, cols)
    rhs = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in rhs_input_names],
        dtype=np.float64,
    ).reshape(rhs_shape)
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("solve adjoint requires a nonsingular matrix") from exc
    cotangent = np.zeros_like(solution, dtype=np.float64)
    if len(rhs_shape) == 1:
        cotangent[output_row] = 1.0
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent)
        matrix_adjoint = -np.outer(rhs_adjoint, solution)
    else:
        cotangent[output_row, output_col] = 1.0
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent)
        matrix_adjoint = -(rhs_adjoint @ solution.T)
    flat_rhs_adjoint = np.asarray(rhs_adjoint, dtype=np.float64).reshape(-1)
    return tuple(
        (
            matrix_input_names[row * cols + col],
            float(matrix_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    ) + tuple(
        (
            rhs_input_names[index],
            float(flat_rhs_adjoint[index]),
        )
        for index in range(rhs_size)
    )


def _program_adjoint_trace_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a trace primitive node."""
    parts = node.op.split(":")
    if len(parts) != 5 or parts[3] != "offset":
        raise ValueError("trace adjoint requires shape and offset metadata")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        offset = int(parts[4])
    except ValueError as exc:
        raise ValueError("trace adjoint metadata is malformed") from exc
    if len(shape) != 2:
        raise ValueError("trace adjoint requires rank-2 matrix metadata")
    rows, cols = shape
    diagonal_length = sum(1 for row in range(rows) if 0 <= row + offset < cols)
    if diagonal_length <= 0 or len(node.inputs) != diagonal_length:
        raise ValueError("trace adjoint inputs must match the selected diagonal")
    return tuple((name, 1.0) for name in node.inputs)


def _program_adjoint_diag_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one diag primitive output node."""
    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "offset" or parts[5] not in {"construct", "extract"}:
        raise ValueError("diag adjoint requires shape, offset, mode, and index metadata")
    if len(node.inputs) != 1:
        raise ValueError("diag adjoint primitive outputs must have one source input")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        offset = int(parts[4])
        source_index = int(parts[6])
    except ValueError as exc:
        raise ValueError("diag adjoint metadata is malformed") from exc
    mode = parts[5]
    if mode == "construct":
        if len(shape) != 1 or source_index < 0 or source_index >= shape[0]:
            raise ValueError("diag construct adjoint source index is outside vector shape")
    else:
        if len(shape) != 2:
            raise ValueError("diag extract adjoint requires rank-2 source metadata")
        rows, cols = shape
        diagonal_length = sum(1 for row in range(rows) if 0 <= row + offset < cols)
        if source_index < 0 or source_index >= diagonal_length:
            raise ValueError("diag extract adjoint output index is outside diagonal shape")
    return ((node.inputs[0], 1.0),)


def _program_adjoint_diagflat_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one diagflat primitive output node."""
    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "offset" or parts[5] != "construct":
        raise ValueError("diagflat adjoint requires shape, offset, construct, and index metadata")
    if len(node.inputs) != 1:
        raise ValueError("diagflat adjoint primitive outputs must have one source input")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        int(parts[4])
        source_index = int(parts[6])
    except ValueError as exc:
        raise ValueError("diagflat adjoint metadata is malformed") from exc
    source_size = int(np.prod(shape, dtype=np.int64))
    if source_index < 0 or source_index >= source_size:
        raise ValueError("diagflat adjoint source index is outside flattened source shape")
    return ((node.inputs[0], 1.0),)


def _program_adjoint_parse_shape_label(label: str) -> tuple[int, ...]:
    """Parse static primitive shape metadata from compact IR labels."""
    if not label:
        raise ValueError("shape label must not be empty")
    shape = tuple(int(part) for part in label.split("x"))
    if any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions must be non-negative")
    return shape


def _program_adjoint_cumulative_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one compact cumulative output."""

    parts = node.op.split(":")
    try:
        if parts[0] in {"cumsum", "cumprod"}:
            if len(parts) != 7 or parts[1] != "shape" or parts[3] != "axis" or parts[5] != "out":
                raise ValueError
            source_shape = _program_adjoint_parse_shape_label(parts[2])
            axis = None if parts[4] == "flat" else int(parts[4])
            output_index = int(parts[6])
            output_shape = (
                (int(np.prod(source_shape, dtype=np.int64)),) if axis is None else source_shape
            )
            rule = (
                program_ad_cumulative_cumsum_derivative_rule(source_shape, axis=axis)
                if parts[0] == "cumsum"
                else program_ad_cumulative_cumprod_derivative_rule(source_shape, axis=axis)
            )
        elif parts[0] == "diff":
            if (
                len(parts) != 9
                or parts[1] != "shape"
                or parts[3] != "n"
                or parts[5] != "axis"
                or parts[7] != "out"
            ):
                raise ValueError
            source_shape = _program_adjoint_parse_shape_label(parts[2])
            order = int(parts[4])
            axis = int(parts[6])
            output_index = int(parts[8])
            if axis < 0 or axis >= len(source_shape):
                raise ValueError
            if order < 0 or order > source_shape[axis]:
                raise ValueError
            output_shape = (
                source_shape[:axis] + (source_shape[axis] - order,) + source_shape[axis + 1 :]
            )
            rule = program_ad_cumulative_diff_derivative_rule(source_shape, order=order, axis=axis)
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError("cumulative adjoint metadata is malformed") from exc

    expected_inputs = int(np.prod(source_shape, dtype=np.int64))
    if expected_inputs <= 0 or len(node.inputs) != expected_inputs:
        raise ValueError("cumulative adjoint inputs must match flattened source shape")
    output_size = int(np.prod(output_shape, dtype=np.int64))
    if output_size <= 0 or output_index < 0 or output_index >= output_size:
        raise ValueError("cumulative adjoint output index is outside output shape")
    if rule.vjp_rule is None:
        raise ValueError("cumulative adjoint requires a VJP rule")
    flat_values = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    )
    cotangent = np.zeros(output_size, dtype=np.float64)
    cotangent[output_index] = 1.0
    local_adjoint = np.asarray(rule.vjp_rule(flat_values, cotangent), dtype=np.float64).reshape(-1)
    return tuple(
        (name, float(value)) for name, value in zip(node.inputs, local_adjoint, strict=True)
    )


def _program_adjoint_matrix_power_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one matrix-power output node."""
    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "power":
        raise ValueError("matrix_power adjoint requires shape, power, and output-index metadata")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        exponent = int(parts[4])
        output_row = int(parts[5])
        output_col = int(parts[6])
    except ValueError as exc:
        raise ValueError("matrix_power adjoint metadata is malformed") from exc
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 0:
        raise ValueError("matrix_power adjoint requires non-empty square matrix metadata")
    rows, cols = shape
    if len(node.inputs) != rows * cols:
        raise ValueError("matrix_power adjoint inputs must contain one flattened square matrix")
    if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= cols:
        raise ValueError("matrix_power adjoint output index is outside matrix shape")
    flat_values = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    )
    cotangent = np.zeros((rows, cols), dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    rule = program_ad_linalg_matrix_power_derivative_rule(exponent)
    if rule.vjp_rule is None:
        raise ValueError("matrix_power adjoint requires a VJP rule")
    try:
        local_adjoint = np.asarray(
            rule.vjp_rule(flat_values, cotangent.reshape(-1)), dtype=np.float64
        ).reshape(-1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix_power adjoint requires a nonsingular matrix") from exc
    return tuple(
        (name, float(value)) for name, value in zip(node.inputs, local_adjoint, strict=True)
    )


def _program_adjoint_multi_dot_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one multi-dot output node."""
    parts = node.op.split(":")
    if len(parts) not in {5, 6} or parts[3] != "out":
        raise ValueError("multi_dot adjoint requires operand-shape and output metadata")
    try:
        operand_shapes = tuple(
            _program_adjoint_parse_shape_label(label) for label in parts[2].split("__")
        )
        output_shape, output_index = _program_adjoint_multi_dot_output_metadata(parts[4:])
    except ValueError as exc:
        raise ValueError("multi_dot adjoint metadata is malformed") from exc
    expected_inputs = sum(int(np.prod(shape, dtype=np.int64)) for shape in operand_shapes)
    if len(node.inputs) != expected_inputs:
        raise ValueError("multi_dot adjoint inputs must match flattened operand shapes")
    output_size = int(np.prod(output_shape, dtype=np.int64)) if output_shape else 1
    if output_index < 0 or output_index >= output_size:
        raise ValueError("multi_dot adjoint output index is outside result shape")
    flat_values = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    )
    cotangent = np.zeros(output_size, dtype=np.float64)
    cotangent[output_index] = 1.0
    rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    if rule.vjp_rule is None:
        raise ValueError("multi_dot adjoint requires a VJP rule")
    local_adjoint = np.asarray(rule.vjp_rule(flat_values, cotangent), dtype=np.float64).reshape(-1)
    return tuple(
        (name, float(value)) for name, value in zip(node.inputs, local_adjoint, strict=True)
    )


def _program_adjoint_multi_dot_output_metadata(parts: list[str]) -> tuple[tuple[int, ...], int]:
    """Parse multi-dot output shape and flat index metadata."""
    if len(parts) == 1 and parts[0] == "scalar":
        return (), 0
    if len(parts) != 2:
        raise ValueError("multi_dot output metadata must be scalar or shape plus index")
    shape = _program_adjoint_parse_shape_label(parts[0])
    if not shape:
        raise ValueError("multi_dot non-scalar output shape must not be empty")
    return shape, int(parts[1])


def _program_adjoint_eigvalsh_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a distinct symmetric eigvalsh node."""
    try:
        eigenvalue_index = int(node.op.rsplit(":", 1)[1])
    except ValueError as exc:
        raise ValueError("eigvalsh adjoint requires an eigenvalue index") from exc
    matrix_size = int(math.isqrt(len(node.inputs)))
    if matrix_size * matrix_size != len(node.inputs):
        raise ValueError("eigvalsh adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= matrix_size:
        raise ValueError("eigvalsh adjoint eigenvalue index is outside the spectrum")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(matrix_size, matrix_size)
    _program_ad_linalg_require_symmetric("eigvalsh adjoint replay", matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh adjoint replay")
    eigenvector = eigenvectors[:, eigenvalue_index]
    return tuple(
        (
            node.inputs[row * matrix_size + col],
            float(eigenvector[row] * eigenvector[col]),
        )
        for row in range(matrix_size)
        for col in range(matrix_size)
    )


def _program_adjoint_eigvals_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eigenvalue."""
    parts = node.op.split(":")
    if len(parts) != 4:
        raise ValueError("eigvals adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        eigenvalue_index = int(parts[3])
    except ValueError as exc:
        raise ValueError("eigvals adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eigvals adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eigvals adjoint eigenvalue index is outside the spectrum")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix(
            "eigvals adjoint replay", matrix
        )
    )
    local_adjoint = np.outer(
        left_eigenvector_rows[eigenvalue_index, :], right_eigenvectors[:, eigenvalue_index]
    )
    return tuple(
        (
            node.inputs[row * rows + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(rows)
    )


def _program_adjoint_eig_eigenvalue_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eig eigenvalue."""
    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
        ],
        _program_adjoint_eig_metadata(node, node_by_name, expect_eigenvector=False),
    )
    del matrix, _eigenvalues
    size = right_eigenvectors.shape[0]
    local_adjoint = np.outer(
        left_eigenvector_rows[eigenvalue_index, :], right_eigenvectors[:, eigenvalue_index]
    )
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eig_eigenvector_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eig eigenvector element."""
    matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index, row_index = (
        cast(
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                int,
                int,
            ],
            _program_adjoint_eig_metadata(node, node_by_name, expect_eigenvector=True),
        )
    )
    del matrix
    size = right_eigenvectors.shape[0]
    local_adjoint = np.zeros((size, size), dtype=np.float64)
    for tangent_row in range(size):
        for tangent_col in range(size):
            tangent_matrix = np.zeros((size, size), dtype=np.float64)
            tangent_matrix[tangent_row, tangent_col] = 1.0
            eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
                eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
            )
            local_adjoint[tangent_row, tangent_col] = eigenvector_tangent[
                row_index, eigenvalue_index
            ]
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eig_metadata(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
    *,
    expect_eigenvector: bool,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
        int,
    ]
):
    """Decode and replay general ``eig`` adjoint metadata from one IR node.

    Parses the shape-qualified ``linalg:eig:...`` operation label, replays the
    real-simple eigendecomposition of the flattened square input matrix, and
    returns the decomposition together with the requested eigenvalue index (and
    eigenvector row index when ``expect_eigenvector`` is set).

    Parameters
    ----------
    node
        Stabilised whole-program AD IR node for one ``eig`` output element.
    node_by_name
        Captured IR nodes keyed by name, used to resolve input primal values.
    expect_eigenvector
        When ``True`` the label and return tuple include an eigenvector row index.

    Returns
    -------
    tuple
        ``(matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows,
        eigenvalue_index)``, extended with ``row_index`` when
        ``expect_eigenvector`` is ``True``.

    Raises
    ------
    ValueError
        If the metadata is malformed or any index falls outside the spectrum or
        matrix.
    """
    parts = node.op.split(":")
    expected_length = 6 if expect_eigenvector else 5
    if len(parts) != expected_length:
        raise ValueError("eig adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[3].split("x", maxsplit=1))
        eigenvalue_index = int(parts[4])
        row_index = int(parts[5]) if expect_eigenvector else -1
    except ValueError as exc:
        raise ValueError("eig adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eig adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eig adjoint eigenvalue index is outside the spectrum")
    if expect_eigenvector and (row_index < 0 or row_index >= rows):
        raise ValueError("eig adjoint eigenvector row index is outside the matrix")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig adjoint replay", matrix)
    )
    if expect_eigenvector:
        return (
            matrix,
            eigenvalues,
            right_eigenvectors,
            left_eigenvector_rows,
            eigenvalue_index,
            row_index,
        )
    return matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index


def _program_adjoint_eigh_eigenvalue_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one symmetric eigh eigenvalue."""
    matrix, eigenvalues, eigenvectors, eigenvalue_index = cast(
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int],
        _program_adjoint_eigh_metadata(node, node_by_name, expect_eigenvector=False),
    )
    del matrix, eigenvalues
    eigenvector = eigenvectors[:, eigenvalue_index]
    size = eigenvectors.shape[0]
    return tuple(
        (
            node.inputs[row * size + col],
            float(eigenvector[row] * eigenvector[col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eigh_eigenvector_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one symmetric eigh eigenvector element."""
    matrix, eigenvalues, eigenvectors, eigenvalue_index, row_index = cast(
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int],
        _program_adjoint_eigh_metadata(node, node_by_name, expect_eigenvector=True),
    )
    del matrix
    size = eigenvectors.shape[0]
    cotangent = np.zeros_like(eigenvectors, dtype=np.float64)
    cotangent[row_index, eigenvalue_index] = 1.0
    local_adjoint = _program_ad_linalg_eigh_vjp_matrix(
        eigenvalues, eigenvectors, np.zeros(size, dtype=np.float64), cotangent
    )
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eigh_metadata(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
    *,
    expect_eigenvector: bool,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int]
):
    """Decode and replay symmetric ``eigh`` adjoint metadata from one IR node.

    Parses the shape- and UPLO-qualified ``linalg:eigh:...`` operation label,
    validates symmetry, replays ``numpy.linalg.eigh`` on the flattened square
    input matrix, requires a distinct spectrum, and returns the decomposition
    with the requested eigenvalue index (and eigenvector row index when
    ``expect_eigenvector`` is set).

    Parameters
    ----------
    node
        Stabilised whole-program AD IR node for one ``eigh`` output element.
    node_by_name
        Captured IR nodes keyed by name, used to resolve input primal values.
    expect_eigenvector
        When ``True`` the label and return tuple include an eigenvector row index.

    Returns
    -------
    tuple
        ``(matrix, eigenvalues, eigenvectors, eigenvalue_index)``, extended with
        ``row_index`` when ``expect_eigenvector`` is ``True``.

    Raises
    ------
    ValueError
        If the metadata is malformed, the matrix is not symmetric, the spectrum
        is degenerate, or any index is out of range.
    """
    parts = node.op.split(":")
    expected_length = 7 if expect_eigenvector else 6
    if len(parts) != expected_length:
        raise ValueError("eigh adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[3].split("x", maxsplit=1))
        uplo = _program_ad_linalg_uplo(parts[4], "eigh adjoint replay")
        eigenvalue_index = int(parts[5])
        row_index = int(parts[6]) if expect_eigenvector else -1
    except ValueError as exc:
        raise ValueError("eigh adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eigh adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eigh adjoint eigenvalue index is outside the spectrum")
    if expect_eigenvector and (row_index < 0 or row_index >= rows):
        raise ValueError("eigh adjoint eigenvector row index is outside the matrix")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    _program_ad_linalg_require_symmetric("eigh adjoint replay", matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigh adjoint replay")
    if expect_eigenvector:
        return matrix, eigenvalues, eigenvectors, eigenvalue_index, row_index
    return matrix, eigenvalues, eigenvectors, eigenvalue_index


def _program_adjoint_svdvals_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a distinct positive SVD singular value."""
    parts = node.op.split(":")
    if len(parts) != 4:
        raise ValueError("svd adjoint requires shape-qualified singular-value metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        singular_value_index = int(parts[3])
    except ValueError as exc:
        raise ValueError("svd adjoint metadata is malformed") from exc
    if rows <= 0 or cols <= 0 or rows * cols != len(node.inputs):
        raise ValueError("svd adjoint requires flattened matrix inputs matching metadata")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    _program_ad_linalg_require_distinct_positive_singular_values(
        singular_values, "svd adjoint replay"
    )
    if singular_value_index < 0 or singular_value_index >= singular_values.size:
        raise ValueError("svd adjoint singular-value index is outside the spectrum")
    return tuple(
        (
            node.inputs[row * cols + col],
            float(left[row, singular_value_index] * right_h[singular_value_index, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_pinv_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a constant-full-rank pinv element."""
    parts = node.op.split(":")
    if len(parts) != 6:
        raise ValueError("pinv adjoint requires shape, cutoff, and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        rcond = float(parts[3])
        output_row = int(parts[4])
        output_col = int(parts[5])
    except ValueError as exc:
        raise ValueError("pinv adjoint metadata is malformed") from exc
    if rows <= 0 or cols <= 0 or rows * cols != len(node.inputs):
        raise ValueError("pinv adjoint requires flattened matrix inputs matching metadata")
    if output_row < 0 or output_row >= cols or output_col < 0 or output_col >= rows:
        raise ValueError("pinv adjoint output index is outside pseudoinverse shape")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=rcond)
    cotangent = np.zeros_like(pinv, dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    local_adjoint = _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_node_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse-mode contributions for one captured IR node."""
    if node.op == "parameter":
        return ()
    if node.op.startswith("branch:"):
        return ()
    if node.op == "mutation:setitem":
        return ()
    if node.op.startswith("mutation:"):
        raise ValueError("mutation adjoints require alias/effect semantics")
    if node.op == "neg":
        return ((node.inputs[0], -1.0),)
    if node.op in {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
    }:
        arg_name = node.inputs[0]
        arg_value = _program_adjoint_input_value(arg_name, node_by_name)
        if node.op == "sin":
            return ((arg_name, float(np.cos(arg_value))),)
        if node.op == "cos":
            return ((arg_name, -float(np.sin(arg_value))),)
        if node.op == "exp":
            return ((arg_name, node.value),)
        if node.op == "expm1":
            return ((arg_name, float(np.exp(arg_value))),)
        if node.op == "log":
            return ((arg_name, 1.0 / arg_value),)
        if node.op == "log1p":
            if arg_value <= -1.0:
                raise ValueError("log1p adjoint requires input greater than -1")
            return ((arg_name, 1.0 / (1.0 + arg_value)),)
        if node.op == "sqrt":
            return ((arg_name, 1.0 / (2.0 * node.value)),)
        if node.op == "tan":
            cosine = float(np.cos(arg_value))
            if abs(cosine) <= 1.0e-15:
                raise ValueError("tan adjoint requires non-zero cosine")
            return ((arg_name, 1.0 / cosine**2),)
        if node.op == "tanh":
            return ((arg_name, 1.0 - node.value**2),)
        if node.op in {"arcsin", "arccos"}:
            if abs(arg_value) >= 1.0:
                raise ValueError(f"{node.op} adjoint requires input strictly inside (-1, 1)")
            scale = 1.0 / float(np.sqrt(1.0 - arg_value**2))
            if node.op == "arccos":
                scale = -scale
            return ((arg_name, scale),)
        if node.op == "reciprocal":
            if arg_value == 0.0:
                raise ValueError("reciprocal adjoint requires non-zero input")
            return ((arg_name, -1.0 / arg_value**2),)
        if node.op == "square":
            return ((arg_name, 2.0 * arg_value),)
        # ``abs`` is the final unary op in the guarding set, so reaching here
        # implies ``node.op == "abs"``.
        if arg_value == 0.0:
            raise ValueError("abs adjoint is undefined at zero")
        return ((arg_name, 1.0 if arg_value > 0.0 else -1.0),)
    if node.op in {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "where",
        "clip",
        "choose",
    }:
        return _program_adjoint_binary_or_selection_contributions(node, node_by_name)
    if node.op.startswith(("cumsum:", "cumprod:", "diff:")):
        return _program_adjoint_cumulative_contributions(node, node_by_name)
    if node.op.startswith("linalg:det:"):
        return _program_adjoint_det_contributions(node, node_by_name)
    if node.op.startswith("linalg:inv:"):
        return _program_adjoint_inv_contributions(node, node_by_name)
    if node.op.startswith("linalg:solve:"):
        return _program_adjoint_solve_contributions(node, node_by_name)
    if node.op.startswith("linalg:trace:"):
        return _program_adjoint_trace_contributions(node)
    if node.op.startswith("linalg:diag:"):
        return _program_adjoint_diag_contributions(node)
    if node.op.startswith("linalg:diagflat:"):
        return _program_adjoint_diagflat_contributions(node)
    if node.op.startswith("linalg:matrix_power:"):
        return _program_adjoint_matrix_power_contributions(node, node_by_name)
    if node.op.startswith("linalg:multi_dot:"):
        return _program_adjoint_multi_dot_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigh:eigenvalue:"):
        return _program_adjoint_eigh_eigenvalue_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigh:eigenvector:"):
        return _program_adjoint_eigh_eigenvector_contributions(node, node_by_name)
    if node.op.startswith("linalg:eig:eigenvalue:"):
        return _program_adjoint_eig_eigenvalue_contributions(node, node_by_name)
    if node.op.startswith("linalg:eig:eigenvector:"):
        return _program_adjoint_eig_eigenvector_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigvalsh:"):
        return _program_adjoint_eigvalsh_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigvals:"):
        return _program_adjoint_eigvals_contributions(node, node_by_name)
    if node.op.startswith("linalg:svdvals:"):
        return _program_adjoint_svdvals_contributions(node, node_by_name)
    if node.op.startswith("linalg:pinv:"):
        return _program_adjoint_pinv_contributions(node, node_by_name)
    raise ValueError(f"unsupported program AD adjoint op {node.op}")


def _program_adjoint_binary_or_selection_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a binary or selection primitive.

    Covers the elementwise binary operations (``add``, ``sub``, ``mul``,
    ``div``, ``pow``, ``maximum``, ``minimum``) and the selection primitives
    (``where``, ``clip``, ``choose``), routing each to its closed-form local
    partial derivatives. Ties, clipping boundaries, and non-positive ``pow``
    bases with a variable exponent fail closed with a ``ValueError``.
    """
    if node.op == "where":
        if len(node.inputs) != 3:
            raise ValueError("where adjoint requires predicate, true value, and false value")
        predicate_truth = _program_adjoint_where_predicate_truth(node.inputs[0])
        left_name = node.inputs[1]
        right_name = node.inputs[2]
        return ((left_name, 1.0),) if predicate_truth else ((right_name, 1.0),)
    if node.op == "clip":
        if len(node.inputs) != 3:
            raise ValueError("clip adjoint requires value, lower, and upper inputs")
        value_name, lower_name, upper_name = node.inputs
        value = _program_adjoint_input_value(value_name, node_by_name)
        lower = _program_adjoint_input_value(lower_name, node_by_name)
        upper = _program_adjoint_input_value(upper_name, node_by_name)
        if value < lower:
            return ((lower_name, 1.0),)
        if value > upper:
            return ((upper_name, 1.0),)
        if value in (lower, upper):
            raise ValueError("clip adjoint is undefined at clipping boundary")
        return ((value_name, 1.0),)
    if node.op == "choose":
        if len(node.inputs) != 2 or not node.inputs[0].startswith("static_selector:"):
            raise ValueError("choose adjoint requires static selector and selected value")
        return ((node.inputs[1], 1.0),)
    left_name = node.inputs[0]
    right_name = node.inputs[1] if len(node.inputs) > 1 else ""
    left = _program_adjoint_input_value(left_name, node_by_name)
    right = _program_adjoint_input_value(right_name, node_by_name) if right_name else 0.0
    if node.op == "add":
        return ((left_name, 1.0), (right_name, 1.0))
    if node.op == "sub":
        return ((left_name, 1.0), (right_name, -1.0))
    if node.op == "mul":
        return ((left_name, right), (right_name, left))
    if node.op == "div":
        return ((left_name, 1.0 / right), (right_name, -left / right**2))
    if node.op == "pow":
        if left <= 0.0 and _program_adjoint_is_ir_value(right_name):
            raise ValueError("variable exponent adjoint requires positive base")
        primal = node.value
        contributions = [(left_name, right * left ** (right - 1.0))]
        if _program_adjoint_is_ir_value(right_name):
            contributions.append((right_name, primal * float(np.log(left))))
        return tuple(contributions)
    if node.op == "maximum":
        if left == right:
            raise ValueError("maximum adjoint is undefined at ties")
        return ((left_name, 1.0),) if node.value == left else ((right_name, 1.0),)
    if node.op == "minimum":
        if left == right:
            raise ValueError("minimum adjoint is undefined at ties")
        return ((left_name, 1.0),) if node.value == left else ((right_name, 1.0),)
    raise ValueError(f"unsupported program AD adjoint op {node.op}")


def _program_adjoint_where_predicate_truth(predicate_name: str) -> bool:
    """Resolve a recorded ``where`` predicate token to its taken branch.

    Returns ``True`` when the predicate selected its true value and ``False``
    when it selected its false value, decoded from the ``:truth:1`` / ``:truth:0``
    suffix or a ``constant:True`` / ``constant:False`` token. An unrecorded
    predicate fails closed with a ``ValueError``.
    """
    if predicate_name.endswith(":truth:1"):
        return True
    if predicate_name.endswith(":truth:0"):
        return False
    if predicate_name == "constant:True":
        return True
    if predicate_name == "constant:False":
        return False
    raise ValueError("where adjoint requires recorded predicate branch")


def _program_adjoint_result_from_nodes(
    *,
    nodes: tuple[WholeProgramIRNode, ...],
    output_name: str,
    parameter_names: tuple[str, ...],
    trainable: tuple[bool, ...],
    program_ir: ProgramADEffectIR | None = None,
) -> ProgramADAdjointResult:
    """Generate reverse-mode adjoints over supported scalar Program AD IR nodes."""
    parameter_count = len(parameter_names)
    unsupported_ops: set[str] = {
        node.op
        for node in nodes
        if node.op.startswith("mutation:") and node.op != "mutation:setitem"
    }
    node_by_name = {f"%{node.index}": node for node in nodes}
    adjoints = {name: 0.0 for name in node_by_name}
    if output_name not in adjoints:
        unsupported_ops.add("output:not_in_ir")
    else:
        adjoints[output_name] = 1.0
        terminal_output_name = f"%{nodes[-1].index}" if nodes else ""
        if output_name != terminal_output_name:
            unsupported_ops.add("output:not_terminal_ir_node")
    for node in reversed(nodes):
        name = f"%{node.index}"
        cotangent = adjoints.get(name, 0.0)
        if cotangent == 0.0:
            continue
        try:
            contributions = _program_adjoint_node_contributions(node, node_by_name)
        except ValueError:
            unsupported_ops.add(node.op)
            continue
        for input_name, contribution in contributions:
            if input_name in adjoints:
                adjoints[input_name] += cotangent * contribution
    gradient = np.zeros(parameter_count, dtype=np.float64)
    for index, (name, trainable_flag) in enumerate(zip(parameter_names, trainable, strict=True)):
        if not trainable_flag:
            continue
        for node in nodes:
            if node.op == "parameter" and node.inputs == (name,):
                gradient[index] = adjoints.get(f"%{node.index}", 0.0)
                break
    supported = not unsupported_ops
    if not supported:
        gradient = np.zeros(parameter_count, dtype=np.float64)
    replay_effect_count = len(program_ir.effects) if program_ir is not None else 0
    replay_control_region_count = len(program_ir.control_regions) if program_ir is not None else 0
    replay_phi_node_count = len(program_ir.phi_nodes) if program_ir is not None else 0
    adjoint_steps = (
        _program_adjoint_steps_from_ir(
            nodes=nodes,
            node_by_name=node_by_name,
            program_ir=program_ir,
            cotangents=adjoints,
        )
        if program_ir is not None
        else ()
    )
    return ProgramADAdjointResult(
        gradient=gradient,
        supported=supported,
        unsupported_ops=tuple(sorted(unsupported_ops)),
        method="program_adjoint_ir_generation",
        claim_boundary=(
            "reverse-mode adjoint generation over stabilized program_ad_effect_ir.v1 "
            "for supported executed scalar Program AD operations; unsupported operations "
            "fail closed without substituting finite differences or forward tangents; "
            "no non-executed branch adjoints or executable Rust/LLVM/JIT lowering claim"
        ),
        replay_node_count=len(nodes),
        replay_effect_count=replay_effect_count,
        replay_control_region_count=replay_control_region_count,
        replay_phi_node_count=replay_phi_node_count,
        executed_branch_replay_count=sum(
            1
            for step in adjoint_steps
            if step.operation.startswith("branch:")
            and step.control_region_kind == "runtime_branch"
            and step.phi_node is not None
            and step.phi_selected is not None
        ),
        blocked_non_executed_phi_input_count=sum(
            len(step.non_executed_phi_inputs) for step in adjoint_steps
        ),
        replay_ir_format="program_ad_effect_ir.v1",
        adjoint_steps=adjoint_steps,
    )


def _program_adjoint_steps_from_ir(
    *,
    nodes: tuple[WholeProgramIRNode, ...],
    node_by_name: Mapping[str, WholeProgramIRNode],
    program_ir: ProgramADEffectIR,
    cotangents: Mapping[str, float],
) -> tuple[ProgramADAdjointStep, ...]:
    """Generate reverse-adjoint steps from stabilized Program AD IR metadata."""
    ssa_by_name = {value.name: value for value in program_ir.ssa_values}
    effect_by_index = {effect.index: effect for effect in program_ir.effects}
    runtime_regions_by_predicate: dict[str, list[ProgramADControlRegion]] = {}
    for region in program_ir.control_regions:
        if region.source_line is None and region.predicate is not None:
            runtime_regions_by_predicate.setdefault(region.predicate, []).append(region)
    runtime_phi_by_region: dict[int, ProgramADPhiNode] = {}
    ambiguous_phi_regions: set[int] = set()
    for phi_node in program_ir.phi_nodes:
        if phi_node.source_line is not None or phi_node.control_region is None:
            continue
        if phi_node.control_region in runtime_phi_by_region:
            ambiguous_phi_regions.add(phi_node.control_region)
            runtime_phi_by_region.pop(phi_node.control_region, None)
        elif phi_node.control_region not in ambiguous_phi_regions:
            runtime_phi_by_region[phi_node.control_region] = phi_node
    steps: list[ProgramADAdjointStep] = []
    for node in reversed(nodes):
        primal_value = f"%{node.index}"
        ssa_value = ssa_by_name.get(primal_value)
        primal_effect = None if ssa_value is None else ssa_value.effect
        effect = None if primal_effect is None else effect_by_index.get(primal_effect)
        effect_kind = None if effect is None else effect.kind
        effect_version = None if effect is None else effect.version
        effect_ordering = None if effect is None else effect.ordering
        unsupported_reason: str | None = None
        supported = True
        contribution_inputs: tuple[str, ...] = ()
        incoming_cotangent = float(cotangents.get(primal_value, 0.0))
        contribution_scales: tuple[float, ...] = ()
        contribution_cotangents: tuple[float, ...] = ()
        control_region: int | None = None
        control_region_kind: str | None = None
        control_region_entered: bool | None = None
        phi_node_index: int | None = None
        phi_selected: str | None = None
        non_executed_phi_inputs: tuple[str, ...] = ()
        if node.op.startswith("branch:"):
            runtime_regions = tuple(runtime_regions_by_predicate.get(node.op, ()))
            if len(runtime_regions) == 1:
                runtime_region = runtime_regions[0]
                control_region = runtime_region.index
                control_region_kind = runtime_region.kind
                control_region_entered = runtime_region.entered
                runtime_phi = runtime_phi_by_region.get(runtime_region.index)
                if runtime_phi is not None:
                    phi_node_index = runtime_phi.index
                    phi_selected = runtime_phi.selected
                    if runtime_phi.selected is not None:
                        non_executed_phi_inputs = tuple(
                            incoming
                            for incoming in runtime_phi.incoming
                            if incoming != runtime_phi.selected
                        )
        if ssa_value is None:
            supported = False
            unsupported_reason = "missing_ssa_value"
        elif primal_effect is not None and effect is None:
            supported = False
            unsupported_reason = "missing_effect"
        else:
            try:
                contributions = _program_adjoint_node_contributions(node, node_by_name)
            except ValueError as exc:
                supported = False
                unsupported_reason = str(exc)
            else:
                scale_by_input: dict[str, float] = {}
                for input_name, scale in contributions:
                    scale_by_input[input_name] = scale_by_input.get(input_name, 0.0) + scale
                contribution_inputs = tuple(sorted(scale_by_input))
                contribution_scales = tuple(
                    scale_by_input[input_name] for input_name in contribution_inputs
                )
                contribution_cotangents = tuple(
                    incoming_cotangent * scale for scale in contribution_scales
                )
        steps.append(
            ProgramADAdjointStep(
                index=len(steps),
                primal_value=primal_value,
                primal_effect=primal_effect,
                effect_kind=effect_kind,
                effect_version=effect_version,
                effect_ordering=effect_ordering,
                control_region=control_region,
                control_region_kind=control_region_kind,
                control_region_entered=control_region_entered,
                phi_node=phi_node_index,
                phi_selected=phi_selected,
                non_executed_phi_inputs=non_executed_phi_inputs,
                operation=node.op,
                input_values=node.inputs,
                contribution_inputs=contribution_inputs,
                incoming_cotangent=incoming_cotangent,
                contribution_scales=contribution_scales,
                contribution_cotangents=contribution_cotangents,
                supported=supported,
                unsupported_reason=unsupported_reason,
            )
        )
    return tuple(steps)
