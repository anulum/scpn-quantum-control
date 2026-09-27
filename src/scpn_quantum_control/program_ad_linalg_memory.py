# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Linear-algebra execution storage declarations
"""Admission scopes for actual Program AD linear-algebra callbacks.

Declarations cover visible NumPy storage and algorithmic workspaces. They are
not measurements of allocator peaks or vendor LAPACK internals.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory


@contextmanager
def matrix_power_execution_scope(
    values: NDArray[np.float64],
    exponent: int,
    *,
    operand: NDArray[np.float64] | None = None,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit matrix-power storage before numeric conversion and materialisation.

    Parameters
    ----------
    values
        Plain real numeric array containing a flattened square matrix.
    exponent
        Fixed integer power. Derivatives retain ``abs(exponent)`` matrices.
    operand
        Tangent or cotangent array, or absent for forward value evaluation.

    Yields
    ------
    ExecutionMemoryReservation
        Active owner for conversions, retained powers and numeric workspaces.

    Raises
    ------
    ValueError
        Input metadata is not a plain real numeric square matrix or numeric
        primal/derivative inputs are non-finite.
    DenseAllocationError
        Declared storage exceeds native addressability or current capacity.

    Notes
    -----
    Input declarations include source storage, real-array conversion and
    finite-validation masks; data inspection follows capacity admission.
    Eight matrix buffers cover inverse, transformed operand, accumulator, two
    products, replacement accumulator and output conversions. Four further
    buffers declare binary matrix-power multiplication workspace. LAPACK's
    implementation-specific internal allocations are not measured here.
    Cancellation and deadlines are inherited and checked at entry and exit;
    callers also checkpoint between repeated matrix operations.

    """
    inputs = (values,) if operand is None else (values, operand)
    buffers: list[ExecutionBuffer] = []
    for index, array in enumerate(inputs):
        if type(array) is not np.ndarray or array.dtype.kind not in "iuf":
            raise ValueError("program AD matrix_power requires plain real numeric arrays")
        buffers.append(
            ExecutionBuffer(
                f"matrix_power_input_{index}",
                "forward",
                (max(1, int(array.size)), max(8, array.dtype.itemsize)),
                "uint8",
                2,
            )
        )
    size = int(values.size)
    rows = math.isqrt(size)
    if rows * rows != size:
        raise ValueError(
            "program AD linalg matrix_power direct rule requires a flattened square matrix"
        )
    buffers.append(
        ExecutionBuffer("matrix_power_workspace", "intermediate", (max(1, size),), "float64", 12)
    )
    if operand is not None and exponent != 0:
        count = abs(exponent)
        buffers.extend(
            (
                ExecutionBuffer(
                    "matrix_power_retained_powers", "adjoint", (max(1, size),), "float64", count
                ),
                ExecutionBuffer("matrix_power_power_references", "adjoint", (count,), "intp"),
            )
        )
    for index, array in enumerate(inputs):
        buffers.append(
            ExecutionBuffer(
                f"matrix_power_validation_{index}",
                "intermediate",
                (max(1, int(array.size)),),
                "bool",
                2,
            )
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("matrix_power", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def linalg_pullback_execution_scope(
    matrix_shape: tuple[int, int], *, rhs_shape: tuple[int, ...] | None = None
) -> Iterator[ExecutionMemoryReservation]:
    """Admit visible inverse or solve pullback storage before constructing arrays.

    Parameters
    ----------
    matrix_shape
        Square matrix shape already validated by the actual adjoint callback.
    rhs_shape
        Vector or matrix RHS shape for solve, or absent for inverse.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for numeric intermediates and input list/name references.

    Raises
    ------
    DenseAllocationError
        Declared buffers exceed native addressability or current capacity.
    ValueError
        Buffer shape metadata cannot describe positive fixed-width storage.

    Notes
    -----
    Inverse declares matrix, inverse, cotangent, two products and negated
    output. Solve declares matrix and signed/unsigned matrix-adjoint outputs,
    plus RHS, solution, cotangent and RHS adjoint. Input metadata includes
    boxed float conversion and two reference slots for lists and name slices.
    These declarations do not measure vendor LAPACK workspace. Lifecycle
    checkpoints are inherited; callers checkpoint after each native operation.

    """
    matrix_size = math.prod(matrix_shape)
    input_size = matrix_size + (math.prod(rhs_shape) if rhs_shape is not None else 0)
    buffers = [
        ExecutionBuffer(
            "pullback_input_metadata",
            "intermediate",
            (max(1, input_size), sys.getsizeof(0.0) + 2 * np.dtype(np.uintp).itemsize),
            "uint8",
        ),
        ExecutionBuffer(
            "pullback_matrix_buffers",
            "adjoint",
            matrix_shape,
            "float64",
            6 if rhs_shape is None else 3,
        ),
    ]
    if rhs_shape is not None:
        buffers.append(ExecutionBuffer("pullback_rhs_buffers", "adjoint", rhs_shape, "float64", 4))
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def determinant_cofactor_execution_scope(
    shape: tuple[int, int],
) -> Iterator[ExecutionMemoryReservation]:
    """Admit source, cofactor and visible minor-delete storage before allocation.

    Parameters
    ----------
    shape
        Square matrix shape validated by the cofactor callback, including empty.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for source/output matrices, two minors and delete masks.

    Raises
    ------
    DenseAllocationError
        Native addressability or current capacity refuses the declared buffers.

    Notes
    -----
    Nested row/column deletion can hold an (n-1)-by-n intermediate and an
    (n-1)-by-(n-1) minor simultaneously. Old/new column minors can overlap
    during replacement, so that minor has two slots. Two length-n boolean
    masks cover their selectors. Vendor determinant workspace is not measured here.

    """
    rows, cols = shape
    buffers = [
        ExecutionBuffer(
            "cofactor_source_output", "adjoint", (max(1, rows), max(1, cols)), "float64", 2
        )
    ]
    if rows > 1:
        buffers.extend(
            (
                ExecutionBuffer("cofactor_row_minor", "intermediate", (rows - 1, cols), "float64"),
                ExecutionBuffer(
                    "cofactor_column_minor", "intermediate", (rows - 1, cols - 1), "float64", 2
                ),
                ExecutionBuffer("cofactor_delete_masks", "intermediate", (rows,), "bool", 2),
            )
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def determinant_execution_scope(
    values: NDArray[np.float64],
    *,
    operand: NDArray[np.float64] | None = None,
    pullback: bool = False,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit determinant value/JVP/VJP conversions and outputs before NumPy.

    Parameters
    ----------
    values
        Plain numeric array holding a flattened square matrix.
    operand
        Tangent or scalar cotangent, absent for the value callback.
    pullback
        True for a VJP's matrix-sized output, false for a scalar value/JVP.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for input/conversion, derivative product and output storage.

    Raises
    ------
    ValueError
        Input metadata is not a plain fixed-width real numeric array.
    DenseAllocationError
        Declared storage exceeds native addressability or current capacity.

    Notes
    -----
    The actual cofactor callback separately owns its source/output and minor
    storage. These scopes may conservatively overlap source charges and do
    not measure LAPACK internals or retained returned arrays.

    """
    buffers = _real_linalg_input_buffers("det", values, operand)
    output_size = max(1, int(values.size)) if pullback else 1
    buffers.append(ExecutionBuffer("det_output", "dense_output", (output_size,), "float64", 2))
    if operand is not None:
        buffers.append(
            ExecutionBuffer("det_product", "intermediate", (max(1, int(values.size)),), "float64")
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("det", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


def _real_linalg_input_buffers(
    primitive: str, values: NDArray[np.float64], operand: NDArray[np.float64] | None
) -> list[ExecutionBuffer]:
    """Declare inspected real source/conversion storage and finite-validation masks."""
    inputs = (values,) if operand is None else (values, operand)
    buffers: list[ExecutionBuffer] = []
    for index, array in enumerate(inputs):
        if type(array) is not np.ndarray or array.dtype.kind not in "iuf":
            raise ValueError(f"program AD {primitive} requires plain real numeric arrays")
        buffers.extend(
            (
                ExecutionBuffer(
                    f"{primitive}_input_{index}",
                    "forward",
                    (max(1, int(array.size)), max(8, array.dtype.itemsize)),
                    "uint8",
                    2,
                ),
                ExecutionBuffer(
                    f"{primitive}_input_validation_{index}",
                    "intermediate",
                    (max(1, int(array.size)),),
                    "bool",
                    2,
                ),
            )
        )
    return buffers


def _require_finite_linalg_inputs(
    primitive: str, values: NDArray[np.float64], operand: NDArray[np.float64] | None
) -> None:
    """Refuse non-finite inputs only inside their already admitted validation scope."""
    for array in (values,) if operand is None else (values, operand):
        if np.any(~np.isfinite(array)):
            raise ValueError(f"program AD {primitive} inputs must be finite")


@contextmanager
def inverse_execution_scope(
    values: NDArray[np.float64], *, operand: NDArray[np.float64] | None = None
) -> Iterator[ExecutionMemoryReservation]:
    """Admit inverse value/JVP/VJP conversions, products and output copies.

    Parameters
    ----------
    values
        Plain real numeric array representing a flattened square matrix.
    operand
        Tangent or cotangent matrix, or absent for value evaluation.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for input/conversion/mask buffers and inverse results.

    Raises
    ------
    ValueError
        Input is opaque, non-real or non-finite.
    DenseAllocationError
        Native addressability or current capacity refuses declared storage.

    Notes
    -----
    Value declares inverse and output copy. Derivatives declare inverse, two
    products, negated result and output copy. Vendor LAPACK workspace and
    retention of returned arrays are not measured by this scope.

    """
    buffers = _real_linalg_input_buffers("inv", values, operand)
    buffers.append(
        ExecutionBuffer(
            "inv_outputs",
            "dense_output",
            (max(1, int(values.size)),),
            "float64",
            2 if operand is None else 5,
        )
    )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("inv", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def solve_execution_scope(
    values: NDArray[np.float64],
    *,
    operand: NDArray[np.float64] | None = None,
    shapes: tuple[tuple[int, int], tuple[int, ...]] | None = None,
    pullback: bool = False,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit solve callback input, intermediate and output storage before NumPy.

    Parameters
    ----------
    values
        Plain real flattened matrix/RHS array.
    operand
        Input tangent or output cotangent, or absent for value evaluation.
    shapes
        Validated fixed matrix/RHS shapes, or absent for legacy vector RHS.
    pullback
        True for a VJP's matrix/RHS gradient, false for value or JVP.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for conversions, validation masks, solve results and products.

    Raises
    ------
    ValueError
        Numeric inputs are opaque, non-real or non-finite.
    DenseAllocationError
        Native addressability or current capacity refuses declared storage.

    Notes
    -----
    Value owns solution and copy; JVP owns solution, product, differential,
    tangent solution and copy. VJP owns solution/RHS-adjoint buffers, signed
    and unsigned matrix-adjoint buffers, and concatenated gradient. Empty
    shapes retain zero numeric extents with a minimum declaration slot.
    Vendor solve workspace and returned-array retention are not measured.

    """
    buffers = _real_linalg_input_buffers("solve", values, operand)
    if shapes is None:
        rows = (math.isqrt(1 + 4 * int(values.size)) - 1) // 2
        matrix_size, rhs_size = rows * rows, rows
    else:
        matrix_shape, rhs_shape = shapes
        matrix_size, rhs_size = math.prod(matrix_shape), math.prod(rhs_shape)
    buffers.append(
        ExecutionBuffer(
            "solve_rhs_workspace",
            "intermediate",
            (max(1, rhs_size),),
            "float64",
            2 if operand is None or pullback else 5,
        )
    )
    if pullback:
        buffers.extend(
            (
                ExecutionBuffer(
                    "solve_matrix_adjoint", "adjoint", (max(1, matrix_size),), "float64", 2
                ),
                ExecutionBuffer(
                    "solve_gradient", "dense_output", (max(1, matrix_size + rhs_size),), "float64"
                ),
            )
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("solve", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def matrix_power_trace_execution_scope(
    shape: tuple[int, int],
    parameter_count: int,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit visible matrix-power trace storage before building numeric arrays.

    Parameters
    ----------
    shape
        Validated static matrix dimensions carried by the public trace array.
    parameter_count
        Number of derivative coordinates in the active trace context.

    Yields
    ------
    ExecutionMemoryReservation
        Temporary numeric, reference and array-metadata storage owner.

    Raises
    ------
    DenseAllocationError
        Native addressability, current capacity or inherited lifecycle refuses.

    Notes
    -----
    Two matrix buffers declare flattened primal and returned value. Three
    matrix-by-parameter buffers declare input tangents, per-parameter JVP
    results and their stacked copy. References include construction lists,
    input names and retained JVP array pointers. Array headers, shape/stride
    storage and list headers are explicit; output trace-container storage is
    separately handed off by the caller. Each actual value/JVP callback owns
    its numerical algorithm workspace in a nested reservation. Vendor scratch
    and full allocator peaks remain separate qualification requirements.

    """
    matrix_size = max(1, math.prod(shape))
    parameter_slots = max(1, parameter_count)
    buffers = (
        ExecutionBuffer("trace_matrix_power_values", "forward", (matrix_size,), "float64", 2),
        ExecutionBuffer(
            "trace_matrix_power_tangents",
            "intermediate",
            (matrix_size, parameter_slots),
            "float64",
            3,
        ),
        ExecutionBuffer("trace_matrix_power_validation", "intermediate", (matrix_size,), "bool"),
        ExecutionBuffer(
            "trace_matrix_power_input_references", "intermediate", (matrix_size,), "uintp", 4
        ),
        ExecutionBuffer(
            "trace_matrix_power_result_references", "intermediate", (parameter_slots,), "uintp", 2
        ),
        ExecutionBuffer(
            "trace_matrix_power_array_metadata",
            "intermediate",
            (
                parameter_slots + 6,
                np.ndarray.__basicsize__ + 4 * np.dtype(np.uintp).itemsize,
            ),
            "uint8",
        ),
        ExecutionBuffer(
            "trace_matrix_power_list_headers", "intermediate", (sys.getsizeof([]),), "uint8", 3
        ),
    )
    with reserve_execution_memory(ExecutionMemoryPlan(buffers)) as reservation:
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def multi_dot_execution_scope(
    values: NDArray[np.float64],
    shapes: tuple[tuple[int, ...], ...],
    *,
    operand: NDArray[np.float64] | None = None,
    pullback: bool = False,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit visible matrix-chain storage before splitting numeric operands.

    Parameters
    ----------
    values
        Plain real flattened operand data.
    shapes
        Validated aligned vector/matrix operand shapes.
    operand
        Input tangent or output cotangent, absent for value evaluation.
    pullback
        Declare retained operand adjoints and basis buffers for a VJP.

    Yields
    ------
    ExecutionMemoryReservation
        Owner for conversion, chain intermediates and derivative workspace.

    Raises
    ------
    ValueError
        Numeric arrays are opaque, non-real or non-finite.
    DenseAllocationError
        Declared storage exceeds native addressability or current capacity.

    Notes
    -----
    Every possible contiguous subchain output is bounded by its two boundary
    dimensions, including promoted vector endpoints. Twice the operand count
    bounds simultaneous chain intermediates; matrix-chain planning tables and
    views are explicit. JVP accumulation and VJP basis/retained/concatenated
    gradients are declared separately. Products use Python checked integers,
    not native NumPy dimension arithmetic. Vendor BLAS and allocator peaks
    are not measurements supplied by this scope.

    """
    input_size = sum(math.prod(shape) for shape in shapes)
    buffers = _real_linalg_input_buffers("multi_dot", values, operand)
    buffers.extend(_multi_dot_workspace_buffers(shapes))
    if pullback:
        buffers.extend(
            (
                ExecutionBuffer(
                    "multi_dot_basis",
                    "intermediate",
                    (max(math.prod(shape) for shape in shapes),),
                    "float64",
                    2,
                ),
                ExecutionBuffer("multi_dot_adjoints", "adjoint", (input_size,), "float64", 2),
            )
        )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("multi_dot", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


def _multi_dot_workspace_buffers(
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[ExecutionBuffer, ...]:
    """Declare chain outputs, planning tables and visible view/header storage."""
    dimensions = (1 if len(shapes[0]) == 1 else shapes[0][0], shapes[0][-1]) + tuple(
        1 if len(shape) == 1 else shape[-1] for shape in shapes[1:]
    )
    maximum_left = dimensions[0]
    maximum_product = 1
    for right in dimensions[1:]:
        maximum_product = max(maximum_product, maximum_left * right)
        maximum_left = max(maximum_left, right)
    output_size = dimensions[0] * dimensions[-1]
    return tuple(
        (
            ExecutionBuffer(
                "multi_dot_chain_workspace",
                "intermediate",
                (maximum_product,),
                "float64",
                2 * len(shapes),
            ),
            ExecutionBuffer(
                "multi_dot_planning_tables",
                "intermediate",
                (len(shapes), len(shapes)),
                "float64",
                2,
            ),
            ExecutionBuffer("multi_dot_outputs", "dense_output", (output_size,), "float64", 6),
            ExecutionBuffer(
                "multi_dot_operand_views",
                "intermediate",
                (len(shapes), np.ndarray.__basicsize__ + 4 * np.dtype(np.uintp).itemsize),
                "uint8",
                8,
            ),
            ExecutionBuffer(
                "multi_dot_temporary_headers",
                "intermediate",
                (np.ndarray.__basicsize__ + 4 * np.dtype(np.uintp).itemsize,),
                "uint8",
                16,
            ),
            ExecutionBuffer(
                "multi_dot_operand_references", "intermediate", (len(shapes),), "uintp", 8
            ),
        )
    )


@contextmanager
def multi_dot_trace_execution_scope(
    shapes: tuple[tuple[int, ...], ...],
    parameter_count: int,
) -> Iterator[ExecutionMemoryReservation]:
    """Own numeric materialisation and stacked derivatives of an actual matrix chain.

    Parameters
    ----------
    shapes
        Validated aligned trace operand shapes.
    parameter_count
        Number of derivative coordinates in the active trace.

    Yields
    ------
    ExecutionMemoryReservation
        Temporary primal, tangent, JVP and matrix-chain workspace owner.

    Raises
    ------
    DenseAllocationError
        Current capacity, native addressability or inherited lifecycle refuses.

    Notes
    -----
    Primal operand arrays and their flat concatenation coexist. Tangent stacks
    coexist with their flat concatenation, and retained JVP results coexist
    with their stacked copy. Nested callbacks own algorithm workspaces while
    output trace containers hand off separately to the active context.
    Vendor BLAS and full allocator peaks are not measured by this declaration.

    """
    input_size = sum(math.prod(shape) for shape in shapes)
    rows = 1 if len(shapes[0]) == 1 else shapes[0][0]
    cols = 1 if len(shapes[-1]) == 1 else shapes[-1][-1]
    output_size = rows * cols
    parameters = max(1, parameter_count)
    plan = ExecutionMemoryPlan(
        (
            *_multi_dot_workspace_buffers(shapes),
            ExecutionBuffer("trace_multi_dot_values", "forward", (input_size,), "float64", 2),
            ExecutionBuffer(
                "trace_multi_dot_input_tangents",
                "intermediate",
                (input_size, parameters),
                "float64",
                2,
            ),
            ExecutionBuffer(
                "trace_multi_dot_output_tangents",
                "intermediate",
                (output_size, parameters),
                "float64",
                2,
            ),
            ExecutionBuffer("trace_multi_dot_validation", "intermediate", (output_size,), "bool"),
            ExecutionBuffer(
                "trace_multi_dot_input_references", "intermediate", (input_size,), "uintp", 4
            ),
            ExecutionBuffer(
                "trace_multi_dot_jvp_headers",
                "intermediate",
                (parameters, np.ndarray.__basicsize__ + 4 * np.dtype(np.uintp).itemsize),
                "uint8",
            ),
            ExecutionBuffer(
                "trace_multi_dot_jvp_references", "intermediate", (parameters,), "uintp", 2
            ),
        )
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()


@contextmanager
def diagonal_execution_scope(
    values: NDArray[np.float64],
    source_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    operand: NDArray[np.float64] | None = None,
) -> Iterator[ExecutionMemoryReservation]:
    """Admit visible diagonal construction, extraction and pullback storage.

    Parameters
    ----------
    values
        Plain real source input, or tangent for a primal-independent linear JVP.
    source_shape
        Validated source dimensions.
    output_shape
        Metadata-only declared output dimensions.
    operand
        Optional input tangent or output cotangent.

    Yields
    ------
    ExecutionMemoryReservation
        Conversion, validation, source/output and coordinate-metadata owner.

    Raises
    ------
    ValueError
        Numeric data is opaque, non-real or non-finite.
    DenseAllocationError
        Checked source/output storage exceeds addressability or current capacity.

    Notes
    -----
    Source/output slots cover constructed diagonal matrices, reverse zero
    arrays and returned flat copies/views. Coordinate tuples are bounded by
    source dimension integer sizes and built only inside the admitted callback.
    General allocator peaks remain outside these visible storage declarations.

    """
    source_size, output_size = math.prod(source_shape), math.prod(output_shape)
    coordinate_count = min(source_size, output_size)
    coordinate_width = (
        sys.getsizeof((0, 0))
        + 2 * sys.getsizeof(max(source_shape, default=0))
        + np.dtype(np.uintp).itemsize
    )
    buffers = _real_linalg_input_buffers("diagonal", values, operand)
    header_rank = max(
        2,
        len(source_shape),
        len(output_shape),
        values.ndim,
        0 if operand is None else operand.ndim,
    )
    buffers.extend(
        (
            ExecutionBuffer(
                "diagonal_source_workspace", "intermediate", (max(1, source_size),), "float64", 3
            ),
            ExecutionBuffer(
                "diagonal_output_workspace", "dense_output", (max(1, output_size),), "float64", 3
            ),
            ExecutionBuffer(
                "diagonal_coordinate_metadata",
                "intermediate",
                (max(1, coordinate_count), coordinate_width),
                "uint8",
            ),
            ExecutionBuffer(
                "diagonal_array_headers",
                "intermediate",
                (np.ndarray.__basicsize__ + 2 * header_rank * np.dtype(np.uintp).itemsize,),
                "uint8",
                8,
            ),
        )
    )
    with reserve_execution_memory(ExecutionMemoryPlan(tuple(buffers))) as reservation:
        reservation.checkpoint()
        _require_finite_linalg_inputs("diagonal", values, operand)
        reservation.checkpoint()
        yield reservation
        reservation.checkpoint()
