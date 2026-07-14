# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value shape tests
# scpn-quantum-control -- trace-value shape and assembly production contracts
"""Real-surface contracts for trace-value shape and assembly operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import (
    TraceADArray,
    TraceADScalar,
    WholeProgramADResult,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]
TraceObjective = Callable[[TraceADArray], object]
ArrayFunction = Callable[..., object]


def _differentiate(
    objective: TraceObjective,
    values: FloatArray,
) -> WholeProgramADResult:
    """Execute one shape contract through the public whole-program API."""

    def public_objective(raw_values: object) -> object:
        return objective(cast(TraceADArray, raw_values))

    return whole_program_value_and_grad(public_objective, values, trace=False)


def _capture_trace_array(values: FloatArray | None = None) -> TraceADArray:
    """Capture the production trace array injected by the public AD API."""
    captured: list[TraceADArray] = []
    parameters = np.arange(1.0, 25.0, dtype=np.float64) if values is None else values

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(TraceADScalar, trace_values[0])

    _differentiate(objective, parameters)
    assert len(captured) == 1
    return captured[0]


def _dispatch(
    trace_values: TraceADArray,
    function: ArrayFunction,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    """Invoke one public NumPy array-function protocol contract."""
    return trace_values.__array_function__(
        function,
        (TraceADArray,),
        args,
        {} if kwargs is None else kwargs,
    )


def _shape(value: object) -> tuple[int, ...]:
    """Return the shape of one traced array result."""
    return cast(TraceADArray, value).shape


def _primals(value: object) -> tuple[float, ...]:
    """Return flattened primal values from one traced array result."""
    array = cast(TraceADArray, value)
    return tuple(cast(TraceADScalar, item).primal for item in array.flatten())


def test_atleast_promotions_cover_every_supported_input_rank() -> None:
    """At-least transforms should match NumPy's scalar-through-ranked shapes."""
    values = _capture_trace_array()
    scalar = TraceADArray((cast(TraceADScalar, values[0]),), (), values.context)
    vector = values[:2]
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))
    cube = cast(TraceADArray, values[:8]).reshape((2, 2, 2))

    assert _shape(_dispatch(values, np.atleast_1d, (scalar,))) == (1,)
    assert _shape(_dispatch(values, np.atleast_1d, (matrix,))) == (2, 2)
    assert _shape(_dispatch(values, np.atleast_2d, (scalar,))) == (1, 1)
    assert _shape(_dispatch(values, np.atleast_2d, (vector,))) == (1, 2)
    assert _shape(_dispatch(values, np.atleast_2d, (matrix,))) == (2, 2)
    assert _shape(_dispatch(values, np.atleast_3d, (scalar,))) == (1, 1, 1)
    assert _shape(_dispatch(values, np.atleast_3d, (vector,))) == (1, 2, 1)
    assert _shape(_dispatch(values, np.atleast_3d, (matrix,))) == (2, 2, 1)
    assert _shape(_dispatch(values, np.atleast_3d, (cube,))) == (2, 2, 2)


def test_atleast_multiple_operands_and_low_rank_transpose_preserve_values() -> None:
    """Multi-operand promotion and low-rank transpose should preserve trace values."""
    values = _capture_trace_array(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    scalar = TraceADArray((cast(TraceADScalar, values[0]),), (), values.context)
    promoted = cast(
        list[TraceADArray],
        _dispatch(values, np.atleast_2d, (scalar, values)),
    )

    assert [array.shape for array in promoted] == [(1, 1), (1, 3)]
    assert _shape(_dispatch(values, np.transpose, (scalar,))) == ()
    assert _shape(_dispatch(values, np.transpose, (values,))) == (3,)
    assert _primals(_dispatch(values, np.transpose, (values,))) == (1.0, 2.0, 3.0)


def test_like_constructors_validate_keyword_and_dtype_contracts() -> None:
    """Like constructors should accept real dtypes and reject shape or complex drift."""
    values = _capture_trace_array(np.array([1.0, 2.0], dtype=np.float64))

    created = cast(
        TraceADArray,
        _dispatch(values, np.ones_like, (values,), {"dtype": np.float32}),
    )
    assert created.shape == (2,)
    assert _primals(created) == (1.0, 1.0)

    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"shape": (1,)}, "do not support shape overrides"),
        ({"unsupported": True}, "support dtype, order, and subok only"),
        ({"dtype": np.complex128}, "require real numeric dtype"),
    )
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            _dispatch(values, np.zeros_like, (values,), kwargs)


def test_basic_indexing_rejects_out_of_bounds_and_dynamic_selectors() -> None:
    """Basic indexing should fail closed for invalid bounds and dynamic selectors."""
    values = _capture_trace_array(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    with pytest.raises(ValueError, match="in-bounds indices"):
        _ = values[9]

    invalid_selectors: tuple[object, ...] = (
        values[0],
        True,
        slice(None, 1.5),
        np.array([1.5]),
        np.array(True),
        [values[0]],
        object(),
    )
    for selector in invalid_selectors:
        with pytest.raises(ValueError, match="static integer"):
            _ = values[selector]


def test_singleton_axis_sort_and_squeeze_contracts() -> None:
    """Singleton axes should sort and squeeze without introducing boundaries."""
    values = _capture_trace_array(np.array([2.0, 1.0], dtype=np.float64))
    column = values.reshape((2, 1))

    sorted_column = cast(
        TraceADArray,
        _dispatch(values, np.sort, (column,), {"axis": 1}),
    )
    squeezed = cast(
        TraceADArray,
        _dispatch(values, np.squeeze, (column,), {"axis": 1}),
    )

    assert sorted_column.shape == (2, 1)
    assert _primals(sorted_column) == (2.0, 1.0)
    assert squeezed.shape == (2,)
    with pytest.raises(ValueError, match="axis must have length one"):
        _dispatch(values, np.squeeze, (column,), {"axis": 0})


@pytest.mark.parametrize("function", [np.concatenate, np.stack, np.hstack, np.vstack])
def test_assembly_operations_reject_empty_sequences(function: ArrayFunction) -> None:
    """Assembly primitives should reject empty operand sequences consistently."""
    values = _capture_trace_array(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="requires at least one array"):
        _dispatch(values, function, ((),))


def test_block_rejects_empty_nested_sequences_and_shape_mismatches() -> None:
    """Block should fail closed for empty nodes and incompatible nested layouts."""
    values = _capture_trace_array(np.arange(1.0, 7.0, dtype=np.float64))
    left = cast(TraceADArray, values[:4]).reshape((2, 2))
    right = cast(TraceADArray, values[4:]).reshape((1, 2))

    for blocks in ([], [[left], []], [[left, right]]):
        with pytest.raises(ValueError, match="non-empty nested sequence"):
            _dispatch(values, np.block, (blocks,))


def test_split_variants_reject_incompatible_rank_axis_and_sections() -> None:
    """Split variants should diagnose scalar ranks, axes, and invalid sections."""
    values = _capture_trace_array(np.arange(1.0, 9.0, dtype=np.float64))
    scalar = TraceADArray((cast(TraceADScalar, values[0]),), (), values.context)
    vector = values[:4]
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))

    cases: tuple[tuple[ArrayFunction, tuple[object, ...], dict[str, object]], ...] = (
        (np.split, (scalar, 1), {}),
        (np.vsplit, (vector, 2), {}),
        (np.dsplit, (matrix, 2), {}),
        (np.split, (vector, 2), {"axis": True}),
        (np.split, (vector, 3), {}),
    )
    for function, args, kwargs in cases:
        with pytest.raises(ValueError, match="static split sections compatible"):
            _dispatch(values, function, args, kwargs)


def test_diagonal_rejects_dynamic_or_duplicate_axes() -> None:
    """Diagonal should require ranked input, static offsets, and distinct axes."""
    values = _capture_trace_array(np.arange(1.0, 9.0, dtype=np.float64))
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))

    cases: tuple[tuple[object, object, object, str], ...] = (
        (0.5, 0, 1, "static integer offset"),
        (0, True, 1, "static integer axes"),
        (0, 0, False, "static integer axes"),
        (0, 1, 1, "distinct axes"),
    )
    for offset, axis1, axis2, message in cases:
        with pytest.raises(ValueError, match=message):
            _dispatch(values, np.diagonal, (matrix, offset, axis1, axis2))

    with pytest.raises(ValueError, match="requires rank >= 2"):
        _dispatch(values, np.diagonal, (values,))


def test_concatenate_stack_and_append_reject_invalid_layouts() -> None:
    """Assembly calls should reject dynamic axes and incompatible operand shapes."""
    values = _capture_trace_array(np.arange(1.0, 7.0, dtype=np.float64))
    vector = values[:2]
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))

    cases: tuple[tuple[ArrayFunction, tuple[object, ...], dict[str, object]], ...] = (
        (np.concatenate, ((vector, matrix),), {}),
        (np.concatenate, ((vector, vector),), {"axis": True}),
        (np.stack, ((vector, values[:3]),), {}),
        (np.stack, ((vector, vector),), {"axis": True}),
        (np.stack, ((vector, vector),), {"axis": 4}),
        (np.append, (matrix, vector), {"axis": 0}),
    )
    for function, args, kwargs in cases:
        with pytest.raises(ValueError, match="rank|shape|axis|matching"):
            _dispatch(values, function, args, kwargs)


def test_concatenate_axis_none_and_stack_conveniences_match_numpy_layouts() -> None:
    """Flattened concatenation and convenience stacks should preserve exact layouts."""
    values = _capture_trace_array(np.arange(1.0, 7.0, dtype=np.float64))
    left = values[:2]
    right = values[2:4]

    flattened = _dispatch(values, np.concatenate, ((left, right),), {"axis": None})
    column = _dispatch(values, np.column_stack, ((left, right),))
    depth = _dispatch(values, np.dstack, ((left, right),))

    assert _shape(flattened) == (4,)
    assert _primals(flattened) == (1.0, 2.0, 3.0, 4.0)
    assert _shape(column) == (2, 2)
    assert _primals(column) == (1.0, 3.0, 2.0, 4.0)
    assert _shape(depth) == (1, 2, 2)


def test_positional_shape_protocol_variants_match_numpy_layouts() -> None:
    """Positional protocol forms should share the registered structural semantics."""
    values = _capture_trace_array(np.arange(1.0, 7.0, dtype=np.float64))
    matrix = values.reshape((2, 3))

    reshaped = _dispatch(values, np.reshape, (values, 6))
    expanded = _dispatch(values, np.expand_dims, (values, 0))
    swapped = _dispatch(
        values,
        np.swapaxes,
        (matrix,),
        {"axis1": 0, "axis2": 1},
    )
    repeated = _dispatch(values, np.repeat, (values, 2, 0))
    flat_rolled = _dispatch(values, np.roll, (values, 1))
    axis_rolled = _dispatch(values, np.roll, (matrix, 1, 0))
    rotated_once = _dispatch(values, np.rot90, (matrix, 1))
    rotated_twice = _dispatch(values, np.rot90, (matrix, 2, (0, 1)))

    assert _shape(reshaped) == (6,)
    assert _shape(expanded) == (1, 6)
    assert _shape(swapped) == (3, 2)
    assert _shape(repeated) == (12,)
    assert _primals(flat_rolled) == (6.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert _shape(axis_rolled) == (2, 3)
    assert _shape(rotated_once) == (3, 2)
    assert _shape(rotated_twice) == (2, 3)
