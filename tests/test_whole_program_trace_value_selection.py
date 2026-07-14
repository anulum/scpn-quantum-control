# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value selection tests
# scpn-quantum-control -- trace-value selection production contracts
"""Real-surface contracts for whole-program trace-value selection."""

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
    values: FloatArray | None = None,
) -> WholeProgramADResult:
    """Execute one selection contract through the public whole-program API."""

    def public_objective(raw_values: object) -> object:
        return objective(cast(TraceADArray, raw_values))

    parameters = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64) if values is None else values
    return whole_program_value_and_grad(public_objective, parameters, trace=False)


def _capture_trace_array(values: FloatArray | None = None) -> TraceADArray:
    """Capture the production trace array injected by the public AD API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(TraceADScalar, trace_values[0] * trace_values[0])

    _differentiate(objective, values)
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


def _primals(value: object) -> tuple[float, ...]:
    """Return primal values from a public traced scalar or array result."""
    if isinstance(value, TraceADScalar):
        return (value.primal,)
    array = cast(TraceADArray, value)
    return tuple(cast(TraceADScalar, item).primal for item in array.flatten())


def test_where_accepts_static_boolean_conditions_and_scalar_predicate_arrays() -> None:
    """Where should support static booleans and broadcast scalar predicate arrays."""
    values = _capture_trace_array()
    scalar_array = TraceADArray(
        (cast(TraceADScalar, values[0]),),
        (),
        values.context,
    )
    scalar_predicate_array = scalar_array > 0.0

    assert _primals(_dispatch(values, np.where, (True, values, -values))) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )
    assert _primals(_dispatch(values, np.where, (np.bool_(False), values, -values))) == (
        -1.0,
        -2.0,
        -3.0,
        -4.0,
    )
    assert _primals(
        _dispatch(
            values,
            np.where,
            (np.array([True, False, True, False]), values, -values),
        )
    ) == (1.0, -2.0, 3.0, -4.0)
    assert _primals(_dispatch(values, np.where, (np.array(True), values, -values))) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )
    assert _primals(_dispatch(values, np.where, (scalar_predicate_array, values, -values))) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_clip_rejects_inverted_bounds_before_selection() -> None:
    """Clip should reject lower bounds that exceed their upper bounds."""
    values = _capture_trace_array()

    with pytest.raises(ValueError, match="lower bound must not exceed upper bound"):
        _dispatch(values, np.clip, (values, 2.0, 1.0))


def test_where_rejects_invalid_and_cross_trace_predicate_arrays() -> None:
    """Where should reject non-boolean, shape-mismatched, and foreign predicates."""
    values = _capture_trace_array()
    foreign = _capture_trace_array(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64))

    with pytest.raises(ValueError, match="selection condition must be boolean"):
        _dispatch(values, np.where, (np.array([1, 0, 1, 0]), values, -values))
    with pytest.raises(ValueError, match="condition shape must be scalar or output-shaped"):
        _dispatch(values, np.where, (np.array([True, False]), values, -values))
    with pytest.raises(ValueError, match="condition shape must be scalar or output-shaped"):
        _dispatch(values, np.where, (values[:2] > 0.0, values, -values))
    with pytest.raises(ValueError, match="predicate array belongs to a different trace"):
        _dispatch(values, np.where, (foreign > 0.0, values, -values))


def test_choose_supports_raise_wrap_and_clip_modes() -> None:
    """Choose should apply each documented static selector mode deterministically."""
    values = _capture_trace_array()
    choices = (values[:2], values[2:])

    assert _primals(_dispatch(values, np.choose, ((0, 1), choices))) == (1.0, 4.0)
    assert _primals(_dispatch(values, np.choose, ((-1, 2), choices), {"mode": "wrap"})) == (
        3.0,
        2.0,
    )
    assert _primals(_dispatch(values, np.choose, ((-1, 2), choices), {"mode": "clip"})) == (
        1.0,
        4.0,
    )


def test_choose_rejects_dynamic_or_invalid_choices_and_selectors() -> None:
    """Choose should fail closed on dynamic choices and non-static selectors."""
    values = _capture_trace_array()
    choices = (values[:2], values[2:])
    object_selector = np.empty(1, dtype=object)
    object_selector[0] = values[0]

    with pytest.raises(ValueError, match="requires a static choice sequence"):
        _dispatch(values, np.choose, ((0, 1), values))
    with pytest.raises(ValueError, match="requires a static choice sequence"):
        _dispatch(values, np.choose, ((0, 1), object()))
    with pytest.raises(ValueError, match="requires at least one choice"):
        _dispatch(values, np.choose, ((0, 1), ()))
    for selector in (values, values[0], values > 0.0, object_selector, (0.5, 1.5)):
        with pytest.raises(ValueError, match="requires a static integer selector"):
            _dispatch(values, np.choose, (selector, choices))
    with pytest.raises(ValueError, match="selector indices out of bounds"):
        _dispatch(values, np.choose, ((0, 2), choices))
    with pytest.raises(ValueError, match="mode must be raise, wrap, or clip"):
        _dispatch(values, np.choose, ((0, 1), choices), {"mode": "invalid"})


def test_compress_and_extract_support_static_boolean_conditions() -> None:
    """Compress and extract should retain selected trace values and aliases."""
    values = _capture_trace_array()
    matrix = values.reshape((2, 2))

    assert _primals(_dispatch(values, np.compress, ((True, False, True, False), values))) == (
        1.0,
        3.0,
    )
    assert _primals(_dispatch(values, np.compress, ((False, True), matrix, 0))) == (3.0, 4.0)
    assert _primals(_dispatch(values, np.extract, ((True, False, False, True), matrix))) == (
        1.0,
        4.0,
    )


def test_compress_and_extract_reject_dynamic_or_malformed_conditions() -> None:
    """Compress and extract should reject dynamic and malformed conditions."""
    values = _capture_trace_array()
    object_condition = np.empty(1, dtype=object)
    object_condition[0] = values[0]

    for condition in (values, values > 0.0, object_condition, (1, 0, 1, 0)):
        with pytest.raises(ValueError, match="static boolean condition"):
            _dispatch(values, np.compress, (condition, values))
        with pytest.raises(ValueError, match="static boolean condition"):
            _dispatch(values, np.extract, (condition, values))
    with pytest.raises(ValueError, match="one-dimensional condition"):
        _dispatch(values, np.compress, (np.array([[True, False]]), values))
    with pytest.raises(ValueError, match="static integer axis or None"):
        _dispatch(values, np.compress, ((True, False, True, False), values, True))
    with pytest.raises(ValueError, match="condition size must match array size"):
        _dispatch(values, np.extract, ((True, False), values))


def test_select_empty_conditions_return_scalar_and_array_defaults() -> None:
    """Select should return its traced default when no condition rows exist."""
    values = _capture_trace_array()
    scalar_default = TraceADArray(
        (cast(TraceADScalar, values[0]),),
        (),
        values.context,
    )

    assert _primals(_dispatch(values, np.select, ((), (), values))) == (1.0, 2.0, 3.0, 4.0)
    assert _primals(_dispatch(values, np.select, ((), (), scalar_default))) == (1.0,)


def test_select_and_piecewise_reject_non_sequence_contracts() -> None:
    """Select and piecewise should require static condition and branch sequences."""
    values = _capture_trace_array()

    for conditions in (values, np.array([True, False]), object()):
        with pytest.raises(ValueError, match="select requires a static condition sequence"):
            _dispatch(values, np.select, (conditions, (values,)))
        with pytest.raises(ValueError, match="piecewise requires a static condition sequence"):
            _dispatch(values, np.piecewise, (values, conditions, (values,)))
    for choices in (values, np.array([1.0, 2.0]), object()):
        with pytest.raises(ValueError, match="select requires a static choice sequence"):
            _dispatch(values, np.select, ((values > 0.0,), choices))
        with pytest.raises(ValueError, match="piecewise requires a static function sequence"):
            _dispatch(values, np.piecewise, (values, (values > 0.0,), choices))


def test_take_supports_scalar_results_and_rejects_invalid_indices() -> None:
    """Take should return traced scalars and reject dynamic or out-of-bounds indices."""
    values = _capture_trace_array()

    assert _primals(_dispatch(values, np.take, (values, 2))) == (3.0,)
    for indices in (values, values[0], (0.5, 1.5)):
        with pytest.raises(ValueError, match="requires static integer indices"):
            _dispatch(values, np.take, (values, indices))
    with pytest.raises(ValueError, match="indices must be in bounds"):
        _dispatch(values, np.take, (values, 5))


def test_take_along_axis_rejects_dynamic_and_incompatible_indices() -> None:
    """Take-along-axis should require static compatible integer index arrays."""
    values = _capture_trace_array()
    matrix = values.reshape((2, 2))

    for indices in (values, values[0]):
        with pytest.raises(ValueError, match="requires static integer indices"):
            _dispatch(values, np.take_along_axis, (matrix, indices, 1))
    with pytest.raises(ValueError, match="requires static integer axis"):
        _dispatch(values, np.take_along_axis, (matrix, np.array([[0], [1]]), True))
    with pytest.raises(ValueError, match="in bounds and shape-compatible"):
        _dispatch(values, np.take_along_axis, (matrix, np.array([0, 1, 0]), 1))


def test_basic_indexing_rejects_dynamic_and_non_integer_selectors() -> None:
    """Basic indexing should reject traced, boolean-scalar, and non-integer selectors."""
    values = _capture_trace_array()
    object_indices = np.empty(1, dtype=object)
    object_indices[0] = values[0]

    for selector in (
        values,
        values[0],
        values > 0.0,
        True,
        object(),
        np.array([0.5, 1.5]),
        np.array(True),
        object_indices,
    ):
        with pytest.raises(ValueError, match="requires static integer or boolean index arrays"):
            values[selector]
    for bound in (1.5, values[0]):
        with pytest.raises(ValueError, match="static integer slice bounds"):
            values[slice(bound, None)]


def test_sort_and_order_statistics_cover_singleton_and_scalar_reductions() -> None:
    """Singleton sort and scalar order statistics should preserve exact gradients."""

    def singleton_objective(values: TraceADArray) -> TraceADScalar:
        sorted_values = cast(
            TraceADArray,
            _dispatch(values, np.sort, (values,), {"axis": None}),
        )
        return cast(TraceADScalar, sorted_values[0])

    singleton = _differentiate(
        singleton_objective,
        np.array([2.0], dtype=np.float64),
    )
    median = _differentiate(
        lambda values: cast(TraceADScalar, _dispatch(values, np.median, (values, 0))),
        np.array([1.0, 2.0, 4.0], dtype=np.float64),
    )
    extreme = _differentiate(
        lambda values: cast(TraceADScalar, values.max(axis=0)),
        np.array([1.0, 3.0, 2.0], dtype=np.float64),
    )

    assert singleton.value == pytest.approx(2.0)
    np.testing.assert_array_equal(singleton.gradient, np.array([1.0], dtype=np.float64))
    assert median.value == pytest.approx(2.0)
    np.testing.assert_array_equal(median.gradient, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert extreme.value == pytest.approx(3.0)
    np.testing.assert_array_equal(extreme.gradient, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def test_sort_rejects_tied_values_at_nondifferentiable_boundaries() -> None:
    """Sort should reject equal values globally and along an explicit axis."""
    values = _capture_trace_array(np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64))

    with pytest.raises(ValueError, match="strictly ordered values"):
        _dispatch(values, np.sort, (values,), {"axis": None})
    with pytest.raises(ValueError, match="strictly ordered values"):
        _dispatch(values, np.sort, (values.reshape((2, 2)),), {"axis": 1})
