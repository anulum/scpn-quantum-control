# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value tests
# scpn-quantum-control -- whole-program trace-value production contracts
"""Production-contract tests for whole-program derivative-carrying values."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from threading import Event
from types import FrameType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import (
    TraceADArray,
    TraceADScalar,
    whole_program_value_and_grad,
)
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
)

FloatArray = NDArray[np.float64]
ArrayFunction = Callable[..., FloatArray]
ArrayFunctionArgs = Callable[[TraceADArray], tuple[object, ...]]
ArrayFunctionKwargs = Callable[[TraceADArray], dict[str, object]]
ArrayFunctionFailure = tuple[
    str,
    ArrayFunction,
    ArrayFunctionArgs,
    ArrayFunctionKwargs,
    str,
]


_MALFORMED_ARRAY_FUNCTION_CASES: tuple[ArrayFunctionFailure, ...] = (
    ("sum-arity", np.sum, lambda _array: (), lambda _array: {}, r"np\.sum supports"),
    (
        "cumsum-keyword",
        np.cumsum,
        lambda array: (array,),
        lambda _array: {"dtype": np.float64},
        r"np\.cumsum supports",
    ),
    (
        "prod-arity",
        np.prod,
        lambda array: (array, array),
        lambda _array: {},
        r"np\.prod supports",
    ),
    (
        "cumprod-keyword",
        np.cumprod,
        lambda array: (array,),
        lambda _array: {"dtype": np.float64},
        r"np\.cumprod supports",
    ),
    (
        "diff-arity",
        np.diff,
        lambda _array: (),
        lambda _array: {},
        r"np\.diff supports",
    ),
    (
        "gradient-keyword",
        np.gradient,
        lambda array: (array,),
        lambda _array: {"unsupported": 1},
        r"np\.gradient supports",
    ),
    (
        "interp-arity",
        np.interp,
        lambda array: (array, (0.0, 1.0)),
        lambda _array: {},
        r"np\.interp supports",
    ),
    (
        "interp-left-twice",
        np.interp,
        lambda array: (array, (0.0, 1.0), (0.0, 1.0), 0.0),
        lambda _array: {"left": 0.0},
        "left must be supplied once",
    ),
    (
        "interp-right-twice",
        np.interp,
        lambda array: (array, (0.0, 1.0), (0.0, 1.0), 0.0, 1.0),
        lambda _array: {"right": 1.0},
        "right must be supplied once",
    ),
    (
        "interp-period-twice",
        np.interp,
        lambda array: (array, (0.0, 1.0), (0.0, 1.0), 0.0, 1.0, 2.0),
        lambda _array: {"period": 2.0},
        "period must be supplied once",
    ),
    (
        "convolve-arity",
        np.convolve,
        lambda array: (array,),
        lambda _array: {},
        r"np\.convolve supports",
    ),
    (
        "convolve-mode-twice",
        np.convolve,
        lambda array: (array, array, "full"),
        lambda _array: {"mode": "full"},
        "mode must be supplied once",
    ),
    (
        "correlate-arity",
        np.correlate,
        lambda array: (array,),
        lambda _array: {},
        r"np\.correlate supports",
    ),
    (
        "correlate-mode-twice",
        np.correlate,
        lambda array: (array, array, "valid"),
        lambda _array: {"mode": "valid"},
        "mode must be supplied once",
    ),
    (
        "zeros-like-arity",
        np.zeros_like,
        lambda _array: (),
        lambda _array: {},
        "like-constructors require one",
    ),
    (
        "full-like-arity",
        np.full_like,
        lambda array: (array,),
        lambda _array: {},
        "full_like requires reference array and fill value",
    ),
    (
        "mean-keyword",
        np.mean,
        lambda array: (array,),
        lambda _array: {"dtype": np.float64},
        r"np\.mean supports",
    ),
    (
        "trapezoid-arity",
        np.trapezoid,
        lambda _array: (),
        lambda _array: {},
        r"np\.trapezoid supports",
    ),
    (
        "trapezoid-x-twice",
        np.trapezoid,
        lambda array: (array, (0.0, 1.0, 2.0, 3.0)),
        lambda _array: {"x": (0.0, 1.0, 2.0, 3.0)},
        "x must be supplied once",
    ),
    (
        "var-keyword",
        np.var,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        r"np\.var supports",
    ),
    (
        "std-keyword",
        np.std,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        r"np\.std supports",
    ),
    (
        "median-arity",
        np.median,
        lambda _array: (),
        lambda _array: {},
        r"np\.median supports",
    ),
    (
        "median-axis-twice",
        np.median,
        lambda array: (array, 0),
        lambda _array: {"axis": 0},
        "axis must be supplied once",
    ),
    (
        "quantile-arity",
        np.quantile,
        lambda array: (array,),
        lambda _array: {},
        r"np\.quantile supports",
    ),
    (
        "quantile-axis-twice",
        np.quantile,
        lambda array: (array, 0.5, 0),
        lambda _array: {"axis": 0},
        "axis must be supplied once",
    ),
    (
        "percentile-method-twice",
        np.percentile,
        lambda array: (array, 50.0),
        lambda _array: {"method": "linear", "interpolation": "linear"},
        "method must be supplied once",
    ),
    (
        "max-keyword",
        np.max,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        r"np\.max supports",
    ),
    (
        "min-keyword",
        np.min,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        r"np\.min supports",
    ),
    ("dot-arity", np.dot, lambda array: (array,), lambda _array: {}, r"np\.dot supports"),
    (
        "vdot-keyword",
        np.vdot,
        lambda array: (array, array),
        lambda _array: {"out": None},
        r"np\.vdot supports",
    ),
    (
        "inner-arity",
        np.inner,
        lambda array: (array,),
        lambda _array: {},
        r"np\.inner supports",
    ),
    (
        "outer-keyword",
        np.outer,
        lambda array: (array, array),
        lambda _array: {"out": None},
        r"np\.outer supports",
    ),
    (
        "tensordot-arity",
        np.tensordot,
        lambda array: (array,),
        lambda _array: {},
        r"np\.tensordot supports",
    ),
    (
        "einsum-arity",
        np.einsum,
        lambda _array: ("i->",),
        lambda _array: {},
        r"np\.einsum supports",
    ),
    (
        "einsum-subscript-type",
        np.einsum,
        lambda array: (1, array),
        lambda _array: {},
        "requires a string subscript",
    ),
    (
        "matmul-keyword",
        np.matmul,
        lambda array: (array, array),
        lambda _array: {"out": None},
        r"np\.matmul supports",
    ),
    (
        "where-arity",
        np.where,
        lambda array: (array > 0.0, array),
        lambda _array: {},
        r"np\.where supports",
    ),
    (
        "select-arity",
        np.select,
        lambda array: ([array > 0.0],),
        lambda _array: {},
        r"np\.select supports",
    ),
    (
        "select-default-twice",
        np.select,
        lambda array: ([array > 0.0], [array], 0.0),
        lambda _array: {"default": 0.0},
        "default must be supplied once",
    ),
    (
        "piecewise-arity",
        np.piecewise,
        lambda array: (array, [array > 0.0]),
        lambda _array: {},
        r"np\.piecewise supports",
    ),
    (
        "choose-keyword",
        np.choose,
        lambda array: ((0, 1, 0, 1), (array, array)),
        lambda _array: {"out": None},
        r"np\.choose supports",
    ),
    (
        "compress-arity",
        np.compress,
        lambda array: ((True, False),),
        lambda _array: {},
        r"np\.compress supports",
    ),
    (
        "compress-axis-twice",
        np.compress,
        lambda array: ((True, False, True, False), array, 0),
        lambda _array: {"axis": 0},
        "axis must be supplied once",
    ),
    (
        "extract-keyword",
        np.extract,
        lambda array: ((True, False, True, False), array),
        lambda _array: {"extra": 1},
        r"np\.extract supports",
    ),
    (
        "reshape-keyword",
        np.reshape,
        lambda array: (array, (2, 2)),
        lambda _array: {"order": "C"},
        r"np\.reshape supports",
    ),
    (
        "broadcast-to-arity",
        np.broadcast_to,
        lambda array: (array,),
        lambda _array: {},
        r"np\.broadcast_to supports",
    ),
    (
        "broadcast-arrays-arity",
        np.broadcast_arrays,
        lambda _array: (),
        lambda _array: {},
        r"np\.broadcast_arrays supports",
    ),
    (
        "ravel-keyword",
        np.ravel,
        lambda array: (array,),
        lambda _array: {"order": "C"},
        r"np\.ravel supports",
    ),
    (
        "atleast-keyword",
        np.atleast_1d,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        "atleast transforms support positional arrays only",
    ),
    (
        "squeeze-keyword",
        np.squeeze,
        lambda array: (array,),
        lambda _array: {"out": None},
        r"np\.squeeze supports",
    ),
    (
        "expand-dims-arity",
        np.expand_dims,
        lambda array: (array,),
        lambda _array: {},
        r"np\.expand_dims supports",
    ),
    (
        "swapaxes-arity",
        np.swapaxes,
        lambda array: (array, 0),
        lambda _array: {},
        r"np\.swapaxes supports",
    ),
    (
        "moveaxis-arity",
        np.moveaxis,
        lambda array: (array, 0),
        lambda _array: {},
        r"np\.moveaxis supports",
    ),
    (
        "repeat-arity",
        np.repeat,
        lambda array: (array,),
        lambda _array: {},
        r"np\.repeat supports",
    ),
    (
        "tile-arity",
        np.tile,
        lambda array: (array,),
        lambda _array: {},
        r"np\.tile supports",
    ),
    (
        "roll-arity",
        np.roll,
        lambda array: (array,),
        lambda _array: {},
        r"np\.roll supports",
    ),
    (
        "rot90-arity",
        np.rot90,
        lambda array: (array, 1, (0, 1), "extra"),
        lambda _array: {},
        r"np\.rot90 supports",
    ),
    (
        "flip-keyword",
        np.flip,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.flip supports",
    ),
    (
        "flipud-keyword",
        np.flipud,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.flipud supports",
    ),
    (
        "fliplr-keyword",
        np.fliplr,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.fliplr supports",
    ),
    (
        "take-arity",
        np.take,
        lambda array: (array,),
        lambda _array: {},
        r"np\.take supports",
    ),
    (
        "take-along-axis-arity",
        np.take_along_axis,
        lambda array: (array,),
        lambda _array: {},
        r"np\.take_along_axis supports",
    ),
    (
        "delete-arity",
        np.delete,
        lambda array: (array,),
        lambda _array: {},
        r"np\.delete supports",
    ),
    (
        "pad-arity",
        np.pad,
        lambda array: (array,),
        lambda _array: {},
        r"np\.pad supports",
    ),
    (
        "insert-arity",
        np.insert,
        lambda array: (array, 0),
        lambda _array: {},
        r"np\.insert supports",
    ),
    (
        "append-arity",
        np.append,
        lambda array: (array,),
        lambda _array: {},
        r"np\.append supports",
    ),
    (
        "transpose-keyword",
        np.transpose,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.transpose supports",
    ),
    (
        "trace-keyword",
        np.trace,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"dtype": np.float64},
        r"np\.trace supports",
    ),
    (
        "diag-keyword",
        np.diag,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.diag supports",
    ),
    (
        "diagflat-keyword",
        np.diagflat,
        lambda array: (array,),
        lambda _array: {"extra": 1},
        r"np\.diagflat supports",
    ),
    (
        "diagonal-arity",
        np.diagonal,
        lambda _array: (),
        lambda _array: {},
        r"np\.diagonal supports",
    ),
    (
        "diagonal-offset-twice",
        np.diagonal,
        lambda array: (array.reshape((2, 2)), 0),
        lambda _array: {"offset": 0},
        "offset must be supplied once",
    ),
    (
        "diagonal-axis1-twice",
        np.diagonal,
        lambda array: (array.reshape((2, 2)), 0, 0),
        lambda _array: {"axis1": 0},
        "axis1 must be supplied once",
    ),
    (
        "diagonal-axis2-twice",
        np.diagonal,
        lambda array: (array.reshape((2, 2)), 0, 0, 1),
        lambda _array: {"axis2": 1},
        "axis2 must be supplied once",
    ),
    (
        "concatenate-arity",
        np.concatenate,
        lambda array: ((array, array), array),
        lambda _array: {},
        r"np\.concatenate supports",
    ),
    (
        "stack-keyword",
        np.stack,
        lambda array: ((array, array),),
        lambda _array: {"out": None},
        r"np\.stack supports",
    ),
    (
        "hstack-keyword",
        np.hstack,
        lambda array: ((array, array),),
        lambda _array: {"extra": 1},
        r"np\.hstack supports",
    ),
    (
        "block-keyword",
        np.block,
        lambda array: ([array, array],),
        lambda _array: {"extra": 1},
        r"np\.block supports",
    ),
    (
        "split-arity",
        np.split,
        lambda array: (array,),
        lambda _array: {},
        r"np\.split supports",
    ),
    (
        "hsplit-keyword",
        np.hsplit,
        lambda array: (array, 2),
        lambda _array: {"extra": 1},
        r"np\.hsplit supports",
    ),
    (
        "tril-arity",
        np.tril,
        lambda array: (array, 0, 1),
        lambda _array: {},
        r"np\.tril supports",
    ),
    (
        "clip-arity",
        np.clip,
        lambda array: (array, -1.0),
        lambda _array: {},
        r"np\.clip supports",
    ),
    (
        "norm-keyword",
        np.linalg.norm,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        r"np\.linalg\.norm supports",
    ),
    (
        "norm-ord-twice",
        np.linalg.norm,
        lambda array: (array, 2),
        lambda _array: {"ord": 2},
        "ord must be supplied once",
    ),
    (
        "norm-axis-twice",
        np.linalg.norm,
        lambda array: (array, 2, 0),
        lambda _array: {"axis": 0},
        "axis must be supplied once",
    ),
    (
        "det-arity",
        np.linalg.det,
        lambda _array: (),
        lambda _array: {},
        r"np\.linalg\.det supports",
    ),
    (
        "inv-keyword",
        np.linalg.inv,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"extra": 1},
        r"np\.linalg\.inv supports",
    ),
    (
        "solve-arity",
        np.linalg.solve,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {},
        r"np\.linalg\.solve supports",
    ),
    (
        "matrix-power-keyword",
        np.linalg.matrix_power,
        lambda array: (array.reshape((2, 2)), 2),
        lambda _array: {"extra": 1},
        r"np\.linalg\.matrix_power supports",
    ),
    (
        "multi-dot-arity",
        np.linalg.multi_dot,
        lambda array: ((array, array), array),
        lambda _array: {},
        r"np\.linalg\.multi_dot supports",
    ),
    (
        "eig-keyword",
        np.linalg.eig,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"extra": 1},
        r"np\.linalg\.eig supports",
    ),
    (
        "eigh-arity",
        np.linalg.eigh,
        lambda _array: (),
        lambda _array: {},
        r"np\.linalg\.eigh supports",
    ),
    (
        "eigvalsh-arity",
        np.linalg.eigvalsh,
        lambda _array: (),
        lambda _array: {},
        r"np\.linalg\.eigvalsh supports",
    ),
    (
        "eigvals-keyword",
        np.linalg.eigvals,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"extra": 1},
        r"np\.linalg\.eigvals supports",
    ),
    (
        "svd-arity",
        np.linalg.svd,
        lambda _array: (),
        lambda _array: {},
        r"np\.linalg\.svd supports",
    ),
    (
        "svd-keyword",
        np.linalg.svd,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"unknown": True},
        "supports full_matrices",
    ),
    (
        "svd-full-matrices-type",
        np.linalg.svd,
        lambda array: (array.reshape((2, 2)), "yes", False, False),
        lambda _array: {},
        "full_matrices must be static boolean",
    ),
    (
        "svd-compute-uv-type",
        np.linalg.svd,
        lambda array: (array.reshape((2, 2)), True, "no", False),
        lambda _array: {},
        "compute_uv must be static boolean",
    ),
    (
        "svd-hermitian-type",
        np.linalg.svd,
        lambda array: (array.reshape((2, 2)), True, False, "no"),
        lambda _array: {},
        "hermitian must be static boolean",
    ),
    (
        "svd-hermitian-true",
        np.linalg.svd,
        lambda array: (array.reshape((2, 2)), True, False, True),
        lambda _array: {},
        "hermitian=False only",
    ),
    (
        "pinv-arity",
        np.linalg.pinv,
        lambda _array: (),
        lambda _array: {},
        r"np\.linalg\.pinv supports",
    ),
    (
        "pinv-keyword",
        np.linalg.pinv,
        lambda array: (array.reshape((2, 2)),),
        lambda _array: {"unknown": True},
        "supports rcond, rtol, and hermitian",
    ),
    (
        "pinv-cutoff-twice",
        np.linalg.pinv,
        lambda array: (array.reshape((2, 2)), 1.0e-15),
        lambda _array: {"rtol": 1.0e-15},
        "only one of rcond or rtol",
    ),
    (
        "pinv-hermitian-type",
        np.linalg.pinv,
        lambda array: (array.reshape((2, 2)), None, "no"),
        lambda _array: {},
        "hermitian must be static boolean",
    ),
    (
        "argmax-arity",
        np.argmax,
        lambda _array: (),
        lambda _array: {},
        r"np\.argmax supports",
    ),
    (
        "argmax-keyword",
        np.argmax,
        lambda array: (array,),
        lambda _array: {"unsupported": 1},
        "only supports axis, out, and keepdims",
    ),
    (
        "argmax-out",
        np.argmax,
        lambda array: (array,),
        lambda _array: {"out": object()},
        "does not support out",
    ),
    (
        "argmin-keepdims",
        np.argmin,
        lambda array: (array,),
        lambda _array: {"keepdims": True},
        "keepdims=False only",
    ),
    (
        "argmin-axis-twice",
        np.argmin,
        lambda array: (array, 0),
        lambda _array: {"axis": 0},
        "received duplicate axis",
    ),
    (
        "sort-arity",
        np.sort,
        lambda _array: (),
        lambda _array: {},
        r"np\.sort expects exactly one",
    ),
    (
        "sort-keyword",
        np.sort,
        lambda array: (array,),
        lambda _array: {"unsupported": 1},
        "only supports axis, kind, and order",
    ),
    (
        "sort-order",
        np.sort,
        lambda array: (array,),
        lambda _array: {"order": "field"},
        "does not support structured-array order",
    ),
    (
        "sort-kind",
        np.sort,
        lambda array: (array,),
        lambda _array: {"kind": "invalid"},
        "kind must be a NumPy sort kind",
    ),
    (
        "sort-kind-non-string",
        np.sort,
        lambda array: (array,),
        lambda _array: {"kind": []},
        "kind must be a NumPy sort kind",
    ),
    (
        "argsort-arity",
        np.argsort,
        lambda _array: (),
        lambda _array: {},
        r"np\.argsort expects exactly one",
    ),
    (
        "argsort-keyword",
        np.argsort,
        lambda array: (array,),
        lambda _array: {"unsupported": 1},
        "only supports axis, kind, order, and stable",
    ),
    (
        "argsort-order",
        np.argsort,
        lambda array: (array,),
        lambda _array: {"order": "field"},
        "does not support structured-array order",
    ),
    (
        "argsort-stable",
        np.argsort,
        lambda array: (array,),
        lambda _array: {"stable": True},
        "does not support stable keyword",
    ),
    (
        "argsort-kind",
        np.argsort,
        lambda array: (array,),
        lambda _array: {"kind": "invalid"},
        "kind must be a NumPy sort kind",
    ),
    (
        "argsort-kind-non-string",
        np.argsort,
        lambda array: (array,),
        lambda _array: {"kind": []},
        "kind must be a NumPy sort kind",
    ),
    (
        "unsupported-function",
        np.all,
        lambda array: (array,),
        lambda _array: {},
        "unsupported whole-program AD NumPy function all",
    ),
)


def _expect_array_function_failure(
    function: ArrayFunction,
    args_factory: ArrayFunctionArgs,
    kwargs_factory: ArrayFunctionKwargs,
    message: str,
) -> str:
    """Exercise one fail-closed NumPy protocol call through the public AD API."""

    def objective(values: FloatArray) -> object:
        traced = cast(TraceADArray, values)
        return traced.__array_function__(
            function,
            (TraceADArray,),
            args_factory(traced),
            kwargs_factory(traced),
        )

    with pytest.raises(ValueError) as error:
        whole_program_value_and_grad(
            objective,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            trace=False,
        )
    rendered = str(error.value)
    assert re.search(message, rendered) is not None
    return rendered


def test_trace_values_are_crosswired_at_the_package_root() -> None:
    """The package root should expose the value classes used by the public AD API."""
    observed: dict[str, object] = {}

    def record_trace_values(values: Any) -> object:
        observed["type"] = type(values)
        observed["scalar_type"] = type(values[0])
        return values[0] * values[0]

    result = whole_program_value_and_grad(
        record_trace_values,
        np.array([2.0], dtype=np.float64),
        trace=False,
    )

    assert observed == {"type": TraceADArray, "scalar_type": TraceADScalar}
    assert result.value == pytest.approx(4.0)
    np.testing.assert_allclose(result.gradient, np.array([4.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("_case_id", "function", "args_factory", "kwargs_factory", "message"),
    _MALFORMED_ARRAY_FUNCTION_CASES,
    ids=[case[0] for case in _MALFORMED_ARRAY_FUNCTION_CASES],
)
def test_array_function_protocol_rejects_malformed_calls(
    _case_id: str,
    function: ArrayFunction,
    args_factory: ArrayFunctionArgs,
    kwargs_factory: ArrayFunctionKwargs,
    message: str,
) -> None:
    """The public trace protocol should reject malformed NumPy dispatch calls."""
    rendered = _expect_array_function_failure(
        function,
        args_factory,
        kwargs_factory,
        message,
    )
    assert re.search(message, rendered) is not None


def test_dot_contract_rejects_invalid_shapes_and_handles_empty_vectors() -> None:
    """Dot should reject non-scalar contracts and define the empty vector result."""
    with pytest.raises(ValueError, match="scalar dot results only"):
        whole_program_value_and_grad(
            lambda values: np.dot(values.reshape((2, 2)), values.reshape((2, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            trace=False,
        )
    with pytest.raises(ValueError, match="vector dimensions must align"):
        whole_program_value_and_grad(
            lambda values: np.dot(values[:2], values[1:]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            trace=False,
        )

    result = whole_program_value_and_grad(
        lambda values: np.dot(values[:0], values[:0]),
        np.array([1.0], dtype=np.float64),
        trace=False,
    )
    assert result.value == pytest.approx(0.0)
    np.testing.assert_array_equal(result.gradient, np.array([0.0], dtype=np.float64))


def test_clip_protocol_rejects_output_buffers() -> None:
    """The trace protocol should reject NumPy clip output buffers instead of ignoring them."""
    positional_error = _expect_array_function_failure(
        np.clip,
        lambda array: (array, -1.0, 1.0, object()),
        lambda _array: {},
        r"np\.clip supports array, lower, and upper",
    )
    keyword_error = _expect_array_function_failure(
        np.clip,
        lambda array: (array, -1.0, 1.0),
        lambda _array: {"out": object()},
        r"np\.clip supports array, lower, and upper",
    )
    assert "np.clip supports array, lower, and upper" in positional_error
    assert "np.clip supports array, lower, and upper" in keyword_error


@pytest.mark.parametrize("diagonal", [np.diag, np.diagflat])
@pytest.mark.parametrize("offset", [10_000, -10_000, sys.maxsize, -sys.maxsize])
def test_public_diagonal_refuses_oversized_pointer_storage_and_recovers(
    diagonal: ArrayFunction, offset: int
) -> None:
    """A large offset cannot allocate a huge trace container with few AD nodes."""
    baseline = active_reserved_bytes()

    def objective(values: Any) -> object:
        return np.sum(diagonal(values, k=offset))

    with pytest.raises(DenseAllocationError, match="(trace array storage|native addressable)"):
        whole_program_value_and_grad(
            objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
        )
    assert active_reserved_bytes() == baseline
    offset = 2
    result = whole_program_value_and_grad(
        objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
    )
    assert result.value == 5.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("diagonal", [np.diag, np.diagflat])
@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_public_diagonal_fixed_storage_preserves_diagonal_and_gradient(
    diagonal: ArrayFunction, offset: int
) -> None:
    """Admitted placement preserves independent diagonal-index weighted sums."""
    values = np.array([2.0, 3.0])
    size = values.size + abs(offset)
    weights = np.arange(1, size * size + 1, dtype=float).reshape(size, size)

    def objective(parameters: Any) -> object:
        return np.sum(diagonal(parameters, k=offset) * weights)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, values, trace=False, max_execution_gib=0.01)
    rows = np.arange(values.size) + max(0, -offset)
    columns = np.arange(values.size) + max(0, offset)
    expected_gradient = weights[rows, columns]
    assert result.value == float(np.dot(values, expected_gradient))
    np.testing.assert_array_equal(result.gradient, expected_gradient)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("extent", [10_000_000, 1 << 40])
def test_public_broadcast_refuses_storage_before_materialisation_and_recovers(
    scalar: bool, extent: int
) -> None:
    """A broadcast with few trace nodes still admits output pointers and index buffers."""

    def objective(values: Any) -> object:
        source = values.reshape(()) if scalar else values
        return np.sum(np.broadcast_to(source, (extent,)))

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective, np.array([2.0]), trace=False, max_execution_gib=0.01
        )
    assert active_reserved_bytes() == baseline
    extent = 3
    result = whole_program_value_and_grad(
        objective, np.array([2.0]), trace=False, max_execution_gib=0.01
    )
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, np.array([3.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("operation", [np.repeat, np.tile])
@pytest.mark.parametrize("count", [10_000_000, sys.maxsize])
def test_public_repetition_refuses_large_storage_and_recovers(
    operation: ArrayFunction, count: int
) -> None:
    """Output index and lineage storage is admitted before a repeated array exists."""

    def objective(values: Any) -> object:
        return np.sum(operation(values, count))

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
        )
    assert active_reserved_bytes() == baseline
    count = 3
    result = whole_program_value_and_grad(
        objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
    )
    assert result.value == 15.0
    np.testing.assert_array_equal(result.gradient, np.array([3.0, 3.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("operation", [np.repeat, np.tile])
def test_public_repetition_zero_output_preserves_empty_sum(operation: ArrayFunction) -> None:
    """Zero replication returns an empty array without constructing intermediate repeats."""

    def objective(values: Any) -> object:
        return np.sum(operation(values, 0))

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, np.zeros(2))
    assert active_reserved_bytes() == baseline


def test_public_repeat_axis_counts_preserve_order_and_gradient() -> None:
    """Per-column counts preserve rectangular output and pullback multiplicities."""

    def objective(values: Any) -> object:
        return np.sum(np.repeat(values.reshape((2, 2)), (1, 2), axis=-1))

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([1.0, 2.0, 3.0, 4.0]), trace=False)
    assert result.value == 16.0
    np.testing.assert_array_equal(result.gradient, np.array([1.0, 2.0, 1.0, 2.0]))
    assert active_reserved_bytes() == baseline


def test_public_tile_rank_extension_preserves_replication_gradient() -> None:
    """Leading rank extension and per-axis tiling retain all six source copies."""

    def objective(values: Any) -> object:
        return np.sum(np.tile(values.reshape((2, 2)), (2, 1, 3)))

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([1.0, 2.0, 3.0, 4.0]), trace=False)
    assert result.value == 60.0
    np.testing.assert_array_equal(result.gradient, np.full(4, 6.0))
    assert active_reserved_bytes() == baseline


def test_public_pad_refuses_large_layout_before_shape_materialisation_and_recovers() -> None:
    """Pad shape dispatch does not allocate an output before the real memory guard."""
    width = 10_000_000

    def objective(values: Any) -> object:
        return np.sum(np.pad(values, width, constant_values=7.0))

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
        )
    assert active_reserved_bytes() == baseline
    width = 2
    result = whole_program_value_and_grad(
        objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
    )
    assert result.value == 33.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert active_reserved_bytes() == baseline


def test_public_padding_storage_remains_owned_until_objective_returns() -> None:
    """Padding object/tangent charges survive the temporary numeric layout scope."""
    observations: list[int] = []

    def observe_storage() -> None:
        observations.append(active_reserved_bytes())

    def objective(values: Any) -> object:
        observe_storage()
        padded = np.pad(values, 3, constant_values=7.0)
        observe_storage()
        return np.sum(padded)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 47.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert len(observations) == 2
    assert observations[1] - observations[0] >= 6 * 2 * np.dtype(np.float64).itemsize
    assert active_reserved_bytes() == baseline


def test_public_padding_failure_releases_retained_objects_and_tangents() -> None:
    """An objective exception after padding drops its retained charge and allows retry."""
    failing = True

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after owned padding")

    def objective(values: Any) -> object:
        padded = np.pad(values, 3, constant_values=7.0)
        fail_if_armed()
        return np.sum(padded)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after owned padding"):
        whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 47.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("selector", [1, (1,)])
def test_public_insertion_storage_remains_owned_until_objective_returns(
    selector: int | tuple[int, ...],
) -> None:
    """Inserted constants retain their scalar and tangent charge after layout exits."""
    observations: list[int] = []

    def observe_storage() -> None:
        observations.append(active_reserved_bytes())

    def objective(values: Any) -> object:
        observe_storage()
        inserted = np.insert(values, selector, (7.0, 8.0, 9.0))
        observe_storage()
        return np.sum(inserted)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 29.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert len(observations) == 2
    assert observations[1] - observations[0] >= 3 * 2 * np.dtype(np.float64).itemsize
    assert active_reserved_bytes() == baseline


def test_public_insertion_failure_releases_retained_objects_and_recovers() -> None:
    """An exception after insertion disposes charges before a successful retry."""
    failing = True

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after owned insertion")

    def objective(values: Any) -> object:
        inserted = np.insert(values, 1, (7.0, 8.0, 9.0))
        fail_if_armed()
        return np.sum(inserted)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after owned insertion"):
        whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 29.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert active_reserved_bytes() == baseline


def test_public_insertion_refuses_constant_storage_and_recovers() -> None:
    """Constant cells are admitted before their trace objects and tangents exist."""
    constants: tuple[float, ...] = (7.0,) * 4096

    def objective(values: Any) -> object:
        return np.sum(np.insert(values, 1, constants))

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.001
        )
    assert active_reserved_bytes() == baseline
    constants = (7.0, 8.0, 9.0)
    result = whole_program_value_and_grad(
        objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.001
    )
    assert result.value == 29.0
    np.testing.assert_array_equal(result.gradient, np.ones(2))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("along_axis", [False, True])
def test_public_take_shape_admission_refuses_large_selection_and_recovers(
    along_axis: bool,
) -> None:
    """The executed registry shape path admits output indices before NumPy selection."""
    indices = np.broadcast_to(np.array(0, dtype=np.int64), (10_000_000,))
    column_indices = indices.reshape((-1, 1))

    def objective(values: Any) -> object:
        if along_axis:
            return np.sum(np.take_along_axis(values.reshape((1, 2)), column_indices, axis=1))
        return np.sum(np.take(values, indices))

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
        )
    assert active_reserved_bytes() == baseline
    indices = np.array([1, 0, 1], dtype=np.int64)
    column_indices = indices.reshape((-1, 1))
    result = whole_program_value_and_grad(
        objective, np.array([2.0, 3.0]), trace=False, max_execution_gib=0.01
    )
    assert result.value == 8.0
    np.testing.assert_array_equal(result.gradient, np.array([1.0, 2.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("along_axis", [False, True])
def test_public_take_storage_stays_owned_after_numeric_selection(along_axis: bool) -> None:
    """Selection container charges survive until the objective result is formed."""
    observations: list[int] = []
    indices = np.array([1, 0, 1, 1], dtype=np.int64)
    row_indices = indices.reshape((1, 4))

    def observe_storage() -> None:
        observations.append(active_reserved_bytes())

    def objective(values: Any) -> object:
        observe_storage()
        selected = (
            np.take_along_axis(values.reshape((1, 2)), row_indices, axis=1)
            if along_axis
            else np.take(values, indices)
        )
        observe_storage()
        return np.sum(selected)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 11.0
    np.testing.assert_array_equal(result.gradient, np.array([1.0, 3.0]))
    assert len(observations) == 2
    assert observations[1] - observations[0] >= 4 * np.dtype(np.uintp).itemsize
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("along_axis", [False, True])
def test_public_take_failure_disposes_retained_selection_and_recovers(along_axis: bool) -> None:
    """An objective exception after selection releases both numeric and container charges."""
    failing = True
    indices = np.array([1, 0, 1, 1], dtype=np.int64)
    row_indices = indices.reshape((1, 4))

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after owned selection")

    def objective(values: Any) -> object:
        selected = (
            np.take_along_axis(values.reshape((1, 2)), row_indices, axis=1)
            if along_axis
            else np.take(values, indices)
        )
        fail_if_armed()
        return np.sum(selected)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after owned selection"):
        whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 11.0
    np.testing.assert_array_equal(result.gradient, np.array([1.0, 3.0]))
    assert active_reserved_bytes() == baseline


def test_public_take_scalar_return_preserves_gradient_and_releases_charge() -> None:
    """Scalar selection takes the early-return path without losing scope disposal."""

    def objective(values: Any) -> object:
        return np.take(values, 1)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 3.0
    np.testing.assert_array_equal(result.gradient, np.array([0.0, 1.0]))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("along_axis", [False, True])
def test_public_take_empty_result_preserves_zero_gradient(along_axis: bool) -> None:
    """Empty integer selectors preserve the empty reduction and owner disposal."""
    indices = np.empty(0, dtype=np.int64)
    row_indices = indices.reshape((1, 0))

    def objective(values: Any) -> object:
        selected = (
            np.take_along_axis(values.reshape((1, 2)), row_indices, axis=1)
            if along_axis
            else np.take(values, indices)
        )
        return np.sum(selected)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0]), trace=False)
    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, np.zeros(2))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("selector_kind", ["scalar", "slice", "boolean"])
def test_public_delete_retains_storage_and_preserves_gradient(selector_kind: str) -> None:
    """Deletion output containers remain charged after numeric layout finishes."""
    observations: list[int] = []
    selector = (
        1
        if selector_kind == "scalar"
        else slice(1, None, 2)
        if selector_kind == "slice"
        else np.array([False, True, False, True])
    )

    def observe_storage() -> None:
        observations.append(active_reserved_bytes())

    def objective(values: Any) -> object:
        observe_storage()
        selected = np.delete(values, selector)
        observe_storage()
        return np.sum(selected)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0, 4.0, 5.0]), trace=False)
    assert result.value == (11.0 if selector_kind == "scalar" else 6.0)
    np.testing.assert_array_equal(
        result.gradient,
        [1.0, 0.0, 1.0, 1.0] if selector_kind == "scalar" else [1.0, 0.0, 1.0, 0.0],
    )
    assert len(observations) == 2
    assert observations[1] - observations[0] >= 2 * np.dtype(np.uintp).itemsize
    assert active_reserved_bytes() == baseline


def test_public_delete_exception_releases_retained_storage_and_recovers() -> None:
    """Failure after deletion releases retained numeric/output declarations."""
    failing = True

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after owned deletion")

    def objective(values: Any) -> object:
        selected = np.delete(values, slice(1, None, 2))
        fail_if_armed()
        return np.sum(selected)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after owned deletion"):
        whole_program_value_and_grad(objective, np.array([2.0, 3.0, 4.0, 5.0]), trace=False)
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, np.array([2.0, 3.0, 4.0, 5.0]), trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [1.0, 0.0, 1.0, 0.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("advanced", [False, True])
def test_public_getitem_trace_value_gradient_and_disposal(advanced: bool) -> None:
    """Basic and broadcast advanced selectors execute through actual trace ownership."""
    selector = (
        (np.array([[0], [1]]), slice(None), np.array([1, 3]))
        if advanced
        else (slice(None), Ellipsis, slice(1, None, 2))
    )
    failing = True

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after owned getitem")

    def objective(values: Any) -> object:
        selected = values.reshape((2, 3, 4))[selector]
        fail_if_armed()
        return np.sum(selected)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after owned getitem"):
        whole_program_value_and_grad(objective, np.arange(24.0), trace=False)
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, np.arange(24.0), trace=False)
    assert result.value == 144.0
    expected_gradient = np.zeros(24)
    expected_gradient[1::2] = 1.0
    np.testing.assert_array_equal(result.gradient, expected_gradient)
    assert active_reserved_bytes() == baseline


def test_public_ranked_cumulative_product_and_predicate_shape_paths_preserve_gradient() -> None:
    """Row-major dimensions preserve an independently calculated combined gradient."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return (
            np.sum(np.cumsum(matrix, axis=1))
            + np.sum(np.prod(matrix, axis=1))
            + np.sum(np.where(matrix > 2.0, matrix, 0.0))
        )

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0, 4.0], trace=False)
    assert result.value == 35.0
    np.testing.assert_array_equal(result.gradient, [4.0, 2.0, 7.0, 5.0])
    assert active_reserved_bytes() == baseline


def test_public_compact_cumsum_refuses_live_cap_before_actual_callback_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real compact entry observes a tightened cap before its value callback.

    Parameters
    ----------
    monkeypatch
        Restores the process budget environment after boundary fault injection.

    """
    entered = False
    callback_entered = False
    baseline = active_reserved_bytes()
    previous = sys.getprofile()

    def objective(values: Any) -> object:
        return np.sum(np.cumsum(values))

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered, callback_entered
        if event != "call":
            return
        if frame.f_code.co_name == "_evaluate_trace_compact_rule":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
        elif (
            frame.f_back is not None
            and frame.f_back.f_code.co_name == "_evaluate_trace_compact_rule"
        ):
            if frame.f_code.co_name == "value_fn":
                callback_entered = True

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", ".01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
        finally:
            sys.setprofile(previous)
    assert entered
    assert not callback_entered
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_public_compact_cumsum_observes_actual_jvp_return_cancellation() -> None:
    """Actual callback completion checkpoints before the next tangent coordinate."""
    cancelled = Event()
    previous = sys.getprofile()
    baseline = active_reserved_bytes()
    callbacks = 0

    def objective(values: Any) -> object:
        return np.sum(np.cumsum(values))

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal callbacks
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code.co_name == "jvp_rule"
            and caller is not None
            and caller.f_code.co_name == "_evaluate_trace_compact_rule"
        ):
            callbacks += 1
            cancelled.set()

    sys.setprofile(profile)
    try:
        with pytest.raises(ExecutionCancelledError):
            whole_program_value_and_grad(
                objective, [1.0, 2.0, 3.0], trace=False, cancelled=cancelled
            )
    finally:
        sys.setprofile(previous)
    assert callbacks == 1
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0, 1.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("fault", ["value-shape", "tangent-shape", "nonfinite"])
def test_public_compact_cumsum_rejects_corrupted_actual_return_and_recovers(fault: str) -> None:
    """Malformed real callback returns unwind ownership before a valid retry.

    Parameters
    ----------
    fault
        Corruption applied once to an actual numerical callback result.

    """
    previous = sys.getprofile()
    baseline = active_reserved_bytes()
    changed = False

    def objective(values: Any) -> object:
        return np.sum(np.cumsum(values))

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal changed
        caller = frame.f_back
        if changed or event != "return" or caller is None:
            return
        if caller.f_code.co_name != "_evaluate_trace_compact_rule" or not isinstance(
            arg, np.ndarray
        ):
            return
        target = "jvp_rule" if fault == "tangent-shape" else "value_fn"
        if frame.f_code.co_name != target:
            return
        changed = True
        if fault == "nonfinite":
            arg.flat[0] = np.nan
        else:
            arg.dtype = np.float32

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="shape mismatch|outputs must be finite"):
            whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    finally:
        sys.setprofile(previous)
    assert changed
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_public_compact_cumsum_failure_after_output_releases_retained_charge_and_recovers() -> (
    None
):
    """Objective failure after a real compact output disposes its retained storage."""
    failing = True
    observations: list[int] = []

    def observe_storage() -> None:
        observations.append(active_reserved_bytes())

    def fail_if_armed() -> None:
        if failing:
            raise RuntimeError("after compact output")

    def objective(values: Any) -> object:
        observe_storage()
        output = np.cumsum(values)
        observe_storage()
        fail_if_armed()
        return np.sum(output)

    baseline = active_reserved_bytes()
    with pytest.raises(RuntimeError, match="after compact output"):
        whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    assert len(observations) == 2
    assert observations[1] > observations[0]
    assert active_reserved_bytes() == baseline
    failing = False
    result = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
    assert result.value == 10.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_public_compact_cumsum_zero_parameter_constant_output_disposes_owners() -> None:
    """Zero-coordinate compact output preserves constant reduction and cleanup."""

    def objective(values: Any) -> object:
        return np.sum(np.cumsum(np.concatenate((values, np.array([1.0, 2.0])))))

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, np.empty(0, dtype=np.float64))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("shape,axis", [((0,), 0), ((0, 2), 0), ((2, 0), 0), ((2, 0), 1)])
def test_public_empty_sum_axis_preserves_zero_gradient(shape: tuple[int, ...], axis: int) -> None:
    """Empty axis reductions retain NumPy identities and release declared pointer storage.

    Parameters
    ----------
    shape
        Empty ranked input layout.
    axis
        Explicit reduction axis, including a nonempty axis of an empty layout.

    """
    baseline = active_reserved_bytes()

    def objective(values: Any) -> object:
        empty = np.repeat(values, 0).reshape(shape)
        return np.sum(np.sum(empty, axis=axis))

    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, np.zeros(2))
    assert active_reserved_bytes() == baseline
