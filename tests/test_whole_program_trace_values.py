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

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import (
    TraceADArray,
    TraceADScalar,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]
ArrayFunction = Callable[..., object]
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
) -> None:
    """Exercise one fail-closed NumPy protocol call through the public AD API."""

    def objective(values: FloatArray) -> object:
        traced = cast(TraceADArray, values)
        return traced.__array_function__(
            function,
            (TraceADArray,),
            args_factory(traced),
            kwargs_factory(traced),
        )

    with pytest.raises(ValueError, match=message):
        whole_program_value_and_grad(
            objective,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            trace=False,
        )


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
    _expect_array_function_failure(
        function,
        args_factory,
        kwargs_factory,
        message,
    )


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
    _expect_array_function_failure(
        np.clip,
        lambda array: (array, -1.0, 1.0, object()),
        lambda _array: {},
        r"np\.clip supports array, lower, and upper",
    )
    _expect_array_function_failure(
        np.clip,
        lambda array: (array, -1.0, 1.0),
        lambda _array: {"out": object()},
        r"np\.clip supports array, lower, and upper",
    )
