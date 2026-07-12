# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable vmap module
# scpn-quantum-control -- eager differentiable vmap transform
"""Eager vectorization transform for differentiable-programming objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import _normalise_axis
from .program_ad_registry import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    PrimitiveBatchingRule,
    PrimitiveIdentity,
)

VMapInAxes = int | None | Sequence[int | None]


def _is_trace_array(value: object) -> bool:
    """Return whether ``value`` is a facade-owned whole-program trace array."""
    return type(value).__name__ == "TraceADArray" and hasattr(value, "context")


def _is_trace_value(value: object) -> bool:
    """Return whether ``value`` is a facade-owned whole-program trace value."""
    return type(value).__name__ in {"TraceADScalar", "TraceADArray"} and hasattr(value, "context")


def _trace_array_shape(value: object) -> tuple[int, ...]:
    """Return a structural shape for a facade-owned trace array."""
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        raise ValueError("vmap trace array leaves must expose a static shape")
    return tuple(int(dimension) for dimension in shape)


def _trace_array_ndim(value: object) -> int:
    """Return a structural rank for a facade-owned trace array."""
    ndim = getattr(value, "ndim", None)
    if not isinstance(ndim, int):
        return len(_trace_array_shape(value))
    return ndim


def _trace_value_context(value: object) -> object:
    """Return the trace context carried by a facade-owned trace value."""
    if not _is_trace_value(value):
        raise ValueError("vmap trace output leaves must carry a trace context")
    return cast(Any, value).context


def _trace_take_value(array: object, item: int, *, axis: int) -> object:
    """Slice a trace array via the trace runtime without a module-load import cycle."""
    from .whole_program_trace_values import _trace_take

    return _trace_take(cast(Any, array), item, axis=axis, mode="raise")


def _coerce_trace_array_value(value: object, context: object) -> object:
    """Coerce a trace scalar/array output into a trace array via the trace runtime."""
    from .whole_program_trace_values import _coerce_trace_array

    return _coerce_trace_array(cast(Any, value), cast(Any, context))


def _trace_stack_values(values: Sequence[object], context: object, *, axis: int) -> object:
    """Stack trace arrays via the trace runtime without creating an import cycle."""
    from .whole_program_trace_values import _trace_stack

    return _trace_stack(tuple(cast(Any, value) for value in values), cast(Any, context), axis=axis)


def vmap(
    function: Callable[..., object],
    in_axes: VMapInAxes = 0,
    out_axes: int = 0,
    *,
    primitive_identity: PrimitiveIdentity | str | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> Callable[..., object]:
    """Return a composable vectorizing transform over leading or selected axes.

    The transform mirrors the practical contract of a JAX-style ``vmap`` for the
    native NumPy differentiable layer: mapped arguments are sliced along their
    declared axes, ``None`` axes are broadcast unchanged, and stackable scalar,
    array, tuple, list, or dict outputs are reassembled with the mapped axis at
    ``out_axes``. It is an eager deterministic transform, not a JIT compiler.
    """
    if not callable(function):
        raise ValueError("vmap function must be callable")
    if not isinstance(out_axes, int):
        raise ValueError("out_axes must be an integer")
    batching_rule: PrimitiveBatchingRule | None = None
    if primitive_identity is not None:
        target_registry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
        batching_rule = target_registry.require_batching_rule(primitive_identity)

    def vectorized(*args: object) -> object:
        if not args:
            raise ValueError("vmap requires at least one argument")
        axes = _normalise_vmap_in_axes(in_axes, len(args))
        mapped: list[tuple[object, int] | None] = []
        batch_size: int | None = None
        for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
            if axis is None:
                mapped.append(None)
                continue
            if _is_trace_array(arg):
                array: object = arg
                ndim = _trace_array_ndim(array)
                shape = _trace_array_shape(array)
            else:
                numeric_array = _as_real_numeric_array(f"vmap argument {index}", arg)
                array = numeric_array
                ndim = numeric_array.ndim
                shape = tuple(int(dimension) for dimension in numeric_array.shape)
            axis_index = _normalise_axis(f"in_axes[{index}]", axis, ndim)
            size = int(shape[axis_index])
            if size <= 0:
                raise ValueError("mapped axes must be non-empty")
            if batch_size is None:
                batch_size = size
            elif size != batch_size:
                raise ValueError("all mapped axes must have the same length")
            mapped.append((array, axis_index))
        if batch_size is None:
            raise ValueError("at least one in_axes entry must be mapped")
        if batching_rule is not None:
            return batching_rule(function, args, axes, out_axes)

        outputs = []
        for item in range(batch_size):
            call_args = []
            for arg, mapping in zip(args, mapped, strict=True):
                if mapping is None:
                    call_args.append(arg)
                    continue
                array, axis_index = mapping
                if _is_trace_array(array):
                    call_args.append(_trace_take_value(array, item, axis=axis_index))
                else:
                    call_args.append(
                        np.take(cast(NDArray[np.float64], array), item, axis=axis_index)
                    )
            outputs.append(function(*call_args))
        return _stack_vmap_outputs(outputs, out_axes)

    return vectorized


def _normalise_vmap_in_axes(in_axes: VMapInAxes, arity: int) -> tuple[int | None, ...]:
    """Return one input-axis declaration per positional argument."""
    if isinstance(in_axes, int) or in_axes is None:
        return tuple(in_axes for _ in range(arity))
    axes = tuple(in_axes)
    if len(axes) != arity:
        raise ValueError("in_axes length must match positional argument count")
    if any(axis is not None and not isinstance(axis, int) for axis in axes):
        raise ValueError("in_axes entries must be integers or None")
    return axes


def _stack_vmap_outputs(outputs: Sequence[object], out_axes: int) -> object:
    """Stack per-example outputs while preserving simple pytree structure."""
    if not outputs:
        raise ValueError("vmap outputs must be non-empty")
    first = outputs[0]
    if _is_trace_value(first):
        context = _trace_value_context(first)
        trace_arrays = [_coerce_trace_array_value(output, context) for output in outputs]
        shape = _trace_array_shape(trace_arrays[0])
        if any(_trace_array_shape(array) != shape for array in trace_arrays):
            raise ValueError("vmap output leaves must have consistent shapes")
        return _trace_stack_values(trace_arrays, context, axis=out_axes)
    if isinstance(first, np.ndarray) or np.isscalar(first):
        numeric_arrays = [np.asarray(output) for output in outputs]
        shape = numeric_arrays[0].shape
        if any(array.shape != shape for array in numeric_arrays):
            raise ValueError("vmap output leaves must have consistent shapes")
        axis = out_axes
        result_rank = numeric_arrays[0].ndim + 1
        if axis < 0:
            axis += result_rank
        if axis < 0 or axis >= result_rank:
            raise ValueError("out_axes is out of bounds for stacked output rank")
        stacked: NDArray[Any] = np.stack(numeric_arrays, axis=axis)
        if stacked.dtype.kind in {"b", "O", "S", "U"}:
            raise ValueError("vmap output leaves must be numeric")
        return stacked
    if isinstance(first, tuple):
        if any(not isinstance(output, tuple) or len(output) != len(first) for output in outputs):
            raise ValueError("vmap tuple outputs must have consistent structure")
        return tuple(
            _stack_vmap_outputs(
                [cast(tuple[object, ...], output)[index] for output in outputs], out_axes
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(output, list) or len(output) != len(first) for output in outputs):
            raise ValueError("vmap list outputs must have consistent structure")
        return [
            _stack_vmap_outputs(
                [cast(list[object], output)[index] for output in outputs], out_axes
            )
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        keys = tuple(first.keys())
        if any(not isinstance(output, dict) or tuple(output.keys()) != keys for output in outputs):
            raise ValueError("vmap dict outputs must have consistent keys")
        return {
            key: _stack_vmap_outputs(
                [cast(dict[object, object], output)[key] for output in outputs], out_axes
            )
            for key in keys
        }
    raise ValueError("vmap output leaves must be numeric arrays, scalars, tuples, lists, or dicts")
