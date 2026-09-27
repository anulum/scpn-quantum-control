# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD array indexing module
# scpn-quantum-control -- Program AD static array indexing rules
"""Static array-indexing derivative rules for Program AD registry dispatch."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .dense_budget import DenseAllocationError
from .differentiable_parameter_contracts import _as_real_numeric_array
from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import reserve_execution_memory
from .program_ad_registry import (
    _PROGRAM_AD_ARRAY_IDENTITIES,
    _PROGRAM_AD_ARRAY_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)

ProgramADArrayTakeMode = Literal["raise", "wrap", "clip"]

_PROGRAM_AD_ARRAY_TAKE_MODES = frozenset(("raise", "wrap", "clip"))
_PROGRAM_AD_STATIC_INDEX_ERROR = (
    "program AD array getitem requires static integer or boolean index arrays, "
    "integer/slice/ellipsis/newaxis selectors, and static integer slice bounds"
)


def _program_ad_array_take_mode(mode: object, *, context: str) -> ProgramADArrayTakeMode:
    if not isinstance(mode, str) or mode not in _PROGRAM_AD_ARRAY_TAKE_MODES:
        raise ValueError(
            f"program AD array take {context} supports mode in ('raise', 'wrap', 'clip')"
        )
    return cast(ProgramADArrayTakeMode, mode)


def _program_ad_array_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD array primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_array_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD array primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_array_derivative_rule(name: str) -> CustomDerivativeRule:
    """Build a fail-closed trace-dispatch contract for an array primitive.

    Parameters
    ----------
    name:
        Registry primitive name used to identify the intercepted Program AD
        array operation.

    Returns
    -------
    CustomDerivativeRule
        Placeholder value and JVP callbacks that reject direct execution until
        trace dispatch supplies the primitive operands.

    """
    return CustomDerivativeRule(
        name=f"program_ad_array_{name}_trace_contract",
        value_fn=_program_ad_array_direct_value,
        jvp_rule=_program_ad_array_direct_jvp,
    )


def _program_ad_array_normalise_static_shape(
    primitive_name: str, source_shape: Sequence[int]
) -> tuple[int, ...]:
    """Return a validated immutable source shape for a static array rule.

    Parameters
    ----------
    primitive_name:
        Array primitive name included in validation diagnostics.
    source_shape:
        Static source-array dimensions supplied by the registry or direct-rule
        factory.

    Returns
    -------
    tuple[int, ...]
        Canonical non-negative integer dimensions.

    Raises
    ------
    ValueError
        Raised when any static dimension is negative.
    DenseAllocationError
        Native dimension or flattened int64 index storage is unaddressable.

    """
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires non-negative dimensions"
        )
    _program_ad_array_static_size(shape)
    return shape


def _program_ad_array_static_size(source_shape: tuple[int, ...]) -> int:
    """Return the scalar storage size for a static array shape.

    Parameters
    ----------
    source_shape:
        Canonical source-array dimensions.

    Returns
    -------
    int
        Product of the static dimensions, with scalars represented as size one.

    Raises
    ------
    DenseAllocationError
        A dimension or flattened int64 index storage is unaddressable.

    """
    if any(dimension < 0 or dimension > sys.maxsize for dimension in source_shape):
        raise DenseAllocationError("static array dimension exceeds native addressability")
    size = math.prod(source_shape)
    if size > sys.maxsize // np.dtype(np.int64).itemsize:
        raise DenseAllocationError("static array index storage exceeds native addressability")
    return size


def _program_ad_array_signature(source_shape: tuple[int, ...]) -> str:
    """Return a deterministic shape fragment for derivative-rule names.

    Parameters
    ----------
    source_shape:
        Canonical source-array dimensions.

    Returns
    -------
    str
        ``"scalar"`` for rank-zero inputs, otherwise dimensions joined by
        ``"x"``.

    """
    return "scalar" if not source_shape else "x".join(str(dimension) for dimension in source_shape)


def _program_ad_array_vector(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    if type(values) is np.ndarray and values.size != expected_size:
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    vector = _as_real_numeric_array(f"program AD array {primitive_name} {role}", values).reshape(
        -1
    )
    if vector.size != expected_size:
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    return vector


@contextmanager
def _program_ad_array_getitem_layout_plan(
    source_shape: tuple[int, ...],
    index: object,
) -> Iterator[tuple[ExecutionMemoryPlan, int, object]]:
    """Own selector snapshots through selection while deriving its buffer plan."""
    source_size = _program_ad_array_static_size(source_shape)
    _validate_static_basic_index(index)
    selectors = list(index if isinstance(index, tuple) else (index,))
    selectors = [
        _static_index_array(item) if isinstance(item, (np.ndarray, list)) else item
        for item in selectors
    ]
    snapshot_bytes = sum(
        int(item.size) * max(int(item.dtype.itemsize), np.dtype(np.intp).itemsize)
        for item in selectors
        if isinstance(item, np.ndarray)
    )
    snapshot_plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer(
                "getitem_selector_snapshot", "intermediate", (max(1, snapshot_bytes),), "uint8", 2
            ),
        )
    )
    with reserve_execution_memory(snapshot_plan) as snapshot_owner:
        snapshot_owner.checkpoint()
        immutable: list[object] = []
        for item in selectors:
            snapshot_owner.checkpoint()
            if isinstance(item, np.ndarray):
                copied = np.array(item, copy=True, order="C", subok=False)
                copied.setflags(write=False)
                immutable.append(copied)
            else:
                immutable.append(item)
        snapshot_owner.checkpoint()
        selectors = immutable
        prepared_index = tuple(selectors) if isinstance(index, tuple) else selectors[0]
        if sum(item is Ellipsis for item in selectors) > 1:
            raise ValueError("only one ellipsis is allowed in a static index")
        consumed = sum(
            int(item.ndim) if isinstance(item, np.ndarray) and item.dtype.kind == "b" else 1
            for item in selectors
            if item is not None and item is not Ellipsis
        )
        missing = len(source_shape) - consumed
        if missing < 0:
            raise ValueError("static index consumes more dimensions than the source")
        if any(item is Ellipsis for item in selectors):
            expanded: list[object] = []
            for item in selectors:
                expanded.extend([slice(None)] * missing if item is Ellipsis else [item])
            selectors = expanded
        else:
            selectors.extend([slice(None)] * missing)
        axis = 0
        basic_dimensions: list[int] = []
        advanced_shapes: list[tuple[int, ...]] = []
        selector_bytes = 0
        coordinate_size = 0
        for item in selectors:
            if item is None:
                basic_dimensions.append(1)
            elif isinstance(item, slice):
                basic_dimensions.append(len(range(*item.indices(source_shape[axis]))))
                axis += 1
            elif isinstance(item, np.ndarray):
                selector_bytes += int(item.size) * max(
                    int(item.dtype.itemsize), np.dtype(np.intp).itemsize
                )
                if item.dtype.kind == "b":
                    rank = int(item.ndim)
                    if tuple(item.shape) != source_shape[axis : axis + rank]:
                        raise ValueError("boolean index dimensions differ from source dimensions")
                    count = int(np.count_nonzero(item))
                    advanced_shapes.append((count,))
                    coordinate_size += count * rank
                    axis += rank
                else:
                    advanced_shapes.append(tuple(int(size) for size in item.shape))
                    axis += 1
            else:
                axis += 1
        broadcast_shape = np.broadcast_shapes(*advanced_shapes) if advanced_shapes else ()
        output_size = math.prod(basic_dimensions) * math.prod(broadcast_shape)
        plan = ExecutionMemoryPlan(
            (
                ExecutionBuffer(
                    "getitem_source",
                    "intermediate",
                    (max(1, source_size),),
                    "int64",
                    2,
                ),
                ExecutionBuffer(
                    "getitem_output", "intermediate", (max(1, output_size),), "int64", 3
                ),
                ExecutionBuffer(
                    "getitem_selectors", "intermediate", (max(1, selector_bytes),), "uint8", 2
                ),
                ExecutionBuffer(
                    "getitem_boolean_coordinates",
                    "intermediate",
                    (max(1, coordinate_size),),
                    "int64",
                    2,
                ),
            )
        )
        yield plan, output_size, prepared_index
        snapshot_owner.checkpoint()


def _program_ad_array_getitem_flat_indices(
    source_shape: tuple[int, ...], index: object
) -> NDArray[np.int64]:
    _validate_static_basic_index(index)
    try:
        with (
            _program_ad_array_getitem_layout_plan(source_shape, index) as (
                plan,
                output_size,
                prepared_index,
            ),
            reserve_execution_memory(plan) as reservation,
        ):
            reservation.checkpoint()
            source_indices = np.arange(
                _program_ad_array_static_size(source_shape), dtype=np.int64
            ).reshape(source_shape)
            try:
                selected = source_indices[cast(Any, prepared_index)]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "program AD array getitem direct rule requires in-bounds indices"
                ) from exc
            reservation.checkpoint()
            if np.asarray(selected).size != output_size:
                raise ValueError("getitem layout differs from admitted output size")
            return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array getitem direct rule requires in-bounds indices"
        ) from exc


def _program_ad_array_take_indices(indices: object) -> NDArray[np.int64]:
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take direct rule requires static integer indices")
    return cast(NDArray[np.int64], np.asarray(raw_indices, dtype=np.int64))


def _program_ad_array_take_target_shape(
    source_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    axis: int | None,
    *,
    along_axis: bool = False,
) -> tuple[int, ...]:
    """Derive output dimensions without creating source or output index arrays."""
    if axis is None:
        return indices_shape
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    if not along_axis:
        return source_shape[:normalised_axis] + indices_shape + source_shape[normalised_axis + 1 :]
    if len(indices_shape) != len(source_shape):
        raise ValueError(
            "program AD array take_along_axis requires indices in bounds and shape-compatible with source rank"
        )
    dimensions = tuple(1 if i == normalised_axis else size for i, size in enumerate(source_shape))
    try:
        return tuple(int(size) for size in np.broadcast_shapes(dimensions, indices_shape))
    except ValueError as exc:
        raise ValueError(
            "program AD array take_along_axis requires indices with shape compatible with the source"
        ) from exc


def _program_ad_array_take_layout_plan(
    source_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    axis: int | None,
    *,
    along_axis: bool = False,
) -> ExecutionMemoryPlan:
    """Declare source, selector, output and non-axis coordinate buffers."""
    source_size = _program_ad_array_static_size(source_shape)
    output_shape = _program_ad_array_take_target_shape(
        source_shape, indices_shape, axis, along_axis=along_axis
    )
    coordinate_size = (
        sum(size for i, size in enumerate(source_shape) if i != axis) if along_axis else 0
    )
    return ExecutionMemoryPlan(
        (
            ExecutionBuffer("take_source", "intermediate", (max(1, source_size),), "int64", 2),
            ExecutionBuffer(
                "take_output", "intermediate", (max(1, math.prod(output_shape)),), "int64", 3
            ),
            ExecutionBuffer(
                "take_indices", "intermediate", (max(1, math.prod(indices_shape)),), "int64", 2
            ),
            ExecutionBuffer(
                "take_coordinates", "intermediate", (max(1, coordinate_size),), "int64"
            ),
        )
    )


def _program_ad_array_take_flat_indices(
    source_shape: tuple[int, ...],
    indices: object,
    axis: int | None,
    mode: str,
) -> NDArray[np.int64]:
    raw_indices = _program_ad_array_take_indices(indices)
    mode_name = _program_ad_array_take_mode(mode, context="direct rule")
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source_shape))
    plan = _program_ad_array_take_layout_plan(
        source_shape,
        tuple(int(size) for size in raw_indices.shape),
        normalised_axis,
        along_axis=False,
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(
            _program_ad_array_static_size(source_shape), dtype=np.int64
        ).reshape(source_shape)
        try:
            selected = np.take(source_indices, raw_indices, axis=normalised_axis, mode=mode_name)
        except (IndexError, ValueError) as exc:
            if mode_name == "raise":
                raise ValueError(
                    "program AD array take direct rule requires in-bounds indices"
                ) from exc
            raise ValueError(
                "program AD array take direct rule requires axis-compatible indices"
            ) from exc
        reservation.checkpoint()
        return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_take_along_axis_flat_indices(
    source_shape: tuple[int, ...],
    indices: object,
    axis: int,
) -> NDArray[np.int64]:
    raw_indices = _program_ad_array_take_indices(indices)
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    plan = _program_ad_array_take_layout_plan(
        source_shape,
        tuple(int(size) for size in raw_indices.shape),
        normalised_axis,
        along_axis=True,
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(
            _program_ad_array_static_size(source_shape), dtype=np.int64
        ).reshape(source_shape)
        try:
            selected = np.take_along_axis(source_indices, raw_indices, axis=normalised_axis)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "program AD array take_along_axis direct rule requires in-bounds indices "
                "with shape compatible with the source"
            ) from exc
        reservation.checkpoint()
        return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_delete_object(obj: object, *, context: str) -> object:
    if isinstance(obj, (bool, np.bool_)):
        raise ValueError(
            f"program AD array delete {context} requires static integer, slice, "
            "or boolean deletion selectors"
        )
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, slice):
        components = (obj.start, obj.stop, obj.step)
        if any(
            component is not None
            and (
                isinstance(component, (bool, np.bool_))
                or not isinstance(component, (int, np.integer))
            )
            for component in components
        ):
            raise ValueError(
                f"program AD array delete {context} requires static integer slice bounds"
            )
        return slice(
            None if obj.start is None else int(cast(int | np.integer, obj.start)),
            None if obj.stop is None else int(cast(int | np.integer, obj.stop)),
            None if obj.step is None else int(cast(int | np.integer, obj.step)),
        )
    raw_obj = np.asarray(obj)
    if raw_obj.dtype.kind == "b":
        return np.asarray(raw_obj, dtype=np.bool_)
    if raw_obj.dtype.kind in {"i", "u"}:
        if raw_obj.shape == ():
            return int(raw_obj)
        return np.asarray(raw_obj, dtype=np.int64)
    raise ValueError(
        f"program AD array delete {context} requires static integer, slice, "
        "or boolean deletion selectors"
    )


def _program_ad_array_delete_layout_plan(
    source_shape: tuple[int, ...],
    selector: object,
    axis: int | None,
) -> ExecutionMemoryPlan:
    """Declare source/output upper bounds and NumPy deletion mask workspace."""
    source_size = _program_ad_array_static_size(source_shape)
    extent = source_size if axis is None else source_shape[axis]
    mask_size = (
        extent
        if (
            isinstance(selector, np.ndarray)
            and not (selector.size == 1 and selector.dtype.kind in {"i", "u"})
        )
        or (isinstance(selector, slice) and selector.step is not None and abs(selector.step) > 1)
        else 0
    )
    selector_size = int(selector.size) if isinstance(selector, np.ndarray) else 1
    return ExecutionMemoryPlan(
        (
            ExecutionBuffer(
                "delete_source_output", "intermediate", (max(1, source_size),), "int64", 4
            ),
            ExecutionBuffer("delete_mask", "intermediate", (max(1, mask_size),), "bool", 2),
            ExecutionBuffer(
                "delete_selector", "intermediate", (max(1, selector_size),), "int64", 2
            ),
        )
    )


def _program_ad_array_delete_flat_indices(
    source_shape: tuple[int, ...],
    obj: object,
    axis: int | None,
) -> NDArray[np.int64]:
    delete_obj = _program_ad_array_delete_object(obj, context="direct rule")
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source_shape))
    plan = _program_ad_array_delete_layout_plan(source_shape, delete_obj, normalised_axis)
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(
            _program_ad_array_static_size(source_shape), dtype=np.int64
        ).reshape(source_shape)
        source = source_indices.reshape(-1) if normalised_axis is None else source_indices
        try:
            selected = np.delete(source, cast(Any, delete_obj), axis=normalised_axis)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "program AD array delete direct rule requires static in-bounds deletion selectors "
                "and an axis-compatible source"
            ) from exc
        reservation.checkpoint()
        return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_pad_mode(mode: object, *, context: str) -> str:
    if not isinstance(mode, str) or mode != "constant":
        raise ValueError(f"program AD array pad {context} supports constant mode only")
    return "constant"


def _program_ad_array_pad_width(
    pad_width: object,
    ndim: int,
    *,
    context: str,
) -> tuple[tuple[int, int], ...]:
    raw_width = np.asarray(pad_width)
    if raw_width.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"program AD array pad {context} requires static non-negative integer pad widths"
        )
    if ndim == 0:
        if raw_width.size == 0:
            return ()
        raise ValueError("program AD array pad scalar sources require empty pad widths")
    if raw_width.shape == ():
        width = int(raw_width)
        pairs = tuple((width, width) for _ in range(ndim))
    elif raw_width.shape == (2,):
        before, after = (int(item) for item in raw_width)
        pairs = tuple((before, after) for _ in range(ndim))
    elif raw_width.shape == (1, 2):
        before, after = (int(item) for item in raw_width.reshape(2))
        pairs = tuple((before, after) for _ in range(ndim))
    elif raw_width.shape == (ndim, 2):
        pairs = tuple((int(row[0]), int(row[1])) for row in raw_width)
    else:
        raise ValueError(
            f"program AD array pad {context} requires scalar, pair, or per-axis pad widths"
        )
    if any(before < 0 or after < 0 for before, after in pairs):
        raise ValueError(
            f"program AD array pad {context} requires static non-negative integer pad widths"
        )
    return pairs


def _program_ad_array_pad_constant_values(value: object, *, context: str) -> object:
    raw_value = np.asarray(value)
    if raw_value.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError(
            f"program AD array pad {context} requires static finite real constant_values"
        )
    constants = np.asarray(raw_value, dtype=np.float64)
    if not np.all(np.isfinite(constants)):
        raise ValueError(
            f"program AD array pad {context} requires static finite real constant_values"
        )
    return value


def _program_ad_array_pad_target_shape(
    source_shape: tuple[int, ...], pad_width: object, constant_values: object, *, context: str
) -> tuple[int, ...]:
    """Validate small static pad metadata without constructing a numeric layout."""
    pairs = _program_ad_array_pad_width(pad_width, len(source_shape), context=context)
    constants = _program_ad_array_pad_constant_values(constant_values, context=context)
    try:
        np.broadcast_to(np.asarray(constants, dtype=np.float64), (len(source_shape), 2))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array pad requires constants compatible with the source rank"
        ) from exc
    return tuple(
        size + before + after for size, (before, after) in zip(source_shape, pairs, strict=True)
    )


def _program_ad_array_pad_layout(
    source_shape: tuple[int, ...],
    pad_width: object,
    constant_values: object,
    *,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], tuple[int, ...]]:
    pairs = _program_ad_array_pad_width(pad_width, len(source_shape), context=context)
    constants = _program_ad_array_pad_constant_values(constant_values, context=context)
    output_shape = _program_ad_array_pad_target_shape(
        source_shape, pairs, constants, context=context
    )
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer(
                "pad_layout_source",
                "intermediate",
                (max(1, math.prod(source_shape)),),
                "float64",
                2,
            ),
            ExecutionBuffer(
                "pad_layout_output",
                "intermediate",
                (max(1, math.prod(output_shape)),),
                "float64",
                4,
            ),
        )
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(
            _program_ad_array_static_size(source_shape), dtype=np.int64
        ).reshape(source_shape)
        source_zeros = np.zeros(source_shape, dtype=np.float64)
        try:
            padded_indices = np.pad(
                source_indices,
                pairs,
                mode="constant",
                constant_values=-1,
            )
            padded_constants = np.pad(
                source_zeros,
                pairs,
                mode="constant",
                constant_values=cast(Any, constants),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "program AD array pad requires static pad widths and constant_values "
                "compatible with the source rank"
            ) from exc
        reservation.checkpoint()
        return (
            cast(NDArray[np.int64], np.asarray(padded_indices, dtype=np.int64).reshape(-1)),
            cast(NDArray[np.float64], np.asarray(padded_constants, dtype=np.float64).reshape(-1)),
            tuple(int(dimension) for dimension in np.asarray(padded_indices).shape),
        )


def _program_ad_array_insert_object(obj: object, *, context: str) -> object:
    if isinstance(obj, (bool, np.bool_)):
        raise ValueError(
            f"program AD array insert {context} requires static integer insertion indices"
        )
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, slice):
        components = (obj.start, obj.stop, obj.step)
        if any(
            component is not None
            and (
                isinstance(component, (bool, np.bool_))
                or not isinstance(component, (int, np.integer))
            )
            for component in components
        ):
            raise ValueError(
                f"program AD array insert {context} requires static integer insertion indices"
            )
        return slice(
            None if obj.start is None else int(cast(int | np.integer, obj.start)),
            None if obj.stop is None else int(cast(int | np.integer, obj.stop)),
            None if obj.step is None else int(cast(int | np.integer, obj.step)),
        )
    raw_obj = np.asarray(obj)
    if raw_obj.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"program AD array insert {context} requires static integer insertion indices"
        )
    if raw_obj.shape == ():
        return int(raw_obj)
    return np.asarray(raw_obj, dtype=np.int64)


def _program_ad_array_insert_values(values: object, *, context: str) -> NDArray[np.float64]:
    raw_values = np.asarray(values)
    if raw_values.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError(
            f"program AD array insert {context} requires static finite real insert values"
        )
    insert_values = np.asarray(raw_values, dtype=np.float64)
    if not np.all(np.isfinite(insert_values)):
        raise ValueError(
            f"program AD array insert {context} requires static finite real insert values"
        )
    return insert_values


def _program_ad_array_insert_axis(axis: object, ndim: int, *, context: str) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            f"program AD array insert {context} requires a static integer axis or None"
        )
    return _normalise_axis("axis", int(axis), ndim)


def _program_ad_array_insert_target_shape(
    source_shape: tuple[int, ...], obj: object, values: object, axis: object, *, context: str
) -> tuple[int, ...]:
    """Derive insertion dimensions and broadcast compatibility without an output array."""
    selector = _program_ad_array_insert_object(obj, context=context)
    constants = _program_ad_array_insert_values(values, context=context)
    selected_axis = _program_ad_array_insert_axis(axis, len(source_shape), context=context)
    source = (math.prod(source_shape),) if selected_axis is None else source_shape
    selected_axis = 0 if selected_axis is None else selected_axis
    extent = source[selected_axis]
    scalar_selector = isinstance(selector, int)
    if isinstance(selector, slice):
        indices: Sequence[int] = range(*selector.indices(extent))
    elif scalar_selector:
        indices = (cast(int, selector),)
    else:
        index_array = np.asarray(selector)
        if index_array.ndim > 1:
            raise ValueError("program AD array insert requires scalar or one-dimensional indices")
        indices = tuple(int(value) for value in index_array.reshape(-1))
    count = len(indices)
    if count == 1 and (indices[0] < -extent or indices[0] > extent):
        raise ValueError("program AD array insert requires static in-bounds insertion indices")
    constant_shape = (1,) * max(0, len(source) - constants.ndim) + constants.shape
    if count == 1:
        if scalar_selector:
            dimensions = list(constant_shape)
            first = dimensions.pop(0)
            dimensions.insert(selected_axis, first)
            constant_shape = tuple(dimensions)
        inserted = constant_shape[selected_axis]
    else:
        inserted = count
    insertion_shape = list(source)
    insertion_shape[selected_axis] = inserted
    try:
        if np.broadcast_shapes(constant_shape, tuple(insertion_shape)) != tuple(insertion_shape):
            raise ValueError("incompatible insertion broadcast")
    except ValueError as exc:
        raise ValueError(
            "program AD array insert values are incompatible with the source shape"
        ) from exc
    target = list(source)
    target[selected_axis] += inserted
    return tuple(target)


def _program_ad_array_insert_layout(
    source_shape: tuple[int, ...],
    obj: object,
    values: object,
    axis: object,
    *,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], tuple[int, ...]]:
    insert_obj = _program_ad_array_insert_object(obj, context=context)
    insert_values = _program_ad_array_insert_values(values, context=context)
    normalised_axis = _program_ad_array_insert_axis(axis, len(source_shape), context=context)
    output_shape = _program_ad_array_insert_target_shape(
        source_shape, obj, values, axis, context=context
    )
    axis_extent = (
        math.prod(source_shape) if normalised_axis is None else source_shape[normalised_axis]
    )
    index_count = (
        len(range(*insert_obj.indices(axis_extent)))
        if isinstance(insert_obj, slice)
        else 1
        if isinstance(insert_obj, int)
        else np.asarray(insert_obj).size
    )
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer(
                "insert_source", "intermediate", (max(1, math.prod(source_shape)),), "float64", 2
            ),
            ExecutionBuffer(
                "insert_output", "intermediate", (max(1, math.prod(output_shape)),), "float64", 4
            ),
            ExecutionBuffer(
                "insert_index_order", "intermediate", (max(1, index_count),), "int64", 4
            ),
            ExecutionBuffer(
                "insert_mask",
                "intermediate",
                (max(1, output_shape[0 if normalised_axis is None else normalised_axis]),),
                "bool",
            ),
            ExecutionBuffer(
                "insert_markers", "intermediate", (max(1, insert_values.size),), "int64"
            ),
        )
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(
            _program_ad_array_static_size(source_shape), dtype=np.int64
        ).reshape(source_shape)
        source_zeros = np.zeros(source_shape, dtype=np.float64)
        marker_values: object
        marker_values = -1 if insert_values.shape == () else np.full(insert_values.shape, -1)
        source = source_indices.reshape(-1) if normalised_axis is None else source_indices
        try:
            inserted_indices = np.insert(
                source,
                cast(Any, insert_obj),
                cast(Any, marker_values),
                axis=normalised_axis,
            )
            inserted_constants = np.insert(
                source_zeros.reshape(-1) if normalised_axis is None else source_zeros,
                cast(Any, insert_obj),
                insert_values,
                axis=normalised_axis,
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "program AD array insert requires static insertion indices and insert values "
                "compatible with the source shape"
            ) from exc
        reservation.checkpoint()
        return (
            cast(NDArray[np.int64], np.asarray(inserted_indices, dtype=np.int64).reshape(-1)),
            cast(
                NDArray[np.float64], np.asarray(inserted_constants, dtype=np.float64).reshape(-1)
            ),
            tuple(int(dimension) for dimension in np.asarray(inserted_indices).shape),
        )


def _program_ad_array_direct_gather(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        vector = _program_ad_array_vector(
            primitive_name,
            "values",
            values,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        result = _program_ad_float64_vector_result(vector[flat_indices])
        reservation.checkpoint()
        return result


def _program_ad_array_direct_gather_jvp(
    primitive_name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        _program_ad_array_vector(
            primitive_name,
            "values",
            values,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        tangent_vector = _program_ad_array_vector(
            primitive_name,
            "tangent",
            tangent,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        result = _program_ad_float64_vector_result(tangent_vector[flat_indices])
        reservation.checkpoint()
        return result


def _program_ad_array_direct_scatter_vjp(
    primitive_name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_size = _program_ad_array_static_size(source_shape)
        _program_ad_array_vector(primitive_name, "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            primitive_name,
            "cotangent",
            cotangent,
            expected_size=int(flat_indices.size),
        )
        result = np.zeros(source_size, dtype=np.float64)
        np.add.at(result, flat_indices, cotangent_vector)
        reservation.checkpoint()
        return result


def _program_ad_array_direct_transform_plan(
    source_shape: tuple[int, ...], output_size: int
) -> ExecutionMemoryPlan:
    """Declare retained layout and float64/indexing transform workspace."""
    return ExecutionMemoryPlan(
        (
            ExecutionBuffer(
                "direct_transform_source",
                "intermediate",
                (max(1, _program_ad_array_static_size(source_shape)),),
                "float64",
                4,
            ),
            ExecutionBuffer(
                "direct_transform_output", "intermediate", (max(1, output_size),), "float64", 6
            ),
            ExecutionBuffer(
                "direct_transform_mask", "intermediate", (max(1, output_size),), "bool"
            ),
        )
    )


def _program_ad_array_direct_pad_value(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
    flat_constants: NDArray[np.float64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        vector = _program_ad_array_vector(
            primitive_name,
            "values",
            values,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        result = np.array(flat_constants, dtype=np.float64, copy=True)
        source_mask = flat_indices >= 0
        result[source_mask] = vector[flat_indices[source_mask]]
        reservation.checkpoint()
        return result


def _program_ad_array_direct_pad_jvp(
    primitive_name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        _program_ad_array_vector(
            primitive_name,
            "values",
            values,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        tangent_vector = _program_ad_array_vector(
            primitive_name,
            "tangent",
            tangent,
            expected_size=_program_ad_array_static_size(source_shape),
        )
        result = np.zeros(int(flat_indices.size), dtype=np.float64)
        source_mask = flat_indices >= 0
        result[source_mask] = tangent_vector[flat_indices[source_mask]]
        reservation.checkpoint()
        return result


def _program_ad_array_direct_pad_vjp(
    primitive_name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    plan = _program_ad_array_direct_transform_plan(source_shape, int(flat_indices.size))
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_size = _program_ad_array_static_size(source_shape)
        _program_ad_array_vector(primitive_name, "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            primitive_name,
            "cotangent",
            cotangent,
            expected_size=int(flat_indices.size),
        )
        result = np.zeros(source_size, dtype=np.float64)
        source_mask = flat_indices >= 0
        np.add.at(result, flat_indices[source_mask], cotangent_vector[source_mask])
        reservation.checkpoint()
        return result


def program_ad_array_getitem_derivative_rule(
    source_shape: Sequence[int],
    index: object,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed static gather index.

    Parameters
    ----------
    source_shape:
        Static shape of the source array being indexed.
    index:
        Static NumPy-compatible index made from integers, slices, ellipses,
        ``None``, or integer/boolean index arrays.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and VJP rule over the flattened source vector.

    Raises
    ------
    ValueError
        Raised when the source shape is negative, the index is not a supported
        static selector, or the selected coordinates are out of bounds.

    DenseAllocationError
        Declared transform buffers, or guarded take-layout construction, exceed
        live host/cgroup or configured capacity.

    """
    source = _program_ad_array_normalise_static_shape("getitem", source_shape)
    flat_indices = _program_ad_array_getitem_flat_indices(source, index)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "getitem", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "getitem",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "getitem",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=f"program_ad_array_getitem_{_program_ad_array_signature(source)}_static_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_take_derivative_rule(
    source_shape: Sequence[int],
    indices: object,
    *,
    axis: int | None = None,
    mode: str = "raise",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy ``take`` call.

    Parameters
    ----------
    source_shape:
        Static shape of the source array being gathered.
    indices:
        Static integer index array used by ``np.take``.
    axis:
        Axis supplied to ``np.take``. ``None`` selects the flattened source.
    mode:
        NumPy take mode: ``"raise"``, ``"wrap"``, or ``"clip"``.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and scatter-add VJP rule over the flattened source.

    Raises
    ------
    ValueError
        Raised when the source shape, indices, axis, or mode cannot define a
        static in-bounds ``np.take`` gather.

    DenseAllocationError
        Declared transform buffers, or guarded take-layout construction, exceed
        live host/cgroup or configured capacity.

    """
    mode_name = _program_ad_array_take_mode(mode, context="direct rule")
    source = _program_ad_array_normalise_static_shape("take", source_shape)
    flat_indices = _program_ad_array_take_flat_indices(source, indices, axis, mode_name)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    axis_signature = "flat" if normalised_axis is None else str(normalised_axis)
    mode_signature = "" if mode_name == "raise" else f"_mode_{mode_name}"

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "take", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "take",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "take",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_take_"
            f"{_program_ad_array_signature(source)}_axis_{axis_signature}"
            f"{mode_signature}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_take_along_axis_derivative_rule(
    source_shape: Sequence[int],
    indices: object,
    *,
    axis: int,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed ``take_along_axis``.

    Parameters
    ----------
    source_shape:
        Static shape of the source array being gathered.
    indices:
        Static integer index array compatible with ``source_shape`` and
        ``axis``.
    axis:
        Static axis supplied to ``np.take_along_axis``.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and scatter-add VJP rule over the flattened source.

    Raises
    ------
    ValueError
        Raised when the source shape, indices, or axis cannot define a static
        shape-compatible ``np.take_along_axis`` gather.

    DenseAllocationError
        Declared transform buffers, or guarded take-layout construction, exceed
        live host/cgroup or configured capacity.

    """
    source = _program_ad_array_normalise_static_shape("take_along_axis", source_shape)
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD array take_along_axis direct rule requires static integer axis"
        )
    normalised_axis = _normalise_axis("axis", int(axis), len(source))
    flat_indices = _program_ad_array_take_along_axis_flat_indices(source, indices, normalised_axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "take_along_axis", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "take_along_axis",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "take_along_axis",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_take_along_axis_"
            f"{_program_ad_array_signature(source)}_axis_{normalised_axis}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_delete_derivative_rule(
    source_shape: Sequence[int],
    obj: object,
    *,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy ``delete`` call.

    Parameters
    ----------
    source_shape:
        Static shape of the source array before deletion.
    obj:
        Static integer, slice, integer-array, or boolean-array deletion
        selector.
    axis:
        Axis supplied to ``np.delete``. ``None`` selects the flattened source.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and scatter-add VJP rule over the flattened source.

    Raises
    ------
    ValueError
        Raised when the source shape, deletion selector, or axis cannot define
        a static in-bounds ``np.delete`` gather.

    DenseAllocationError
        Declared transform buffers, or guarded take-layout construction, exceed
        live host/cgroup or configured capacity.

    """
    source = _program_ad_array_normalise_static_shape("delete", source_shape)
    flat_indices = _program_ad_array_delete_flat_indices(source, obj, axis)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    axis_signature = "flat" if normalised_axis is None else str(normalised_axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "delete", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "delete",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "delete",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_delete_"
            f"{_program_ad_array_signature(source)}_axis_{axis_signature}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_pad_derivative_rule(
    source_shape: Sequence[int],
    pad_width: object,
    *,
    constant_values: object = 0.0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed constant padding.

    Parameters
    ----------
    source_shape:
        Static shape of the source array before padding.
    pad_width:
        Static NumPy-compatible pad-width declaration.
    constant_values:
        Static finite real constants used for padded cells.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and source-scatter VJP rule over the flattened source.

    Raises
    ------
    ValueError
        Raised when the source shape, pad width, or constant values cannot
        define a finite static constant-padding layout.
    DenseAllocationError
        Declared layout or transform buffers exceed live or configured capacity.

    """
    source = _program_ad_array_normalise_static_shape("pad", source_shape)
    flat_indices, flat_constants, _ = _program_ad_array_pad_layout(
        source,
        pad_width,
        constant_values,
        context="direct rule",
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_value(
            "pad",
            values,
            source_shape=source,
            flat_indices=flat_indices,
            flat_constants=flat_constants,
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_jvp(
            "pad",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_vjp(
            "pad",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_pad_"
            f"{_program_ad_array_signature(source)}_static_constant_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_insert_derivative_rule(
    source_shape: Sequence[int],
    obj: object,
    values: object,
    *,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed constant insertion.

    Parameters
    ----------
    source_shape:
        Static shape of the source array before insertion.
    obj:
        Static integer, slice, or integer-array insertion selector.
    values:
        Static finite real insertion values.
    axis:
        Axis supplied to ``np.insert``. ``None`` selects the flattened source.

    Returns
    -------
    CustomDerivativeRule
        Direct value, JVP, and source-scatter VJP rule over the flattened source.

    Raises
    ------
    ValueError
        Raised when the source shape, insertion selector, insertion constants,
        or axis cannot define a finite static insertion layout.
    DenseAllocationError
        Declared insertion layout or transform exceeds live or configured capacity.

    """
    source = _program_ad_array_normalise_static_shape("insert", source_shape)
    flat_indices, flat_constants, _ = _program_ad_array_insert_layout(
        source,
        obj,
        values,
        axis,
        context="direct rule",
    )
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    axis_signature = "flat" if normalised_axis is None else str(normalised_axis)

    def value_fn(source_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_value(
            "insert",
            source_values,
            source_shape=source,
            flat_indices=flat_indices,
            flat_constants=flat_constants,
        )

    def jvp_rule(
        source_values: NDArray[np.float64], tangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_jvp(
            "insert",
            source_values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        source_values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_pad_vjp(
            "insert",
            source_values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_insert_"
            f"{_program_ad_array_signature(source)}_axis_{axis_signature}"
            "_static_constant_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _normalise_axis(name: str, axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _validate_static_basic_index(index: object) -> None:
    if isinstance(index, tuple):
        for selector in index:
            _validate_static_basic_index_selector(selector)
        return
    _validate_static_basic_index_selector(index)


def _validate_static_basic_index_selector(selector: object) -> None:
    if isinstance(selector, (bool, np.bool_)):
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if selector is Ellipsis or selector is None:
        return
    if isinstance(selector, (int, np.integer)):
        return
    if isinstance(selector, slice):
        for item in (selector.start, selector.stop, selector.step):
            if item is not None and (
                isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
            ):
                raise ValueError("program AD basic indexing requires static integer slice bounds")
        return
    if isinstance(selector, (np.ndarray, list)):
        _static_index_array(selector)
        return
    raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)


def _static_index_array(selector: object) -> NDArray[Any]:
    array = np.asarray(selector)
    if array.dtype.kind not in {"i", "u", "b"}:
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if array.shape == () and array.dtype.kind == "b":
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    return array


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(values, dtype=np.float64).reshape(-1))


def _is_program_ad_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""
    return type(value).__name__ == "TraceADArray"


def _program_ad_trace_array_shape(value: object) -> tuple[int, ...]:
    """Return a static shape from a structural trace-array value."""
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        raise ValueError("program AD array trace shape must be static")
    return tuple(int(dimension) for dimension in shape)


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static shape for a trace array or concrete array-like value.

    Parameters
    ----------
    value:
        Whole-program trace array or concrete array-like operand.

    Returns
    -------
    tuple[int, ...]
        Static dimensions used by Program AD shape, dtype, and static-argument
        rules.

    Raises
    ------
    ValueError
        Raised when a structural trace array does not expose a static tuple
        shape.

    """
    if _is_program_ad_trace_array(value):
        return _program_ad_trace_array_shape(value)
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name for a trace array or concrete array-like value."""
    if _is_program_ad_trace_array(value):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD array primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _program_ad_array_getitem_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD getitem primitive."""
    if len(args) != 2:
        raise ValueError("program AD array getitem shape rule requires array and index")
    _validate_static_basic_index(args[1])
    source_shape = _program_ad_array_shape_of(args[0])
    try:
        with (
            _program_ad_array_getitem_layout_plan(source_shape, args[1]) as (
                plan,
                output_size,
                prepared_index,
            ),
            reserve_execution_memory(plan) as reservation,
        ):
            reservation.checkpoint()
            source = np.arange(math.prod(source_shape), dtype=np.int64).reshape(source_shape)
            try:
                selected = source[cast(Any, prepared_index)]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "program AD array getitem shape rule requires in-bounds indices"
                ) from exc
            reservation.checkpoint()
            if np.asarray(selected).size != output_size:
                raise ValueError("getitem layout differs from admitted output size")
            return tuple(int(dimension) for dimension in np.asarray(selected).shape)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("program AD array getitem shape rule requires in-bounds indices") from exc


def _program_ad_array_take_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD take primitive."""
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array take shape rule requires array, indices, axis, and mode"
        )
    indices = args[1]
    axis = cast(int | None, args[2]) if len(args) >= 3 else None
    mode = cast(str, args[3]) if len(args) == 4 else "raise"
    mode_name = _program_ad_array_take_mode(mode, context="shape rule")
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take shape rule requires static integer indices")
    source_shape = _program_ad_array_shape_of(args[0])
    normalised_axis = (
        None
        if axis is None or not source_shape
        else _normalise_axis("axis", int(axis), len(source_shape))
    )
    plan = _program_ad_array_take_layout_plan(
        source_shape,
        tuple(int(size) for size in raw_indices.shape),
        normalised_axis,
        along_axis=False,
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source = np.arange(math.prod(source_shape), dtype=np.int64).reshape(source_shape)
        try:
            selected = np.take(source, raw_indices, axis=axis, mode=mode_name)
        except (IndexError, ValueError) as exc:
            if mode_name == "raise":
                raise ValueError(
                    "program AD array take shape rule indices must be in bounds"
                ) from exc
            raise ValueError(
                "program AD array take shape rule requires axis-compatible indices"
            ) from exc
        reservation.checkpoint()
        return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_take_along_axis_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD take-along-axis primitive."""
    if len(args) != 3:
        raise ValueError(
            "program AD array take_along_axis shape rule requires array, indices, and axis"
        )
    axis = args[2]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD array take_along_axis shape rule requires static integer axis"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "program AD array take_along_axis shape rule requires static integer indices"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    normalised_axis = (
        None if axis is None else _normalise_axis("axis", int(axis), len(source_shape))
    )
    plan = _program_ad_array_take_layout_plan(
        source_shape,
        tuple(int(size) for size in raw_indices.shape),
        normalised_axis,
        along_axis=True,
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source = np.arange(math.prod(source_shape), dtype=np.int64).reshape(source_shape)
        try:
            selected = np.take_along_axis(source, raw_indices, axis=int(axis))
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "program AD array take_along_axis shape rule indices must be in bounds "
                "and shape-compatible"
            ) from exc
        reservation.checkpoint()
        return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_delete_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD delete primitive."""
    if len(args) not in {2, 3}:
        raise ValueError("program AD array delete shape rule requires array, object, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    delete_obj = _program_ad_array_delete_object(args[1], context="shape rule")
    axis = args[2] if len(args) == 3 else None
    source: NDArray[np.int64]
    if axis is None:
        normalised_axis = None
    else:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError("program AD array delete shape rule requires static integer axis")
        normalised_axis = _normalise_axis("axis", int(axis), len(source_shape))
    plan = _program_ad_array_delete_layout_plan(source_shape, delete_obj, normalised_axis)
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        source_indices = np.arange(math.prod(source_shape), dtype=np.int64).reshape(source_shape)
        source = source_indices.reshape(-1) if normalised_axis is None else source_indices
        try:
            selected = np.delete(source, cast(Any, delete_obj), axis=normalised_axis)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "program AD array delete shape rule requires static in-bounds deletion selectors"
            ) from exc
        reservation.checkpoint()
        return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_pad_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD pad primitive."""
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array pad shape rule requires array, pad_width, mode, and constants"
        )
    mode = args[2] if len(args) >= 3 else "constant"
    _program_ad_array_pad_mode(mode, context="shape rule")
    return _program_ad_array_pad_target_shape(
        _program_ad_array_shape_of(args[0]),
        args[1],
        args[3] if len(args) == 4 else 0.0,
        context="shape rule",
    )


def _program_ad_array_insert_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD insert primitive."""
    if len(args) not in {3, 4}:
        raise ValueError(
            "program AD array insert shape rule requires array, object, values, and axis"
        )
    return _program_ad_array_insert_target_shape(
        _program_ad_array_shape_of(args[0]),
        args[1],
        args[2],
        args[3] if len(args) == 4 else None,
        context="shape rule",
    )


def _program_ad_array_dtype_rule(args: tuple[object, ...]) -> str:
    """Return the dtype emitted by a Program AD array primitive."""
    if not args:
        raise ValueError("program AD array dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_array_getitem_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD getitem primitive."""
    if len(args) != 2:
        raise ValueError("program AD array getitem static rule requires array and index")
    _validate_static_basic_index(args[1])
    index = args[1]
    if isinstance(index, tuple):
        return (tuple(_program_ad_array_static_index_component(item) for item in index),)
    return (_program_ad_array_static_index_component(index),)


def _program_ad_array_static_index_component(selector: object) -> object:
    """Return a canonical immutable representation of a static index selector."""
    if selector is Ellipsis or selector is None:
        return selector
    if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
        return int(selector)
    if isinstance(selector, slice):
        return slice(
            None if selector.start is None else int(selector.start),
            None if selector.stop is None else int(selector.stop),
            None if selector.step is None else int(selector.step),
        )
    if isinstance(selector, (np.ndarray, list)):
        array = _static_index_array(selector)
        dtype_name = "bool" if array.dtype.kind == "b" else "int64"
        values = tuple(
            bool(item) if array.dtype.kind == "b" else int(item) for item in array.reshape(-1)
        )
        return (
            "static_index_array",
            dtype_name,
            tuple(int(dimension) for dimension in array.shape),
            values,
        )
    raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)


def _program_ad_array_take_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD take primitive."""
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array take static rule requires array, indices, axis, and mode"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take static rule requires static integer indices")
    axis = cast(int | None, args[2]) if len(args) >= 3 else None
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, (int, np.integer))):
        raise ValueError("program AD array take static rule requires static integer axis")
    mode = cast(str, args[3]) if len(args) == 4 else "raise"
    mode_name = _program_ad_array_take_mode(mode, context="static rule")
    return (
        tuple(int(index) for index in raw_indices.reshape(-1)),
        None if axis is None else int(axis),
        mode_name,
    )


def _program_ad_array_take_along_axis_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD take-along-axis primitive."""
    if len(args) != 3:
        raise ValueError(
            "program AD array take_along_axis static rule requires array, indices, and axis"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "program AD array take_along_axis static rule requires static integer indices"
        )
    axis = args[2]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD array take_along_axis static rule requires static integer axis"
        )
    normalised_axis = _normalise_axis("axis", int(axis), len(_program_ad_array_shape_of(args[0])))
    return (
        tuple(int(index) for index in raw_indices.reshape(-1)),
        tuple(int(dimension) for dimension in raw_indices.shape),
        normalised_axis,
    )


def _program_ad_array_delete_static_object(obj: object) -> object:
    """Return a canonical immutable representation of a delete selector."""
    delete_obj = _program_ad_array_delete_object(obj, context="static rule")
    if isinstance(delete_obj, int):
        return delete_obj
    if isinstance(delete_obj, slice):
        return delete_obj
    delete_array = np.asarray(delete_obj)
    if delete_array.dtype.kind == "b":
        return (
            "static_delete_mask",
            tuple(int(dimension) for dimension in delete_array.shape),
            tuple(bool(item) for item in delete_array.reshape(-1)),
        )
    return tuple(int(index) for index in delete_array.reshape(-1))


def _program_ad_array_delete_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD delete primitive."""
    if len(args) not in {2, 3}:
        raise ValueError("program AD array delete static rule requires array, object, and axis")
    axis = args[2] if len(args) == 3 else None
    if axis is None:
        normalised_axis = None
    else:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError("program AD array delete static rule requires static integer axis")
        normalised_axis = _normalise_axis(
            "axis", int(axis), len(_program_ad_array_shape_of(args[0]))
        )
    return (_program_ad_array_delete_static_object(args[1]), normalised_axis)


def _program_ad_array_pad_static_constants(value: object) -> object:
    """Return canonical static pad constants."""
    constants = _program_ad_array_pad_constant_values(value, context="static rule")
    constant_array = np.asarray(constants, dtype=np.float64)
    if constant_array.shape == ():
        return float(constant_array)
    return (
        "static_pad_constants",
        tuple(int(dimension) for dimension in constant_array.shape),
        tuple(float(item) for item in constant_array.reshape(-1)),
    )


def _program_ad_array_pad_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD pad primitive."""
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array pad static rule requires array, pad_width, mode, and constants"
        )
    mode = args[2] if len(args) >= 3 else "constant"
    mode_name = _program_ad_array_pad_mode(mode, context="static rule")
    pad_width = _program_ad_array_pad_width(
        args[1],
        len(_program_ad_array_shape_of(args[0])),
        context="static rule",
    )
    return (
        pad_width,
        mode_name,
        _program_ad_array_pad_static_constants(args[3] if len(args) == 4 else 0.0),
    )


def _program_ad_array_insert_static_object(obj: object) -> object:
    """Return a canonical immutable representation of an insert selector."""
    insert_obj = _program_ad_array_insert_object(obj, context="static rule")
    if isinstance(insert_obj, int):
        return insert_obj
    if isinstance(insert_obj, slice):
        return insert_obj
    return tuple(int(index) for index in np.asarray(insert_obj).reshape(-1))


def _program_ad_array_insert_static_values(values: object) -> object:
    """Return canonical static insert values."""
    insert_values = _program_ad_array_insert_values(values, context="static rule")
    if insert_values.shape == ():
        return float(insert_values)
    return (
        "static_insert_values",
        tuple(int(dimension) for dimension in insert_values.shape),
        tuple(float(item) for item in insert_values.reshape(-1)),
    )


def _program_ad_array_insert_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD insert primitive."""
    if len(args) not in {3, 4}:
        raise ValueError(
            "program AD array insert static rule requires array, object, values, and axis"
        )
    normalised_axis = _program_ad_array_insert_axis(
        args[3] if len(args) == 4 else None,
        len(_program_ad_array_shape_of(args[0])),
        context="static rule",
    )
    return (
        _program_ad_array_insert_static_object(args[1]),
        _program_ad_array_insert_static_values(args[2]),
        normalised_axis,
    )


_PROGRAM_AD_ARRAY_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "getitem": _program_ad_array_getitem_shape,
    "take": _program_ad_array_take_shape,
    "take_along_axis": _program_ad_array_take_along_axis_shape,
    "delete": _program_ad_array_delete_shape,
    "pad": _program_ad_array_pad_shape,
    "insert": _program_ad_array_insert_shape,
}

_PROGRAM_AD_ARRAY_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "getitem": _program_ad_array_getitem_static_arguments,
    "take": _program_ad_array_take_static_arguments,
    "take_along_axis": _program_ad_array_take_along_axis_static_arguments,
    "delete": _program_ad_array_delete_static_arguments,
    "pad": _program_ad_array_pad_static_arguments,
    "insert": _program_ad_array_insert_static_arguments,
}


def _program_ad_array_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Map an array-indexing primitive over a batch axis."""
    if len(args) != len(axes):
        raise ValueError("program AD array batching axes must match argument count")
    if not args:
        raise ValueError("program AD array batching requires an array operand")
    array = _as_real_numeric_array("program AD array batched operand", args[0])
    axis = axes[0]
    if axis is None:
        raise ValueError("program AD array batching requires the array operand to be mapped")
    axis_index = _normalise_axis("axes[0]", axis, array.ndim)
    batch_size = int(array.shape[axis_index])
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD array batching supports static non-array arguments only")
    outputs = [
        _as_real_numeric_array(
            "program AD array batched output",
            function(np.take(array, batch_index, axis=axis_index), *args[1:]),
        )
        for batch_index in range(batch_size)
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_array_lowering_metadata(name: str) -> Mapping[str, str]:
    """Return lowering metadata for a Program AD array primitive."""
    static_signature = {
        "getitem": "source_shape:ranked_tensor_shape;index:static_gather_index",
        "take": "source_shape:ranked_tensor_shape;indices_axis_mode",
        "take_along_axis": "source_shape:ranked_tensor_shape;indices_shape_axis",
        "delete": "source_shape:ranked_tensor_shape;object_axis",
        "pad": "source_shape:ranked_tensor_shape;pad_width_constant_values",
        "insert": "source_shape:ranked_tensor_shape;object_values_axis",
    }[name]
    static_factory = {
        "getitem": "program_ad_array_getitem_derivative_rule",
        "take": "program_ad_array_take_derivative_rule",
        "take_along_axis": "program_ad_array_take_along_axis_derivative_rule",
        "delete": "program_ad_array_delete_derivative_rule",
        "pad": "program_ad_array_pad_derivative_rule",
        "insert": "program_ad_array_insert_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "getitem": "static_gather_index_scatter_add",
        "take": "static_integer_gather_scatter_add",
        "take_along_axis": "static_along_axis_gather_scatter_add",
        "delete": "static_delete_gather_scatter_add",
        "pad": "static_constant_pad_scatter_add",
        "insert": "static_constant_insert_scatter_add",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff array dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.array.{name}",
        "llvm": "blocked_until_executable_array_lowering",
        "rust": "blocked_until_polyglot_array_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_array_primitive_contracts() -> None:
    """Register fail-closed Program AD array primitive contracts.

    Returns
    -------
    None
        Registration mutates the default custom-derivative registry in place
        and leaves existing contracts untouched.

    """
    for name, identity in _PROGRAM_AD_ARRAY_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_array_derivative_rule(name),
                batching_rule=_program_ad_array_batching_rule,
                lowering_metadata=_program_ad_array_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_ARRAY_SHAPE_RULES[name],
                dtype_rule=_program_ad_array_dtype_rule,
                static_argument_rule=_PROGRAM_AD_ARRAY_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_ARRAY_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_array_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate array primitive dispatch helpers against concrete arguments."""
    if contract.static_argument_rule is None:
        raise ValueError(
            f"program AD primitive {contract.identity.key} missing static argument rule"
        )
    if contract.shape_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing shape rule")
    if contract.dtype_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing dtype rule")
    static_arguments = contract.static_argument_rule(args)
    if not isinstance(static_arguments, tuple):
        raise ValueError(
            f"program AD primitive {contract.identity.key} static rule must return a tuple"
        )
    shape = contract.shape_rule(args)
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"program AD primitive {contract.identity.key} shape rule must return "
            "non-negative integer dimensions"
        )
    dtype = contract.dtype_rule(args)
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(
            f"program AD primitive {contract.identity.key} dtype rule must return a dtype name"
        )


def _require_program_ad_array_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered array primitive runtime contract.

    Parameters
    ----------
    name:
        Program AD array primitive name, such as ``"getitem"`` or ``"pad"``.
    args:
        Optional concrete or trace operands used to validate static-argument,
        shape, and dtype dispatch helpers.

    Returns
    -------
    PrimitiveContract
        Complete registry contract for the requested fail-closed array
        primitive.

    Raises
    ------
    ValueError
        Raised when the primitive name is unknown, the registry contract is
        incomplete, the metadata policy is not fail-closed, or supplied
        arguments violate the contract dispatch helpers.

    """
    identity: PrimitiveIdentity | None = _PROGRAM_AD_ARRAY_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD array primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_ARRAY_POLICY:
        raise ValueError(f"invalid program AD array primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD array primitive effect for {identity.key}")

    missing: list[str] = []
    if contract.batching_rule is None:
        missing.append("batching_rule")
    if not contract.lowering_metadata:
        missing.append("lowering_metadata")
    if not contract.lowering_metadata.get("mlir_op"):
        missing.append("mlir_op")
    if not contract.lowering_metadata.get("nondifferentiable_boundary"):
        missing.append("nondifferentiable_boundary")
    if contract.lowering_metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
        missing.append("nondifferentiable_boundary_policy")
    if contract.shape_rule is None:
        missing.append("shape_rule")
    if contract.dtype_rule is None:
        missing.append("dtype_rule")
    if contract.static_argument_rule is None:
        missing.append("static_argument_rule")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"incomplete program AD array primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_array_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_normalise_axis",
    "_program_ad_array_derivative_rule",
    "_program_ad_array_normalise_static_shape",
    "_program_ad_array_shape_of",
    "_program_ad_array_signature",
    "_program_ad_array_static_size",
    "_register_program_ad_array_primitive_contracts",
    "_require_program_ad_array_contract",
    "program_ad_array_delete_derivative_rule",
    "program_ad_array_getitem_derivative_rule",
    "program_ad_array_insert_derivative_rule",
    "program_ad_array_pad_derivative_rule",
    "program_ad_array_take_along_axis_derivative_rule",
    "program_ad_array_take_derivative_rule",
)
