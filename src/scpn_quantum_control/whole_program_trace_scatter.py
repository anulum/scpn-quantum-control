# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — unbuffered trace array accumulation
"""Validate and execute bounded NumPy scatter through existing trace primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from .execution_memory import ExecutionBuffer

if TYPE_CHECKING:
    from .whole_program_trace_values import TraceADArray, TraceADScalar

_INTEGER_TYPES = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_REAL_TYPES = (*_INTEGER_TYPES, float, np.float16, np.float32, np.float64, np.longdouble)


def _trace_add_at(
    ufunc: np.ufunc,
    inputs: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> None:
    """Accumulate rank-one indexed updates without buffering duplicate targets.

    Parameters
    ----------
    ufunc
        Exact NumPy ``add`` ufunc invoking its ``at`` method.
    inputs
        Trace destination, static integer indices and finite real updates.
        A same-trace scalar broadcasts; rank-one updates match index count.
    kwargs
        Empty keyword mapping required by the native ``ufunc.at`` signature.

    Raises
    ------
    ValueError
        If operands are opaque, dynamic, nonfinite, foreign-trace, out of bounds
        or outside the declared rank-one subset. Admission precedes mutation.

    Notes
    -----
    Updates snapshot their original scalar references before writing, including
    when destination and update arrays alias. Each addition and assignment uses
    the existing trace primitives and ordered mutation-version evidence.

    """
    from .whole_program_trace_values import TraceADArray

    if ufunc is not np.add or kwargs or len(inputs) != 3:
        raise ValueError("whole-program AD scatter supports np.add.at without keywords only")
    destination, indices, updates = inputs
    if type(destination) is not TraceADArray or destination.ndim != 1:
        raise ValueError("whole-program AD scatter requires a rank-one trace destination")
    count = _index_count(indices)
    _update_count(updates, count, destination)
    with destination.context.array_storage(
        (max(1, count),),
        workspaces=(
            ExecutionBuffer("scatter_indices", "intermediate", (max(1, count),), "intp", 2),
            ExecutionBuffer(
                "scatter_update_tangents",
                "intermediate",
                (max(1, count), max(1, destination.context.parameter_count)),
                "float64",
                4,
            ),
        ),
    ) as reservation:
        targets = tuple(
            _index_at(indices, position, destination.size) for position in range(count)
        )
        scalars = _snapshot_updates(updates, count, destination)
        for target, scalar in zip(targets, scalars, strict=True):
            reservation.checkpoint()
            current = destination._items[target]
            destination._set_flat_item(target, current + scalar)


def _index_count(indices: object) -> int:
    """Inspect exact builtin or native integer storage without opaque conversion."""
    if any(type(indices) is kind for kind in _INTEGER_TYPES):
        return 1
    if (type(indices) is list or type(indices) is tuple) and isinstance(indices, list | tuple):
        return len(indices)
    if type(indices) is np.ndarray and indices.ndim == 1 and indices.dtype.kind in {"i", "u"}:
        return int(indices.size)
    raise ValueError("whole-program AD scatter indices must be static rank-one integers")


def _index_at(indices: object, position: int, size: int) -> int:
    """Normalize one inspected index and reject every invalid target before writes."""
    raw = indices[position] if isinstance(indices, list | tuple | np.ndarray) else indices
    if not any(type(raw) is kind for kind in _INTEGER_TYPES) or not isinstance(
        raw, int | np.integer
    ):
        raise ValueError("whole-program AD scatter indices must be static integers")
    index = int(raw)
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise ValueError("whole-program AD scatter index out of bounds")
    return index


def _update_count(updates: object, count: int, destination: TraceADArray) -> None:
    """Validate update storage, shape and trace ownership before snapshots."""
    from .whole_program_trace_values import TraceADArray, TraceADScalar

    if type(updates) is TraceADScalar:
        if updates.context is not destination.context:
            raise ValueError("whole-program AD scatter updates belong to a different trace")
        return
    if type(updates) is TraceADArray:
        if updates.context is not destination.context:
            raise ValueError("whole-program AD scatter updates belong to a different trace")
        if updates.shape == () or updates.shape == (count,):
            return
    elif any(type(updates) is kind for kind in _REAL_TYPES):
        return
    elif (type(updates) is list or type(updates) is tuple) and isinstance(updates, list | tuple):
        if len(updates) == count:
            return
    elif type(updates) is np.ndarray:
        if updates.dtype.kind in {"i", "u", "f"} and updates.shape in ((), (count,)):
            return
    raise ValueError(
        "whole-program AD scatter updates require real scalar or matching rank-one storage"
    )


def _snapshot_updates(
    updates: object, count: int, destination: TraceADArray
) -> tuple[TraceADScalar, ...]:
    """Freeze finite update values and aliases before the first accumulation."""
    from .whole_program_trace_values import TraceADArray, TraceADScalar, _coerce_trace_scalar

    if isinstance(updates, TraceADArray):
        source: tuple[object, ...] = (
            (updates.item(),) * max(1, count) if updates.shape == () else tuple(updates._items)
        )
    elif isinstance(updates, list | tuple):
        source = tuple(updates)
    elif isinstance(updates, np.ndarray):
        source = (updates[()],) * max(1, count) if updates.shape == () else tuple(updates)
    else:
        source = (updates,) * max(1, count)
    scalars: list[TraceADScalar] = []
    for item in source:
        if type(item) is TraceADScalar:
            if item.context is not destination.context:
                raise ValueError("whole-program AD scatter updates belong to a different trace")
            scalar = item
        elif any(type(item) is kind for kind in _REAL_TYPES) and isinstance(
            item, int | float | np.integer | np.floating
        ):
            try:
                value = float(item)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    "whole-program AD scatter updates must be finite real scalars"
                ) from error
            if not np.isfinite(value):
                raise ValueError("whole-program AD scatter updates must be finite real scalars")
            scalar = _coerce_trace_scalar(value, destination.context)
        else:
            raise ValueError("whole-program AD scatter updates must be finite real scalars")
        scalars.append(scalar)
    return tuple(scalars[:count])
