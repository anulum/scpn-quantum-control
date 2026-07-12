# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program trace metadata module
# scpn-quantum-control -- Whole-program AD trace operand metadata normalisation
"""Static normalisation of whole-program AD trace operation shapes and axes.

These helpers canonicalise and validate the static (value-independent) shape and
axis arguments of whole-program AD trace operations -- target reshape and
broadcast shapes, shape-transform and axis-permutation axes, repeat and tile
counts, roll shifts, ``rot90`` rotations, sort and reduction axes, single-axis
normalisation, and broadcast shapes. They depend only on shape/axis metadata,
never on trace values or the trace context, so the operator-intercepted trace
runtime can keep these contracts separate from the value-carrying classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np


def _normalise_trace_reshape_shape(shape: object, size: int) -> tuple[int, ...]:
    dimensions: tuple[int, ...]
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        dimensions = (int(shape),)
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        raw_dimensions = tuple(cast(Any, shape))
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer))
            for dimension in raw_dimensions
        ):
            raise ValueError("program AD reshape shape dimensions must be static integers")
        dimensions = tuple(int(dimension) for dimension in raw_dimensions)
    else:
        raise ValueError("program AD reshape shape must be a static integer or shape tuple")
    inferred_axes = tuple(index for index, dimension in enumerate(dimensions) if dimension == -1)
    if len(inferred_axes) > 1:
        raise ValueError("program AD reshape supports at most one inferred dimension")
    if any(dimension < -1 for dimension in dimensions):
        raise ValueError("program AD reshape dimensions must be non-negative or -1")
    known_product = int(np.prod(tuple(dimension for dimension in dimensions if dimension != -1)))
    if inferred_axes:
        if known_product == 0:
            raise ValueError("program AD reshape cannot infer dimension from zero product")
        if size % known_product != 0:
            raise ValueError("program AD reshape inferred dimension must preserve size")
        inferred = size // known_product
        dimensions = tuple(inferred if dimension == -1 else dimension for dimension in dimensions)
    if int(np.prod(dimensions)) != size:
        raise ValueError("program AD reshape must preserve size")
    return dimensions


def _normalise_trace_broadcast_shape(shape: object) -> tuple[int, ...]:
    dimensions: tuple[int, ...]
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        dimensions = (int(shape),)
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        dimensions = tuple(int(dimension) for dimension in shape)
    else:
        raise ValueError("program AD np.broadcast_to requires an integer shape")
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("program AD np.broadcast_to shape dimensions must be non-negative")
    return dimensions


def _normalise_shape_transform_axes(
    name: str, axis: int | tuple[int, ...], *, output_rank: int
) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalised: list[int] = []
    for item in axes:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"program AD {name} axes must be static integers")
        value = int(item)
        if value < 0:
            value += output_rank
        if value < 0 or value >= output_rank:
            raise ValueError(f"program AD {name} axis out of bounds")
        if value in normalised:
            raise ValueError(f"program AD {name} axes must be unique")
        normalised.append(value)
    return tuple(sorted(normalised))


def _normalise_axis_permutation_axis(name: str, axis: object, *, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(f"program AD {name} axes must be static integers")
    value = int(axis)
    if value < 0:
        value += rank
    if value < 0 or value >= rank:
        raise ValueError(f"program AD {name} axis out of bounds")
    return value


def _normalise_axis_permutation_axes(
    name: str, axes: object, *, rank: int, role: str
) -> tuple[int, ...]:
    if isinstance(axes, (int, np.integer)):
        raw_axes = (axes,)
    else:
        try:
            raw_axes = tuple(cast(Any, axes))
        except TypeError as exc:
            raise ValueError(f"program AD {name} {role} axes must be static integers") from exc
    normalised = tuple(
        _normalise_axis_permutation_axis(name, axis, rank=rank) for axis in raw_axes
    )
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"program AD {name} {role} axes must be unique")
    return normalised


def _normalise_repeat_count(count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("program AD repeat counts must be static non-negative integers")
    value = int(count)
    if value < 0:
        raise ValueError("program AD repeat counts must be static non-negative integers")
    return value


def _normalise_repeat_counts(repeats: object, selected_size: int) -> int | tuple[int, ...]:
    if isinstance(repeats, (int, np.integer)) and not isinstance(repeats, bool):
        return _normalise_repeat_count(repeats)
    try:
        raw_repeats = tuple(cast(Any, repeats))
    except TypeError as exc:
        raise ValueError("program AD repeat counts must be static non-negative integers") from exc
    if len(raw_repeats) != selected_size:
        raise ValueError("program AD repeat counts length must match selected axis")
    return tuple(_normalise_repeat_count(item) for item in raw_repeats)


def _normalise_tile_reps(reps: object) -> tuple[int, ...]:
    values: tuple[int, ...]
    if isinstance(reps, (int, np.integer)) and not isinstance(reps, bool):
        values = (int(reps),)
    else:
        try:
            values = tuple(cast(Any, reps))
        except TypeError as exc:
            raise ValueError("program AD tile reps must be static non-negative integers") from exc
        if not values:
            raise ValueError("program AD tile reps must contain at least one axis")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in values
        ):
            raise ValueError("program AD tile reps must be static non-negative integers")
        values = tuple(int(item) for item in values)
    if any(value < 0 for value in values):
        raise ValueError("program AD tile reps must be static non-negative integers")
    return values


def _normalise_roll_shift_scalar(shift: object) -> int:
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        raise ValueError("program AD roll shift must be static integers")
    return int(shift)


def _normalise_roll_shift_tuple(shift: object, axis_count: int) -> tuple[int, ...]:
    if isinstance(shift, (int, np.integer)) and not isinstance(shift, bool):
        return tuple(int(shift) for _ in range(axis_count))
    try:
        raw_shifts = tuple(cast(Any, shift))
    except TypeError as exc:
        raise ValueError("program AD roll shift must be static integers") from exc
    if len(raw_shifts) != axis_count:
        raise ValueError("program AD roll shift and axis lengths must match")
    return tuple(_normalise_roll_shift_scalar(item) for item in raw_shifts)


def _normalise_rot90_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("program AD rot90 k must be a static integer")
    return int(k)


def _normalise_rot90_axes(axes: object, *, rank: int) -> tuple[int, int]:
    normalised = _normalise_axis_permutation_axes("rot90", axes, rank=rank, role="axes")
    if len(normalised) != 2:
        raise ValueError("program AD rot90 axes must contain exactly two axes")
    return (normalised[0], normalised[1])


def _normalise_sort_axis(axis: object, rank: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.sort axis must be a static integer or None")
    axis_index = int(axis)
    if axis_index < 0:
        axis_index += rank
    if axis_index < 0 or axis_index >= rank:
        raise ValueError("program AD np.sort axis out of bounds")
    return axis_index


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return a NumPy-compatible broadcast shape or fail closed."""
    try:
        shape: tuple[int, ...] = np.broadcast_shapes(*shapes)
        return shape
    except ValueError as exc:
        raise ValueError(
            "whole-program AD array operands must follow NumPy broadcasting rules"
        ) from exc


def _normalise_trapezoid_axis(axis: object, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.trapezoid axis must be a static integer")
    try:
        return _normalise_axis("axis", int(axis), ndim)
    except ValueError as exc:
        if "out of bounds" in str(exc):
            raise ValueError("program AD np.trapezoid axis out of bounds") from exc
        raise


def _normalise_axis(name: str, axis: int, ndim: int) -> int:
    """Return a non-negative axis for an array with ``ndim`` dimensions."""
    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis
