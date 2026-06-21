# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD shape transform rules
"""Static shape-transform derivative rules for Program AD registry dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_registry import CustomDerivativeRule


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(values, dtype=np.float64).reshape(-1))


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


def _program_ad_shape_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD shape primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_shape_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD shape primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_shape_derivative_rule(name: str) -> CustomDerivativeRule:
    return CustomDerivativeRule(
        name=f"program_ad_shape_{name}_trace_contract",
        value_fn=_program_ad_shape_direct_value,
        jvp_rule=_program_ad_shape_direct_jvp,
    )


def _program_ad_shape_normalise_static_shape(
    primitive_name: str, shape: Sequence[int]
) -> tuple[int, ...]:
    normalised = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in normalised):
        raise ValueError(
            f"program AD shape {primitive_name} direct rule requires non-negative dimensions"
        )
    return normalised


def _program_ad_shape_static_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _program_ad_shape_signature(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def _program_ad_shape_vector(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD shape {primitive_name} {role}", values).reshape(
        -1
    )
    if vector.size != expected_size:
        raise ValueError(
            f"program AD shape {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    return vector


def program_ad_shape_reshape_derivative_rule(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed reshape signature."""

    source = _program_ad_shape_normalise_static_shape("reshape", source_shape)
    target = _program_ad_shape_normalise_static_shape("reshape", target_shape)
    source_size = _program_ad_shape_static_size(source)
    target_size = _program_ad_shape_static_size(target)
    if source_size != target_size:
        raise ValueError(
            "program AD shape reshape direct rule requires source and target "
            "with the same element count"
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(vector.reshape(source).reshape(target))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "reshape", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(tangent_vector.reshape(source).reshape(target))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "reshape", "cotangent", cotangent, expected_size=target_size
        )
        return _program_ad_float64_vector_result(cotangent_vector.reshape(target).reshape(source))

    return CustomDerivativeRule(
        name=(
            "program_ad_shape_reshape_"
            f"{_program_ad_shape_signature(source)}_to_"
            f"{_program_ad_shape_signature(target)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_shape_ravel_derivative_rule(source_shape: Sequence[int]) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed ravel signature."""

    source = _program_ad_shape_normalise_static_shape("ravel", source_shape)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)
        return _program_ad_shape_vector("ravel", "tangent", tangent, expected_size=source_size)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)
        return _program_ad_shape_vector("ravel", "cotangent", cotangent, expected_size=source_size)

    return CustomDerivativeRule(
        name=f"program_ad_shape_ravel_{_program_ad_shape_signature(source)}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_static_axes(
    source_shape: tuple[int, ...],
    axes: Sequence[int] | None,
) -> tuple[int, ...]:
    if axes is None:
        return tuple(reversed(range(len(source_shape))))
    normalised = tuple(int(axis) for axis in axes)
    if len(normalised) != len(source_shape) or set(normalised) != set(range(len(source_shape))):
        raise ValueError("program AD shape transpose direct rule requires axes permutation")
    return normalised


def program_ad_shape_transpose_derivative_rule(
    source_shape: Sequence[int],
    axes: Sequence[int] | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed transpose signature."""

    source = _program_ad_shape_normalise_static_shape("transpose", source_shape)
    normalised_axes = _program_ad_shape_normalise_static_axes(source, axes)
    inverse_axes = tuple(int(axis) for axis in np.argsort(normalised_axes))
    source_size = _program_ad_shape_static_size(source)
    target = tuple(source[axis] for axis in normalised_axes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(vector.reshape(source).transpose(normalised_axes))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "transpose", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            tangent_vector.reshape(source).transpose(normalised_axes)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "transpose", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            cotangent_vector.reshape(target).transpose(inverse_axes)
        )

    axes_signature = "_".join(str(axis) for axis in normalised_axes)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_transpose_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_insert_singleton_axes(
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    target_shape = list(source_shape)
    for axis in axes:
        target_shape.insert(axis, 1)
    return tuple(target_shape)


def _program_ad_shape_normalise_expand_dims_axes(
    source_shape: tuple[int, ...],
    axis: int | Sequence[int],
) -> tuple[int, ...]:
    raw_axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    output_rank = len(source_shape) + len(raw_axes)
    return _normalise_shape_transform_axes("expand_dims", raw_axes, output_rank=output_rank)


def program_ad_shape_expand_dims_derivative_rule(
    source_shape: Sequence[int],
    axis: int | Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed singleton-axis insertion."""

    source = _program_ad_shape_normalise_static_shape("expand_dims", source_shape)
    axes = _program_ad_shape_normalise_expand_dims_axes(source, axis)
    target = _program_ad_shape_insert_singleton_axes(source, axes)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector(
            "expand_dims", "values", values, expected_size=source_size
        )
        return _program_ad_float64_vector_result(np.expand_dims(vector.reshape(source), axes))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("expand_dims", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "expand_dims", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.expand_dims(tangent_vector.reshape(source), axes)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("expand_dims", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "expand_dims", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(cotangent_vector.reshape(target).reshape(source))

    axes_signature = "_".join(str(axis_item) for axis_item in axes)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_expand_dims_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_squeeze_axes(
    source_shape: tuple[int, ...],
    axis: int | Sequence[int] | None,
) -> tuple[int, ...]:
    if axis is None:
        return tuple(index for index, dimension in enumerate(source_shape) if dimension == 1)
    raw_axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    axes = _normalise_shape_transform_axes("squeeze", raw_axes, output_rank=len(source_shape))
    if any(source_shape[axis_item] != 1 for axis_item in axes):
        raise ValueError("program AD squeeze axis must have length one")
    return axes


def _program_ad_shape_remove_axes(
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        dimension for index, dimension in enumerate(source_shape) if index not in set(axes)
    )


def program_ad_shape_squeeze_derivative_rule(
    source_shape: Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed singleton-axis removal."""

    source = _program_ad_shape_normalise_static_shape("squeeze", source_shape)
    axes = _program_ad_shape_normalise_squeeze_axes(source, axis)
    target = _program_ad_shape_remove_axes(source, axes)
    source_size = _program_ad_shape_static_size(source)
    numpy_axis: tuple[int, ...] | None = None if axis is None else axes

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("squeeze", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.squeeze(vector.reshape(source), axis=numpy_axis)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("squeeze", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "squeeze", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.squeeze(tangent_vector.reshape(source), axis=numpy_axis)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("squeeze", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "squeeze", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(cotangent_vector.reshape(target).reshape(source))

    axes_signature = (
        "all"
        if axis is None
        else "none"
        if not axes
        else "_".join(str(axis_item) for axis_item in axes)
    )
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_squeeze_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_shape_swapaxes_derivative_rule(
    source_shape: Sequence[int],
    axis1: int,
    axis2: int,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed two-axis exchange."""

    source = _program_ad_shape_normalise_static_shape("swapaxes", source_shape)
    first = _normalise_axis_permutation_axis("swapaxes", axis1, rank=len(source))
    second = _normalise_axis_permutation_axis("swapaxes", axis2, rank=len(source))
    target_shape = list(source)
    target_shape[first], target_shape[second] = target_shape[second], target_shape[first]
    target = tuple(target_shape)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("swapaxes", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.swapaxes(vector.reshape(source), first, second)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("swapaxes", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "swapaxes", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.swapaxes(tangent_vector.reshape(source), first, second)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("swapaxes", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "swapaxes", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.swapaxes(cotangent_vector.reshape(target), first, second)
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_shape_swapaxes_"
            f"{_program_ad_shape_signature(source)}_axes_{first}_{second}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_moveaxis_order(
    rank: int,
    source_axes: tuple[int, ...],
    destination_axes: tuple[int, ...],
) -> tuple[int, ...]:
    order = [axis for axis in range(rank) if axis not in set(source_axes)]
    for destination_axis, source_axis in sorted(zip(destination_axes, source_axes)):
        order.insert(destination_axis, source_axis)
    return tuple(order)


def _program_ad_shape_normalise_moveaxis_axes(
    source_shape: tuple[int, ...],
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    source_axes = _normalise_axis_permutation_axes(
        "moveaxis", source, rank=len(source_shape), role="source"
    )
    destination_axes = _normalise_axis_permutation_axes(
        "moveaxis", destination, rank=len(source_shape), role="destination"
    )
    if len(source_axes) != len(destination_axes):
        raise ValueError("program AD moveaxis source and destination lengths must match")
    order = _program_ad_shape_moveaxis_order(len(source_shape), source_axes, destination_axes)
    return source_axes, destination_axes, order


def program_ad_shape_moveaxis_derivative_rule(
    source_shape: Sequence[int],
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed axis relocation."""

    source_shape_tuple = _program_ad_shape_normalise_static_shape("moveaxis", source_shape)
    source_axes, destination_axes, order = _program_ad_shape_normalise_moveaxis_axes(
        source_shape_tuple, source, destination
    )
    target = tuple(source_shape_tuple[axis] for axis in order)
    source_size = _program_ad_shape_static_size(source_shape_tuple)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("moveaxis", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.moveaxis(vector.reshape(source_shape_tuple), source_axes, destination_axes)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("moveaxis", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "moveaxis", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.moveaxis(tangent_vector.reshape(source_shape_tuple), source_axes, destination_axes)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("moveaxis", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "moveaxis", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.moveaxis(cotangent_vector.reshape(target), destination_axes, source_axes)
        )

    source_signature = "_".join(str(axis) for axis in source_axes) or "none"
    destination_signature = "_".join(str(axis) for axis in destination_axes) or "none"
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_moveaxis_"
            f"{_program_ad_shape_signature(source_shape_tuple)}_"
            f"source_{source_signature}_destination_{destination_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_roll_signature(
    source_shape: tuple[int, ...],
    shift: object,
    axis: object,
) -> tuple[int | tuple[int, ...], tuple[int, ...] | None]:
    if axis is None:
        return _normalise_roll_shift_scalar(shift), None
    axes = _normalise_axis_permutation_axes("roll", axis, rank=len(source_shape), role="axis")
    return _normalise_roll_shift_tuple(shift, len(axes)), axes


def _program_ad_shape_negate_roll_shift(
    shift: int | tuple[int, ...],
) -> int | tuple[int, ...]:
    if isinstance(shift, tuple):
        return tuple(-item for item in shift)
    return -shift


def program_ad_shape_roll_derivative_rule(
    source_shape: Sequence[int],
    shift: object,
    axis: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed integer roll permutations."""

    source = _program_ad_shape_normalise_static_shape("roll", source_shape)
    normalised_shift, normalised_axis = _program_ad_shape_normalise_roll_signature(
        source, shift, axis
    )
    inverse_shift = _program_ad_shape_negate_roll_shift(normalised_shift)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("roll", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.roll(vector.reshape(source), normalised_shift, axis=normalised_axis)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("roll", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "roll", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.roll(tangent_vector.reshape(source), normalised_shift, axis=normalised_axis)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("roll", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "roll", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.roll(cotangent_vector.reshape(source), inverse_shift, axis=normalised_axis)
        )

    shift_signature = (
        "_".join(str(item) for item in normalised_shift)
        if isinstance(normalised_shift, tuple)
        else str(normalised_shift)
    )
    axis_signature = (
        "flat"
        if normalised_axis is None
        else "none"
        if not normalised_axis
        else "_".join(str(item) for item in normalised_axis)
    )
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_roll_"
            f"{_program_ad_shape_signature(source)}_"
            f"shift_{shift_signature}_axis_{axis_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_flip_axis(
    source_shape: tuple[int, ...],
    axis: object,
) -> tuple[int, ...] | None:
    if axis is None:
        return None
    return _normalise_axis_permutation_axes("flip", axis, rank=len(source_shape), role="axis")


def program_ad_shape_flip_derivative_rule(
    source_shape: Sequence[int],
    axis: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed axis-flip permutations."""

    source = _program_ad_shape_normalise_static_shape("flip", source_shape)
    normalised_axis = _program_ad_shape_normalise_flip_axis(source, axis)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("flip", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.flip(vector.reshape(source), axis=normalised_axis)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("flip", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "flip", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.flip(tangent_vector.reshape(source), axis=normalised_axis)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("flip", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "flip", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.flip(cotangent_vector.reshape(source), axis=normalised_axis)
        )

    axis_signature = (
        "all"
        if normalised_axis is None
        else "none"
        if not normalised_axis
        else "_".join(str(item) for item in normalised_axis)
    )
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_flip_"
            f"{_program_ad_shape_signature(source)}_axis_{axis_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_shape_flipud_derivative_rule(source_shape: Sequence[int]) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed first-axis flips."""

    source = _program_ad_shape_normalise_static_shape("flipud", source_shape)
    if len(source) < 1:
        raise ValueError("program AD flipud direct rule requires at least rank-1 arrays")
    rule = program_ad_shape_flip_derivative_rule(source, axis=0)
    return CustomDerivativeRule(
        name=f"program_ad_shape_flipud_{_program_ad_shape_signature(source)}_direct_rule",
        value_fn=rule.value_fn,
        jvp_rule=rule.jvp_rule,
        vjp_rule=rule.vjp_rule,
    )


def program_ad_shape_fliplr_derivative_rule(source_shape: Sequence[int]) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed second-axis flips."""

    source = _program_ad_shape_normalise_static_shape("fliplr", source_shape)
    if len(source) < 2:
        raise ValueError("program AD fliplr direct rule requires at least rank-2 arrays")
    rule = program_ad_shape_flip_derivative_rule(source, axis=1)
    return CustomDerivativeRule(
        name=f"program_ad_shape_fliplr_{_program_ad_shape_signature(source)}_direct_rule",
        value_fn=rule.value_fn,
        jvp_rule=rule.jvp_rule,
        vjp_rule=rule.vjp_rule,
    )


def _program_ad_shape_rot90_target_shape(
    source_shape: tuple[int, ...],
    k: int,
    axes: tuple[int, int],
) -> tuple[int, ...]:
    target = list(source_shape)
    if k % 2:
        first, second = axes
        target[first], target[second] = target[second], target[first]
    return tuple(target)


def program_ad_shape_rot90_derivative_rule(
    source_shape: Sequence[int],
    k: object = 1,
    axes: object = (0, 1),
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static quarter-turns."""

    source = _program_ad_shape_normalise_static_shape("rot90", source_shape)
    k_value = _normalise_rot90_k(k)
    axes_value = _normalise_rot90_axes(axes, rank=len(source))
    target = _program_ad_shape_rot90_target_shape(source, k_value, axes_value)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("rot90", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(
            np.rot90(vector.reshape(source), k=k_value, axes=axes_value)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("rot90", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "rot90", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.rot90(tangent_vector.reshape(source), k=k_value, axes=axes_value)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("rot90", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "rot90", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            np.rot90(cotangent_vector.reshape(target), k=-k_value, axes=axes_value)
        )

    axes_signature = "_".join(str(axis) for axis in axes_value)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_rot90_"
            f"{_program_ad_shape_signature(source)}_k_{k_value}_"
            f"axes_{axes_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_repeat_signature(
    source_shape: tuple[int, ...],
    repeats: object,
    axis: object,
) -> tuple[int | tuple[int, ...], int | None, tuple[int, ...]]:
    if axis is None:
        repeat_counts = _normalise_repeat_counts(
            repeats, _program_ad_shape_static_size(source_shape)
        )
        target_size = (
            sum(repeat_counts)
            if isinstance(repeat_counts, tuple)
            else _program_ad_shape_static_size(source_shape) * repeat_counts
        )
        return repeat_counts, None, (int(target_size),)

    axis_index = _normalise_axis_permutation_axis("repeat", axis, rank=len(source_shape))
    repeat_counts = _normalise_repeat_counts(repeats, source_shape[axis_index])
    target = list(source_shape)
    target[axis_index] = int(
        sum(repeat_counts)
        if isinstance(repeat_counts, tuple)
        else source_shape[axis_index] * repeat_counts
    )
    return repeat_counts, axis_index, tuple(target)


def program_ad_shape_repeat_derivative_rule(
    source_shape: Sequence[int],
    repeats: object,
    axis: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static repeat signatures."""

    source = _program_ad_shape_normalise_static_shape("repeat", source_shape)
    repeat_counts, axis_index, target = _program_ad_shape_normalise_repeat_signature(
        source, repeats, axis
    )
    source_size = _program_ad_shape_static_size(source)
    target_size = _program_ad_shape_static_size(target)
    source_indices = np.arange(source_size, dtype=np.int64).reshape(source)
    if axis_index is None:
        repeated_indices = np.repeat(source_indices.reshape(-1), repeat_counts).reshape(-1)
    else:
        repeated_indices = np.repeat(source_indices, repeat_counts, axis=axis_index).reshape(-1)

    def _repeat_flat(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        if axis_index is None:
            return _program_ad_float64_vector_result(np.repeat(vector, repeat_counts))
        return _program_ad_float64_vector_result(
            np.repeat(vector.reshape(source), repeat_counts, axis=axis_index)
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("repeat", "values", values, expected_size=source_size)
        return _repeat_flat(vector)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("repeat", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "repeat", "tangent", tangent, expected_size=source_size
        )
        return _repeat_flat(tangent_vector)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("repeat", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "repeat", "cotangent", cotangent, expected_size=target_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, repeated_indices, cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    repeat_signature = (
        "_".join(str(value) for value in repeat_counts)
        if isinstance(repeat_counts, tuple)
        else str(repeat_counts)
    )
    axis_signature = "flat" if axis_index is None else str(axis_index)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_repeat_"
            f"{_program_ad_shape_signature(source)}_repeats_{repeat_signature}_"
            f"axis_{axis_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_tile_signature(
    source_shape: tuple[int, ...],
    reps: object,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    reps_tuple = _normalise_tile_reps(reps)
    rank = max(len(source_shape), len(reps_tuple))
    source_aligned = (1,) * (rank - len(source_shape)) + source_shape
    reps_aligned = (1,) * (rank - len(reps_tuple)) + reps_tuple
    target = tuple(int(size * rep) for size, rep in zip(source_aligned, reps_aligned))
    return reps_tuple, reps_aligned, target


def program_ad_shape_tile_derivative_rule(
    source_shape: Sequence[int],
    reps: object,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static tile signatures."""

    source = _program_ad_shape_normalise_static_shape("tile", source_shape)
    reps_tuple, _, target = _program_ad_shape_normalise_tile_signature(source, reps)
    source_size = _program_ad_shape_static_size(source)
    target_size = _program_ad_shape_static_size(target)
    source_indices = np.arange(source_size, dtype=np.int64).reshape(source)
    tiled_indices = np.tile(source_indices, reps_tuple).reshape(-1)

    def _tile_flat(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(np.tile(vector.reshape(source), reps_tuple))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("tile", "values", values, expected_size=source_size)
        return _tile_flat(vector)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("tile", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "tile", "tangent", tangent, expected_size=source_size
        )
        return _tile_flat(tangent_vector)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("tile", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "tile", "cotangent", cotangent, expected_size=target_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, tiled_indices, cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    reps_signature = "_".join(str(value) for value in reps_tuple)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_tile_"
            f"{_program_ad_shape_signature(source)}_reps_{reps_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_atleast_target_shape(
    source_shape: tuple[int, ...],
    rank: Literal[1, 2, 3],
) -> tuple[int, ...]:
    if rank == 1:
        return source_shape if len(source_shape) >= 1 else (1,)
    if rank == 2:
        if len(source_shape) == 0:
            return (1, 1)
        if len(source_shape) == 1:
            return (1, source_shape[0])
        return source_shape
    if len(source_shape) == 0:
        return (1, 1, 1)
    if len(source_shape) == 1:
        return (1, source_shape[0], 1)
    if len(source_shape) == 2:
        return (source_shape[0], source_shape[1], 1)
    return source_shape


def _program_ad_shape_atleast_derivative_rule(
    source_shape: Sequence[int],
    rank: Literal[1, 2, 3],
) -> CustomDerivativeRule:
    name = f"atleast_{rank}d"
    source = _program_ad_shape_normalise_static_shape(name, source_shape)
    target = _program_ad_shape_atleast_target_shape(source, rank)
    source_size = _program_ad_shape_static_size(source)

    def _reshape_to_target(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(vector.reshape(target))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector(name, "values", values, expected_size=source_size)
        return _reshape_to_target(vector)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector(name, "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            name, "tangent", tangent, expected_size=source_size
        )
        return _reshape_to_target(tangent_vector)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector(name, "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            name, "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(cotangent_vector.reshape(source_size))

    return CustomDerivativeRule(
        name=(
            f"program_ad_shape_{name}_"
            f"{_program_ad_shape_signature(source)}_to_"
            f"{_program_ad_shape_signature(target)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_shape_atleast_1d_derivative_rule(
    source_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static atleast-1D promotion."""

    return _program_ad_shape_atleast_derivative_rule(source_shape, 1)


def program_ad_shape_atleast_2d_derivative_rule(
    source_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static atleast-2D promotion."""

    return _program_ad_shape_atleast_derivative_rule(source_shape, 2)


def program_ad_shape_atleast_3d_derivative_rule(
    source_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static atleast-3D promotion."""

    return _program_ad_shape_atleast_derivative_rule(source_shape, 3)
