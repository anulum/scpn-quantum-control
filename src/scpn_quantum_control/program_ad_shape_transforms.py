# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD shape transforms module
# scpn-quantum-control -- Program AD shape transform rules
"""Static shape-transform derivative rules for Program AD registry dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_dtype_of,
    _program_ad_array_shape_of,
)
from .program_ad_registry import (
    _PROGRAM_AD_SHAPE_IDENTITIES,
    _PROGRAM_AD_SHAPE_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .whole_program_trace_metadata import (
    _normalise_axis_permutation_axes,
    _normalise_axis_permutation_axis,
    _normalise_repeat_counts,
    _normalise_roll_shift_scalar,
    _normalise_roll_shift_tuple,
    _normalise_rot90_axes,
    _normalise_rot90_k,
    _normalise_shape_transform_axes,
    _normalise_tile_reps,
)


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(values, dtype=np.float64).reshape(-1))


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
    """Build the trace-dispatch placeholder rule for a shape primitive.

    Parameters
    ----------
    name:
        Primitive name from the Program AD shape registry.

    Returns
    -------
    CustomDerivativeRule
        Rule whose direct value/JVP callables fail closed because runtime
        execution must pass through operator-intercepted Program AD traces.
    """
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
    """Build a direct derivative rule for a fixed reshape signature.

    Parameters
    ----------
    source_shape:
        Static input tensor shape before reshape.
    target_shape:
        Static output tensor shape after reshape.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that preserve element order while changing
        only the static layout.

    Raises
    ------
    ValueError
        If either shape has a negative dimension, if the element counts differ,
        or if the returned callables receive values, tangents, or cotangents
        with the wrong flattened length.
    """
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
    """Build a direct derivative rule for a fixed ravel signature.

    Parameters
    ----------
    source_shape:
        Static tensor shape to flatten.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that expose the source tensor as a flat
        float64 vector without reordering entries.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension or if the returned
        callables receive values, tangents, or cotangents with the wrong
        flattened length.
    """
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
    """Build a direct derivative rule for a fixed transpose signature.

    Parameters
    ----------
    source_shape:
        Static tensor shape before axis permutation.
    axes:
        Optional permutation of all source axes; ``None`` reverses the axes.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule using the transpose axes and their inverse.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if ``axes`` is not a
        complete permutation, or if the returned callables receive vectors with
        incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed singleton-axis insertion.

    Parameters
    ----------
    source_shape:
        Static tensor shape before singleton axes are inserted.
    axis:
        Static axis or axes where singleton dimensions are inserted.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that inserts singleton dimensions while
        preserving the flattened element order.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if any axis is invalid,
        or if the returned callables receive vectors with incompatible
        flattened length.
    """
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
    """Build a direct derivative rule for fixed singleton-axis removal.

    Parameters
    ----------
    source_shape:
        Static tensor shape before singleton axes are removed.
    axis:
        Optional static axis or axes to squeeze; ``None`` removes all singleton
        axes.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that removes only statically singleton axes and
        reshapes the cotangent back to the source layout.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if a selected axis is not
        length one, or if the returned callables receive vectors with
        incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed two-axis exchange.

    Parameters
    ----------
    source_shape:
        Static tensor shape before the swap.
    axis1:
        First static axis to exchange.
    axis2:
        Second static axis to exchange.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that applies the same two-axis swap to primal,
        tangent, and cotangent layouts.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if either axis is outside
        the source rank, or if the returned callables receive vectors with
        incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed axis relocation.

    Parameters
    ----------
    source_shape:
        Static tensor shape before axis relocation.
    source:
        Static source axis or axes moved from the input layout.
    destination:
        Static destination axis or axes in the output layout.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that applies ``moveaxis`` forward and the
        inverse relocation to cotangents.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if source and destination
        axis counts differ, if an axis is invalid, or if the returned callables
        receive vectors with incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed integer roll permutations.

    Parameters
    ----------
    source_shape:
        Static tensor shape before the roll.
    shift:
        Static integer shift or shift tuple accepted by ``numpy.roll``.
    axis:
        Optional static axis or axis tuple; ``None`` rolls the flattened array.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that rolls primal/tangent values forward and
        rolls cotangents by the inverse shift.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if shift/axis metadata is
        not static and compatible, or if the returned callables receive vectors
        with incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed axis-flip permutations.

    Parameters
    ----------
    source_shape:
        Static tensor shape before the flip.
    axis:
        Optional static axis or axis tuple; ``None`` flips every axis.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that applies the same axis reversal to primal,
        tangent, and cotangent values.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if an axis is outside the
        source rank, or if the returned callables receive vectors with
        incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed first-axis flips.

    Parameters
    ----------
    source_shape:
        Static tensor shape before ``flipud`` is applied.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule for the first-axis reversal.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if the rank is zero, or
        if the returned callables receive vectors with incompatible flattened
        length.
    """
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
    """Build a direct derivative rule for fixed second-axis flips.

    Parameters
    ----------
    source_shape:
        Static tensor shape before ``fliplr`` is applied.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule for the second-axis reversal.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if the rank is below two,
        or if the returned callables receive vectors with incompatible
        flattened length.
    """
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
    """Build a direct derivative rule for fixed static quarter-turns.

    Parameters
    ----------
    source_shape:
        Static tensor shape before rotation.
    k:
        Static number of quarter-turns.
    axes:
        Static pair of axes defining the rotation plane.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule that applies ``rot90`` forward and the inverse
        quarter-turn to cotangents.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if ``k`` or ``axes`` is
        not static and valid, or if the returned callables receive vectors with
        incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed static repeat signatures.

    Parameters
    ----------
    source_shape:
        Static tensor shape before repeating entries.
    repeats:
        Static repeat count or repeat-count sequence.
    axis:
        Optional static axis; ``None`` repeats the flattened input.

    Returns
    -------
    CustomDerivativeRule
        Value and JVP rules that repeat entries plus a VJP rule that scatter-adds
        repeated cotangents back to source positions.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if repeat metadata is
        invalid for the selected axis, or if the returned callables receive
        vectors with incompatible flattened length.
    """
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
    """Build a direct derivative rule for fixed static tile signatures.

    Parameters
    ----------
    source_shape:
        Static tensor shape before tiling.
    reps:
        Static tile repetition signature.

    Returns
    -------
    CustomDerivativeRule
        Value and JVP rules that tile the input plus a VJP rule that scatter-adds
        tiled cotangents back to source positions.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension, if tile metadata is
        invalid, or if the returned callables receive vectors with incompatible
        flattened length.
    """
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
    """Build a direct derivative rule for fixed static atleast-1D promotion.

    Parameters
    ----------
    source_shape:
        Static tensor shape before rank promotion.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule for ``numpy.atleast_1d`` shape promotion.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension or if the returned
        callables receive values, tangents, or cotangents with the wrong
        flattened length.
    """
    return _program_ad_shape_atleast_derivative_rule(source_shape, 1)


def program_ad_shape_atleast_2d_derivative_rule(
    source_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct derivative rule for fixed static atleast-2D promotion.

    Parameters
    ----------
    source_shape:
        Static tensor shape before rank promotion.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule for ``numpy.atleast_2d`` shape promotion.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension or if the returned
        callables receive values, tangents, or cotangents with the wrong
        flattened length.
    """
    return _program_ad_shape_atleast_derivative_rule(source_shape, 2)


def program_ad_shape_atleast_3d_derivative_rule(
    source_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct derivative rule for fixed static atleast-3D promotion.

    Parameters
    ----------
    source_shape:
        Static tensor shape before rank promotion.

    Returns
    -------
    CustomDerivativeRule
        Value, JVP, and VJP rule for ``numpy.atleast_3d`` shape promotion.

    Raises
    ------
    ValueError
        If the source shape has a negative dimension or if the returned
        callables receive values, tangents, or cotangents with the wrong
        flattened length.
    """
    return _program_ad_shape_atleast_derivative_rule(source_shape, 3)


def _normalise_trace_reshape_shape(shape: object, size: int) -> tuple[int, ...]:
    """Return a concrete reshape target that preserves ``size``."""
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


def _program_ad_shape_reshape_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD reshape primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape reshape rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return _normalise_trace_reshape_shape(args[1], int(np.prod(source_shape)))


def _program_ad_shape_expand_dims_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD expand-dims primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape expand_dims rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalise_expand_dims_axes(source_shape, cast(Any, args[1]))
    return _program_ad_shape_insert_singleton_axes(source_shape, axes)


def _program_ad_shape_ravel_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD ravel primitive."""
    if len(args) != 1:
        raise ValueError("program AD shape ravel rule requires one array")
    return (int(np.prod(_program_ad_array_shape_of(args[0]))),)


def _program_ad_shape_normalised_transpose_axes(
    array_shape: tuple[int, ...],
    axes: object,
) -> tuple[int, ...]:
    """Return normalised transpose axes for a static shape contract."""
    if len(array_shape) < 2:
        return ()
    if axes is None:
        return tuple(reversed(range(len(array_shape))))
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise ValueError("program AD shape transpose axes must be a static axis sequence")
    raw_axes = tuple(cast(Any, axes))
    if len(raw_axes) != len(array_shape):
        raise ValueError("program AD shape transpose axes must match array rank")
    normalised_axes = tuple(
        _normalise_axis("axis", cast(int, axis), len(array_shape)) for axis in raw_axes
    )
    if sorted(normalised_axes) != list(range(len(array_shape))):
        raise ValueError("program AD shape transpose axes must be a permutation")
    return normalised_axes


def _program_ad_shape_transpose_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD transpose primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape transpose rule requires array and optional axes")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalised_transpose_axes(
        source_shape, args[1] if len(args) == 2 else None
    )
    if not axes:
        return source_shape
    return tuple(source_shape[axis] for axis in axes)


def _program_ad_shape_atleast_rank_shape(
    args: tuple[object, ...], *, rank: Literal[1, 2, 3]
) -> tuple[int, ...]:
    """Return the static output shape for an atleast-rank primitive."""
    if len(args) != 1:
        raise ValueError(f"program AD shape atleast_{rank}d rule requires one array")
    return _program_ad_shape_atleast_target_shape(_program_ad_array_shape_of(args[0]), rank)


def _program_ad_shape_atleast_1d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD atleast-1D primitive."""
    return _program_ad_shape_atleast_rank_shape(args, rank=1)


def _program_ad_shape_atleast_2d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD atleast-2D primitive."""
    return _program_ad_shape_atleast_rank_shape(args, rank=2)


def _program_ad_shape_atleast_3d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD atleast-3D primitive."""
    return _program_ad_shape_atleast_rank_shape(args, rank=3)


def _program_ad_shape_swapaxes_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD swapaxes primitive."""
    if len(args) != 3:
        raise ValueError("program AD shape swapaxes rule requires array, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    first = _normalise_axis_permutation_axis(
        "swapaxes", cast(int, args[1]), rank=len(source_shape)
    )
    second = _normalise_axis_permutation_axis(
        "swapaxes", cast(int, args[2]), rank=len(source_shape)
    )
    target = list(source_shape)
    target[first], target[second] = target[second], target[first]
    return tuple(target)


def _program_ad_shape_moveaxis_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD moveaxis primitive."""
    if len(args) != 3:
        raise ValueError("program AD shape moveaxis rule requires array, source, and destination")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, order = _program_ad_shape_normalise_moveaxis_axes(
        source_shape, cast(Any, args[1]), cast(Any, args[2])
    )
    return tuple(source_shape[axis] for axis in order)


def _program_ad_shape_roll_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD roll primitive."""
    if len(args) not in {2, 3}:
        raise ValueError("program AD shape roll rule requires array, shift, and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _program_ad_shape_normalise_roll_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return source_shape


def _program_ad_shape_flip_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD flip primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape flip rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _program_ad_shape_normalise_flip_axis(source_shape, args[1] if len(args) == 2 else None)
    return source_shape


def _program_ad_shape_flipud_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD flipud primitive."""
    if len(args) != 1:
        raise ValueError("program AD shape flipud rule requires one array")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 1:
        raise ValueError("program AD flipud requires at least rank-1 arrays")
    return source_shape


def _program_ad_shape_fliplr_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD fliplr primitive."""
    if len(args) != 1:
        raise ValueError("program AD shape fliplr rule requires one array")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD fliplr requires at least rank-2 arrays")
    return source_shape


def _program_ad_shape_rot90_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD rot90 primitive."""
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD shape rot90 rule requires array, k, and axes")
    source_shape = _program_ad_array_shape_of(args[0])
    k_value = _normalise_rot90_k(args[1] if len(args) >= 2 else 1)
    axes_value = _normalise_rot90_axes(
        args[2] if len(args) == 3 else (0, 1), rank=len(source_shape)
    )
    return _program_ad_shape_rot90_target_shape(source_shape, k_value, axes_value)


def _program_ad_shape_repeat_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD repeat primitive."""
    if len(args) not in {2, 3}:
        raise ValueError("program AD shape repeat rule requires array, repeats, and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, target_shape = _program_ad_shape_normalise_repeat_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return target_shape


def _program_ad_shape_tile_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD tile primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape tile rule requires array and reps")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, target_shape = _program_ad_shape_normalise_tile_signature(source_shape, args[1])
    return target_shape


def _program_ad_shape_squeeze_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a Program AD squeeze primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape squeeze rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalise_squeeze_axes(
        source_shape, cast(Any, args[1]) if len(args) == 2 else None
    )
    return _program_ad_shape_remove_axes(source_shape, axes)


def _program_ad_shape_dtype_rule(args: tuple[object, ...]) -> str:
    """Return the dtype emitted by a Program AD shape primitive."""
    if not args:
        raise ValueError("program AD shape dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_shape_reshape_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD reshape primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape reshape static rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return (_normalise_trace_reshape_shape(args[1], int(np.prod(source_shape))),)


def _program_ad_shape_expand_dims_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD expand-dims primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape expand_dims static rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (_program_ad_shape_normalise_expand_dims_axes(source_shape, cast(Any, args[1])),)


def _program_ad_shape_no_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return an empty static signature for a unary Program AD shape primitive."""
    if len(args) != 1:
        raise ValueError("program AD shape static rule requires one array")
    return ()


def _program_ad_shape_atleast_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for an atleast-rank shape primitive."""
    if len(args) != 1:
        raise ValueError("program AD shape atleast static rule requires one array")
    return ()


def _program_ad_shape_transpose_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD transpose primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape transpose static rule requires array and optional axes")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalised_transpose_axes(
        source_shape, args[1] if len(args) == 2 else None
    )
    return () if not axes else (axes,)


def _program_ad_shape_swapaxes_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD swapaxes primitive."""
    if len(args) != 3:
        raise ValueError("program AD shape swapaxes static rule requires array, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _normalise_axis_permutation_axis("swapaxes", cast(int, args[1]), rank=len(source_shape)),
        _normalise_axis_permutation_axis("swapaxes", cast(int, args[2]), rank=len(source_shape)),
    )


def _program_ad_shape_moveaxis_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD moveaxis primitive."""
    if len(args) != 3:
        raise ValueError(
            "program AD shape moveaxis static rule requires array, source, and destination"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    source_axes, destination_axes, _ = _program_ad_shape_normalise_moveaxis_axes(
        source_shape, cast(Any, args[1]), cast(Any, args[2])
    )
    return source_axes, destination_axes


def _program_ad_shape_roll_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD roll primitive."""
    if len(args) not in {2, 3}:
        raise ValueError(
            "program AD shape roll static rule requires array, shift, and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    return _program_ad_shape_normalise_roll_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )


def _program_ad_shape_flip_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD flip primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape flip static rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _program_ad_shape_normalise_flip_axis(source_shape, args[1] if len(args) == 2 else None),
    )


def _program_ad_shape_rot90_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD rot90 primitive."""
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD shape rot90 static rule requires array, k, and axes")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _normalise_rot90_k(args[1] if len(args) >= 2 else 1),
        _normalise_rot90_axes(args[2] if len(args) == 3 else (0, 1), rank=len(source_shape)),
    )


def _program_ad_shape_repeat_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD repeat primitive."""
    if len(args) not in {2, 3}:
        raise ValueError(
            "program AD shape repeat static rule requires array, repeats, and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    repeat_counts, axis_index, _ = _program_ad_shape_normalise_repeat_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return repeat_counts, axis_index


def _program_ad_shape_tile_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD tile primitive."""
    if len(args) != 2:
        raise ValueError("program AD shape tile static rule requires array and reps")
    source_shape = _program_ad_array_shape_of(args[0])
    reps_tuple, _, _ = _program_ad_shape_normalise_tile_signature(source_shape, args[1])
    return (reps_tuple,)


def _program_ad_shape_squeeze_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Return canonical static arguments for a Program AD squeeze primitive."""
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape squeeze static rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _program_ad_shape_normalise_squeeze_axes(
            source_shape, cast(Any, args[1]) if len(args) == 2 else None
        ),
    )


_PROGRAM_AD_SHAPE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "atleast_1d": _program_ad_shape_atleast_1d_shape,
    "atleast_2d": _program_ad_shape_atleast_2d_shape,
    "atleast_3d": _program_ad_shape_atleast_3d_shape,
    "expand_dims": _program_ad_shape_expand_dims_shape,
    "flip": _program_ad_shape_flip_shape,
    "fliplr": _program_ad_shape_fliplr_shape,
    "flipud": _program_ad_shape_flipud_shape,
    "moveaxis": _program_ad_shape_moveaxis_shape,
    "reshape": _program_ad_shape_reshape_shape,
    "ravel": _program_ad_shape_ravel_shape,
    "repeat": _program_ad_shape_repeat_shape,
    "roll": _program_ad_shape_roll_shape,
    "rot90": _program_ad_shape_rot90_shape,
    "squeeze": _program_ad_shape_squeeze_shape,
    "swapaxes": _program_ad_shape_swapaxes_shape,
    "tile": _program_ad_shape_tile_shape,
    "transpose": _program_ad_shape_transpose_shape,
}

_PROGRAM_AD_SHAPE_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "atleast_1d": _program_ad_shape_atleast_static_arguments,
    "atleast_2d": _program_ad_shape_atleast_static_arguments,
    "atleast_3d": _program_ad_shape_atleast_static_arguments,
    "expand_dims": _program_ad_shape_expand_dims_static_arguments,
    "flip": _program_ad_shape_flip_static_arguments,
    "fliplr": _program_ad_shape_no_static_arguments,
    "flipud": _program_ad_shape_no_static_arguments,
    "moveaxis": _program_ad_shape_moveaxis_static_arguments,
    "reshape": _program_ad_shape_reshape_static_arguments,
    "ravel": _program_ad_shape_no_static_arguments,
    "repeat": _program_ad_shape_repeat_static_arguments,
    "roll": _program_ad_shape_roll_static_arguments,
    "rot90": _program_ad_shape_rot90_static_arguments,
    "squeeze": _program_ad_shape_squeeze_static_arguments,
    "swapaxes": _program_ad_shape_swapaxes_static_arguments,
    "tile": _program_ad_shape_tile_static_arguments,
    "transpose": _program_ad_shape_transpose_static_arguments,
}


def _program_ad_shape_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Map a shape-transform primitive over a batch axis."""
    if len(args) != len(axes):
        raise ValueError("program AD shape batching axes must match argument count")
    if not args:
        raise ValueError("program AD shape batching requires an array operand")
    array = _as_real_numeric_array("program AD shape batched operand", args[0])
    axis = axes[0]
    if axis is None:
        return function(*args)
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD shape batching supports static non-array arguments only")
    axis_index = _normalise_axis("axes[0]", axis, array.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD shape batched output",
            function(np.take(array, batch_index, axis=axis_index), *args[1:]),
        )
        for batch_index in range(int(array.shape[axis_index]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_shape_lowering_metadata(name: str) -> Mapping[str, str]:
    """Return lowering metadata for a Program AD shape primitive."""
    static_factory = {
        "atleast_1d": "program_ad_shape_atleast_1d_derivative_rule",
        "atleast_2d": "program_ad_shape_atleast_2d_derivative_rule",
        "atleast_3d": "program_ad_shape_atleast_3d_derivative_rule",
        "expand_dims": "program_ad_shape_expand_dims_derivative_rule",
        "flip": "program_ad_shape_flip_derivative_rule",
        "fliplr": "program_ad_shape_fliplr_derivative_rule",
        "flipud": "program_ad_shape_flipud_derivative_rule",
        "moveaxis": "program_ad_shape_moveaxis_derivative_rule",
        "repeat": "program_ad_shape_repeat_derivative_rule",
        "reshape": "program_ad_shape_reshape_derivative_rule",
        "ravel": "program_ad_shape_ravel_derivative_rule",
        "roll": "program_ad_shape_roll_derivative_rule",
        "rot90": "program_ad_shape_rot90_derivative_rule",
        "squeeze": "program_ad_shape_squeeze_derivative_rule",
        "swapaxes": "program_ad_shape_swapaxes_derivative_rule",
        "tile": "program_ad_shape_tile_derivative_rule",
        "transpose": "program_ad_shape_transpose_derivative_rule",
    }[name]
    static_signature = {
        "atleast_1d": "source_shape:ranked_tensor_shape",
        "atleast_2d": "source_shape:ranked_tensor_shape",
        "atleast_3d": "source_shape:ranked_tensor_shape",
        "expand_dims": "source_shape:ranked_tensor_shape;axis",
        "flip": "source_shape:ranked_tensor_shape;axis",
        "fliplr": "source_shape:rank_ge_2",
        "flipud": "source_shape:rank_ge_1",
        "moveaxis": "source_shape:ranked_tensor_shape;source_destination",
        "repeat": "source_shape:ranked_tensor_shape;repeats_axis",
        "reshape": "source_shape:ranked_tensor_shape;target_shape",
        "ravel": "source_shape:ranked_tensor_shape",
        "roll": "source_shape:ranked_tensor_shape;shift_axis",
        "rot90": "source_shape:ranked_tensor_shape;k_axes",
        "squeeze": "source_shape:ranked_tensor_shape;axis",
        "swapaxes": "source_shape:ranked_tensor_shape;axis1_axis2",
        "tile": "source_shape:ranked_tensor_shape;reps",
        "transpose": "source_shape:ranked_tensor_shape;axes",
    }[name]
    nondifferentiable_boundaries = {
        "atleast_1d": "static_rank_promotion",
        "atleast_2d": "static_rank_promotion",
        "atleast_3d": "static_rank_promotion",
        "expand_dims": "static_singleton_axis_insertion",
        "flip": "static_axis_flip_permutation",
        "fliplr": "static_second_axis_flip_permutation",
        "flipud": "static_first_axis_flip_permutation",
        "moveaxis": "static_axis_move_permutation",
        "repeat": "static_repeat_scatter_add",
        "reshape": "element_count_preserving_static_shape",
        "ravel": "contiguous_flat_view_shape",
        "roll": "static_integer_roll_permutation",
        "rot90": "static_quarter_turn_axis_permutation",
        "squeeze": "static_singleton_axis_removal",
        "swapaxes": "static_axis_swap_permutation",
        "tile": "static_tile_scatter_add",
        "transpose": "static_axis_permutation",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff shape dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.shape.{name}",
        "llvm": "blocked_until_executable_shape_lowering",
        "rust": "blocked_until_polyglot_shape_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_shape_primitive_contracts() -> None:
    """Register fail-closed Program AD shape primitive contracts.

    Returns
    -------
    None
        Registration is applied to the default custom derivative registry. An
        already registered shape primitive is left unchanged.
    """
    for name, identity in _PROGRAM_AD_SHAPE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_shape_derivative_rule(name),
                batching_rule=_program_ad_shape_batching_rule,
                lowering_metadata=_program_ad_shape_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SHAPE_SHAPE_RULES[name],
                dtype_rule=_program_ad_shape_dtype_rule,
                static_argument_rule=_PROGRAM_AD_SHAPE_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SHAPE_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_shape_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate shape primitive dispatch helpers against concrete arguments."""
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


def _require_program_ad_shape_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered shape primitive runtime contract.

    Parameters
    ----------
    name:
        Shape primitive name in the Program AD registry.
    args:
        Optional concrete runtime arguments used to validate static argument,
        shape, and dtype rules for dispatch.

    Returns
    -------
    PrimitiveContract
        Registered contract with derivative, batching, lowering metadata,
        shape, dtype, static-argument, purity, and fail-closed policy fields
        verified.

    Raises
    ------
    ValueError
        If ``name`` is unknown, if the registered contract is incomplete or has
        the wrong policy/effect metadata, or if concrete ``args`` fail dispatch
        validation.
    """
    identity: PrimitiveIdentity | None = _PROGRAM_AD_SHAPE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD shape primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_SHAPE_POLICY:
        raise ValueError(f"invalid program AD shape primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD shape primitive effect for {identity.key}")

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
            f"incomplete program AD shape primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_shape_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_program_ad_shape_derivative_rule",
    "_register_program_ad_shape_primitive_contracts",
    "_require_program_ad_shape_contract",
    "program_ad_shape_atleast_1d_derivative_rule",
    "program_ad_shape_atleast_2d_derivative_rule",
    "program_ad_shape_atleast_3d_derivative_rule",
    "program_ad_shape_expand_dims_derivative_rule",
    "program_ad_shape_flip_derivative_rule",
    "program_ad_shape_fliplr_derivative_rule",
    "program_ad_shape_flipud_derivative_rule",
    "program_ad_shape_moveaxis_derivative_rule",
    "program_ad_shape_ravel_derivative_rule",
    "program_ad_shape_repeat_derivative_rule",
    "program_ad_shape_reshape_derivative_rule",
    "program_ad_shape_roll_derivative_rule",
    "program_ad_shape_rot90_derivative_rule",
    "program_ad_shape_squeeze_derivative_rule",
    "program_ad_shape_swapaxes_derivative_rule",
    "program_ad_shape_tile_derivative_rule",
    "program_ad_shape_transpose_derivative_rule",
)
