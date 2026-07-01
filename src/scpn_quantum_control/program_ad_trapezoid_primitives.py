# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD trapezoid primitive rules
"""Static trapezoidal-integration derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array, _as_real_scalar
from .program_ad_registry import CustomDerivativeRule
from .program_ad_shape_transforms import (
    _program_ad_float64_vector_result,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
)


def _is_program_ad_trace_value(value: object) -> bool:
    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _normalise_trapezoid_axis(axis: object, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.trapezoid axis must be a static integer")
    if ndim == 0:
        raise ValueError("axis cannot map over a scalar")
    axis_value = int(axis)
    if axis_value < 0:
        axis_value += ndim
    if axis_value < 0 or axis_value >= ndim:
        raise ValueError("program AD np.trapezoid axis out of bounds")
    return axis_value


def _program_ad_trapezoid_vector(role: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD reduction trapezoid {role}", values).reshape(-1)
    if vector.size == 0:
        raise ValueError("program AD reduction trapezoid direct rule requires at least one value")
    return vector


def _program_ad_trapezoid_tangent_pair(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_trapezoid_vector("values", values)
    tangent_vector = _as_real_numeric_array(
        "program AD reduction trapezoid tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD reduction trapezoid tangent shape must match values shape")
    return vector, tangent_vector


def _program_ad_trapezoid_scalar_cotangent(cotangent: NDArray[np.float64]) -> float:
    cotangent_vector = _as_real_numeric_array(
        "program AD reduction trapezoid cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError("program AD reduction trapezoid VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_reduction_trapezoid_flat_weights(size: int) -> NDArray[np.float64]:
    if size < 2:
        raise ValueError("program AD reduction trapezoid direct rule requires at least two values")
    weights = np.ones(size, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    return weights


def _program_ad_reduction_trapezoid_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_trapezoid_vector("values", values)
    weights = _program_ad_reduction_trapezoid_flat_weights(vector.size)
    return np.array([float(np.dot(weights, vector))], dtype=np.float64)


def _program_ad_reduction_trapezoid_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_trapezoid_tangent_pair(values, tangent)
    weights = _program_ad_reduction_trapezoid_flat_weights(vector.size)
    return np.array([float(np.dot(weights, tangent_vector))], dtype=np.float64)


def _program_ad_reduction_trapezoid_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_trapezoid_vector("values", values)
    scalar_cotangent = _program_ad_trapezoid_scalar_cotangent(cotangent)
    return scalar_cotangent * _program_ad_reduction_trapezoid_flat_weights(vector.size)


def _program_ad_trapezoid_normalise_static_shape(source_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            "program AD reduction trapezoid direct rule requires non-negative dimensions"
        )
    if _program_ad_shape_static_size(shape) == 0:
        raise ValueError("program AD reduction trapezoid direct rule requires at least one value")
    return shape


def _program_ad_trapezoid_output_shape(
    source_shape: tuple[int, ...],
    axis: int,
) -> tuple[int, ...]:
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_trapezoid_source_vector(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD reduction trapezoid {role}", values).reshape(-1)
    if vector.size != _program_ad_shape_static_size(source_shape):
        raise ValueError(
            "program AD reduction trapezoid direct rule requires "
            f"{role} with {_program_ad_shape_static_size(source_shape)} values"
        )
    return vector


def _program_ad_trapezoid_cotangent_array(
    cotangent: NDArray[np.float64],
    *,
    output_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    cotangent_vector = _as_real_numeric_array(
        "program AD reduction trapezoid cotangent", cotangent
    ).reshape(-1)
    expected_size = _program_ad_shape_static_size(output_shape)
    if cotangent_vector.size != expected_size:
        raise ValueError(
            "program AD reduction trapezoid direct rule requires cotangent "
            f"with {expected_size} values"
        )
    return cotangent_vector.reshape(output_shape)


def _program_ad_reduction_trapezoid_static_widths(
    source_shape: tuple[int, ...],
    *,
    x: object,
    dx: object,
    axis: int,
) -> NDArray[np.float64]:
    axis_size = source_shape[axis]
    if axis_size < 2:
        raise ValueError(
            "program AD reduction trapezoid direct rule requires at least two samples along axis"
        )
    width_shape = source_shape[:axis] + (axis_size - 1,) + source_shape[axis + 1 :]
    if _is_program_ad_trace_value(x):
        raise ValueError("program AD reduction trapezoid grid x must be static real numeric")
    if x is None:
        dx_value = _as_real_scalar("program AD reduction trapezoid dx", dx)
        return np.full(width_shape, dx_value, dtype=np.float64)
    dx_value = _as_real_scalar("program AD reduction trapezoid dx", dx)
    if dx_value != 1.0:
        raise ValueError("program AD reduction trapezoid accepts either x or dx, not both")
    x_array = _as_real_numeric_array("program AD reduction trapezoid x", x)
    if not bool(np.all(np.isfinite(x_array))):
        raise ValueError("program AD reduction trapezoid x must contain only finite values")
    if x_array.ndim == 1:
        if x_array.shape[0] != axis_size:
            raise ValueError("program AD reduction trapezoid x must match the integration axis")
        reshape = [1 for _ in source_shape]
        reshape[axis] = axis_size - 1
        return np.broadcast_to(np.diff(x_array).reshape(tuple(reshape)), width_shape).copy()
    if tuple(x_array.shape) != source_shape:
        raise ValueError(
            "program AD reduction trapezoid x must match the integration axis or full array shape"
        )
    return np.diff(x_array, axis=axis)


def _program_ad_reduction_trapezoid_static_weights(
    source_shape: tuple[int, ...],
    *,
    x: object,
    dx: object,
    axis: int,
) -> NDArray[np.float64]:
    widths = _program_ad_reduction_trapezoid_static_widths(source_shape, x=x, dx=dx, axis=axis)
    weights = np.zeros(source_shape, dtype=np.float64)
    reduced_shape = source_shape[:axis] + source_shape[axis + 1 :]
    for reduced_index in np.ndindex(reduced_shape):
        reduced_index_tuple = tuple(reduced_index)
        for segment_index in range(source_shape[axis] - 1):
            left_index = reduced_index_tuple[:axis] + (segment_index,) + reduced_index_tuple[axis:]
            right_index = (
                reduced_index_tuple[:axis] + (segment_index + 1,) + reduced_index_tuple[axis:]
            )
            width_index = (
                reduced_index_tuple[:axis] + (segment_index,) + reduced_index_tuple[axis:]
            )
            half_width = 0.5 * float(widths[width_index])
            weights[left_index] += half_width
            weights[right_index] += half_width
    return weights


def program_ad_reduction_trapezoid_derivative_rule(
    source_shape: Sequence[int],
    *,
    x: object = None,
    dx: object = 1.0,
    axis: int = -1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed trapezoid integration signature."""

    source = _program_ad_trapezoid_normalise_static_shape(source_shape)
    normalised_axis = _normalise_trapezoid_axis(axis, len(source))
    weights = _program_ad_reduction_trapezoid_static_weights(
        source, x=x, dx=dx, axis=normalised_axis
    )
    output_shape = _program_ad_trapezoid_output_shape(source, normalised_axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        values_array = _program_ad_trapezoid_source_vector(
            "values", values, source_shape=source
        ).reshape(source)
        return _program_ad_float64_vector_result(
            np.sum(values_array * weights, axis=normalised_axis)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_trapezoid_source_vector("values", values, source_shape=source)
        tangent_array = _program_ad_trapezoid_source_vector(
            "tangent", tangent, source_shape=source
        ).reshape(source)
        return _program_ad_float64_vector_result(
            np.sum(tangent_array * weights, axis=normalised_axis)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_trapezoid_source_vector("values", values, source_shape=source)
        cotangent_array = _program_ad_trapezoid_cotangent_array(
            cotangent, output_shape=output_shape
        )
        if output_shape == ():
            return _program_ad_float64_vector_result(float(cotangent_array) * weights)
        expanded = np.expand_dims(cotangent_array, axis=normalised_axis)
        return _program_ad_float64_vector_result(expanded * weights)

    return CustomDerivativeRule(
        name=(
            "program_ad_reduction_trapezoid_"
            f"{_program_ad_shape_signature(source)}_axis_{normalised_axis}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
