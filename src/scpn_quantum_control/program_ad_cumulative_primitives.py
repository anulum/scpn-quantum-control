# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD cumulative primitive rules
"""Static cumulative derivative rules for Program AD registry dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_dtype_of,
    _program_ad_array_shape_of,
)
from .program_ad_registry import (
    _PROGRAM_AD_CUMULATIVE_IDENTITIES,
    _PROGRAM_AD_CUMULATIVE_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .program_ad_shape_transforms import (
    _program_ad_float64_vector_result,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
)


def _normalise_cumulative_axis(name: str, axis: int, ndim: int) -> int:
    """Return a non-negative cumulative axis for a ranked static source shape."""

    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    value = axis + ndim if axis < 0 else axis
    if value < 0 or value >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return value


def _program_ad_cumulative_cumsum_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    return np.cumsum(vector).astype(np.float64)


def _program_ad_cumulative_cumsum_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array(
        "program AD cumulative cumsum tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumsum tangent shape must match values shape")
    return np.cumsum(tangent_vector).astype(np.float64)


def _program_ad_cumulative_cumsum_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative cumsum cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumsum cotangent shape must match output shape")
    return _program_ad_float64_vector_result(np.flip(np.cumsum(np.flip(cotangent_vector))))


def _program_ad_cumulative_cumprod_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    return np.cumprod(vector).astype(np.float64)


def _program_ad_cumulative_cumprod_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array(
        "program AD cumulative cumprod tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumprod tangent shape must match values shape")
    result = np.zeros_like(vector, dtype=np.float64)
    for output_index in range(vector.size):
        total = 0.0
        for tangent_index in range(output_index + 1):
            product = 1.0
            for factor_index in range(output_index + 1):
                product *= (
                    tangent_vector[factor_index]
                    if factor_index == tangent_index
                    else vector[factor_index]
                )
            total += product
        result[output_index] = total
    return result


def _program_ad_cumulative_cumprod_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative cumprod cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumprod cotangent shape must match output shape")
    result = np.zeros_like(vector, dtype=np.float64)
    for input_index in range(vector.size):
        total = 0.0
        for output_index in range(input_index, vector.size):
            product = 1.0
            for factor_index in range(output_index + 1):
                if factor_index != input_index:
                    product *= vector[factor_index]
            total += cotangent_vector[output_index] * product
        result[input_index] = total
    return result


def _program_ad_cumulative_diff_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    return np.diff(vector).astype(np.float64)


def _program_ad_cumulative_diff_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array("program AD cumulative diff tangent", tangent).reshape(
        -1
    )
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative diff tangent shape must match values shape")
    return np.diff(tangent_vector).astype(np.float64)


def _program_ad_cumulative_diff_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative diff cotangent", cotangent
    ).reshape(-1)
    if vector.size == 0:
        raise ValueError("program AD cumulative diff direct rule requires at least one value")
    if cotangent_vector.shape != (max(vector.size - 1, 0),):
        raise ValueError("program AD cumulative diff cotangent shape must match output shape")
    result = np.zeros_like(vector, dtype=np.float64)
    if cotangent_vector.size == 0:
        return result
    result[0] = -cotangent_vector[0]
    result[-1] = cotangent_vector[-1]
    if vector.size > 2:
        result[1:-1] = cotangent_vector[:-1] - cotangent_vector[1:]
    return result


def _program_ad_cumulative_derivative_rule(name: str) -> CustomDerivativeRule:
    """Return the registry direct rule for a flat cumulative primitive."""

    if name == "cumsum":
        return CustomDerivativeRule(
            name="program_ad_cumulative_cumsum_direct_rule",
            value_fn=_program_ad_cumulative_cumsum_value,
            jvp_rule=_program_ad_cumulative_cumsum_jvp,
            vjp_rule=_program_ad_cumulative_cumsum_vjp,
        )
    if name == "cumprod":
        return CustomDerivativeRule(
            name="program_ad_cumulative_cumprod_direct_rule",
            value_fn=_program_ad_cumulative_cumprod_value,
            jvp_rule=_program_ad_cumulative_cumprod_jvp,
            vjp_rule=_program_ad_cumulative_cumprod_vjp,
        )
    if name == "diff":
        return CustomDerivativeRule(
            name="program_ad_cumulative_diff_direct_rule",
            value_fn=_program_ad_cumulative_diff_value,
            jvp_rule=_program_ad_cumulative_diff_jvp,
            vjp_rule=_program_ad_cumulative_diff_vjp,
        )
    raise ValueError(f"unsupported program AD cumulative primitive {name}")


def _program_ad_cumulative_normalise_static_shape(
    name: str, source_shape: Sequence[int]
) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD cumulative {name} direct rule requires non-negative dimensions"
        )
    if _program_ad_shape_static_size(shape) == 0:
        raise ValueError(f"program AD cumulative {name} direct rule requires at least one value")
    return shape


def _program_ad_cumulative_static_axis(
    source_shape: tuple[int, ...], axis: int | None
) -> int | None:
    return None if axis is None else _normalise_cumulative_axis("axis", axis, len(source_shape))


def _program_ad_cumulative_axis_signature(axis: int | None) -> str:
    return "flat" if axis is None else str(axis)


def _program_ad_cumulative_source_array(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD cumulative {name} {role}", values).reshape(-1)
    if vector.size != _program_ad_shape_static_size(source_shape):
        raise ValueError(
            f"program AD cumulative {name} direct rule requires {role} "
            f"with {_program_ad_shape_static_size(source_shape)} values"
        )
    return vector.reshape(source_shape)


def _program_ad_cumulative_cumsum_static_vjp(
    cotangent_array: NDArray[np.float64],
    axis: int | None,
) -> NDArray[np.float64]:
    if axis is None:
        vector = cotangent_array.reshape(-1)
        return _program_ad_float64_vector_result(np.flip(np.cumsum(np.flip(vector))))
    return _program_ad_float64_vector_result(
        np.flip(np.cumsum(np.flip(cotangent_array, axis=axis), axis=axis), axis=axis)
    )


def _program_ad_cumulative_cumprod_static_jvp_array(
    values_array: NDArray[np.float64],
    tangent_array: NDArray[np.float64],
    axis: int | None,
) -> NDArray[np.float64]:
    if axis is None:
        return _program_ad_cumulative_cumprod_jvp(
            values_array.reshape(-1), tangent_array.reshape(-1)
        ).reshape(values_array.shape)
    result = np.zeros_like(values_array, dtype=np.float64)
    axis_size = values_array.shape[axis]
    output_shape = values_array.shape[:axis] + values_array.shape[axis + 1 :]
    for output_index in np.ndindex(output_shape):
        for end_index in range(axis_size):
            total = 0.0
            for tangent_index in range(end_index + 1):
                product = 1.0
                for factor_index in range(end_index + 1):
                    full_index = output_index[:axis] + (factor_index,) + output_index[axis:]
                    product *= float(
                        tangent_array[full_index]
                        if factor_index == tangent_index
                        else values_array[full_index]
                    )
                total += product
            result[output_index[:axis] + (end_index,) + output_index[axis:]] = total
    return result


def _program_ad_cumulative_cumprod_static_vjp_array(
    values_array: NDArray[np.float64],
    cotangent_array: NDArray[np.float64],
    axis: int | None,
) -> NDArray[np.float64]:
    if axis is None:
        return _program_ad_cumulative_cumprod_vjp(
            values_array.reshape(-1), cotangent_array.reshape(-1)
        ).reshape(values_array.shape)
    result = np.zeros_like(values_array, dtype=np.float64)
    axis_size = values_array.shape[axis]
    output_shape = values_array.shape[:axis] + values_array.shape[axis + 1 :]
    for output_index in np.ndindex(output_shape):
        for input_index in range(axis_size):
            total = 0.0
            for end_index in range(input_index, axis_size):
                product = 1.0
                for factor_index in range(end_index + 1):
                    if factor_index != input_index:
                        full_index = output_index[:axis] + (factor_index,) + output_index[axis:]
                        product *= float(values_array[full_index])
                full_output_index = output_index[:axis] + (end_index,) + output_index[axis:]
                total += float(cotangent_array[full_output_index]) * product
            result[output_index[:axis] + (input_index,) + output_index[axis:]] = total
    return result


def _program_ad_cumulative_diff_once_vjp_axis(
    cotangent_array: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int,
) -> NDArray[np.float64]:
    result = np.zeros(source_shape, dtype=np.float64)
    source_axis_size = source_shape[axis]
    output_shape = source_shape[:axis] + (source_axis_size - 1,) + source_shape[axis + 1 :]
    for output_index in np.ndindex(output_shape):
        lower_index = output_index
        upper_index = output_index[:axis] + (output_index[axis] + 1,) + output_index[axis + 1 :]
        result[lower_index] -= cotangent_array[output_index]
        result[upper_index] += cotangent_array[output_index]
    return result


def _program_ad_cumulative_diff_static_vjp_array(
    cotangent_array: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    order: int,
    axis: int,
) -> NDArray[np.float64]:
    current = cotangent_array
    for step in range(order, 0, -1):
        next_source_shape = (
            source_shape[:axis] + (source_shape[axis] - step + 1,) + source_shape[axis + 1 :]
        )
        current = _program_ad_cumulative_diff_once_vjp_axis(
            current, source_shape=next_source_shape, axis=axis
        )
    if current.shape != source_shape:
        raise ValueError("program AD cumulative diff VJP internal shape mismatch")
    return current


def program_ad_cumulative_cumsum_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed cumsum signature."""

    source = _program_ad_cumulative_normalise_static_shape("cumsum", source_shape)
    normalised_axis = _program_ad_cumulative_static_axis(source, axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        values_array = _program_ad_cumulative_source_array(
            "cumsum", "values", values, source_shape=source
        )
        return _program_ad_float64_vector_result(np.cumsum(values_array, axis=normalised_axis))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_cumulative_source_array("cumsum", "values", values, source_shape=source)
        tangent_array = _program_ad_cumulative_source_array(
            "cumsum", "tangent", tangent, source_shape=source
        )
        return _program_ad_float64_vector_result(np.cumsum(tangent_array, axis=normalised_axis))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_cumulative_source_array("cumsum", "values", values, source_shape=source)
        cotangent_array = _program_ad_cumulative_source_array(
            "cumsum", "cotangent", cotangent, source_shape=source
        )
        return _program_ad_cumulative_cumsum_static_vjp(cotangent_array, normalised_axis)

    return CustomDerivativeRule(
        name=(
            f"program_ad_cumulative_cumsum_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_cumulative_axis_signature(normalised_axis)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_cumulative_cumprod_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed cumprod signature."""

    source = _program_ad_cumulative_normalise_static_shape("cumprod", source_shape)
    normalised_axis = _program_ad_cumulative_static_axis(source, axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        values_array = _program_ad_cumulative_source_array(
            "cumprod", "values", values, source_shape=source
        )
        return _program_ad_float64_vector_result(np.cumprod(values_array, axis=normalised_axis))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        values_array = _program_ad_cumulative_source_array(
            "cumprod", "values", values, source_shape=source
        )
        tangent_array = _program_ad_cumulative_source_array(
            "cumprod", "tangent", tangent, source_shape=source
        )
        return _program_ad_float64_vector_result(
            _program_ad_cumulative_cumprod_static_jvp_array(
                values_array, tangent_array, normalised_axis
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        values_array = _program_ad_cumulative_source_array(
            "cumprod", "values", values, source_shape=source
        )
        cotangent_array = _program_ad_cumulative_source_array(
            "cumprod", "cotangent", cotangent, source_shape=source
        )
        return _program_ad_float64_vector_result(
            _program_ad_cumulative_cumprod_static_vjp_array(
                values_array, cotangent_array, normalised_axis
            )
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_cumulative_cumprod_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_cumulative_axis_signature(normalised_axis)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_cumulative_diff_derivative_rule(
    source_shape: Sequence[int],
    *,
    order: int = 1,
    axis: int = -1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed diff signature."""

    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) < 0:
        raise ValueError(
            "program AD cumulative diff direct rule requires non-negative integer order"
        )
    source = _program_ad_cumulative_normalise_static_shape("diff", source_shape)
    normalised_order = int(order)
    normalised_axis = _normalise_cumulative_axis("axis", int(axis), len(source))
    if normalised_order > source[normalised_axis]:
        raise ValueError("program AD cumulative diff direct rule order exceeds axis length")
    output_shape = (
        source[:normalised_axis]
        + (source[normalised_axis] - normalised_order,)
        + source[normalised_axis + 1 :]
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        values_array = _program_ad_cumulative_source_array(
            "diff", "values", values, source_shape=source
        )
        return _program_ad_float64_vector_result(
            np.diff(values_array, n=normalised_order, axis=normalised_axis)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_cumulative_source_array("diff", "values", values, source_shape=source)
        tangent_array = _program_ad_cumulative_source_array(
            "diff", "tangent", tangent, source_shape=source
        )
        return _program_ad_float64_vector_result(
            np.diff(tangent_array, n=normalised_order, axis=normalised_axis)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_cumulative_source_array("diff", "values", values, source_shape=source)
        cotangent_array = _program_ad_cumulative_source_array(
            "diff", "cotangent", cotangent, source_shape=output_shape
        )
        return _program_ad_float64_vector_result(
            _program_ad_cumulative_diff_static_vjp_array(
                cotangent_array,
                source_shape=source,
                order=normalised_order,
                axis=normalised_axis,
            )
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_cumulative_diff_{_program_ad_shape_signature(source)}_order_"
            f"{normalised_order}_axis_{normalised_axis}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_cumulative_axis(args: tuple[object, ...]) -> int | None:
    """Return the optional static axis argument for a cumulative primitive."""

    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD cumulative rule requires array and static parameters")
    if len(args) == 1 or args[1] is None:
        return None
    axis = args[1]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD cumulative axis must be a static integer or None")
    return int(axis)


def _program_ad_cumulative_diff_order(args: tuple[object, ...]) -> int:
    """Return the static finite-difference order for a cumulative diff primitive."""

    if len(args) < 2:
        return 1
    order = args[1]
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("program AD np.diff requires non-negative integer n")
    order = int(order)
    if order < 0:
        raise ValueError("program AD np.diff requires non-negative integer n")
    return order


def _program_ad_cumulative_scan_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for cumsum and cumprod primitives."""

    if len(args) not in {1, 2}:
        raise ValueError("program AD cumulative scan shape rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD cumulative scan requires at least one element")
    axis = _program_ad_cumulative_axis(args)
    return (int(np.prod(source_shape)),) if axis is None else source_shape


def _program_ad_cumulative_diff_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a cumulative diff primitive."""

    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD diff shape rule requires array, order, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    order = _program_ad_cumulative_diff_order(args)
    axis = args[2] if len(args) == 3 else -1
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD diff axis must be a static integer")
    axis_index = _normalise_axis("axis", int(axis), len(source_shape))
    axis_size = max(source_shape[axis_index] - order, 0)
    return source_shape[:axis_index] + (axis_size,) + source_shape[axis_index + 1 :]


def _program_ad_cumulative_dtype_rule(args: tuple[object, ...]) -> str:
    """Return the dtype emitted by a cumulative primitive."""

    if not args:
        raise ValueError("program AD cumulative dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_cumulative_scan_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return canonical static arguments for cumsum and cumprod primitives."""

    return (_program_ad_cumulative_axis(args),)


def _program_ad_cumulative_diff_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return canonical static arguments for a cumulative diff primitive."""

    order = _program_ad_cumulative_diff_order(args)
    axis = args[2] if len(args) == 3 else -1
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD diff axis must be a static integer")
    axis_index = _normalise_axis("axis", int(axis), len(_program_ad_array_shape_of(args[0])))
    return (order, axis_index)


_PROGRAM_AD_CUMULATIVE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "cumsum": _program_ad_cumulative_scan_shape,
    "cumprod": _program_ad_cumulative_scan_shape,
    "diff": _program_ad_cumulative_diff_shape,
}

_PROGRAM_AD_CUMULATIVE_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "cumsum": _program_ad_cumulative_scan_static_arguments,
    "cumprod": _program_ad_cumulative_scan_static_arguments,
    "diff": _program_ad_cumulative_diff_static_arguments,
}


def _program_ad_cumulative_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Map a cumulative primitive over a batch axis."""

    if len(args) != len(axes):
        raise ValueError("program AD cumulative batching axes must match argument count")
    if not args:
        raise ValueError("program AD cumulative batching requires an array operand")
    array = _as_real_numeric_array("program AD cumulative batched operand", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return _as_real_numeric_array("program AD cumulative batched output", function(*args))
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD cumulative batching supports static parameters only")
    batch_axis = _normalise_axis("axes[0]", batch_axis, array.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD cumulative batched output",
            function(np.take(array, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(int(array.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_cumulative_lowering_metadata(name: str) -> Mapping[str, str]:
    """Return lowering metadata for a Program AD cumulative primitive."""

    static_factory = {
        "cumsum": "program_ad_cumulative_cumsum_derivative_rule",
        "cumprod": "program_ad_cumulative_cumprod_derivative_rule",
        "diff": "program_ad_cumulative_diff_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "cumsum": "ordered_axis_sequence",
        "cumprod": "ordered_axis_zero_factor_sensitive",
        "diff": "finite_difference_order_and_spacing",
    }
    static_signature = (
        "source_shape:ranked_tensor_shape;order_axis"
        if name == "diff"
        else "source_shape:ranked_tensor_shape;axis"
    )
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff cumulative dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.cumulative.{name}",
        "llvm": "blocked_until_executable_cumulative_lowering",
        "rust": "blocked_until_polyglot_cumulative_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_cumulative_primitive_contracts() -> None:
    """Register fail-closed Program AD cumulative primitive contracts."""

    for name, identity in _PROGRAM_AD_CUMULATIVE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_cumulative_derivative_rule(name),
                batching_rule=_program_ad_cumulative_batching_rule,
                lowering_metadata=_program_ad_cumulative_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_CUMULATIVE_SHAPE_RULES[name],
                dtype_rule=_program_ad_cumulative_dtype_rule,
                static_argument_rule=_PROGRAM_AD_CUMULATIVE_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_CUMULATIVE_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_cumulative_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate cumulative primitive dispatch helpers against concrete arguments."""

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


def _require_program_ad_cumulative_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered cumulative primitive runtime contract."""

    identity: PrimitiveIdentity | None = _PROGRAM_AD_CUMULATIVE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD cumulative primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_CUMULATIVE_POLICY:
        raise ValueError(f"invalid program AD cumulative primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD cumulative primitive effect for {identity.key}")

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
            f"incomplete program AD cumulative primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_cumulative_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_program_ad_cumulative_derivative_rule",
    "_register_program_ad_cumulative_primitive_contracts",
    "_require_program_ad_cumulative_contract",
    "program_ad_cumulative_cumprod_derivative_rule",
    "program_ad_cumulative_cumsum_derivative_rule",
    "program_ad_cumulative_diff_derivative_rule",
)
