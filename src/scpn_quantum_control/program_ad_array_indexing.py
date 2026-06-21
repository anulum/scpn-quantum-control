# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD static array indexing rules
"""Static array-indexing derivative rules for Program AD registry dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_registry import CustomDerivativeRule

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
    return CustomDerivativeRule(
        name=f"program_ad_array_{name}_trace_contract",
        value_fn=_program_ad_array_direct_value,
        jvp_rule=_program_ad_array_direct_jvp,
    )


def _program_ad_array_normalise_static_shape(
    primitive_name: str, source_shape: Sequence[int]
) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires non-negative dimensions"
        )
    return shape


def _program_ad_array_static_size(source_shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in source_shape:
        size *= dimension
    return size


def _program_ad_array_signature(source_shape: tuple[int, ...]) -> str:
    return "scalar" if not source_shape else "x".join(str(dimension) for dimension in source_shape)


def _program_ad_array_vector(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD array {primitive_name} {role}", values).reshape(
        -1
    )
    if vector.size != expected_size:
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    return vector


def _program_ad_array_getitem_flat_indices(
    source_shape: tuple[int, ...], index: object
) -> NDArray[np.int64]:
    _validate_static_basic_index(index)
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    try:
        selected = source_indices[cast(Any, index)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array getitem direct rule requires in-bounds indices"
        ) from exc
    return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_take_indices(indices: object) -> NDArray[np.int64]:
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take direct rule requires static integer indices")
    return cast(NDArray[np.int64], np.asarray(raw_indices, dtype=np.int64))


def _program_ad_array_take_flat_indices(
    source_shape: tuple[int, ...],
    indices: object,
    axis: int | None,
    mode: str,
) -> NDArray[np.int64]:
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    raw_indices = _program_ad_array_take_indices(indices)
    mode_name = _program_ad_array_take_mode(mode, context="direct rule")
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source_shape))
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
    return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_take_along_axis_flat_indices(
    source_shape: tuple[int, ...],
    indices: object,
    axis: int,
) -> NDArray[np.int64]:
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    raw_indices = _program_ad_array_take_indices(indices)
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    try:
        selected = np.take_along_axis(source_indices, raw_indices, axis=normalised_axis)
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "program AD array take_along_axis direct rule requires in-bounds indices "
            "with shape compatible with the source"
        ) from exc
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


def _program_ad_array_delete_flat_indices(
    source_shape: tuple[int, ...],
    obj: object,
    axis: int | None,
) -> NDArray[np.int64]:
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    delete_obj = _program_ad_array_delete_object(obj, context="direct rule")
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source_shape))
    source = source_indices.reshape(-1) if normalised_axis is None else source_indices
    try:
        selected = np.delete(source, cast(Any, delete_obj), axis=normalised_axis)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array delete direct rule requires static in-bounds deletion selectors "
            "and an axis-compatible source"
        ) from exc
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


def _program_ad_array_pad_layout(
    source_shape: tuple[int, ...],
    pad_width: object,
    constant_values: object,
    *,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], tuple[int, ...]]:
    pairs = _program_ad_array_pad_width(pad_width, len(source_shape), context=context)
    constants = _program_ad_array_pad_constant_values(constant_values, context=context)
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
    return (
        cast(NDArray[np.int64], np.asarray(inserted_indices, dtype=np.int64).reshape(-1)),
        cast(NDArray[np.float64], np.asarray(inserted_constants, dtype=np.float64).reshape(-1)),
        tuple(int(dimension) for dimension in np.asarray(inserted_indices).shape),
    )


def _program_ad_array_direct_gather(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    vector = _program_ad_array_vector(
        primitive_name,
        "values",
        values,
        expected_size=_program_ad_array_static_size(source_shape),
    )
    return _program_ad_float64_vector_result(vector[flat_indices])


def _program_ad_array_direct_gather_jvp(
    primitive_name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
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
    return _program_ad_float64_vector_result(tangent_vector[flat_indices])


def _program_ad_array_direct_scatter_vjp(
    primitive_name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
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
    return result


def _program_ad_array_direct_pad_value(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
    flat_constants: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_array_vector(
        primitive_name,
        "values",
        values,
        expected_size=_program_ad_array_static_size(source_shape),
    )
    result = np.array(flat_constants, dtype=np.float64, copy=True)
    source_mask = flat_indices >= 0
    result[source_mask] = vector[flat_indices[source_mask]]
    return result


def _program_ad_array_direct_pad_jvp(
    primitive_name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
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
    return result


def _program_ad_array_direct_pad_vjp(
    primitive_name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
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
