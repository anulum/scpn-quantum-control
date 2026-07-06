# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD selection primitive rules
"""Program AD selection derivative factories and registry contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import _normalise_axis
from .program_ad_registry import (
    _PROGRAM_AD_SELECTION_IDENTITIES,
    _PROGRAM_AD_SELECTION_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveBatchingRule,
    PrimitiveContract,
    PrimitiveDTypeRule,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .program_ad_shape_transforms import (
    _program_ad_float64_vector_result,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
)


def _is_trace_predicate(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program scalar predicate."""
    return type(value).__name__ == "_TracePredicate" and hasattr(value, "context")


def _is_trace_predicate_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program predicate array."""
    return type(value).__name__ == "TraceADPredicateArray" and hasattr(value, "predicates")


def _is_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""
    return type(value).__name__ == "TraceADArray" and hasattr(value, "context")


def _is_trace_scalar(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace scalar."""
    return type(value).__name__ == "TraceADScalar" and hasattr(value, "context")


def _is_runtime_trace_payload(value: object) -> bool:
    """Return whether ``value`` is a runtime trace payload rather than static data."""
    return (
        _is_trace_scalar(value)
        or _is_trace_array(value)
        or _is_trace_predicate(value)
        or _is_trace_predicate_array(value)
    )


def _trace_predicate_array_shape(value: object) -> tuple[int, ...]:
    """Return a predicate-array shape through the protocol boundary."""
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        raise ValueError("program AD selection predicate array shape must be static")
    return tuple(int(dimension) for dimension in shape)


def _program_ad_array_static_size(source_shape: Sequence[int]) -> int:
    """Return the element count for a static tensor shape."""
    size = 1
    for dimension in source_shape:
        size *= int(dimension)
    return int(size)


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static array shape recorded by a trace value or array-like input."""
    if _is_trace_array(value):
        return _trace_predicate_array_shape(value)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name recorded by a trace value or array-like input."""
    if _is_trace_array(value):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD selection primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return a NumPy-compatible broadcast shape or fail closed."""
    try:
        shape: tuple[int, ...] = np.broadcast_shapes(*shapes)
        return shape
    except ValueError as exc:
        raise ValueError(
            "program AD selection operands must follow NumPy broadcasting rules"
        ) from exc


def _normalise_sort_axis(axis: object, rank: int) -> int:
    """Return the normalised static axis for selection ordering primitives."""
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.sort axis must be a static integer or None")
    axis_index = int(axis)
    if axis_index < 0:
        axis_index += rank
    if axis_index < 0 or axis_index >= rank:
        raise ValueError("program AD np.sort axis out of bounds")
    return axis_index


def _program_ad_elementwise_unbroadcast(
    values: NDArray[np.float64],
    *,
    target_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Reduce a broadcasted adjoint back to ``target_shape``."""
    result = np.asarray(values, dtype=np.float64)
    if target_shape == ():
        return np.array([float(np.sum(result))], dtype=np.float64)
    while result.ndim > len(target_shape):
        result = np.sum(result, axis=0)
    for axis, dimension in enumerate(target_shape):
        if dimension == 1 and result.shape[axis] != 1:
            result = np.sum(result, axis=axis, keepdims=True)
    return _program_ad_float64_vector_result(result.reshape(target_shape))


def _trace_choose_selector_indices(
    selector: object,
    *,
    choice_count: int,
    mode: str,
) -> NDArray[np.int64]:
    """Return static choose selector indices or reject runtime trace payloads."""
    if _is_runtime_trace_payload(selector):
        raise ValueError("program AD np.choose requires a static integer selector")
    raw = np.asarray(selector)
    if raw.dtype == object and any(_is_runtime_trace_payload(item) for item in raw.reshape(-1)):
        raise ValueError("program AD np.choose requires a static integer selector")
    if raw.dtype.kind not in {"i", "u", "b"}:
        raise ValueError("program AD np.choose requires a static integer selector")
    indices = raw.astype(np.int64, copy=False)
    if mode == "raise":
        if bool(np.any(indices < 0)) or bool(np.any(indices >= choice_count)):
            raise ValueError("program AD np.choose selector indices out of bounds")
        return indices
    if mode == "wrap":
        return cast(NDArray[np.int64], np.mod(indices, choice_count).astype(np.int64))
    if mode == "clip":
        return cast(NDArray[np.int64], np.clip(indices, 0, choice_count - 1).astype(np.int64))
    raise ValueError("program AD np.choose mode must be raise, wrap, or clip")


def _trace_compress_condition_indices(condition: object) -> NDArray[np.int64]:
    """Return static compress condition indices or reject runtime trace payloads."""
    if _is_runtime_trace_payload(condition):
        raise ValueError("program AD np.compress requires a static boolean condition")
    raw = np.asarray(condition)
    if raw.dtype == object and any(_is_runtime_trace_payload(item) for item in raw.reshape(-1)):
        raise ValueError("program AD np.compress requires a static boolean condition")
    if raw.ndim != 1:
        raise ValueError("program AD np.compress requires a one-dimensional condition")
    if raw.dtype.kind != "b":
        raise ValueError("program AD np.compress requires a static boolean condition")
    return cast(NDArray[np.int64], np.flatnonzero(raw).astype(np.int64))


def _trace_extract_condition_indices(condition: object, array_size: int) -> NDArray[np.int64]:
    """Return static extract condition indices or reject runtime trace payloads."""
    if _is_runtime_trace_payload(condition):
        raise ValueError("program AD np.extract requires a static boolean condition")
    raw = np.asarray(condition)
    if raw.dtype == object and any(_is_runtime_trace_payload(item) for item in raw.reshape(-1)):
        raise ValueError("program AD np.extract requires a static boolean condition")
    if raw.dtype.kind != "b":
        raise ValueError("program AD np.extract requires a static boolean condition")
    if raw.size != array_size:
        raise ValueError("program AD np.extract condition size must match array size")
    return cast(NDArray[np.int64], np.flatnonzero(raw.reshape(-1)).astype(np.int64))


def _validate_program_ad_selection_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate selection primitive dispatch helpers against concrete arguments."""
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


def _program_ad_selection_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD selection primitive contracts require static derivative factories "
        "or operator-intercepted trace dispatch"
    )


def _program_ad_selection_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD selection primitive contracts require static derivative factories "
        "or operator-intercepted trace dispatch"
    )


def _program_ad_selection_derivative_rule(name: str) -> CustomDerivativeRule:
    return CustomDerivativeRule(
        name=f"program_ad_selection_{name}_trace_contract",
        value_fn=_program_ad_selection_direct_value,
        jvp_rule=_program_ad_selection_direct_jvp,
    )


def _program_ad_selection_normalise_shapes(
    primitive_name: str,
    true_shape: Sequence[int],
    false_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    true_static_shape = tuple(int(dimension) for dimension in true_shape)
    false_static_shape = tuple(int(dimension) for dimension in false_shape)
    if any(dimension < 0 for dimension in (*true_static_shape, *false_static_shape)):
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires non-negative dimensions"
        )
    try:
        output_shape = np.broadcast_shapes(true_static_shape, false_static_shape)
    except ValueError as exc:
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires "
            "broadcast-compatible branch shapes"
        ) from exc
    return true_static_shape, false_static_shape, tuple(int(dim) for dim in output_shape)


def _program_ad_selection_condition_mask(
    condition: object,
    output_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    raw_condition = np.asarray(condition)
    if raw_condition.dtype.kind != "b":
        raise ValueError("program AD selection where direct rule requires a boolean condition")
    if tuple(raw_condition.shape) not in {(), output_shape}:
        raise ValueError(
            "program AD selection where direct rule requires scalar or output-shaped condition"
        )
    return np.broadcast_to(raw_condition, output_shape).astype(np.bool_, copy=False)


def _program_ad_selection_split_pair(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    true_shape: tuple[int, ...],
    false_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(
        f"program AD selection {primitive_name} {role}", values
    ).reshape(-1)
    true_size = _program_ad_shape_static_size(true_shape)
    false_size = _program_ad_shape_static_size(false_shape)
    if vector.size != true_size + false_size:
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires flattened true branch "
            "followed by false branch"
        )
    return (
        vector[:true_size].reshape(true_shape),
        vector[true_size:].reshape(false_shape),
    )


def program_ad_selection_where_derivative_rule(
    condition: object,
    true_shape: Sequence[int],
    false_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy where signature."""
    true_static_shape, false_static_shape, output_shape = _program_ad_selection_normalise_shapes(
        "where", true_shape, false_shape
    )
    condition_mask = _program_ad_selection_condition_mask(condition, output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        true_values, false_values = _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.where(
                condition_mask,
                np.broadcast_to(true_values, output_shape),
                np.broadcast_to(false_values, output_shape),
            )
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        true_tangent, false_tangent = _program_ad_selection_split_pair(
            "where",
            "tangent",
            tangent,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.where(
                condition_mask,
                np.broadcast_to(true_tangent, output_shape),
                np.broadcast_to(false_tangent, output_shape),
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        cotangent_array = _as_real_numeric_array(
            "program AD selection where cotangent", cotangent
        ).reshape(-1)
        if cotangent_array.size != _program_ad_shape_static_size(output_shape):
            raise ValueError(
                "program AD selection where VJP cotangent shape must match output shape"
            )
        cotangent_output = cotangent_array.reshape(output_shape)
        true_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(condition_mask, cotangent_output, 0.0), target_shape=true_static_shape
        )
        false_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(condition_mask, 0.0, cotangent_output), target_shape=false_static_shape
        )
        return _program_ad_float64_vector_result(np.concatenate((true_adjoint, false_adjoint)))

    return CustomDerivativeRule(
        name=(
            f"program_ad_selection_where_{_program_ad_shape_signature(true_static_shape)}_by_"
            f"{_program_ad_shape_signature(false_static_shape)}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_selection_normalise_clip_shapes(
    source_shape: Sequence[int],
    lower_shape: Sequence[int],
    upper_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    source_static_shape = tuple(int(dimension) for dimension in source_shape)
    lower_static_shape = tuple(int(dimension) for dimension in lower_shape)
    upper_static_shape = tuple(int(dimension) for dimension in upper_shape)
    if any(
        dimension < 0
        for dimension in (*source_static_shape, *lower_static_shape, *upper_static_shape)
    ):
        raise ValueError("program AD selection clip direct rule requires non-negative dimensions")
    try:
        lower_output = np.broadcast_shapes(source_static_shape, lower_static_shape)
        upper_output = np.broadcast_shapes(source_static_shape, upper_static_shape)
    except ValueError as exc:
        raise ValueError(
            "program AD selection clip direct rule requires bounds broadcastable to source shape"
        ) from exc
    if tuple(lower_output) != source_static_shape or tuple(upper_output) != source_static_shape:
        raise ValueError(
            "program AD selection clip direct rule requires bounds broadcastable to source shape"
        )
    return source_static_shape, lower_static_shape, upper_static_shape


def _program_ad_selection_split_clip(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    lower_shape: tuple[int, ...],
    upper_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD selection clip {role}", values).reshape(-1)
    source_size = _program_ad_shape_static_size(source_shape)
    lower_size = _program_ad_shape_static_size(lower_shape)
    upper_size = _program_ad_shape_static_size(upper_shape)
    if vector.size != source_size + lower_size + upper_size:
        raise ValueError(
            "program AD selection clip direct rule requires flattened source, lower, and upper"
        )
    lower_start = source_size
    upper_start = source_size + lower_size
    return (
        vector[:source_size].reshape(source_shape),
        vector[lower_start:upper_start].reshape(lower_shape),
        vector[upper_start:].reshape(upper_shape),
    )


def _program_ad_selection_clip_domain(
    values: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    derivative: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower_broadcast = np.broadcast_to(lower, values.shape)
    upper_broadcast = np.broadcast_to(upper, values.shape)
    if np.any(lower_broadcast > upper_broadcast):
        raise ValueError("program AD selection clip lower bound must not exceed upper bound")
    if derivative and np.any((values == lower_broadcast) | (values == upper_broadcast)):
        raise ValueError("program AD selection clip derivative is undefined at clipping boundary")
    return lower_broadcast, upper_broadcast


def program_ad_selection_clip_derivative_rule(
    source_shape: Sequence[int],
    *,
    lower_shape: Sequence[int] = (),
    upper_shape: Sequence[int] = (),
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy clip signature."""
    source_static_shape, lower_static_shape, upper_static_shape = (
        _program_ad_selection_normalise_clip_shapes(source_shape, lower_shape, upper_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=False
        )
        return _program_ad_float64_vector_result(np.clip(source, lower_broadcast, upper_broadcast))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        source_tangent, lower_tangent, upper_tangent = _program_ad_selection_split_clip(
            "tangent",
            tangent,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=True
        )
        lower_tangent = np.broadcast_to(lower_tangent, source_static_shape)
        upper_tangent = np.broadcast_to(upper_tangent, source_static_shape)
        return _program_ad_float64_vector_result(
            np.where(
                source < lower_broadcast,
                lower_tangent,
                np.where(source > upper_broadcast, upper_tangent, source_tangent),
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        cotangent_array = _as_real_numeric_array(
            "program AD selection clip cotangent", cotangent
        ).reshape(-1)
        if cotangent_array.size != _program_ad_shape_static_size(source_static_shape):
            raise ValueError(
                "program AD selection clip VJP cotangent shape must match output shape"
            )
        cotangent_output = cotangent_array.reshape(source_static_shape)
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=True
        )
        below = source < lower_broadcast
        above = source > upper_broadcast
        source_adjoint = np.where(~below & ~above, cotangent_output, 0.0)
        lower_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(below, cotangent_output, 0.0), target_shape=lower_static_shape
        )
        upper_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(above, cotangent_output, 0.0), target_shape=upper_static_shape
        )
        return _program_ad_float64_vector_result(
            np.concatenate((source_adjoint.reshape(-1), lower_adjoint, upper_adjoint))
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_selection_clip_{_program_ad_shape_signature(source_static_shape)}_bounds_"
            f"{_program_ad_shape_signature(lower_static_shape)}_by_"
            f"{_program_ad_shape_signature(upper_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_selection_condition_shape(condition: object) -> tuple[int, ...]:
    if _is_trace_predicate(condition):
        return ()
    if _is_trace_predicate_array(condition):
        return _trace_predicate_array_shape(condition)
    raw = np.asarray(condition)
    if raw.dtype.kind != "b":
        raise ValueError("program AD selection condition must be boolean")
    return tuple(int(dimension) for dimension in raw.shape)


def _program_ad_selection_where_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection where shape rule requires condition, true, false")
    output_shape = _broadcast_shape(
        _program_ad_array_shape_of(args[1]),
        _program_ad_array_shape_of(args[2]),
    )
    condition_shape = _program_ad_selection_condition_shape(args[0])
    if condition_shape not in {(), output_shape}:
        raise ValueError(
            "program AD selection where condition shape must be scalar or output-shaped"
        )
    return output_shape


def _program_ad_selection_clip_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection clip shape rule requires source, lower, and upper")
    source_shape = _program_ad_array_shape_of(args[0])
    try:
        lower_shape = np.broadcast_shapes(source_shape, _program_ad_array_shape_of(args[1]))
        upper_shape = np.broadcast_shapes(source_shape, _program_ad_array_shape_of(args[2]))
    except ValueError as exc:
        raise ValueError("program AD selection clip bounds must broadcast to source") from exc
    if tuple(lower_shape) != source_shape or tuple(upper_shape) != source_shape:
        raise ValueError("program AD selection clip bounds must broadcast to source")
    return source_shape


def _program_ad_selection_sort_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort shape rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    if axis is None:
        return (int(np.prod(source_shape)),)
    _normalise_sort_axis(axis, len(source_shape))
    return source_shape


def _program_ad_selection_index_reduce_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD index selection shape rule requires source and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) == 2 else None
    if axis is None:
        return ()
    axis_index = _normalise_sort_axis(axis, len(source_shape))
    return tuple(dimension for index, dimension in enumerate(source_shape) if index != axis_index)


def _program_ad_selection_argsort_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD argsort shape rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD argsort shape rule requires a NumPy sort kind")
    if axis is None:
        return (int(np.prod(source_shape)),)
    _normalise_sort_axis(axis, len(source_shape))
    return source_shape


def _program_ad_selection_sequence(name: str, value: object, role: str) -> tuple[object, ...]:
    if _is_trace_array(value) or isinstance(value, np.ndarray) or not isinstance(value, Sequence):
        raise ValueError(f"program AD {name} requires a static {role} sequence")
    return tuple(value)


def _program_ad_selection_select_parts(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], tuple[object, ...], object]:
    if len(args) != 3:
        raise ValueError("program AD select contract requires conditions, choices, and default")
    conditions = _program_ad_selection_sequence("select", args[0], "condition")
    choices = _program_ad_selection_sequence("select", args[1], "choice")
    if len(conditions) != len(choices):
        raise ValueError("program AD select requires matching condition and choice counts")
    return conditions, choices, args[2]


def _program_ad_selection_select_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    conditions, choices, default = _program_ad_selection_select_parts(args)
    output_shape = _program_ad_array_shape_of(default)
    for condition, choice in reversed(tuple(zip(conditions, choices, strict=True))):
        choice_shape = _program_ad_array_shape_of(choice)
        try:
            output_shape = tuple(
                int(dim) for dim in np.broadcast_shapes(choice_shape, output_shape)
            )
        except ValueError as exc:
            raise ValueError("program AD select choices must broadcast with default") from exc
        condition_shape = _program_ad_selection_condition_shape(condition)
        if condition_shape not in {(), output_shape}:
            raise ValueError("program AD select condition shape must be scalar or output-shaped")
    return output_shape


def _program_ad_selection_piecewise_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[object, ...], tuple[object, ...]]:
    if len(args) != 3:
        raise ValueError("program AD piecewise contract requires source, conditions, functions")
    source_shape = _program_ad_array_shape_of(args[0])
    conditions = _program_ad_selection_sequence("piecewise", args[1], "condition")
    functions = _program_ad_selection_sequence("piecewise", args[2], "function")
    if len(functions) not in {len(conditions), len(conditions) + 1}:
        raise ValueError(
            "program AD piecewise requires one function per condition and optional default"
        )
    for condition in conditions:
        condition_shape = _program_ad_selection_condition_shape(condition)
        if condition_shape not in {(), source_shape}:
            raise ValueError(
                "program AD piecewise condition shape must be scalar or source-shaped"
            )
    return source_shape, conditions, functions


def _program_ad_selection_piecewise_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, _conditions, _functions = _program_ad_selection_piecewise_parts(args)
    return source_shape


def _program_ad_selection_choose_parts(
    args: tuple[object, ...],
) -> tuple[NDArray[np.int64], tuple[tuple[int, ...], ...], str]:
    if len(args) != 3:
        raise ValueError("program AD choose contract requires selector, choices, and mode")
    raw_choices = args[1]
    if _is_trace_array(raw_choices):
        raise ValueError("program AD choose requires a static choice sequence")
    if isinstance(raw_choices, (np.ndarray, Sequence)):
        choices = tuple(raw_choices)
    else:
        raise ValueError("program AD choose requires a static choice sequence")
    if not choices:
        raise ValueError("program AD choose requires at least one choice")
    mode = args[2]
    if not isinstance(mode, str):
        raise ValueError("program AD choose mode must be raise, wrap, or clip")
    selector = _trace_choose_selector_indices(args[0], choice_count=len(choices), mode=mode)
    choice_shapes = tuple(_program_ad_array_shape_of(choice) for choice in choices)
    return selector, choice_shapes, mode


def _program_ad_selection_choose_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    selector, choice_shapes, _mode = _program_ad_selection_choose_parts(args)
    try:
        return tuple(
            int(dim) for dim in np.broadcast_shapes(tuple(selector.shape), *choice_shapes)
        )
    except ValueError as exc:
        raise ValueError(
            "program AD choose selector and choices must be broadcast-compatible"
        ) from exc


def _program_ad_selection_compress_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], NDArray[np.int64], int | None]:
    if len(args) != 3:
        raise ValueError("program AD compress contract requires condition, array, and axis")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_compress_condition_indices(args[0])
    axis_arg = args[2]
    if axis_arg is None:
        source_size = _program_ad_array_static_size(source_shape)
        if bool(np.any(indices >= source_size)):
            raise ValueError("program AD compress condition length exceeds flattened array")
        return source_shape, indices, None
    if isinstance(axis_arg, (bool, np.bool_)) or not isinstance(axis_arg, (int, np.integer)):
        raise ValueError("program AD compress requires a static integer axis or None")
    axis = _normalise_axis("axis", int(axis_arg), len(source_shape))
    if bool(np.any(indices >= source_shape[axis])):
        raise ValueError("program AD compress condition length exceeds selected axis")
    return source_shape, indices, axis


def _program_ad_selection_compress_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, indices, axis = _program_ad_selection_compress_parts(args)
    if axis is None:
        return (int(indices.size),)
    result_shape = list(source_shape)
    result_shape[axis] = int(indices.size)
    return tuple(result_shape)


def _program_ad_selection_extract_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD extract contract requires condition and array")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_extract_condition_indices(
        args[0], _program_ad_array_static_size(source_shape)
    )
    return (int(indices.size),)


def _program_ad_selection_where_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD selection where dtype rule requires condition, true, false")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args[1:])
    return str(np.result_type(*dtypes))


def _program_ad_selection_clip_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD selection clip dtype rule requires source, lower, and upper")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args)
    return str(np.result_type(*dtypes))


def _program_ad_selection_sort_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort dtype rule requires source, axis, and kind")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_selection_index_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD index selection dtype rule requires a source operand")
    return "int64"


def _program_ad_selection_select_dtype_rule(args: tuple[object, ...]) -> str:
    _conditions, choices, default = _program_ad_selection_select_parts(args)
    dtypes = [np.dtype(_program_ad_array_dtype_of(choice)) for choice in choices]
    dtypes.append(np.dtype(_program_ad_array_dtype_of(default)))
    return str(np.result_type(*dtypes))


def _program_ad_selection_piecewise_dtype_rule(args: tuple[object, ...]) -> str:
    _source_shape, _conditions, _functions = _program_ad_selection_piecewise_parts(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_selection_choose_dtype_rule(args: tuple[object, ...]) -> str:
    _selector, choice_shapes, _mode = _program_ad_selection_choose_parts(args)
    if not choice_shapes:
        raise ValueError("program AD choose requires at least one choice")
    raw_choices = tuple(cast(Sequence[object], args[1]))
    return str(
        np.result_type(*(np.dtype(_program_ad_array_dtype_of(choice)) for choice in raw_choices))
    )


def _program_ad_selection_compress_dtype_rule(args: tuple[object, ...]) -> str:
    _program_ad_selection_compress_parts(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[1])))


def _program_ad_selection_extract_dtype_rule(args: tuple[object, ...]) -> str:
    _program_ad_selection_extract_shape(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[1])))


def _program_ad_selection_where_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection where static rule requires condition, true, false")
    condition = args[0]
    output_shape = _program_ad_selection_where_shape(args)
    if _is_trace_predicate(condition):
        return ("runtime_predicate", (), output_shape)
    if _is_trace_predicate_array(condition):
        return ("runtime_predicate", _trace_predicate_array_shape(condition), output_shape)
    raw = np.asarray(condition)
    if raw.dtype.kind != "b":
        raise ValueError("program AD selection where static rule requires boolean condition")
    if tuple(raw.shape) not in {(), output_shape}:
        raise ValueError(
            "program AD selection where condition shape must be scalar or output-shaped"
        )
    mask = np.broadcast_to(raw, output_shape).reshape(-1)
    return ("static_condition", tuple(bool(item) for item in mask), output_shape)


def _program_ad_selection_clip_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection clip static rule requires source, lower, and upper")
    _program_ad_selection_clip_shape(args)
    return ()


def _program_ad_selection_sort_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort static rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if axis is not None:
        axis = _normalise_sort_axis(axis, len(source_shape))
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD selection sort static rule requires a NumPy sort kind")
    return ("axis", axis, "kind", "quicksort" if kind is None else kind)


def _program_ad_selection_index_reduce_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD index selection static rule requires source and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) == 2 else None
    if axis is None:
        return ("axis", None)
    return ("axis", _normalise_sort_axis(axis, len(source_shape)))


def _program_ad_selection_argsort_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD argsort static rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if axis is not None:
        axis = _normalise_sort_axis(axis, len(source_shape))
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD argsort static rule requires a NumPy sort kind")
    return ("axis", axis, "kind", "quicksort" if kind is None else kind)


def _program_ad_selection_select_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    conditions, choices, default = _program_ad_selection_select_parts(args)
    output_shape = _program_ad_selection_select_shape(args)
    condition_signatures: list[tuple[str, object, tuple[int, ...]]] = []
    for condition in conditions:
        condition_shape = _program_ad_selection_condition_shape(condition)
        if _is_trace_predicate(condition):
            condition_signatures.append(("runtime_predicate", (), output_shape))
        elif _is_trace_predicate_array(condition):
            condition_signatures.append(
                ("runtime_predicate", _trace_predicate_array_shape(condition), output_shape)
            )
        else:
            raw = np.asarray(condition)
            condition_signatures.append(
                (
                    "static_condition",
                    tuple(bool(item) for item in np.broadcast_to(raw, output_shape).reshape(-1)),
                    output_shape,
                )
            )
        if condition_shape not in {(), output_shape}:
            raise ValueError("program AD select condition shape must be scalar or output-shaped")
    return (
        "branch_count",
        len(conditions),
        "condition_signatures",
        tuple(condition_signatures),
        "choice_shapes",
        tuple(_program_ad_array_shape_of(choice) for choice in choices),
        "default_shape",
        _program_ad_array_shape_of(default),
        "output_shape",
        output_shape,
    )


def _program_ad_selection_piecewise_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, conditions, functions = _program_ad_selection_piecewise_parts(args)
    return (
        "source_shape",
        source_shape,
        "condition_shapes",
        tuple(_program_ad_selection_condition_shape(condition) for condition in conditions),
        "function_count",
        len(functions),
        "has_default",
        len(functions) == len(conditions) + 1,
    )


def _program_ad_selection_choose_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    selector, choice_shapes, mode = _program_ad_selection_choose_parts(args)
    return (
        "selector",
        tuple(int(item) for item in selector.reshape(-1)),
        "selector_shape",
        tuple(int(dimension) for dimension in selector.shape),
        "choice_shapes",
        choice_shapes,
        "mode",
        mode,
    )


def _program_ad_selection_compress_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, indices, axis = _program_ad_selection_compress_parts(args)
    return (
        "source_shape",
        source_shape,
        "indices",
        tuple(int(item) for item in indices.reshape(-1)),
        "axis",
        axis,
    )


def _program_ad_selection_extract_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD extract static rule requires condition and array")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_extract_condition_indices(
        args[0], _program_ad_array_static_size(source_shape)
    )
    return (
        "source_shape",
        source_shape,
        "indices",
        tuple(int(item) for item in indices.reshape(-1)),
    )


_PROGRAM_AD_SELECTION_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "where": _program_ad_selection_where_shape,
    "clip": _program_ad_selection_clip_shape,
    "sort": _program_ad_selection_sort_shape,
    "select": _program_ad_selection_select_shape,
    "piecewise": _program_ad_selection_piecewise_shape,
    "choose": _program_ad_selection_choose_shape,
    "compress": _program_ad_selection_compress_shape,
    "extract": _program_ad_selection_extract_shape,
    "argmax": _program_ad_selection_index_reduce_shape,
    "argmin": _program_ad_selection_index_reduce_shape,
    "argsort": _program_ad_selection_argsort_shape,
}

_PROGRAM_AD_SELECTION_DTYPE_RULES: Mapping[str, PrimitiveDTypeRule] = {
    "where": _program_ad_selection_where_dtype_rule,
    "clip": _program_ad_selection_clip_dtype_rule,
    "sort": _program_ad_selection_sort_dtype_rule,
    "select": _program_ad_selection_select_dtype_rule,
    "piecewise": _program_ad_selection_piecewise_dtype_rule,
    "choose": _program_ad_selection_choose_dtype_rule,
    "compress": _program_ad_selection_compress_dtype_rule,
    "extract": _program_ad_selection_extract_dtype_rule,
    "argmax": _program_ad_selection_index_dtype_rule,
    "argmin": _program_ad_selection_index_dtype_rule,
    "argsort": _program_ad_selection_index_dtype_rule,
}

_PROGRAM_AD_SELECTION_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "where": _program_ad_selection_where_static_arguments,
    "clip": _program_ad_selection_clip_static_arguments,
    "sort": _program_ad_selection_sort_static_arguments,
    "select": _program_ad_selection_select_static_arguments,
    "piecewise": _program_ad_selection_piecewise_static_arguments,
    "choose": _program_ad_selection_choose_static_arguments,
    "compress": _program_ad_selection_compress_static_arguments,
    "extract": _program_ad_selection_extract_static_arguments,
    "argmax": _program_ad_selection_index_reduce_static_arguments,
    "argmin": _program_ad_selection_index_reduce_static_arguments,
    "argsort": _program_ad_selection_argsort_static_arguments,
}


def _program_ad_selection_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD selection batching requires three operands and axes")
    arrays = tuple(np.asarray(arg) for arg in args)
    if all(axis is None for axis in axes):
        return function(*args)
    mapped_axes: list[int | None] = [
        None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
        for index, (axis, array) in enumerate(zip(axes, arrays, strict=True))
    ]
    batch_sizes = {
        int(array.shape[axis])
        for array, axis in zip(arrays, mapped_axes, strict=True)
        if axis is not None
    }
    if len(batch_sizes) != 1:
        raise ValueError("program AD selection batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(np.asarray(function(*sliced_args)))
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_selection_sort_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) not in {1, 2, 3} or len(args) != len(axes):
        raise ValueError("program AD selection sort batching requires source, axis, and kind")
    if len(axes) >= 2 and any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD selection sort batching requires static axis and kind")
    source = _as_real_numeric_array("program AD selection sort batched source", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return function(*args)
    batch_axis_index = _normalise_axis("axes[0]", batch_axis, source.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD selection sort batched output",
            function(np.take(source, batch_index, axis=batch_axis_index), *args[1:]),
        )
        for batch_index in range(int(source.shape[batch_axis_index]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_selection_index_batching_rule(
    _function: Callable[..., object],
    _args: tuple[object, ...],
    _axes: tuple[int | None, ...],
    _out_axes: int,
) -> NoReturn:
    raise ValueError(
        "program AD argmax/argmin/argsort batching is unsupported because integer "
        "index selection is nondifferentiable"
    )


def _program_ad_selection_batching_rule_for(name: str) -> PrimitiveBatchingRule:
    if name == "sort":
        return _program_ad_selection_sort_batching_rule
    if name in {"argmax", "argmin", "argsort"}:
        return _program_ad_selection_index_batching_rule
    return _program_ad_selection_batching_rule


def _program_ad_selection_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factories = {
        "where": "program_ad_selection_where_derivative_rule",
        "clip": "program_ad_selection_clip_derivative_rule",
        "sort": "operator_intercepted_sort_permutation_trace",
        "select": "operator_intercepted_static_select_fold_trace",
        "piecewise": "operator_intercepted_static_piecewise_fold_trace",
        "choose": "operator_intercepted_static_choose_gather_trace",
        "compress": "operator_intercepted_static_compress_gather_trace",
        "extract": "operator_intercepted_static_extract_gather_trace",
        "argmax": "unsupported_nondifferentiable_index_selection",
        "argmin": "unsupported_nondifferentiable_index_selection",
        "argsort": "unsupported_nondifferentiable_index_selection",
    }
    static_signatures = {
        "where": (
            "condition:static_bool_mask;true_shape:ranked_tensor_shape;"
            "false_shape:ranked_tensor_shape"
        ),
        "clip": (
            "source_shape:ranked_tensor_shape;lower_shape:ranked_tensor_shape;"
            "upper_shape:ranked_tensor_shape"
        ),
        "sort": "source_shape:ranked_tensor_shape;axis_kind",
        "select": "condition_sequence;choice_shapes;default_shape",
        "piecewise": "source_shape;condition_sequence;function_count",
        "choose": "selector_shape;choice_shapes;mode",
        "compress": "source_shape;condition_indices;axis",
        "extract": "source_shape;condition_indices",
        "argmax": "source_shape:ranked_tensor_shape;axis",
        "argmin": "source_shape:ranked_tensor_shape;axis",
        "argsort": "source_shape:ranked_tensor_shape;axis_kind",
    }
    nondifferentiable_boundaries = {
        "where": "predicate_branch_boundary",
        "clip": "clipping_boundary_and_bound_order",
        "sort": "strict_total_order_required",
        "select": "static_condition_sequence_branch_fold",
        "piecewise": "static_condition_sequence_callable_fold",
        "choose": "static_integer_selector_gather",
        "compress": "static_boolean_mask_gather",
        "extract": "static_boolean_mask_flat_gather",
        "argmax": "integer_index_selection_nondifferentiable",
        "argmin": "integer_index_selection_nondifferentiable",
        "argsort": "integer_index_permutation_nondifferentiable",
    }
    return {
        "program_ad": (
            "unsupported_index_selection_fail_closed"
            if name in {"argmax", "argmin", "argsort"}
            else "operator_intercepted_trace"
        ),
        "mlir": "available: scpn_diff selection dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.selection.{name}",
        "llvm": "blocked_until_executable_selection_lowering",
        "rust": "blocked_until_polyglot_selection_ad",
        "static_argument_rule": (
            "required"
            if name
            in {
                "where",
                "select",
                "piecewise",
                "choose",
                "compress",
                "extract",
                "argmax",
                "argmin",
                "argsort",
            }
            else "none"
        ),
        "static_derivative_factory": static_factories[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_selection_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_SELECTION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_selection_derivative_rule(name),
                batching_rule=_program_ad_selection_batching_rule_for(name),
                lowering_metadata=_program_ad_selection_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SELECTION_SHAPE_RULES[name],
                dtype_rule=_PROGRAM_AD_SELECTION_DTYPE_RULES[name],
                static_argument_rule=_PROGRAM_AD_SELECTION_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SELECTION_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_selection_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered Program AD selection primitive contract."""
    identity = _PROGRAM_AD_SELECTION_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD selection primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_SELECTION_POLICY:
        raise ValueError(f"invalid program AD selection primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD selection primitive effect for {identity.key}")
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
            "incomplete program AD selection primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_selection_contract_dispatch(contract, args)
    return contract
