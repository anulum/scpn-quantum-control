# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD assembly registry primitives
"""Program AD assembly derivative factories and registry contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_normalise_static_shape,
    _program_ad_array_static_size,
    _program_ad_array_vector,
    _program_ad_float64_vector_result,
)
from .program_ad_broadcast_assembly import (
    _normalise_program_ad_broadcast_shape,
    _program_ad_assembly_broadcast_to_shapes,
    program_ad_assembly_broadcast_arrays_derivative_rule,
    program_ad_assembly_broadcast_to_derivative_rule,
)
from .program_ad_registry import (
    _PROGRAM_AD_ASSEMBLY_IDENTITIES,
    _PROGRAM_AD_ASSEMBLY_POLICY,
    _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveBatchingRule,
    PrimitiveContract,
    PrimitiveDTypeRule,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .program_ad_stack_block_assembly import (
    _program_ad_assembly_append_output_shape,
    _program_ad_assembly_block_output_shape,
    _program_ad_assembly_concatenate_axis,
    _program_ad_assembly_concatenate_output_shape,
    _program_ad_assembly_stack_axis,
    _program_ad_assembly_stack_convenience_selected_indices,
    _program_ad_assembly_stack_output_shape,
    program_ad_assembly_append_derivative_rule,
    program_ad_assembly_block_derivative_rule,
    program_ad_assembly_column_stack_derivative_rule,
    program_ad_assembly_concatenate_derivative_rule,
    program_ad_assembly_dstack_derivative_rule,
    program_ad_assembly_hstack_derivative_rule,
    program_ad_assembly_stack_derivative_rule,
    program_ad_assembly_vstack_derivative_rule,
)


def _is_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""

    return type(value).__name__ == "TraceADArray" and hasattr(value, "context")


def _is_trace_scalar(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace scalar."""

    return type(value).__name__ == "TraceADScalar" and hasattr(value, "context")


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static shape for a trace value or array-like input."""

    if _is_trace_array(value):
        shape = getattr(value, "shape", None)
        if not isinstance(shape, tuple):
            raise ValueError("program AD assembly trace array shape must be static")
        return tuple(int(dimension) for dimension in shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name for a trace value or array-like input."""

    if _is_trace_array(value):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD assembly primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _validate_program_ad_assembly_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate assembly primitive dispatch helpers against concrete arguments."""

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


def _program_ad_assembly_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD assembly primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_assembly_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD assembly primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_assembly_split_source_shape(source_shape: Sequence[int]) -> tuple[int, ...]:
    shape = _program_ad_array_normalise_static_shape("assembly split source", source_shape)
    if not shape:
        raise ValueError("program AD assembly split direct rule requires ranked source arrays")
    return shape


def program_ad_assembly_split_derivative_rule(
    source_shape: Sequence[int],
    indices_or_sections: object,
    *,
    axis: object = 0,
    split_name: str = "split",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static split-family layouts."""

    if split_name not in _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES:
        raise ValueError(f"unsupported program AD assembly split primitive {split_name}")
    shape = _program_ad_assembly_split_source_shape(source_shape)
    axis_index = _program_ad_assembly_split_axis(axis, rank=len(shape))
    sections = _program_ad_assembly_split_sections(indices_or_sections)
    selected_indices = _program_ad_assembly_split_selected_indices(
        split_name,
        shape,
        sections,
        axis=axis_index,
    )
    source_size = _program_ad_array_static_size(shape)
    part_count = len(selected_indices)

    def split_array(source: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        if split_name == "array_split":
            parts = np.array_split(source, cast(Any, sections), axis=axis_index)
        else:
            parts = np.split(source, cast(Any, sections), axis=axis_index)
        return tuple(np.asarray(part, dtype=np.float64) for part in parts)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = _program_ad_array_vector("split", "values", values, expected_size=source_size)
        parts = split_array(source.reshape(shape))
        return _program_ad_float64_vector_result(
            np.concatenate([part.reshape(-1) for part in parts])
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("split", "values", values, expected_size=source_size)
        tangent_source = _program_ad_array_vector(
            "split", "tangent", tangent, expected_size=source_size
        )
        tangent_parts = split_array(tangent_source.reshape(shape))
        return _program_ad_float64_vector_result(
            np.concatenate([part.reshape(-1) for part in tangent_parts])
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("split", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "split", "cotangent", cotangent, expected_size=source_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        offset = 0
        for index_part in selected_indices:
            part_size = int(index_part.size)
            np.add.at(
                adjoint,
                index_part.reshape(-1),
                cotangent_vector[offset : offset + part_size],
            )
            offset += part_size
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=f"program_ad_assembly_{split_name}_axis{axis_index}_{part_count}_parts_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_triangular_source_shape(
    source_shape: Sequence[int],
) -> tuple[int, ...]:
    shape = _program_ad_array_normalise_static_shape(
        "assembly triangular mask source", source_shape
    )
    if len(shape) < 2:
        raise ValueError("program AD assembly triangular mask direct rule requires rank >= 2")
    return shape


def _program_ad_assembly_triangular_k(name: str, k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"program AD assembly {name} direct rule requires static integer k")
    return int(k)


def _program_ad_assembly_triangular_derivative_rule(
    name: str,
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    shape = _program_ad_assembly_triangular_source_shape(source_shape)
    k_value = _program_ad_assembly_triangular_k(name, k)
    source_size = _program_ad_array_static_size(shape)
    numpy_fn = np.tril if name == "tril" else np.triu

    def masked_array(values: NDArray[np.float64], role: str) -> NDArray[np.float64]:
        vector = _program_ad_array_vector(name, role, values, expected_size=source_size)
        return cast(NDArray[np.float64], numpy_fn(vector.reshape(shape), k=k_value))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(masked_array(values, "values"))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector(name, "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(masked_array(tangent, "tangent"))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector(name, "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(masked_array(cotangent, "cotangent"))

    return CustomDerivativeRule(
        name=f"program_ad_assembly_{name}_k{k_value}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_assembly_tril_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.tril`` masks."""

    return _program_ad_assembly_triangular_derivative_rule("tril", source_shape, k=k)


def program_ad_assembly_triu_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.triu`` masks."""

    return _program_ad_assembly_triangular_derivative_rule("triu", source_shape, k=k)


def program_ad_assembly_diagonal_derivative_rule(
    source_shape: Sequence[int],
    *,
    offset: object = 0,
    axis1: object = 0,
    axis2: object = 1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.diagonal`` gathers."""

    source = np.empty(
        _program_ad_array_normalise_static_shape("assembly diagonal source", source_shape),
        dtype=np.float64,
    )
    shape, offset_value, axis1_value, axis2_value, output_shape = (
        _program_ad_assembly_diagonal_static_parts((source, offset, axis1, axis2))
    )
    source_size = _program_ad_array_static_size(shape)
    output_size = _program_ad_array_static_size(output_shape)
    source_indices = np.arange(source_size, dtype=np.int64).reshape(shape)
    selected_indices = np.asarray(
        np.diagonal(source_indices, offset=offset_value, axis1=axis1_value, axis2=axis2_value),
        dtype=np.int64,
    ).reshape(-1)

    def gather_array(values: NDArray[np.float64], role: str) -> NDArray[np.float64]:
        vector = _program_ad_array_vector("diagonal", role, values, expected_size=source_size)
        return np.asarray(
            np.diagonal(
                vector.reshape(shape),
                offset=offset_value,
                axis1=axis1_value,
                axis2=axis2_value,
            ),
            dtype=np.float64,
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(gather_array(values, "values"))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("diagonal", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(gather_array(tangent, "tangent"))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("diagonal", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "diagonal", "cotangent", cotangent, expected_size=output_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, selected_indices, cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_assembly_diagonal_"
            f"offset{offset_value}_axis{axis1_value}_{axis2_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in _PROGRAM_AD_ASSEMBLY_IDENTITIES:
        return CustomDerivativeRule(
            name=f"program_ad_assembly_{name}_trace_contract",
            value_fn=_program_ad_assembly_direct_value,
            jvp_rule=_program_ad_assembly_direct_jvp,
        )
    raise ValueError(f"unsupported program AD assembly primitive {name}")


def _program_ad_assembly_like_reference_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if not args:
        raise ValueError("program AD like-constructor requires a reference operand")
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD like-constructor requires at least one element")
    return source_shape


def _program_ad_assembly_zeros_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD zeros_like requires one reference operand")
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_ones_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD ones_like requires one reference operand")
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_full_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD full_like requires reference and scalar fill operands")
    _program_ad_assembly_static_scalar_fill(args[1])
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_like_dtype_rule(_args: tuple[object, ...]) -> str:
    return "float64"


def _program_ad_assembly_static_scalar_fill(value: object) -> str:
    if _is_trace_scalar(value):
        return "trace_scalar"
    if _is_trace_array(value):
        raise ValueError("program AD full_like fill value must be scalar")
    array = np.asarray(value)
    if array.shape not in {(), (1,)}:
        raise ValueError("program AD full_like fill value must be scalar")
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD full_like fill value must be real numeric")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError("program AD full_like fill value must be finite")
    return "static_scalar"


def _program_ad_assembly_zeros_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_assembly_zeros_like_shape(args),)


def _program_ad_assembly_ones_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_assembly_ones_like_shape(args),)


def _program_ad_assembly_full_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        _program_ad_assembly_full_like_shape(args),
        _program_ad_assembly_static_scalar_fill(args[1]),
    )


def _program_ad_assembly_concatenate_operands(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], object]:
    if len(args) != 2:
        raise ValueError("program AD assembly concatenate requires operands and axis")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError("program AD assembly concatenate requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError("program AD assembly concatenate requires operands")
    return operand_tuple, args[1]


def _program_ad_assembly_concatenate_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], int | None]:
    operands, axis = _program_ad_assembly_concatenate_operands(args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    if axis is None:
        _program_ad_assembly_concatenate_output_shape(operand_shapes, None)
        return operand_shapes, None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly concatenate requires a static integer axis or None")
    normalised_axis = _program_ad_assembly_concatenate_axis(axis, rank=len(operand_shapes[0]))
    _program_ad_assembly_concatenate_output_shape(operand_shapes, normalised_axis)
    return operand_shapes, normalised_axis


def _program_ad_assembly_concatenate_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, axis = _program_ad_assembly_concatenate_static_parts(args)
    return _program_ad_assembly_concatenate_output_shape(operand_shapes, axis)


def _program_ad_assembly_concatenate_dtype_rule(args: tuple[object, ...]) -> str:
    operands, _axis = _program_ad_assembly_concatenate_operands(args)
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_concatenate_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    operand_shapes, axis = _program_ad_assembly_concatenate_static_parts(args)
    return operand_shapes, axis


def _program_ad_assembly_stack_operands(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], object]:
    if len(args) != 2:
        raise ValueError("program AD assembly stack requires operands and axis")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError("program AD assembly stack requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError("program AD assembly stack requires operands")
    return operand_tuple, args[1]


def _program_ad_assembly_stack_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    operands, axis = _program_ad_assembly_stack_operands(args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    normalised_axis = _program_ad_assembly_stack_axis(axis, rank=len(operand_shapes[0]))
    _program_ad_assembly_stack_output_shape(operand_shapes, normalised_axis)
    return operand_shapes, normalised_axis


def _program_ad_assembly_stack_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, axis = _program_ad_assembly_stack_static_parts(args)
    return _program_ad_assembly_stack_output_shape(operand_shapes, axis)


def _program_ad_assembly_stack_dtype_rule(args: tuple[object, ...]) -> str:
    operands, _axis = _program_ad_assembly_stack_operands(args)
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_stack_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    operand_shapes, axis = _program_ad_assembly_stack_static_parts(args)
    return operand_shapes, axis


def _program_ad_assembly_append_operands(
    args: tuple[object, ...],
) -> tuple[object, object, object]:
    if len(args) != 3:
        raise ValueError("program AD assembly append requires source, values, and axis")
    return args[0], args[1], args[2]


def _program_ad_assembly_append_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    source, values, axis = _program_ad_assembly_append_operands(args)
    source_shape = _program_ad_array_shape_of(source)
    values_shape = _program_ad_array_shape_of(values)
    if axis is None:
        _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=None)
        return source_shape, values_shape, None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly append requires a static integer axis or None")
    normalised_axis = _program_ad_assembly_concatenate_axis(axis, rank=len(source_shape))
    _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=normalised_axis)
    return source_shape, values_shape, normalised_axis


def _program_ad_assembly_append_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, values_shape, axis = _program_ad_assembly_append_static_parts(args)
    return _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=axis)


def _program_ad_assembly_append_dtype_rule(args: tuple[object, ...]) -> str:
    source, values, _axis = _program_ad_assembly_append_operands(args)
    return str(
        np.result_type(
            np.dtype(_program_ad_array_dtype_of(source)),
            np.dtype(_program_ad_array_dtype_of(values)),
        )
    )


def _program_ad_assembly_append_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, values_shape, axis = _program_ad_assembly_append_static_parts(args)
    return source_shape, values_shape, axis


def _program_ad_assembly_block_layout(args: tuple[object, ...]) -> object:
    if len(args) != 1:
        raise ValueError("program AD assembly block requires one nested layout argument")
    layout = args[0]
    if not isinstance(layout, (tuple, list)):
        raise ValueError("program AD assembly block requires a nested layout")
    if not layout:
        raise ValueError("program AD assembly block requires a non-empty nested layout")
    return layout


def _program_ad_assembly_block_shape_layout_from_operands(layout: object) -> tuple[object, ...]:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        return tuple(
            _program_ad_assembly_block_shape_layout_from_operands(item) for item in layout
        )
    return _program_ad_array_shape_of(layout)


def _program_ad_assembly_block_dtype_leaves(layout: object) -> tuple[np.dtype[Any], ...]:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        dtypes: list[np.dtype[Any]] = []
        for item in layout:
            dtypes.extend(_program_ad_assembly_block_dtype_leaves(item))
        return tuple(dtypes)
    return (np.dtype(_program_ad_array_dtype_of(layout)),)


def _program_ad_assembly_block_numpy_layout(layout: object) -> object:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        return [_program_ad_assembly_block_numpy_layout(item) for item in layout]
    return layout


def _program_ad_assembly_block_static_parts(args: tuple[object, ...]) -> tuple[object, ...]:
    layout = _program_ad_assembly_block_layout(args)
    layout_shapes = _program_ad_assembly_block_shape_layout_from_operands(layout)
    _program_ad_assembly_block_output_shape(layout_shapes)
    return layout_shapes


def _program_ad_assembly_block_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    return _program_ad_assembly_block_output_shape(_program_ad_assembly_block_static_parts(args))


def _program_ad_assembly_block_dtype_rule(args: tuple[object, ...]) -> str:
    layout = _program_ad_assembly_block_layout(args)
    return str(np.result_type(*_program_ad_assembly_block_dtype_leaves(layout)))


def _program_ad_assembly_block_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_block_static_parts(args)


def _program_ad_assembly_stack_convenience_operands(
    name: str,
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError(f"program AD assembly {name} requires one operand sequence")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError(f"program AD assembly {name} requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError(f"program AD assembly {name} requires operands")
    return operand_tuple


def _program_ad_assembly_stack_convenience_static_parts(
    name: str,
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    operands = _program_ad_assembly_stack_convenience_operands(name, args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    _shapes, selected = _program_ad_assembly_stack_convenience_selected_indices(
        name, operand_shapes
    )
    return operand_shapes, tuple(int(dimension) for dimension in selected.shape)


def _program_ad_assembly_stack_convenience_shape_rule_for(name: str) -> PrimitiveShapeRule:
    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        _operand_shapes, output_shape = _program_ad_assembly_stack_convenience_static_parts(
            name, args
        )
        return output_shape

    return shape_rule


def _program_ad_assembly_stack_convenience_dtype_rule_for(name: str) -> PrimitiveDTypeRule:
    def dtype_rule(args: tuple[object, ...]) -> str:
        operands = _program_ad_assembly_stack_convenience_operands(name, args)
        operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
        return str(np.result_type(*operand_dtypes))

    return dtype_rule


def _program_ad_assembly_stack_convenience_static_arguments_rule_for(
    name: str,
) -> PrimitiveStaticArgumentRule:
    def static_argument_rule(args: tuple[object, ...]) -> tuple[object, ...]:
        operand_shapes, output_shape = _program_ad_assembly_stack_convenience_static_parts(
            name, args
        )
        return operand_shapes, output_shape

    return static_argument_rule


def _program_ad_assembly_broadcast_to_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(args) != 2:
        raise ValueError("program AD assembly broadcast_to requires source and output shape")
    return _program_ad_assembly_broadcast_to_shapes(_program_ad_array_shape_of(args[0]), args[1])


def _program_ad_assembly_broadcast_to_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    _source_shape, output_shape = _program_ad_assembly_broadcast_to_static_parts(args)
    return output_shape


def _program_ad_assembly_broadcast_to_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 2:
        raise ValueError("program AD assembly broadcast_to requires source and output shape")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_broadcast_to_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_broadcast_to_static_parts(args)


def _program_ad_assembly_broadcast_arrays_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays requires at least one operand")
    operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args)
    try:
        output_shape = tuple(int(dimension) for dimension in np.broadcast_shapes(*operand_shapes))
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_arrays requires operands compatible "
            "with broadcasting rules"
        ) from exc
    return operand_shapes, output_shape


def _program_ad_assembly_broadcast_arrays_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, output_shape = _program_ad_assembly_broadcast_arrays_static_parts(args)
    return (len(operand_shapes) * _program_ad_array_static_size(output_shape),)


def _program_ad_assembly_broadcast_arrays_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays requires at least one operand")
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(arg)) for arg in args]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_broadcast_arrays_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_broadcast_arrays_static_parts(args)


@dataclass(frozen=True)
class _ProgramADAssemblyBlockMappedLeaf:
    array: NDArray[np.float64]
    axis: int


def _program_ad_assembly_split_sections(indices_or_sections: object) -> int | tuple[int, ...]:
    if isinstance(indices_or_sections, bool):
        raise ValueError("program AD assembly split requires static integer sections")
    if isinstance(indices_or_sections, (int, np.integer)):
        sections = int(indices_or_sections)
        if sections <= 0:
            raise ValueError("program AD assembly split requires positive static sections")
        return sections
    array = np.asarray(indices_or_sections)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("program AD assembly split requires static integer split indices")
    return tuple(int(index) for index in array.tolist())


def _program_ad_assembly_split_axis(axis: object, *, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly split requires a static integer axis")
    if rank <= 0:
        raise ValueError("program AD assembly split requires ranked source arrays")
    return _normalise_axis("axis", int(axis), rank)


def _program_ad_assembly_split_selected_indices(
    split_name: str,
    source_shape: Sequence[int],
    indices_or_sections: object,
    *,
    axis: object,
) -> tuple[NDArray[np.int64], ...]:
    if split_name not in _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES:
        raise ValueError(f"unsupported program AD assembly split primitive {split_name}")
    shape = _program_ad_array_normalise_static_shape("assembly split source", source_shape)
    axis_index = _program_ad_assembly_split_axis(axis, rank=len(shape))
    sections = _program_ad_assembly_split_sections(indices_or_sections)
    index_array = np.arange(_program_ad_array_static_size(shape), dtype=np.int64).reshape(shape)
    try:
        if split_name == "array_split":
            selected = np.array_split(index_array, cast(Any, sections), axis=axis_index)
        else:
            selected = np.split(index_array, cast(Any, sections), axis=axis_index)
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD assembly split requires static split sections compatible with source shape"
        ) from exc
    return tuple(np.asarray(part, dtype=np.int64) for part in selected)


def _program_ad_assembly_split_static_parts(
    split_name: str,
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int | tuple[int, ...], int, tuple[tuple[int, ...], ...]]:
    if len(args) != 3:
        raise ValueError("program AD assembly split requires source, sections, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = _program_ad_assembly_split_axis(args[2], rank=len(source_shape))
    sections = _program_ad_assembly_split_sections(args[1])
    selected = _program_ad_assembly_split_selected_indices(
        split_name,
        source_shape,
        sections,
        axis=axis,
    )
    part_shapes = tuple(tuple(int(dimension) for dimension in part.shape) for part in selected)
    return source_shape, sections, axis, part_shapes


def _program_ad_assembly_split_shape_rule_for(split_name: str) -> PrimitiveShapeRule:
    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        source_shape, _sections, _axis, _part_shapes = _program_ad_assembly_split_static_parts(
            split_name, args
        )
        return (_program_ad_array_static_size(source_shape),)

    return shape_rule


def _program_ad_assembly_split_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD assembly split requires source, sections, and axis")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_split_static_arguments_rule_for(
    split_name: str,
) -> PrimitiveStaticArgumentRule:
    def static_argument_rule(args: tuple[object, ...]) -> tuple[object, ...]:
        return _program_ad_assembly_split_static_parts(split_name, args)

    return static_argument_rule


def _program_ad_assembly_triangular_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int]:
    if len(args) != 2:
        raise ValueError("program AD assembly triangular mask requires source and k")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD assembly triangular mask requires rank >= 2")
    k = args[1]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("program AD assembly triangular mask requires static integer k")
    return source_shape, int(k)


def _program_ad_assembly_triangular_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, _k = _program_ad_assembly_triangular_static_parts(args)
    return source_shape


def _program_ad_assembly_triangular_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 2:
        raise ValueError("program AD assembly triangular mask requires source and k")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_triangular_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_triangular_static_parts(args)


def _program_ad_assembly_diagonal_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int, int, int, tuple[int, ...]]:
    if len(args) != 4:
        raise ValueError("program AD assembly diagonal requires source, offset, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD assembly diagonal requires rank >= 2")
    offset, axis1, axis2 = args[1], args[2], args[3]
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer offset")
    if isinstance(axis1, bool) or not isinstance(axis1, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer axes")
    if isinstance(axis2, bool) or not isinstance(axis2, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer axes")
    axis1_value = _normalise_axis("axis1", int(axis1), len(source_shape))
    axis2_value = _normalise_axis("axis2", int(axis2), len(source_shape))
    if axis1_value == axis2_value:
        raise ValueError("program AD assembly diagonal requires distinct axes")
    try:
        output = np.diagonal(
            np.empty(source_shape, dtype=np.float64),
            offset=int(offset),
            axis1=axis1_value,
            axis2=axis2_value,
        )
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD assembly diagonal requires static offset and axes "
            "compatible with source shape"
        ) from exc
    return (
        source_shape,
        int(offset),
        axis1_value,
        axis2_value,
        tuple(int(dimension) for dimension in output.shape),
    )


def _program_ad_assembly_diagonal_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    _source_shape, _offset, _axis1, _axis2, output_shape = (
        _program_ad_assembly_diagonal_static_parts(args)
    )
    return output_shape


def _program_ad_assembly_diagonal_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 4:
        raise ValueError("program AD assembly diagonal requires source, offset, axis1, and axis2")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_diagonal_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_diagonal_static_parts(args)


def _program_ad_assembly_split_move_output_batch_axis(output: object, out_axes: int) -> object:
    if isinstance(output, tuple):
        return tuple(
            _program_ad_assembly_split_move_output_batch_axis(item, out_axes) for item in output
        )
    if isinstance(output, list):
        return [
            _program_ad_assembly_split_move_output_batch_axis(item, out_axes) for item in output
        ]
    array = _as_real_numeric_array("program AD assembly split batched output", output)
    return np.moveaxis(array, 0, _normalise_axis("out_axes", out_axes, array.ndim))


def _program_ad_assembly_split_stack_outputs(outputs: Sequence[object], out_axes: int) -> object:
    if not outputs:
        raise ValueError("program AD assembly split batching requires non-empty outputs")
    first = outputs[0]
    if isinstance(first, tuple):
        if any(not isinstance(output, tuple) or len(output) != len(first) for output in outputs):
            raise ValueError(
                "program AD assembly split batching requires stable output partitions"
            )
        return tuple(
            _program_ad_assembly_split_stack_outputs(
                [cast(tuple[object, ...], output)[index] for output in outputs],
                out_axes,
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(output, list) or len(output) != len(first) for output in outputs):
            raise ValueError(
                "program AD assembly split batching requires stable output partitions"
            )
        return [
            _program_ad_assembly_split_stack_outputs(
                [cast(list[object], output)[index] for output in outputs],
                out_axes,
            )
            for index in range(len(first))
        ]
    arrays = [
        _as_real_numeric_array("program AD assembly split batched output", output)
        for output in outputs
    ]
    stacked = np.stack(arrays, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_split_batching_rule_for(split_name: str) -> PrimitiveBatchingRule:
    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 3 or len(axes) != 3:
            raise ValueError(
                "program AD assembly split batching requires source, sections, and axis"
            )
        if axes[1] is not None or axes[2] is not None:
            raise ValueError("program AD assembly split batching keeps split metadata static")
        source = _as_real_numeric_array("program AD assembly split batched source", args[0])
        sections = _program_ad_assembly_split_sections(args[1])
        split_axis = _program_ad_assembly_split_axis(args[2], rank=source.ndim)
        source_axis = axes[0]
        if source_axis is None:
            return function(source, sections, split_axis)
        source_axis_index = _normalise_axis("source axis", source_axis, source.ndim)
        if source_axis_index == split_axis:
            raise ValueError("program AD assembly split batching cannot map the split axis")
        adjusted_split_axis = split_axis - 1 if source_axis_index < split_axis else split_axis
        outputs = [
            function(
                np.take(source, batch_index, axis=source_axis_index),
                sections,
                adjusted_split_axis,
            )
            for batch_index in range(int(source.shape[source_axis_index]))
        ]
        return _program_ad_assembly_split_stack_outputs(outputs, out_axes)

    return batching_rule


def _program_ad_assembly_broadcast_to_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError(
            "program AD assembly broadcast_to batching requires source and output shape"
        )
    source, output_shape_arg = args
    source_axis, output_shape_axis = axes
    if output_shape_axis is not None:
        raise ValueError("program AD assembly broadcast_to batching keeps output shape static")
    output_shape = _normalise_program_ad_broadcast_shape(output_shape_arg)
    if source_axis is None:
        return function(source, output_shape)
    source_array = _as_real_numeric_array(
        "program AD assembly broadcast_to batched source", source
    )
    axis_index = _normalise_axis("source axis", source_axis, source_array.ndim)
    moved = np.moveaxis(source_array, axis_index, 0)
    outputs = [
        _as_real_numeric_array(
            "program AD assembly broadcast_to batched output",
            function(moved[index], output_shape),
        )
        for index in range(moved.shape[0])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_broadcast_arrays_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays batching requires operands")
    if len(args) != len(axes):
        raise ValueError(
            "program AD assembly broadcast_arrays batching requires one axis per operand"
        )
    moved_args: list[object] = []
    batch_size: int | None = None
    for operand_index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            moved_args.append(arg)
            continue
        operand = _as_real_numeric_array(
            f"program AD assembly broadcast_arrays operand {operand_index}", arg
        )
        axis_index = _normalise_axis("operand axis", axis, operand.ndim)
        moved = np.moveaxis(operand, axis_index, 0)
        if batch_size is None:
            batch_size = int(moved.shape[0])
        elif int(moved.shape[0]) != batch_size:
            raise ValueError(
                "program AD assembly broadcast_arrays batching requires same batch size"
            )
        moved_args.append(moved)
    if batch_size is None:
        return function(*args)
    outputs: list[tuple[NDArray[np.float64], ...]] = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            cast(NDArray[np.float64], arg)[batch_index] if axis is not None else arg
            for arg, axis in zip(moved_args, axes, strict=True)
        )
        result = function(*sliced_args)
        if not isinstance(result, (tuple, list)):
            raise ValueError(
                "program AD assembly broadcast_arrays batching requires tuple/list outputs"
            )
        outputs.append(
            tuple(
                _as_real_numeric_array("program AD assembly broadcast_arrays batched output", item)
                for item in result
            )
        )
    return _program_ad_assembly_split_stack_outputs(outputs, out_axes)


def _program_ad_assembly_triangular_batching_rule_for(name: str) -> PrimitiveBatchingRule:
    if name not in {"tril", "triu"}:
        raise ValueError(f"unsupported program AD assembly triangular primitive {name}")

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 2 or len(axes) != 2:
            raise ValueError(f"program AD assembly {name} batching requires source and k")
        source, k = args
        source_axis, k_axis = axes
        if k_axis is not None:
            raise ValueError(f"program AD assembly {name} batching keeps k static")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"program AD assembly {name} batching requires static integer k")
        if source_axis is None:
            return function(source, int(k))
        source_array = _as_real_numeric_array(f"program AD assembly {name} batched source", source)
        if source_array.ndim < 3:
            raise ValueError(
                f"program AD assembly {name} batching requires an outer batch axis "
                "separate from matrix axes"
            )
        axis_index = _normalise_axis("source axis", source_axis, source_array.ndim)
        if axis_index >= source_array.ndim - 2:
            raise ValueError(f"program AD assembly {name} batching cannot map matrix axes")
        moved = np.moveaxis(source_array, axis_index, 0)
        outputs = [
            _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(moved[index], int(k)),
            )
            for index in range(moved.shape[0])
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))

    return batching_rule


def _program_ad_assembly_diagonal_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 4 or len(axes) != 4:
        raise ValueError(
            "program AD assembly diagonal batching requires source, offset, axis1, and axis2"
        )
    source, offset, axis1, axis2 = args
    source_axis, offset_axis, axis1_axis, axis2_axis = axes
    if offset_axis is not None or axis1_axis is not None or axis2_axis is not None:
        raise ValueError("program AD assembly diagonal batching keeps offset and axes static")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer offset")
    if isinstance(axis1, bool) or not isinstance(axis1, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer axes")
    if isinstance(axis2, bool) or not isinstance(axis2, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer axes")
    if source_axis is None:
        return function(source, int(offset), int(axis1), int(axis2))
    source_array = _as_real_numeric_array("program AD assembly diagonal batched source", source)
    if source_array.ndim < 3:
        raise ValueError(
            "program AD assembly diagonal batching requires an outer batch axis "
            "separate from diagonal axes"
        )
    batch_axis = _normalise_axis("source axis", source_axis, source_array.ndim)
    axis1_value = _normalise_axis("axis1", int(axis1), source_array.ndim)
    axis2_value = _normalise_axis("axis2", int(axis2), source_array.ndim)
    if axis1_value == axis2_value:
        raise ValueError("program AD assembly diagonal batching requires distinct diagonal axes")
    if batch_axis in {axis1_value, axis2_value}:
        raise ValueError("program AD assembly diagonal batching cannot map diagonal axes")

    def adjusted_axis(axis: int) -> int:
        if batch_axis < axis:
            return axis - 1
        return axis

    moved = np.moveaxis(source_array, batch_axis, 0)
    outputs = [
        _as_real_numeric_array(
            "program AD assembly diagonal batched output",
            function(
                moved[index],
                int(offset),
                adjusted_axis(axis1_value),
                adjusted_axis(axis2_value),
            ),
        )
        for index in range(moved.shape[0])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD assembly concatenate batching requires operands and axis")
    if axes[1] is not None:
        raise ValueError("program AD assembly concatenate batching keeps axis static")
    operands, axis = _program_ad_assembly_concatenate_operands(args)
    operand_axes = cast(Any, axes[0])
    if operand_axes is None:
        return _as_real_numeric_array(
            "program AD assembly concatenate batched output",
            function(operands, axis),
        )
    if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
        raise ValueError(
            "program AD assembly concatenate batching requires one operand axis per operand"
        )

    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None
    if axis is None:
        adjusted_axis = None
    elif isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly concatenate batching keeps axis static")
    else:
        adjusted_axis = None

    for operand_index, (operand, operand_axis) in enumerate(
        zip(operands, operand_axes, strict=True)
    ):
        if operand_axis is None:
            if axis is not None:
                raise ValueError(
                    "program AD assembly concatenate batching maps every operand for ranked axes"
                )
            mapped.append(None)
            continue
        array = _as_real_numeric_array(
            f"program AD assembly concatenate batched operand {operand_index}",
            operand,
        )
        batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly concatenate batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError(
                "program AD assembly concatenate batching axes must share one batch size"
            )
        if axis is not None:
            axis_index = _normalise_axis("axis", int(axis), array.ndim)
            if axis_index == batch_axis:
                raise ValueError(
                    "program AD assembly concatenate batching cannot map the concatenate axis"
                )
            operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
            if adjusted_axis is None:
                adjusted_axis = operand_adjusted_axis
            elif adjusted_axis != operand_adjusted_axis:
                raise ValueError(
                    "program AD assembly concatenate batching requires one adjusted axis"
                )
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly concatenate batched output",
            function(operands, axis),
        )

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_operands: list[object] = []
        for operand, mapped_operand in zip(operands, mapped, strict=True):
            if mapped_operand is None:
                sliced_operands.append(operand)
                continue
            array, batch_axis = mapped_operand
            sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly concatenate batched output",
                function(tuple(sliced_operands), adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_append_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD assembly append batching requires source, values, and axis")
    if axes[2] is not None:
        raise ValueError("program AD assembly append batching keeps axis static")
    source, values, axis = _program_ad_assembly_append_operands(args)
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, (int, np.integer))):
        raise ValueError("program AD assembly append batching keeps axis static")
    arrays = (
        _as_real_numeric_array("program AD assembly append batched source", source),
        _as_real_numeric_array("program AD assembly append batched values", values),
    )
    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None = None
    for operand_index, (array, operand_axis) in enumerate(zip(arrays, axes[:2], strict=True)):
        if operand_axis is None:
            if axis is not None:
                raise ValueError(
                    "program AD assembly append batching maps source and values for ranked axes"
                )
            mapped.append(None)
            continue
        batch_axis = _normalise_axis(f"axes[{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly append batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD assembly append batching axes must share one batch size")
        if axis is not None:
            axis_index = _normalise_axis("axis", int(axis), array.ndim)
            if axis_index == batch_axis:
                raise ValueError("program AD assembly append batching cannot map the append axis")
            operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
            if adjusted_axis is None:
                adjusted_axis = operand_adjusted_axis
            elif adjusted_axis != operand_adjusted_axis:
                raise ValueError("program AD assembly append batching requires one adjusted axis")
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly append batched output",
            function(source, values, axis),
        )

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_args: list[object] = []
        for original, mapped_operand in zip((source, values), mapped, strict=True):
            if mapped_operand is None:
                sliced_args.append(original)
                continue
            array, batch_axis = mapped_operand
            sliced_args.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly append batched output",
                function(sliced_args[0], sliced_args[1], adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_block_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[object, ...],
    out_axes: int,
) -> object:
    if len(args) != 1 or len(axes) != 1:
        raise ValueError("program AD assembly block batching requires one nested layout argument")
    layout = _program_ad_assembly_block_layout(args)
    layout_axes = axes[0]
    if layout_axes is None:
        return _as_real_numeric_array(
            "program AD assembly block batched output",
            function(_program_ad_assembly_block_numpy_layout(layout)),
        )

    batch_size: int | None = None

    def map_layout(
        node: object,
        axis_node: object,
        path: str,
    ) -> object:
        nonlocal batch_size
        if isinstance(node, (tuple, list)):
            if not isinstance(axis_node, (tuple, list)) or len(axis_node) != len(node):
                raise ValueError(
                    "program AD assembly block batching requires axes matching layout"
                )
            if not node:
                raise ValueError("program AD assembly block requires a non-empty nested layout")
            return tuple(
                map_layout(child, child_axis, f"{path}.{index}")
                for index, (child, child_axis) in enumerate(zip(node, axis_node, strict=True))
            )
        if axis_node is None:
            return node
        if isinstance(axis_node, bool) or not isinstance(axis_node, (int, np.integer)):
            raise ValueError("program AD assembly block batching requires axes matching layout")
        array = _as_real_numeric_array(
            "program AD assembly block batched leaf",
            node,
        )
        axis_index = _normalise_axis(f"{path} axis", int(axis_node), array.ndim)
        current_batch_size = int(array.shape[axis_index])
        if batch_size is None:
            batch_size = current_batch_size
        elif batch_size != current_batch_size:
            raise ValueError("program AD assembly block batching requires equal batch sizes")
        return _ProgramADAssemblyBlockMappedLeaf(array=array, axis=axis_index)

    mapped_layout = map_layout(layout, layout_axes, "layout")
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly block batched output",
            function(_program_ad_assembly_block_numpy_layout(layout)),
        )

    def slice_layout(node: object, index: int) -> object:
        if isinstance(node, _ProgramADAssemblyBlockMappedLeaf):
            return np.take(node.array, index, axis=node.axis)
        if isinstance(node, tuple):
            return [slice_layout(child, index) for child in node]
        return node

    outputs = [
        _as_real_numeric_array(
            "program AD assembly block batched output",
            function(slice_layout(mapped_layout, batch_index)),
        )
        for batch_index in range(batch_size)
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_stack_convenience_batching_rule_for(
    name: str,
) -> PrimitiveBatchingRule:
    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 1 or len(axes) != 1:
            raise ValueError(f"program AD assembly {name} batching requires operands")
        operands = _program_ad_assembly_stack_convenience_operands(name, args)
        operand_axes = cast(Any, axes[0])
        if operand_axes is None:
            return _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(operands),
            )
        if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
            raise ValueError(
                f"program AD assembly {name} batching requires one operand axis per operand"
            )

        mapped: list[tuple[NDArray[np.float64], int] | None] = []
        batch_size: int | None = None
        for operand_index, (operand, operand_axis) in enumerate(
            zip(operands, operand_axes, strict=True)
        ):
            if operand_axis is None:
                mapped.append(None)
                continue
            array = _as_real_numeric_array(
                f"program AD assembly {name} batched operand {operand_index}",
                operand,
            )
            batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
            current_batch_size = int(array.shape[batch_axis])
            if current_batch_size <= 0:
                raise ValueError(f"program AD assembly {name} batching axes must be non-empty")
            if batch_size is None:
                batch_size = current_batch_size
            elif batch_size != current_batch_size:
                raise ValueError(
                    f"program AD assembly {name} batching axes must share one batch size"
                )
            mapped.append((array, batch_axis))
        if batch_size is None:
            return _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(operands),
            )

        outputs: list[NDArray[np.float64]] = []
        for batch_index in range(batch_size):
            sliced_operands: list[object] = []
            for operand, mapped_operand in zip(operands, mapped, strict=True):
                if mapped_operand is None:
                    sliced_operands.append(operand)
                    continue
                array, batch_axis = mapped_operand
                sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
            outputs.append(
                _as_real_numeric_array(
                    f"program AD assembly {name} batched output",
                    function(tuple(sliced_operands)),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))

    return batching_rule


def _program_ad_assembly_stack_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD assembly stack batching requires operands and axis")
    if axes[1] is not None:
        raise ValueError("program AD assembly stack batching keeps axis static")
    operands, axis = _program_ad_assembly_stack_operands(args)
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly stack batching keeps axis static")
    operand_axes = cast(Any, axes[0])
    if operand_axes is None:
        return _as_real_numeric_array(
            "program AD assembly stack batched output",
            function(operands, axis),
        )
    if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
        raise ValueError(
            "program AD assembly stack batching requires one operand axis per operand"
        )

    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None = None
    for operand_index, (operand, operand_axis) in enumerate(
        zip(operands, operand_axes, strict=True)
    ):
        if operand_axis is None:
            mapped.append(None)
            continue
        array = _as_real_numeric_array(
            f"program AD assembly stack batched operand {operand_index}",
            operand,
        )
        batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly stack batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD assembly stack batching axes must share one batch size")
        axis_index = _normalise_axis("axis", int(axis), array.ndim + 1)
        if axis_index == batch_axis:
            raise ValueError("program AD assembly stack batching cannot map the stack axis")
        operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
        if adjusted_axis is None:
            adjusted_axis = operand_adjusted_axis
        elif adjusted_axis != operand_adjusted_axis:
            raise ValueError("program AD assembly stack batching requires one adjusted axis")
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly stack batched output",
            function(operands, axis),
        )
    if adjusted_axis is None:
        raise ValueError("program AD assembly stack batching requires a mapped operand axis")

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_operands: list[object] = []
        for operand, mapped_operand in zip(operands, mapped, strict=True):
            if mapped_operand is None:
                sliced_operands.append(operand)
                continue
            array, batch_axis = mapped_operand
            sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly stack batched output",
                function(tuple(sliced_operands), adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_like_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD like-constructor batching axes must match arguments")
    if not args:
        raise ValueError("program AD like-constructor batching requires a reference operand")
    if axes[0] is None:
        return _as_real_numeric_array(
            "program AD like-constructor batched output", function(*args)
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD like-constructor batching keeps fill values static")
    reference = _as_real_numeric_array("program AD like-constructor batched reference", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], reference.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD like-constructor batched output",
            function(np.take(reference, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(int(reference.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_lowering_metadata(name: str) -> Mapping[str, str]:
    if name not in _PROGRAM_AD_ASSEMBLY_IDENTITIES:
        raise ValueError(f"unsupported program AD assembly primitive {name}")
    factory_names = {
        "append": "program_ad_assembly_append_derivative_rule",
        "block": "program_ad_assembly_block_derivative_rule",
        "broadcast_arrays": "program_ad_assembly_broadcast_arrays_derivative_rule",
        "broadcast_to": "program_ad_assembly_broadcast_to_derivative_rule",
        "concatenate": "program_ad_assembly_concatenate_derivative_rule",
        "diagonal": "program_ad_assembly_diagonal_derivative_rule",
        "dstack": "program_ad_assembly_dstack_derivative_rule",
        "full_like": "program_ad_assembly_full_like_derivative_rule",
        "hstack": "program_ad_assembly_hstack_derivative_rule",
        "column_stack": "program_ad_assembly_column_stack_derivative_rule",
        "vstack": "program_ad_assembly_vstack_derivative_rule",
        "ones_like": "program_ad_assembly_ones_like_derivative_rule",
        "split": "program_ad_assembly_split_derivative_rule",
        "array_split": "program_ad_assembly_split_derivative_rule",
        "hsplit": "program_ad_assembly_split_derivative_rule",
        "vsplit": "program_ad_assembly_split_derivative_rule",
        "dsplit": "program_ad_assembly_split_derivative_rule",
        "stack": "program_ad_assembly_stack_derivative_rule",
        "tril": "program_ad_assembly_tril_derivative_rule",
        "triu": "program_ad_assembly_triu_derivative_rule",
        "zeros_like": "program_ad_assembly_zeros_like_derivative_rule",
    }
    boundaries = {
        "append": "static_source_values_shape_axis_append",
        "block": "static_nested_block_shape_layout",
        "broadcast_arrays": "static_operand_shape_broadcast_arrays",
        "broadcast_to": "static_source_shape_broadcast_to",
        "concatenate": "static_operand_shape_axis_concatenate",
        "diagonal": "static_diagonal_offset_axis_gather_scatter",
        "dstack": "static_operand_shape_dstack",
        "full_like": "static_reference_shape_scalar_fill",
        "hstack": "static_operand_shape_hstack",
        "column_stack": "static_operand_shape_column_stack",
        "vstack": "static_operand_shape_vstack",
        "ones_like": "static_reference_shape_unit_fill",
        "split": "static_split_sections_gather_scatter",
        "array_split": "static_array_split_sections_gather_scatter",
        "hsplit": "static_hsplit_sections_gather_scatter",
        "vsplit": "static_vsplit_sections_gather_scatter",
        "dsplit": "static_dsplit_sections_gather_scatter",
        "stack": "static_operand_shape_axis_stack",
        "tril": "static_lower_triangular_mask",
        "triu": "static_upper_triangular_mask",
        "zeros_like": "static_reference_shape_zero_fill",
    }
    static_signatures = {
        "append": "source_shape:ranked_tensor_shape;values_shape:ranked_tensor_shape;axis",
        "block": "layout_shapes:nested_ranked_tensor_shapes",
        "broadcast_arrays": "operand_shapes:ranked_tensor_shapes;output_shape",
        "broadcast_to": "source_shape:ranked_tensor_shape;output_shape",
        "concatenate": "operand_shapes:ranked_tensor_shapes;axis",
        "diagonal": "source_shape:rank_ge_2;offset_axis_pair;output_shape",
        "full_like": "source_shape:ranked_tensor_shape;scalar_fill",
        "hstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "vstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "column_stack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "dstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "ones_like": "source_shape:ranked_tensor_shape",
        "split": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "array_split": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "hsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "vsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "dsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "stack": "operand_shapes:ranked_tensor_shapes;axis",
        "tril": "source_shape:rank_ge_2;k",
        "triu": "source_shape:rank_ge_2;k",
        "zeros_like": "source_shape:ranked_tensor_shape",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff assembly dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.assembly.{name}",
        "llvm": "blocked_until_executable_assembly_lowering",
        "rust": "blocked_until_polyglot_assembly_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": factory_names[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_assembly_primitive_contracts() -> None:
    """Register fail-closed Program AD assembly primitive contracts."""

    batching_rules: Mapping[str, PrimitiveBatchingRule] = {
        "append": _program_ad_assembly_append_batching_rule,
        "block": _program_ad_assembly_block_batching_rule,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_batching_rule,
        "broadcast_to": _program_ad_assembly_broadcast_to_batching_rule,
        "concatenate": _program_ad_assembly_batching_rule,
        "diagonal": _program_ad_assembly_diagonal_batching_rule,
        "dstack": _program_ad_assembly_stack_convenience_batching_rule_for("dstack"),
        "full_like": _program_ad_assembly_like_batching_rule,
        "hstack": _program_ad_assembly_stack_convenience_batching_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_batching_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_like_batching_rule,
        "split": _program_ad_assembly_split_batching_rule_for("split"),
        "array_split": _program_ad_assembly_split_batching_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_batching_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_batching_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_batching_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_batching_rule,
        "tril": _program_ad_assembly_triangular_batching_rule_for("tril"),
        "triu": _program_ad_assembly_triangular_batching_rule_for("triu"),
        "vstack": _program_ad_assembly_stack_convenience_batching_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_like_batching_rule,
    }
    shape_rules: Mapping[str, PrimitiveShapeRule] = {
        "append": _program_ad_assembly_append_shape,
        "block": _program_ad_assembly_block_shape,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_shape,
        "broadcast_to": _program_ad_assembly_broadcast_to_shape,
        "concatenate": _program_ad_assembly_concatenate_shape,
        "diagonal": _program_ad_assembly_diagonal_shape,
        "dstack": _program_ad_assembly_stack_convenience_shape_rule_for("dstack"),
        "full_like": _program_ad_assembly_full_like_shape,
        "hstack": _program_ad_assembly_stack_convenience_shape_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_shape_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_ones_like_shape,
        "split": _program_ad_assembly_split_shape_rule_for("split"),
        "array_split": _program_ad_assembly_split_shape_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_shape_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_shape_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_shape_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_shape,
        "tril": _program_ad_assembly_triangular_shape,
        "triu": _program_ad_assembly_triangular_shape,
        "vstack": _program_ad_assembly_stack_convenience_shape_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_zeros_like_shape,
    }
    dtype_rules: Mapping[str, PrimitiveDTypeRule] = {
        "append": _program_ad_assembly_append_dtype_rule,
        "block": _program_ad_assembly_block_dtype_rule,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_dtype_rule,
        "broadcast_to": _program_ad_assembly_broadcast_to_dtype_rule,
        "concatenate": _program_ad_assembly_concatenate_dtype_rule,
        "diagonal": _program_ad_assembly_diagonal_dtype_rule,
        "dstack": _program_ad_assembly_stack_convenience_dtype_rule_for("dstack"),
        "full_like": _program_ad_assembly_like_dtype_rule,
        "hstack": _program_ad_assembly_stack_convenience_dtype_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_dtype_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_like_dtype_rule,
        "split": _program_ad_assembly_split_dtype_rule,
        "array_split": _program_ad_assembly_split_dtype_rule,
        "hsplit": _program_ad_assembly_split_dtype_rule,
        "vsplit": _program_ad_assembly_split_dtype_rule,
        "dsplit": _program_ad_assembly_split_dtype_rule,
        "stack": _program_ad_assembly_stack_dtype_rule,
        "tril": _program_ad_assembly_triangular_dtype_rule,
        "triu": _program_ad_assembly_triangular_dtype_rule,
        "vstack": _program_ad_assembly_stack_convenience_dtype_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_like_dtype_rule,
    }
    static_argument_rules: Mapping[str, PrimitiveStaticArgumentRule] = {
        "append": _program_ad_assembly_append_static_arguments,
        "block": _program_ad_assembly_block_static_arguments,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_static_arguments,
        "broadcast_to": _program_ad_assembly_broadcast_to_static_arguments,
        "concatenate": _program_ad_assembly_concatenate_static_arguments,
        "diagonal": _program_ad_assembly_diagonal_static_arguments,
        "dstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("dstack"),
        "full_like": _program_ad_assembly_full_like_static_arguments,
        "hstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_static_arguments_rule_for(
            "column_stack"
        ),
        "ones_like": _program_ad_assembly_ones_like_static_arguments,
        "split": _program_ad_assembly_split_static_arguments_rule_for("split"),
        "array_split": _program_ad_assembly_split_static_arguments_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_static_arguments_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_static_arguments_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_static_arguments_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_static_arguments,
        "tril": _program_ad_assembly_triangular_static_arguments,
        "triu": _program_ad_assembly_triangular_static_arguments,
        "vstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_zeros_like_static_arguments,
    }
    for name, identity in _PROGRAM_AD_ASSEMBLY_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_assembly_derivative_rule(name),
                batching_rule=batching_rules[name],
                lowering_metadata=_program_ad_assembly_lowering_metadata(name),
                shape_rule=shape_rules[name],
                dtype_rule=dtype_rules[name],
                static_argument_rule=static_argument_rules[name],
                nondifferentiable_policy=_PROGRAM_AD_ASSEMBLY_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_assembly_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered assembly primitive runtime contract."""

    identity = _PROGRAM_AD_ASSEMBLY_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD assembly primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_ASSEMBLY_POLICY:
        raise ValueError(f"invalid program AD assembly primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD assembly primitive effect for {identity.key}")

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
            f"incomplete program AD assembly primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_assembly_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_register_program_ad_assembly_primitive_contracts",
    "_require_program_ad_assembly_contract",
    "program_ad_assembly_append_derivative_rule",
    "program_ad_assembly_block_derivative_rule",
    "program_ad_assembly_broadcast_arrays_derivative_rule",
    "program_ad_assembly_broadcast_to_derivative_rule",
    "program_ad_assembly_column_stack_derivative_rule",
    "program_ad_assembly_concatenate_derivative_rule",
    "program_ad_assembly_diagonal_derivative_rule",
    "program_ad_assembly_dstack_derivative_rule",
    "program_ad_assembly_hstack_derivative_rule",
    "program_ad_assembly_split_derivative_rule",
    "program_ad_assembly_stack_derivative_rule",
    "program_ad_assembly_tril_derivative_rule",
    "program_ad_assembly_triu_derivative_rule",
    "program_ad_assembly_vstack_derivative_rule",
)
