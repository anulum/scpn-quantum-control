# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD stack block assembly module
# scpn-quantum-control -- Program AD stack/block assembly primitives
"""Static stack, append, concatenate, and block assembly rules for Program AD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_normalise_static_shape,
    _program_ad_array_static_size,
    _program_ad_array_vector,
    _program_ad_float64_vector_result,
)
from .program_ad_registry import CustomDerivativeRule


def _program_ad_assembly_stack_convenience_numpy(
    name: str,
    operands: Sequence[object],
) -> object:
    if name == "hstack":
        return np.hstack(cast(Any, operands))
    if name == "vstack":
        return np.vstack(cast(Any, operands))
    if name == "column_stack":
        return np.column_stack(cast(Any, operands))
    if name == "dstack":
        return np.dstack(cast(Any, operands))
    raise ValueError(f"unsupported program AD assembly stack convenience primitive {name}")


def _program_ad_assembly_stack_convenience_selected_indices(
    name: str,
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], NDArray[np.int64]]:
    shapes = tuple(
        _program_ad_array_normalise_static_shape(f"assembly {name} operand", shape)
        for shape in operand_shapes
    )
    if not shapes:
        raise ValueError(f"program AD assembly {name} direct rule requires operands")
    index_operands: list[NDArray[np.int64]] = []
    offset = 0
    for shape in shapes:
        size = _program_ad_array_static_size(shape)
        index_operands.append(np.arange(offset, offset + size, dtype=np.int64).reshape(shape))
        offset += size
    try:
        selected = _program_ad_assembly_stack_convenience_numpy(name, index_operands)
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            f"program AD assembly {name} requires shape-compatible static operands"
        ) from exc
    return shapes, np.asarray(selected, dtype=np.int64)


def _program_ad_assembly_concatenate_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(
        _program_ad_array_normalise_static_shape("assembly concatenate operand", shape)
        for shape in operand_shapes
    )
    if not shapes:
        raise ValueError("program AD assembly concatenate direct rule requires operands")
    return shapes


def _program_ad_assembly_concatenate_axis(
    axis: object,
    *,
    rank: int,
) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD assembly concatenate direct rule requires a static integer axis or None"
        )
    if rank <= 0:
        raise ValueError("program AD assembly concatenate direct rule requires ranked operands")
    return _normalise_axis("axis", int(axis), rank)


def _program_ad_assembly_concatenate_output_shape(
    operand_shapes: Sequence[Sequence[int]],
    axis: object = 0,
) -> tuple[int, ...]:
    shapes = _program_ad_assembly_concatenate_shapes(operand_shapes)
    if axis is None:
        return (sum(_program_ad_array_static_size(shape) for shape in shapes),)
    rank = len(shapes[0])
    axis_index = cast(int, _program_ad_assembly_concatenate_axis(axis, rank=rank))
    for shape in shapes:
        if len(shape) != rank:
            raise ValueError(
                "program AD assembly concatenate direct rule requires equal operand ranks"
            )
        for dimension_index, dimension in enumerate(shape):
            if dimension_index != axis_index and dimension != shapes[0][dimension_index]:
                raise ValueError(
                    "program AD assembly concatenate direct rule requires matching "
                    "non-concatenate dimensions"
                )
    output = list(shapes[0])
    output[axis_index] = sum(shape[axis_index] for shape in shapes)
    return tuple(output)


def _program_ad_assembly_concatenate_source_size(
    operand_shapes: tuple[tuple[int, ...], ...],
) -> int:
    return sum(_program_ad_array_static_size(shape) for shape in operand_shapes)


def _program_ad_assembly_concatenate_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _program_ad_array_vector(
        "concatenate",
        role,
        values,
        expected_size=_program_ad_assembly_concatenate_source_size(operand_shapes),
    )
    operands: list[NDArray[np.float64]] = []
    offset = 0
    for shape in operand_shapes:
        size = _program_ad_array_static_size(shape)
        operands.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return tuple(operands)


def program_ad_assembly_concatenate_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
    *,
    axis: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.concatenate`` operands."""
    shapes = _program_ad_assembly_concatenate_shapes(operand_shapes)
    axis_index = (
        None if axis is None else _program_ad_assembly_concatenate_axis(axis, rank=len(shapes[0]))
    )
    output_shape = _program_ad_assembly_concatenate_output_shape(shapes, axis_index)
    output_size = _program_ad_array_static_size(output_shape)
    axis_signature = "flat" if axis_index is None else str(axis_index)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _program_ad_assembly_concatenate_split_source(
            "values", values, operand_shapes=shapes
        )
        return _program_ad_float64_vector_result(
            np.concatenate(operands, axis=axis_index).reshape(-1)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_assembly_concatenate_split_source("values", values, operand_shapes=shapes)
        tangent_operands = _program_ad_assembly_concatenate_split_source(
            "tangent", tangent, operand_shapes=shapes
        )
        return _program_ad_float64_vector_result(
            np.concatenate(tangent_operands, axis=axis_index).reshape(-1)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_assembly_concatenate_split_source("values", values, operand_shapes=shapes)
        cotangent_vector = _program_ad_array_vector(
            "concatenate", "cotangent", cotangent, expected_size=output_size
        )
        if axis_index is None:
            return _program_ad_float64_vector_result(cotangent_vector)
        cotangent_array = cotangent_vector.reshape(output_shape)
        split_points = np.cumsum([shape[axis_index] for shape in shapes[:-1]], dtype=np.int64)
        split_adjoints = np.split(cotangent_array, split_points.tolist(), axis=axis_index)
        return _program_ad_float64_vector_result(
            np.concatenate([adjoint.reshape(-1) for adjoint in split_adjoints])
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_assembly_concatenate_"
            f"{len(shapes)}_operands_axis{axis_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_stack_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(
        _program_ad_array_normalise_static_shape("assembly stack operand", shape)
        for shape in operand_shapes
    )
    if not shapes:
        raise ValueError("program AD assembly stack direct rule requires operands")
    reference = shapes[0]
    for shape in shapes:
        if shape != reference:
            raise ValueError(
                "program AD assembly stack direct rule requires matching operand shapes"
            )
    return shapes


def _program_ad_assembly_stack_axis(
    axis: object,
    *,
    rank: int,
) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly stack direct rule requires a static integer axis")
    output_rank = rank + 1
    return _normalise_axis("axis", int(axis), output_rank)


def _program_ad_assembly_stack_output_shape(
    operand_shapes: Sequence[Sequence[int]],
    axis: object = 0,
) -> tuple[int, ...]:
    shapes = _program_ad_assembly_stack_shapes(operand_shapes)
    axis_index = _program_ad_assembly_stack_axis(axis, rank=len(shapes[0]))
    output = list(shapes[0])
    output.insert(axis_index, len(shapes))
    return tuple(output)


def _program_ad_assembly_stack_source_size(
    operand_shapes: tuple[tuple[int, ...], ...],
) -> int:
    return sum(_program_ad_array_static_size(shape) for shape in operand_shapes)


def _program_ad_assembly_stack_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _program_ad_array_vector(
        "stack",
        role,
        values,
        expected_size=_program_ad_assembly_stack_source_size(operand_shapes),
    )
    operands: list[NDArray[np.float64]] = []
    offset = 0
    for shape in operand_shapes:
        size = _program_ad_array_static_size(shape)
        operands.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return tuple(operands)


def program_ad_assembly_stack_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
    *,
    axis: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.stack`` operands."""
    shapes = _program_ad_assembly_stack_shapes(operand_shapes)
    axis_index = _program_ad_assembly_stack_axis(axis, rank=len(shapes[0]))
    output_shape = _program_ad_assembly_stack_output_shape(shapes, axis_index)
    output_size = _program_ad_array_static_size(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _program_ad_assembly_stack_split_source("values", values, operand_shapes=shapes)
        return _program_ad_float64_vector_result(np.stack(operands, axis=axis_index).reshape(-1))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_assembly_stack_split_source("values", values, operand_shapes=shapes)
        tangent_operands = _program_ad_assembly_stack_split_source(
            "tangent", tangent, operand_shapes=shapes
        )
        return _program_ad_float64_vector_result(
            np.stack(tangent_operands, axis=axis_index).reshape(-1)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_assembly_stack_split_source("values", values, operand_shapes=shapes)
        cotangent_vector = _program_ad_array_vector(
            "stack", "cotangent", cotangent, expected_size=output_size
        )
        cotangent_array = cotangent_vector.reshape(output_shape)
        adjoints = [
            np.take(cotangent_array, operand_index, axis=axis_index).reshape(-1)
            for operand_index in range(len(shapes))
        ]
        return _program_ad_float64_vector_result(np.concatenate(adjoints))

    return CustomDerivativeRule(
        name=f"program_ad_assembly_stack_{len(shapes)}_operands_axis{axis_index}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_stack_convenience_source_size(
    operand_shapes: tuple[tuple[int, ...], ...],
) -> int:
    return sum(_program_ad_array_static_size(shape) for shape in operand_shapes)


def _program_ad_assembly_stack_convenience_split_source(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _program_ad_array_vector(
        name,
        role,
        values,
        expected_size=_program_ad_assembly_stack_convenience_source_size(operand_shapes),
    )
    operands: list[NDArray[np.float64]] = []
    offset = 0
    for shape in operand_shapes:
        size = _program_ad_array_static_size(shape)
        operands.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return tuple(operands)


def _program_ad_assembly_stack_convenience_derivative_rule(
    name: str,
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    shapes, selected = _program_ad_assembly_stack_convenience_selected_indices(
        name, operand_shapes
    )
    output_shape = tuple(int(dimension) for dimension in selected.shape)
    output_size = _program_ad_array_static_size(output_shape)
    source_size = _program_ad_assembly_stack_convenience_source_size(shapes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _program_ad_assembly_stack_convenience_split_source(
            name, "values", values, operand_shapes=shapes
        )
        output = _program_ad_assembly_stack_convenience_numpy(name, operands)
        return _program_ad_float64_vector_result(np.asarray(output, dtype=np.float64).reshape(-1))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_assembly_stack_convenience_split_source(
            name, "values", values, operand_shapes=shapes
        )
        tangent_operands = _program_ad_assembly_stack_convenience_split_source(
            name, "tangent", tangent, operand_shapes=shapes
        )
        output = _program_ad_assembly_stack_convenience_numpy(name, tangent_operands)
        return _program_ad_float64_vector_result(np.asarray(output, dtype=np.float64).reshape(-1))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_assembly_stack_convenience_split_source(
            name, "values", values, operand_shapes=shapes
        )
        cotangent_vector = _program_ad_array_vector(
            name, "cotangent", cotangent, expected_size=output_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, selected.reshape(-1), cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=f"program_ad_assembly_{name}_{len(shapes)}_operands_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_assembly_hstack_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.hstack`` operands."""
    return _program_ad_assembly_stack_convenience_derivative_rule("hstack", operand_shapes)


def program_ad_assembly_vstack_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.vstack`` operands."""
    return _program_ad_assembly_stack_convenience_derivative_rule("vstack", operand_shapes)


def program_ad_assembly_column_stack_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.column_stack`` operands."""
    return _program_ad_assembly_stack_convenience_derivative_rule("column_stack", operand_shapes)


def program_ad_assembly_dstack_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.dstack`` operands."""
    return _program_ad_assembly_stack_convenience_derivative_rule("dstack", operand_shapes)


def _program_ad_assembly_append_shapes(
    source_shape: Sequence[int],
    values_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = _program_ad_array_normalise_static_shape("assembly append source", source_shape)
    values = _program_ad_array_normalise_static_shape("assembly append values", values_shape)
    return source, values


def _program_ad_assembly_append_output_shape(
    source_shape: Sequence[int],
    values_shape: Sequence[int],
    *,
    axis: object = None,
) -> tuple[int, ...]:
    source, values = _program_ad_assembly_append_shapes(source_shape, values_shape)
    if axis is None:
        return (_program_ad_array_static_size(source) + _program_ad_array_static_size(values),)
    axis_index = _program_ad_assembly_concatenate_axis(axis, rank=len(source))
    return _program_ad_assembly_concatenate_output_shape((source, values), axis_index)


def _program_ad_assembly_append_source_size(
    source_shape: tuple[int, ...],
    values_shape: tuple[int, ...],
) -> int:
    return _program_ad_array_static_size(source_shape) + _program_ad_array_static_size(
        values_shape
    )


def _program_ad_assembly_append_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    values_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_array_vector(
        "append",
        role,
        values,
        expected_size=_program_ad_assembly_append_source_size(source_shape, values_shape),
    )
    source_size = _program_ad_array_static_size(source_shape)
    return (
        vector[:source_size].reshape(source_shape),
        vector[source_size:].reshape(values_shape),
    )


def program_ad_assembly_append_derivative_rule(
    source_shape: Sequence[int],
    values_shape: Sequence[int],
    *,
    axis: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.append`` operands."""
    source_static, values_static = _program_ad_assembly_append_shapes(source_shape, values_shape)
    axis_index = (
        None
        if axis is None
        else _program_ad_assembly_concatenate_axis(axis, rank=len(source_static))
    )
    output_shape = _program_ad_assembly_append_output_shape(
        source_static, values_static, axis=axis_index
    )
    output_size = _program_ad_array_static_size(output_shape)
    axis_signature = "flat" if axis_index is None else str(axis_index)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source_operand, values_operand = _program_ad_assembly_append_split_source(
            "values", values, source_shape=source_static, values_shape=values_static
        )
        return _program_ad_float64_vector_result(
            np.append(source_operand, values_operand, axis=axis_index).reshape(-1)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_assembly_append_split_source(
            "values", values, source_shape=source_static, values_shape=values_static
        )
        tangent_source, tangent_values = _program_ad_assembly_append_split_source(
            "tangent", tangent, source_shape=source_static, values_shape=values_static
        )
        return _program_ad_float64_vector_result(
            np.append(tangent_source, tangent_values, axis=axis_index).reshape(-1)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_assembly_append_split_source(
            "values", values, source_shape=source_static, values_shape=values_static
        )
        cotangent_vector = _program_ad_array_vector(
            "append", "cotangent", cotangent, expected_size=output_size
        )
        if axis_index is None:
            return _program_ad_float64_vector_result(cotangent_vector)
        cotangent_array = cotangent_vector.reshape(output_shape)
        source_extent = source_static[axis_index]
        source_adjoint, values_adjoint = np.split(
            cotangent_array, [source_extent], axis=axis_index
        )
        return _program_ad_float64_vector_result(
            np.concatenate([source_adjoint.reshape(-1), values_adjoint.reshape(-1)])
        )

    return CustomDerivativeRule(
        name=f"program_ad_assembly_append_axis{axis_signature}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_block_is_static_shape(value: object) -> bool:
    return isinstance(value, (tuple, list)) and all(
        not isinstance(dimension, bool) and isinstance(dimension, (int, np.integer))
        for dimension in value
    )


def _program_ad_assembly_block_shapes(layout_shapes: object) -> tuple[object, ...]:
    if _program_ad_assembly_block_is_static_shape(layout_shapes):
        raise ValueError("program AD assembly block direct rule requires nested layout shapes")
    if not isinstance(layout_shapes, (tuple, list)) or not layout_shapes:
        raise ValueError("program AD assembly block direct rule requires nested layout shapes")

    def normalise(node: object) -> object:
        if _program_ad_assembly_block_is_static_shape(node):
            return _program_ad_array_normalise_static_shape(
                "assembly block operand", cast(Any, node)
            )
        if not isinstance(node, (tuple, list)) or not node:
            raise ValueError("program AD assembly block direct rule requires nested layout shapes")
        return tuple(normalise(item) for item in node)

    return cast(tuple[object, ...], tuple(normalise(item) for item in layout_shapes))


def _program_ad_assembly_block_shape_leaves(layout_shapes: object) -> tuple[tuple[int, ...], ...]:
    if _program_ad_assembly_block_is_static_shape(layout_shapes):
        return (tuple(int(dimension) for dimension in cast(Sequence[int], layout_shapes)),)
    leaves: list[tuple[int, ...]] = []
    if not isinstance(layout_shapes, (tuple, list)) or not layout_shapes:
        raise ValueError("program AD assembly block direct rule requires nested layout shapes")
    for item in layout_shapes:
        leaves.extend(_program_ad_assembly_block_shape_leaves(item))
    return tuple(leaves)


def _program_ad_assembly_block_probe_layout(layout_shapes: object) -> object:
    if _program_ad_assembly_block_is_static_shape(layout_shapes):
        return np.empty(tuple(int(dimension) for dimension in cast(Sequence[int], layout_shapes)))
    if not isinstance(layout_shapes, (tuple, list)) or not layout_shapes:
        raise ValueError("program AD assembly block direct rule requires nested layout shapes")
    return [_program_ad_assembly_block_probe_layout(item) for item in layout_shapes]


def _program_ad_assembly_block_output_shape(layout_shapes: object) -> tuple[int, ...]:
    shapes = _program_ad_assembly_block_shapes(layout_shapes)
    try:
        output = np.block(cast(Any, _program_ad_assembly_block_probe_layout(shapes)))
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD assembly block direct rule requires shape-compatible nested layout"
        ) from exc
    return tuple(int(dimension) for dimension in output.shape)


def _program_ad_assembly_block_source_size(layout_shapes: object) -> int:
    return sum(
        _program_ad_array_static_size(shape)
        for shape in _program_ad_assembly_block_shape_leaves(layout_shapes)
    )


def _program_ad_assembly_block_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    layout_shapes: tuple[object, ...],
) -> object:
    vector = _program_ad_array_vector(
        "block",
        role,
        values,
        expected_size=_program_ad_assembly_block_source_size(layout_shapes),
    )
    offset = 0

    def split(node: object) -> object:
        nonlocal offset
        if _program_ad_assembly_block_is_static_shape(node):
            shape = tuple(int(dimension) for dimension in cast(Sequence[int], node))
            size = _program_ad_array_static_size(shape)
            operand = vector[offset : offset + size].reshape(shape)
            offset += size
            return operand
        return [split(item) for item in cast(tuple[object, ...], node)]

    return split(layout_shapes)


def _program_ad_assembly_block_index_layout(layout_shapes: tuple[object, ...]) -> object:
    offset = 0

    def build(node: object) -> object:
        nonlocal offset
        if _program_ad_assembly_block_is_static_shape(node):
            shape = tuple(int(dimension) for dimension in cast(Sequence[int], node))
            size = _program_ad_array_static_size(shape)
            indices = np.arange(offset, offset + size, dtype=np.int64).reshape(shape)
            offset += size
            return indices
        return [build(item) for item in cast(tuple[object, ...], node)]

    return build(layout_shapes)


def program_ad_assembly_block_derivative_rule(layout_shapes: object) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.block`` layouts."""
    shapes = _program_ad_assembly_block_shapes(layout_shapes)
    output_shape = _program_ad_assembly_block_output_shape(shapes)
    output_size = _program_ad_array_static_size(output_shape)
    source_size = _program_ad_assembly_block_source_size(shapes)
    operand_count = len(_program_ad_assembly_block_shape_leaves(shapes))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        layout = _program_ad_assembly_block_split_source("values", values, layout_shapes=shapes)
        return _program_ad_float64_vector_result(np.block(cast(Any, layout)).reshape(-1))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_assembly_block_split_source("values", values, layout_shapes=shapes)
        tangent_layout = _program_ad_assembly_block_split_source(
            "tangent", tangent, layout_shapes=shapes
        )
        return _program_ad_float64_vector_result(np.block(cast(Any, tangent_layout)).reshape(-1))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_assembly_block_split_source("values", values, layout_shapes=shapes)
        cotangent_vector = _program_ad_array_vector(
            "block", "cotangent", cotangent, expected_size=output_size
        )
        selected = np.asarray(np.block(cast(Any, _program_ad_assembly_block_index_layout(shapes))))
        cotangent_array = cotangent_vector.reshape(output_shape)
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, selected.reshape(-1), cotangent_array.reshape(-1))
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=f"program_ad_assembly_block_{operand_count}_operands_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
