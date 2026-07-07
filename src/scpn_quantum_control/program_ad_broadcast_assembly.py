# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD broadcast assembly primitives
"""Static broadcast assembly derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .program_ad_array_indexing import (
    _program_ad_array_normalise_static_shape,
    _program_ad_array_signature,
    _program_ad_array_static_size,
    _program_ad_array_vector,
    _program_ad_float64_vector_result,
)
from .program_ad_registry import CustomDerivativeRule


def _normalise_program_ad_broadcast_shape(shape: object) -> tuple[int, ...]:
    """Normalise a static broadcast target shape for Program AD assembly."""
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        dimensions: tuple[int, ...] = (int(shape),)
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        dimensions = tuple(int(dimension) for dimension in shape)
    else:
        raise ValueError("program AD np.broadcast_to requires an integer shape")
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("program AD np.broadcast_to shape dimensions must be non-negative")
    return dimensions


def _program_ad_assembly_broadcast_to_shapes(
    source_shape: Sequence[int],
    output_shape: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate static ``broadcast_to`` source and target shapes."""
    source = _program_ad_array_normalise_static_shape("assembly broadcast_to source", source_shape)
    output = _normalise_program_ad_broadcast_shape(output_shape)
    try:
        np.broadcast_to(np.empty(source, dtype=np.float64), output)
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_to requires output shape compatible "
            "with source broadcasting rules"
        ) from exc
    return source, output


def _program_ad_assembly_broadcast_adjoint(
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Reduce a broadcast cotangent back to the original source shape."""
    adjoint = np.asarray(cotangent, dtype=np.float64)
    if not source_shape:
        return np.asarray(float(np.sum(adjoint)), dtype=np.float64).reshape(())
    if adjoint.ndim > len(source_shape):
        leading_axes = tuple(range(adjoint.ndim - len(source_shape)))
        adjoint = np.sum(adjoint, axis=leading_axes)
    if adjoint.ndim != len(source_shape):
        raise ValueError("program AD assembly broadcast adjoint rank mismatch")
    for axis, dimension in enumerate(source_shape):
        if dimension == 1 and adjoint.shape[axis] != 1:
            adjoint = np.sum(adjoint, axis=axis, keepdims=True)
        elif dimension != adjoint.shape[axis]:
            raise ValueError("program AD assembly broadcast adjoint shape mismatch")
    return np.asarray(adjoint, dtype=np.float64).reshape(source_shape)


def program_ad_assembly_broadcast_to_derivative_rule(
    source_shape: Sequence[int],
    output_shape: object,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.broadcast_to``."""
    source_static_shape, output_static_shape = _program_ad_assembly_broadcast_to_shapes(
        source_shape, output_shape
    )
    source_size = _program_ad_array_static_size(source_static_shape)
    output_size = _program_ad_array_static_size(output_static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = _program_ad_array_vector(
            "broadcast_to", "values", values, expected_size=source_size
        ).reshape(source_static_shape)
        return _program_ad_float64_vector_result(np.broadcast_to(source, output_static_shape))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_to", "values", values, expected_size=source_size)
        tangent_source = _program_ad_array_vector(
            "broadcast_to", "tangent", tangent, expected_size=source_size
        ).reshape(source_static_shape)
        return _program_ad_float64_vector_result(
            np.broadcast_to(tangent_source, output_static_shape)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_to", "values", values, expected_size=source_size)
        cotangent_array = _program_ad_array_vector(
            "broadcast_to", "cotangent", cotangent, expected_size=output_size
        ).reshape(output_static_shape)
        return _program_ad_float64_vector_result(
            _program_ad_assembly_broadcast_adjoint(
                cotangent_array,
                source_shape=source_static_shape,
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_assembly_broadcast_to_"
            f"{_program_ad_array_signature(source_static_shape)}_to_"
            f"{_program_ad_array_signature(output_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_broadcast_arrays_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate static ``broadcast_arrays`` operand shapes and output shape."""
    shapes = tuple(
        _program_ad_array_normalise_static_shape("assembly broadcast_arrays operand", shape)
        for shape in operand_shapes
    )
    if not shapes:
        raise ValueError("program AD assembly broadcast_arrays direct rule requires operands")
    try:
        output_shape = tuple(int(dimension) for dimension in np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_arrays direct rule requires broadcast-compatible operands"
        ) from exc
    return shapes, output_shape


def program_ad_assembly_broadcast_arrays_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.broadcast_arrays``."""
    shapes, output_shape = _program_ad_assembly_broadcast_arrays_shapes(operand_shapes)
    source_sizes = tuple(_program_ad_array_static_size(shape) for shape in shapes)
    source_size = sum(source_sizes)
    output_size = _program_ad_array_static_size(output_shape)
    flat_output_size = len(shapes) * output_size

    def split_sources(
        role: str,
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        vector = _program_ad_array_vector(
            "broadcast_arrays", role, values, expected_size=source_size
        )
        offset = 0
        operands: list[NDArray[np.float64]] = []
        for shape, size in zip(shapes, source_sizes, strict=True):
            operands.append(vector[offset : offset + size].reshape(shape))
            offset += size
        return tuple(operands)

    def broadcast_flat(operands: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(
            np.concatenate(
                [
                    np.asarray(item, dtype=np.float64).reshape(-1)
                    for item in np.broadcast_arrays(*operands)
                ]
            )
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return broadcast_flat(split_sources("values", values))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_arrays", "values", values, expected_size=source_size)
        return broadcast_flat(split_sources("tangent", tangent))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_arrays", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "broadcast_arrays", "cotangent", cotangent, expected_size=flat_output_size
        )
        adjoints: list[NDArray[np.float64]] = []
        offset = 0
        for shape in shapes:
            cotangent_array = cotangent_vector[offset : offset + output_size].reshape(output_shape)
            adjoints.append(
                _program_ad_assembly_broadcast_adjoint(cotangent_array, source_shape=shape)
            )
            offset += output_size
        return _program_ad_float64_vector_result(
            np.concatenate([item.reshape(-1) for item in adjoints])
        )

    return CustomDerivativeRule(
        name=f"program_ad_assembly_broadcast_arrays_{len(shapes)}_operands_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
