# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD stencil primitive rules
"""Static finite-difference stencil derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array, _as_real_scalar
from .program_ad_registry import CustomDerivativeRule
from .program_ad_shape_transforms import _program_ad_shape_signature

_GradientSpacing = (
    tuple[Literal["scalar"], float] | tuple[Literal["coordinates"], NDArray[np.float64]]
)


def _normalise_gradient_edge_order(edge_order: object) -> int:
    if isinstance(edge_order, (bool, np.bool_)) or not isinstance(edge_order, (int, np.integer)):
        raise ValueError("program AD np.gradient edge_order must be 1 or 2")
    edge = int(edge_order)
    if edge not in {1, 2}:
        raise ValueError("program AD np.gradient edge_order must be 1 or 2")
    return edge


def _normalise_gradient_axis(axis: int, ndim: int) -> int:
    value = axis
    if value < 0:
        value += ndim
    if value < 0 or value >= ndim:
        raise ValueError("program AD np.gradient axis out of bounds")
    return value


def _normalise_gradient_axes(axis: object, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (bool, np.bool_)):
        raise ValueError("program AD np.gradient axis must be a static integer")
    if isinstance(axis, (int, np.integer)):
        return (_normalise_gradient_axis(int(axis), ndim),)
    if not isinstance(axis, tuple) or not axis:
        raise ValueError("program AD np.gradient axis must be a static integer")
    axes: list[int] = []
    for item in axis:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("program AD np.gradient axis must be a static integer")
        axes.append(_normalise_gradient_axis(int(item), ndim))
    if len(set(axes)) != len(axes):
        raise ValueError("program AD np.gradient axes must be unique")
    return tuple(axes)


def _is_program_ad_trace_value(value: object) -> bool:
    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _is_static_gradient_scalar_spacing(spacing: object) -> bool:
    if _is_program_ad_trace_value(spacing):
        return False
    try:
        raw = np.asarray(spacing)
    except ValueError:
        return False
    return raw.shape == () and raw.dtype.kind not in {"b", "O", "S", "U", "c"}


def _normalise_gradient_spacing(spacing: object, axis_size: int) -> _GradientSpacing:
    if _is_program_ad_trace_value(spacing):
        raise ValueError("program AD np.gradient spacing must be static real numeric")
    raw = np.asarray(spacing)
    if raw.shape == ():
        scalar = _as_real_scalar("program AD np.gradient spacing", spacing)
        if scalar == 0.0:
            raise ValueError("program AD np.gradient spacing must be non-zero")
        return ("scalar", scalar)
    coordinates = _as_real_numeric_array("program AD np.gradient coordinates", spacing)
    if coordinates.ndim != 1 or coordinates.shape[0] != axis_size:
        raise ValueError("program AD np.gradient coordinates must match the differentiation axis")
    if not bool(np.all(np.isfinite(coordinates))):
        raise ValueError("program AD np.gradient coordinates must contain only finite values")
    deltas = np.diff(coordinates)
    if not bool(np.all(deltas > 0.0)) and not bool(np.all(deltas < 0.0)):
        raise ValueError("program AD np.gradient coordinates must be strictly monotonic")
    return ("coordinates", coordinates)


def _normalise_gradient_spacings(
    spacings: tuple[object, ...], axes: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[_GradientSpacing, ...]:
    if not spacings:
        raw_spacings: tuple[object, ...] = tuple(1.0 for _ in axes)
    elif len(spacings) == 1:
        if len(axes) == 1 or _is_static_gradient_scalar_spacing(spacings[0]):
            raw_spacings = tuple(spacings[0] for _ in axes)
        else:
            raise ValueError("program AD np.gradient spacing count must match axes")
    elif len(spacings) == len(axes):
        raw_spacings = spacings
    else:
        raise ValueError("program AD np.gradient spacing count must match axes")
    return tuple(
        _normalise_gradient_spacing(spacing, shape[axis])
        for spacing, axis in zip(raw_spacings, axes, strict=True)
    )


def _gradient_axis_coefficients(
    position: int,
    axis_size: int,
    spacing: _GradientSpacing,
    edge_order: int,
) -> tuple[tuple[int, float], ...]:
    if axis_size < edge_order + 1:
        raise ValueError(
            f"program AD np.gradient edge_order {edge_order} requires "
            f"at least {edge_order + 1} samples"
        )
    if spacing[0] == "scalar":
        dx = spacing[1]
        if position == 0:
            if edge_order == 1:
                return ((0, -1.0 / dx), (1, 1.0 / dx))
            return ((0, -1.5 / dx), (1, 2.0 / dx), (2, -0.5 / dx))
        if position == axis_size - 1:
            if edge_order == 1:
                return ((axis_size - 2, -1.0 / dx), (axis_size - 1, 1.0 / dx))
            return (
                (axis_size - 3, 0.5 / dx),
                (axis_size - 2, -2.0 / dx),
                (axis_size - 1, 1.5 / dx),
            )
        return ((position - 1, -0.5 / dx), (position + 1, 0.5 / dx))

    coordinates = spacing[1]
    if position == 0:
        dx1 = float(coordinates[1] - coordinates[0])
        if edge_order == 1:
            return ((0, -1.0 / dx1), (1, 1.0 / dx1))
        dx2 = float(coordinates[2] - coordinates[1])
        return (
            (0, -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))),
            (1, (dx1 + dx2) / (dx1 * dx2)),
            (2, -dx1 / (dx2 * (dx1 + dx2))),
        )
    if position == axis_size - 1:
        dx1 = float(coordinates[-2] - coordinates[-3])
        dx2 = float(coordinates[-1] - coordinates[-2])
        if edge_order == 1:
            return ((axis_size - 2, -1.0 / dx2), (axis_size - 1, 1.0 / dx2))
        return (
            (axis_size - 3, dx2 / (dx1 * (dx1 + dx2))),
            (axis_size - 2, -(dx1 + dx2) / (dx1 * dx2)),
            (axis_size - 1, (dx1 + 2.0 * dx2) / (dx2 * (dx1 + dx2))),
        )
    dx1 = float(coordinates[position] - coordinates[position - 1])
    dx2 = float(coordinates[position + 1] - coordinates[position])
    return (
        (position - 1, -dx2 / (dx1 * (dx1 + dx2))),
        (position, (dx2 - dx1) / (dx1 * dx2)),
        (position + 1, dx1 / (dx2 * (dx1 + dx2))),
    )


def _program_ad_stencil_gradient_source_vector(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD stencil gradient {role}", values).reshape(-1)
    expected_size = int(np.prod(source_shape, dtype=np.int64))
    if vector.size != expected_size:
        raise ValueError(
            f"program AD stencil gradient direct rule requires {expected_size} {role} values"
        )
    return vector


def _program_ad_stencil_gradient_flat_components(
    source: NDArray[np.float64],
    *,
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> tuple[NDArray[np.float64], ...]:
    components: list[NDArray[np.float64]] = []
    for axis_index, spacing in zip(axes, spacings, strict=True):
        component = np.zeros_like(source, dtype=np.float64)
        for flat_index in range(source.size):
            target_index = np.unravel_index(flat_index, source.shape)
            total = 0.0
            for source_axis_index, coefficient in _gradient_axis_coefficients(
                int(target_index[axis_index]),
                int(source.shape[axis_index]),
                spacing,
                edge_order,
            ):
                source_index = (
                    target_index[:axis_index]
                    + (source_axis_index,)
                    + target_index[axis_index + 1 :]
                )
                total += float(source[source_index]) * coefficient
            component[target_index] = total
        components.append(component.reshape(-1))
    return tuple(components)


def _program_ad_stencil_gradient_flat_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> NDArray[np.float64]:
    source = _program_ad_stencil_gradient_source_vector(
        "values", values, source_shape=source_shape
    ).reshape(source_shape)
    components = _program_ad_stencil_gradient_flat_components(
        source,
        axes=axes,
        spacings=spacings,
        edge_order=edge_order,
    )
    return np.concatenate(components).astype(np.float64, copy=False)


def _program_ad_stencil_gradient_flat_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> NDArray[np.float64]:
    _program_ad_stencil_gradient_source_vector("values", values, source_shape=source_shape)
    source_size = int(np.prod(source_shape, dtype=np.int64))
    cotangent_vector = _as_real_numeric_array(
        "program AD stencil gradient cotangent", cotangent
    ).reshape(-1)
    expected_cotangent_size = source_size * len(axes)
    if cotangent_vector.size != expected_cotangent_size:
        raise ValueError(
            f"program AD stencil gradient VJP requires {expected_cotangent_size} cotangent values"
        )
    adjoint = np.zeros(source_shape, dtype=np.float64)
    cotangent_components = cotangent_vector.reshape((len(axes), source_size))
    for component_cotangent, axis_index, spacing in zip(
        cotangent_components,
        axes,
        spacings,
        strict=True,
    ):
        cotangent_array = component_cotangent.reshape(source_shape)
        for flat_index in range(source_size):
            target_index = np.unravel_index(flat_index, source_shape)
            scalar_cotangent = float(cotangent_array[target_index])
            for source_axis_index, coefficient in _gradient_axis_coefficients(
                int(target_index[axis_index]),
                int(source_shape[axis_index]),
                spacing,
                edge_order,
            ):
                source_index = (
                    target_index[:axis_index]
                    + (source_axis_index,)
                    + target_index[axis_index + 1 :]
                )
                adjoint[source_index] += scalar_cotangent * coefficient
    return adjoint.reshape(-1)


def _program_ad_gradient_spacing_signature(
    spacing: _GradientSpacing,
) -> tuple[str, float] | tuple[str, tuple[int, ...], tuple[float, ...]]:
    if spacing[0] == "scalar":
        return ("scalar", float(spacing[1]))
    coordinates = np.asarray(spacing[1], dtype=np.float64)
    return (
        "coordinates",
        tuple(int(dimension) for dimension in coordinates.shape),
        tuple(float(value) for value in coordinates.reshape(-1)),
    )


def program_ad_stencil_gradient_derivative_rule(
    source_shape: Sequence[int],
    spacings: Sequence[object] = (),
    *,
    axis: object = None,
    edge_order: object = 1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed ``np.gradient`` signature."""

    source = tuple(int(dimension) for dimension in source_shape)
    if any(dimension <= 0 for dimension in source):
        raise ValueError(
            "program AD stencil gradient direct rule requires positive source dimensions"
        )
    edge = _normalise_gradient_edge_order(edge_order)
    axes = _normalise_gradient_axes(axis, len(source))
    spacing_specs = _normalise_gradient_spacings(tuple(spacings), axes, source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_stencil_gradient_flat_value(
            values,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_stencil_gradient_source_vector("values", values, source_shape=source)
        return _program_ad_stencil_gradient_flat_value(
            tangent,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_stencil_gradient_flat_vjp(
            values,
            cotangent,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    axes_signature = "_".join(str(axis_index) for axis_index in axes)
    return CustomDerivativeRule(
        name=(
            "program_ad_stencil_gradient_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_edge_{edge}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
