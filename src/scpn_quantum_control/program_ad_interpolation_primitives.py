# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD interpolation primitive rules
"""Static interpolation derivative rules for Program AD."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array, _as_real_scalar
from .program_ad_array_indexing import (
    _program_ad_array_normalise_static_shape,
    _program_ad_array_signature,
    _program_ad_array_static_size,
)
from .program_ad_registry import CustomDerivativeRule


def _is_program_ad_trace_value(value: object) -> bool:
    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _normalise_interp_grid(xp: object) -> NDArray[np.float64]:
    if _is_program_ad_trace_value(xp):
        raise ValueError("program AD np.interp xp grid must be static real numeric")
    grid = _as_real_numeric_array("program AD np.interp xp grid", xp)
    if grid.ndim != 1:
        raise ValueError("program AD np.interp xp grid must be one-dimensional")
    if grid.size < 2:
        raise ValueError("program AD np.interp xp grid requires at least two samples")
    if not bool(np.all(np.isfinite(grid))):
        raise ValueError("program AD np.interp xp grid must contain only finite values")
    if not bool(np.all(np.diff(grid) > 0.0)):
        raise ValueError("program AD np.interp xp grid must be strictly increasing")
    return grid


def _program_ad_interp_static_boundary(name: str, value: object) -> float | None:
    if value is None:
        return None
    if _is_program_ad_trace_value(value):
        raise ValueError(f"program AD np.interp {name} boundary must be static real numeric")
    return _as_real_scalar(f"program AD interpolation interp {name}", value)


def _program_ad_interp_split_source(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(
        f"program AD interpolation {primitive_name} {role}", values
    ).reshape(-1)
    sample_size = _program_ad_array_static_size(sample_shape)
    expected_size = sample_size + grid_size
    if vector.size != expected_size:
        raise ValueError(
            f"program AD interpolation {primitive_name} direct rule requires "
            f"{expected_size} {role} values"
        )
    return vector[:sample_size], vector[sample_size:]


def _program_ad_interp_segment(
    sample: float,
    grid: NDArray[np.float64],
) -> tuple[Literal["left", "right", "interior"], int, float]:
    if not math.isfinite(sample):
        raise ValueError("program AD interpolation interp samples must be finite")
    if bool(np.any(grid == sample)):
        raise ValueError("program AD interpolation interp samples must avoid grid knots")
    if sample < float(grid[0]):
        return ("left", 0, 0.0)
    if sample > float(grid[-1]):
        return ("right", grid.size - 1, 0.0)
    segment = int(np.searchsorted(grid, sample, side="right") - 1)
    lower = float(grid[segment])
    upper = float(grid[segment + 1])
    weight = (sample - lower) / (upper - lower)
    return ("interior", segment, weight)


def _program_ad_interp_direct_value(
    values: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    outputs = np.zeros(samples.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        if region == "left":
            outputs[index] = fp_values[0] if left is None else left
        elif region == "right":
            outputs[index] = fp_values[-1] if right is None else right
        else:
            outputs[index] = (1.0 - weight) * fp_values[segment] + weight * fp_values[segment + 1]
    return outputs


def _program_ad_interp_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    sample_tangent, fp_tangent = _program_ad_interp_split_source(
        "interp", "tangent", tangent, sample_shape=sample_shape, grid_size=grid.size
    )
    outputs = np.zeros(samples.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        if region == "left":
            outputs[index] = fp_tangent[0] if left is None else 0.0
        elif region == "right":
            outputs[index] = fp_tangent[-1] if right is None else 0.0
        else:
            width = float(grid[segment + 1] - grid[segment])
            slope = (fp_values[segment + 1] - fp_values[segment]) / width
            outputs[index] = (
                slope * sample_tangent[index]
                + (1.0 - weight) * fp_tangent[segment]
                + weight * fp_tangent[segment + 1]
            )
    return outputs


def _program_ad_interp_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD interpolation interp cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != samples.size:
        raise ValueError(
            "program AD interpolation interp VJP requires cotangent matching sample size"
        )
    sample_adjoint = np.zeros(samples.size, dtype=np.float64)
    fp_adjoint = np.zeros(grid.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        scalar_cotangent = float(cotangent_vector[index])
        if region == "left":
            if left is None:
                fp_adjoint[0] += scalar_cotangent
        elif region == "right":
            if right is None:
                fp_adjoint[-1] += scalar_cotangent
        else:
            width = float(grid[segment + 1] - grid[segment])
            slope = (fp_values[segment + 1] - fp_values[segment]) / width
            sample_adjoint[index] += scalar_cotangent * slope
            fp_adjoint[segment] += scalar_cotangent * (1.0 - weight)
            fp_adjoint[segment + 1] += scalar_cotangent * weight
    return np.concatenate([sample_adjoint, fp_adjoint])


def program_ad_interpolation_interp_derivative_rule(
    sample_shape: Sequence[int],
    xp: object,
    value_shape: Sequence[int],
    *,
    left: object = None,
    right: object = None,
    period: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.interp`` grids."""

    if period is not None:
        raise ValueError("program AD interpolation interp direct rule does not support period")
    samples = _program_ad_array_normalise_static_shape("interp", sample_shape)
    grid = _normalise_interp_grid(xp)
    fp_shape = tuple(int(dimension) for dimension in value_shape)
    if fp_shape != (grid.size,):
        raise ValueError(
            "program AD interpolation interp direct rule requires fp shape to match xp"
        )
    left_value = _program_ad_interp_static_boundary("left", left)
    right_value = _program_ad_interp_static_boundary("right", right)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_interp_direct_value(
            values,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_interp_direct_jvp(
            values,
            tangent,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_interp_direct_vjp(
            values,
            cotangent,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_interpolation_interp_"
            f"x{_program_ad_array_signature(samples)}_grid{grid.size}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )
