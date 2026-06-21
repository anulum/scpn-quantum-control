# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable transform helper contracts
"""Shared scalar, parameter, bounds, and tape helpers for native transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    Parameter,
    ParameterBounds,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_scalar_kernels import DualNumber, ReverseNode


def _as_scalar(value: float | int | np.floating[Any] | NDArray[np.float64]) -> float:
    """Return a finite scalar objective value for native scalar transforms."""

    try:
        scalar = _as_real_scalar("differentiable objective", value)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("differentiable objective must return a scalar") from exc
        raise
    if not np.isfinite(scalar):
        raise ValueError("differentiable objective returned a non-finite scalar")
    return scalar


def _as_forward_mode_scalar(value: object) -> DualNumber:
    """Return a scalar dual objective value."""

    if isinstance(value, DualNumber):
        return value
    try:
        return DualNumber(_as_real_scalar("forward-mode objective", value), 0.0)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("forward-mode objective must return a scalar") from exc
        raise


def _as_reverse_mode_scalar(value: object) -> ReverseNode:
    """Return a scalar reverse-mode objective value."""

    if isinstance(value, ReverseNode):
        return value
    try:
        return ReverseNode(_as_real_scalar("reverse-mode objective", value))
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("reverse-mode objective must return a scalar") from exc
        raise


def _reverse_topological_order(root: ReverseNode) -> tuple[ReverseNode, ...]:
    """Return reverse-mode tape nodes in parent-before-child order."""

    ordered: list[ReverseNode] = []
    seen: set[int] = set()

    def visit(node: ReverseNode) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        for parent, _local_derivative in node.parents:
            visit(parent)
        ordered.append(node)

    visit(root)
    return tuple(ordered)


def _as_complex_step_scalar(value: object) -> complex:
    """Return a scalar objective value that may carry a complex-step signal."""

    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError("complex-step objective must return a scalar")
    try:
        scalar = complex(raw.item())
    except (TypeError, ValueError) as exc:
        raise ValueError("complex-step objective must return a numeric scalar") from exc
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError("complex-step objective returned a non-finite scalar")
    return scalar


def _as_vector_output(value: ArrayLike) -> NDArray[np.float64]:
    """Return a finite one-dimensional vector objective value."""

    vector = _as_real_numeric_array("differentiable vector objective", value)
    if vector.ndim != 1:
        raise ValueError("differentiable vector objective must return a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("differentiable vector objective returned non-finite values")
    return vector


def _normalise_parameters(
    values: NDArray[np.float64],
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    """Return parameter metadata aligned to a numeric parameter vector."""

    if parameters is None:
        return tuple(Parameter(f"theta_{index}") for index in range(values.size))
    normalised = tuple(parameters)
    if len(normalised) != values.size:
        raise ValueError("parameters length must match values length")
    if len({parameter.name for parameter in normalised}) != len(normalised):
        raise ValueError("parameter names must be unique")
    return normalised


def _normalise_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds] | None,
) -> tuple[ParameterBounds, ...]:
    """Return bounds metadata aligned to a numeric parameter vector."""

    if bounds is None:
        return tuple(ParameterBounds() for _ in range(values.size))
    normalised = tuple(bounds)
    if len(normalised) != values.size:
        raise ValueError("bounds length must match values length")
    if any(not isinstance(item, ParameterBounds) for item in normalised):
        raise ValueError("bounds must contain ParameterBounds instances")
    return normalised


def _project_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds],
) -> NDArray[np.float64]:
    """Project parameter values through box or periodic bounds."""

    projected = values.copy()
    for index, bound in enumerate(bounds):
        if bound.periodic:
            lower = cast(float, bound.lower)
            upper = cast(float, bound.upper)
            width = upper - lower
            projected[index] = ((projected[index] - lower) % width) + lower
            continue
        if bound.lower is not None and projected[index] < bound.lower:
            projected[index] = bound.lower
        if bound.upper is not None and projected[index] > bound.upper:
            projected[index] = bound.upper
    typed_projected: NDArray[np.float64] = projected
    return typed_projected


def _validate_max_gradient_norm(max_gradient_norm: float | None) -> float | None:
    """Return a positive finite gradient-norm cap or ``None``."""

    if max_gradient_norm is None:
        return None
    max_norm = _as_real_scalar("max_gradient_norm", max_gradient_norm)
    if max_norm <= 0.0:
        raise ValueError("max_gradient_norm must be finite and positive")
    return max_norm


def _clip_gradient(
    gradient: NDArray[np.float64],
    trainable: NDArray[np.bool_],
    *,
    max_gradient_norm: float | None,
) -> NDArray[np.float64]:
    """Clip trainable gradient entries to an optional Euclidean norm cap."""

    max_norm = _validate_max_gradient_norm(max_gradient_norm)
    clipped = gradient.copy()
    if max_norm is None or not np.any(trainable):
        typed_clipped: NDArray[np.float64] = clipped
        return typed_clipped
    trainable_norm = float(np.linalg.norm(clipped[trainable], ord=2))
    if trainable_norm > max_norm:
        clipped[trainable] *= max_norm / trainable_norm
    clipped_gradient: NDArray[np.float64] = clipped
    return clipped_gradient


__all__ = [
    "_as_complex_step_scalar",
    "_as_forward_mode_scalar",
    "_as_reverse_mode_scalar",
    "_as_scalar",
    "_as_vector_output",
    "_clip_gradient",
    "_normalise_bounds",
    "_normalise_parameters",
    "_project_bounds",
    "_reverse_topological_order",
    "_validate_max_gradient_norm",
]
