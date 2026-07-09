# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable synchronisation objectives
"""Differentiable synchronisation losses for phase-control objectives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .objectives import ComposedPhaseObjective, ObjectiveTerm

FloatArray: TypeAlias = NDArray[np.float64]

SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY = (
    "local differentiable synchronisation objective over explicit phase vectors; "
    "Kuramoto order-parameter, phase-locking, and cluster-synchronisation losses "
    "use analytic gradients and do not claim provider execution, hardware "
    "sampling, benchmark performance, or oscillatools package ownership"
)


def kuramoto_order_parameter(params: ArrayLike) -> float:
    """Return the global Kuramoto order parameter magnitude."""
    phases = _as_phase_vector("params", params)
    return _order_parameter_and_gradient(phases, min_order_parameter=0.0)[0]


def kuramoto_order_parameter_gradient(
    params: ArrayLike,
    *,
    min_order_parameter: float = 1.0e-12,
) -> FloatArray:
    """Return the analytic gradient of the global order parameter."""
    phases = _as_phase_vector("params", params)
    _order, gradient = _order_parameter_and_gradient(
        phases,
        min_order_parameter=min_order_parameter,
    )
    return gradient


def kuramoto_order_parameter_target_term(
    width: int,
    *,
    target: float = 1.0,
    min_order_parameter: float = 1.0e-12,
    term_weight: float = 1.0,
    name: str = "kuramoto_order_parameter_target",
) -> ObjectiveTerm:
    """Build ``0.5 * (R(theta) - target)^2`` for the Kuramoto order parameter."""
    size = _positive_width(width)
    target_value = _unit_interval("target", target)
    threshold = _non_negative_scalar("min_order_parameter", min_order_parameter)

    def value(params: FloatArray) -> float:
        phases = _as_phase_vector("params", params, width=size)
        order = _order_parameter_and_gradient(phases, min_order_parameter=0.0)[0]
        return float(0.5 * (order - target_value) ** 2)

    def gradient(params: FloatArray) -> FloatArray:
        phases = _as_phase_vector("params", params, width=size)
        order, order_gradient = _order_parameter_and_gradient(
            phases,
            min_order_parameter=threshold,
        )
        return cast(
            FloatArray,
            ((order - target_value) * order_gradient).astype(np.float64, copy=False),
        )

    return ObjectiveTerm(
        name=name,
        kind="synchronisation_order_parameter",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description=(
            "analytic global Kuramoto order-parameter target loss; not a "
            "parameter-shift quantum term"
        ),
    )


def phase_locking_target_term(
    width: int,
    pairs: Sequence[tuple[int, int]],
    *,
    offsets: ArrayLike | float = 0.0,
    term_weight: float = 1.0,
    name: str = "phase_locking_target",
) -> ObjectiveTerm:
    """Build a periodic pairwise phase-locking loss over selected pairs."""
    size = _positive_width(width)
    pair_tuple = _validate_pairs(pairs, size)
    offset_vector = _broadcast_vector("offsets", offsets, len(pair_tuple))

    def value(params: FloatArray) -> float:
        phases = _as_phase_vector("params", params, width=size)
        total = 0.0
        for index, (left, right) in enumerate(pair_tuple):
            total += 1.0 - np.cos(phases[left] - phases[right] - offset_vector[index])
        return float(total / len(pair_tuple))

    def gradient(params: FloatArray) -> FloatArray:
        phases = _as_phase_vector("params", params, width=size)
        grad = np.zeros(size, dtype=np.float64)
        scale = 1.0 / len(pair_tuple)
        for index, (left, right) in enumerate(pair_tuple):
            difference = phases[left] - phases[right] - offset_vector[index]
            contribution = float(np.sin(difference) * scale)
            grad[left] += contribution
            grad[right] -= contribution
        return grad

    return ObjectiveTerm(
        name=name,
        kind="phase_locking",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="parameter_shift",
        parameter_shift_compatible=True,
        description="periodic pairwise phase-locking loss with exact sinusoidal gradient",
    )


def cluster_synchronisation_target_term(
    width: int,
    clusters: Sequence[Sequence[int]],
    *,
    targets: ArrayLike | float = 1.0,
    min_order_parameter: float = 1.0e-12,
    term_weight: float = 1.0,
    name: str = "cluster_synchronisation_target",
) -> ObjectiveTerm:
    """Build a mean cluster-order target loss over disjoint phase clusters."""
    size = _positive_width(width)
    cluster_tuple = _validate_clusters(clusters, size)
    target_vector = _broadcast_unit_interval_vector("targets", targets, len(cluster_tuple))
    threshold = _non_negative_scalar("min_order_parameter", min_order_parameter)

    def value(params: FloatArray) -> float:
        phases = _as_phase_vector("params", params, width=size)
        total = 0.0
        for index, cluster in enumerate(cluster_tuple):
            order = _order_parameter_and_gradient(
                phases[np.asarray(cluster, dtype=np.int64)],
                min_order_parameter=0.0,
            )[0]
            total += 0.5 * (order - target_vector[index]) ** 2
        return float(total / len(cluster_tuple))

    def gradient(params: FloatArray) -> FloatArray:
        phases = _as_phase_vector("params", params, width=size)
        grad = np.zeros(size, dtype=np.float64)
        scale = 1.0 / len(cluster_tuple)
        for index, cluster in enumerate(cluster_tuple):
            indices = np.asarray(cluster, dtype=np.int64)
            order, local_gradient = _order_parameter_and_gradient(
                phases[indices],
                min_order_parameter=threshold,
            )
            grad[indices] += scale * (order - target_vector[index]) * local_gradient
        return grad

    return ObjectiveTerm(
        name=name,
        kind="cluster_synchronisation",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description=(
            "analytic cluster Kuramoto order-parameter target loss; not a "
            "provider or hardware synchronisation claim"
        ),
    )


def build_synchronisation_objective(
    width: int,
    *,
    order_parameter_target: float | None = 1.0,
    order_parameter_weight: float = 1.0,
    phase_locking_pairs: Sequence[tuple[int, int]] | None = None,
    phase_locking_offsets: ArrayLike | float = 0.0,
    phase_locking_weight: float = 0.0,
    clusters: Sequence[Sequence[int]] | None = None,
    cluster_targets: ArrayLike | float = 1.0,
    cluster_weight: float = 0.0,
    min_order_parameter: float = 1.0e-12,
    name: str = "synchronisation_objective",
) -> ComposedPhaseObjective:
    """Build a composed differentiable synchronisation objective."""
    size = _positive_width(width)
    terms: list[ObjectiveTerm] = []
    if order_parameter_target is not None and order_parameter_weight > 0.0:
        terms.append(
            kuramoto_order_parameter_target_term(
                size,
                target=order_parameter_target,
                min_order_parameter=min_order_parameter,
                term_weight=order_parameter_weight,
            )
        )
    if phase_locking_pairs is not None and phase_locking_weight > 0.0:
        terms.append(
            phase_locking_target_term(
                size,
                phase_locking_pairs,
                offsets=phase_locking_offsets,
                term_weight=phase_locking_weight,
            )
        )
    if clusters is not None and cluster_weight > 0.0:
        terms.append(
            cluster_synchronisation_target_term(
                size,
                clusters,
                targets=cluster_targets,
                min_order_parameter=min_order_parameter,
                term_weight=cluster_weight,
            )
        )
    return ComposedPhaseObjective(
        terms=tuple(terms),
        name=name,
        claim_boundary=SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY,
    )


def _order_parameter_and_gradient(
    phases: FloatArray,
    *,
    min_order_parameter: float,
) -> tuple[float, FloatArray]:
    complex_mean = np.mean(np.exp(1j * phases))
    order = float(np.abs(complex_mean))
    if min_order_parameter > 0.0 and order <= min_order_parameter:
        raise ValueError(
            "Kuramoto order-parameter gradient is singular at the incoherent boundary"
        )
    phase = float(np.angle(complex_mean))
    gradient = np.sin(phase - phases) / phases.size
    return order, cast(FloatArray, gradient.astype(np.float64, copy=False))


def _as_phase_vector(name: str, values: ArrayLike, *, width: int | None = None) -> FloatArray:
    phases = np.asarray(values, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional phase vector")
    if width is not None and phases.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {phases.shape}")
    if not np.all(np.isfinite(phases)):
        raise ValueError(f"{name} must contain only finite phases")
    return phases.astype(np.float64, copy=True)


def _positive_width(width: int) -> int:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    return width


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _non_negative_scalar(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _unit_interval(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value)
    if scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return scalar


def _broadcast_vector(name: str, values: ArrayLike | float, width: int) -> FloatArray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim == 0:
        return np.full(width, float(raw), dtype=np.float64)
    return _as_phase_vector(name, raw, width=width)


def _broadcast_unit_interval_vector(
    name: str,
    values: ArrayLike | float,
    width: int,
) -> FloatArray:
    vector = _broadcast_vector(name, values, width)
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return vector


def _validate_pairs(
    pairs: Sequence[tuple[int, int]],
    width: int,
) -> tuple[tuple[int, int], ...]:
    if not pairs:
        raise ValueError("phase-locking pairs must contain at least one pair")
    validated: list[tuple[int, int]] = []
    for left, right in pairs:
        if isinstance(left, bool) or isinstance(right, bool):
            raise ValueError("phase-locking pair indices must be integers")
        if left == right:
            raise ValueError("phase-locking pairs must reference distinct parameters")
        if left < 0 or right < 0 or left >= width or right >= width:
            raise ValueError("phase-locking pair index out of bounds")
        validated.append((int(left), int(right)))
    return tuple(validated)


def _validate_clusters(
    clusters: Sequence[Sequence[int]],
    width: int,
) -> tuple[tuple[int, ...], ...]:
    if not clusters:
        raise ValueError("clusters must contain at least one cluster")
    seen: set[int] = set()
    validated: list[tuple[int, ...]] = []
    for cluster in clusters:
        indices = tuple(cluster)
        if len(indices) < 2:
            raise ValueError("each cluster must contain at least two indices")
        if len(set(indices)) != len(indices):
            raise ValueError("cluster indices must be unique within each cluster")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("cluster indices must be integers")
            if index < 0 or index >= width:
                raise ValueError("cluster index out of bounds")
            if index in seen:
                raise ValueError("clusters must be disjoint")
            seen.add(index)
        validated.append(indices)
    return tuple(validated)


__all__ = [
    "SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY",
    "build_synchronisation_objective",
    "cluster_synchronisation_target_term",
    "kuramoto_order_parameter",
    "kuramoto_order_parameter_gradient",
    "kuramoto_order_parameter_target_term",
    "phase_locking_target_term",
]
