# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Loss Landscapes
"""Deterministic loss-landscape evidence for bounded phase-QNN training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]

EVIDENCE_CLASS = "local_deterministic_qnn_loss_landscape"
CLAIM_BOUNDARY = (
    "local deterministic bounded phase-QNN loss-landscape evidence; not hardware, "
    "not finite-shot training, not isolated benchmark evidence, and not arbitrary "
    "QNN/QGNN/QSNN architecture evidence"
)


@dataclass(frozen=True)
class _QNNLossLandscapeCase:
    name: str
    features: FloatArray
    labels: FloatArray
    center_params: FloatArray


@dataclass(frozen=True)
class ParameterShiftQNNLossLandscapePoint:
    """One bounded QNN loss-landscape grid point."""

    params: tuple[float, ...]
    loss: float
    gradient: tuple[float, ...]
    gradient_norm: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready point evidence."""
        return {
            "params": list(self.params),
            "loss": self.loss,
            "gradient": list(self.gradient),
            "gradient_norm": self.gradient_norm,
        }


@dataclass(frozen=True)
class ParameterShiftQNNLossLandscapeCaseResult:
    """Loss-landscape evidence for one bounded phase-QNN case."""

    name: str
    n_samples: int
    n_features: int
    center_params: tuple[float, ...]
    grid_radius: float
    points_per_axis: int
    axis_values: tuple[tuple[float, ...], ...]
    points: tuple[ParameterShiftQNNLossLandscapePoint, ...]
    min_loss_span: float
    evidence_class: str
    claim_boundary: str
    production_benchmark: bool

    @property
    def point_count(self) -> int:
        """Return the number of evaluated grid points."""
        return len(self.points)

    @property
    def losses(self) -> tuple[float, ...]:
        """Return all sampled losses."""
        return tuple(point.loss for point in self.points)

    @property
    def min_loss(self) -> float:
        """Return the smallest sampled loss."""
        return float(min(self.losses))

    @property
    def max_loss(self) -> float:
        """Return the largest sampled loss."""
        return float(max(self.losses))

    @property
    def center_loss(self) -> float:
        """Return the loss at the declared center parameters."""
        center = tuple(float(value) for value in self.center_params)
        for point in self.points:
            if point.params == center:
                return point.loss
        raise RuntimeError("loss landscape grid does not contain its center point")

    @property
    def loss_span(self) -> float:
        """Return max(loss) - min(loss) across the grid."""
        return float(self.max_loss - self.min_loss)

    @property
    def min_gradient_norm(self) -> float:
        """Return the smallest sampled gradient norm."""
        return float(min(point.gradient_norm for point in self.points))

    @property
    def max_gradient_norm(self) -> float:
        """Return the largest sampled gradient norm."""
        return float(max(point.gradient_norm for point in self.points))

    @property
    def argmin_params(self) -> tuple[float, ...]:
        """Return the parameters at the smallest sampled loss."""
        best = min(self.points, key=lambda point: point.loss)
        return best.params

    @property
    def loss_span_passed(self) -> bool:
        """Return whether the grid has enough observable loss variation."""
        return self.loss_span >= self.min_loss_span

    @property
    def passed(self) -> bool:
        """Return whether this loss-landscape case passed."""
        return self.loss_span_passed and all(np.isfinite(point.loss) for point in self.points)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready loss-landscape evidence."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "center_params": list(self.center_params),
            "grid_radius": self.grid_radius,
            "points_per_axis": self.points_per_axis,
            "axis_values": [list(axis) for axis in self.axis_values],
            "point_count": self.point_count,
            "min_loss": self.min_loss,
            "max_loss": self.max_loss,
            "center_loss": self.center_loss,
            "loss_span": self.loss_span,
            "min_loss_span": self.min_loss_span,
            "min_gradient_norm": self.min_gradient_norm,
            "max_gradient_norm": self.max_gradient_norm,
            "argmin_params": list(self.argmin_params),
            "loss_span_passed": self.loss_span_passed,
            "passed": self.passed,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "production_benchmark": self.production_benchmark,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class ParameterShiftQNNLossLandscapeSuiteResult:
    """Bundled deterministic bounded-QNN loss-landscape evidence."""

    cases: tuple[ParameterShiftQNNLossLandscapeCaseResult, ...]
    evidence_class: str
    claim_boundary: str
    production_benchmark: bool

    @property
    def passed(self) -> bool:
        """Return whether every loss-landscape case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of landscape cases."""
        return len(self.cases)

    @property
    def passed_count(self) -> int:
        """Return the number of passing landscape cases."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed_count(self) -> int:
        """Return the number of failing landscape cases."""
        return self.case_count - self.passed_count

    @property
    def total_point_count(self) -> int:
        """Return the number of sampled grid points across cases."""
        return sum(case.point_count for case in self.cases)

    def case_by_name(self, name: str) -> ParameterShiftQNNLossLandscapeCaseResult:
        """Return a loss-landscape case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN loss landscape case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_point_count": self.total_point_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "production_benchmark": self.production_benchmark,
            "cases": [case.to_dict() for case in self.cases],
        }


def _default_cases() -> tuple[_QNNLossLandscapeCase, ...]:
    return (
        _QNNLossLandscapeCase(
            name="single_feature_phase_flip",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            center_params=np.array([0.8], dtype=np.float64),
        ),
        _QNNLossLandscapeCase(
            name="two_feature_phase_flip",
            features=np.array([[0.0, 0.0], [np.pi, np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            center_params=np.array([0.8, 0.8], dtype=np.float64),
        ),
    )


def _case_map() -> dict[str, _QNNLossLandscapeCase]:
    return {case.name: case for case in _default_cases()}


def _selected_cases(case_names: Sequence[str] | None) -> tuple[_QNNLossLandscapeCase, ...]:
    cases = _case_map()
    if case_names is None:
        return tuple(cases.values())
    selected: list[_QNNLossLandscapeCase] = []
    for name in case_names:
        if name not in cases:
            raise ValueError(f"unknown QNN loss landscape case: {name}")
        selected.append(cases[name])
    return tuple(selected)


def _as_positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


def _as_non_negative_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _as_points_per_axis(value: int) -> int:
    integer = int(value)
    if integer < 3 or integer % 2 == 0:
        raise ValueError("points_per_axis must be an odd integer greater than or equal to 3")
    return integer


def _as_max_features(value: int) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError("max_features must be a positive integer")
    return integer


def _axis_values(
    center: FloatArray, *, grid_radius: float, points_per_axis: int
) -> tuple[tuple[float, ...], ...]:
    offsets = np.linspace(-grid_radius, grid_radius, points_per_axis, dtype=np.float64)
    return tuple(tuple(float(value + offset) for offset in offsets) for value in center)


def _landscape_point(
    case: _QNNLossLandscapeCase,
    params: Sequence[float],
) -> ParameterShiftQNNLossLandscapePoint:
    vector = np.asarray(params, dtype=np.float64)
    loss = parameter_shift_qnn_classifier_loss(case.features, case.labels, vector)
    gradient = parameter_shift_qnn_classifier_gradient(case.features, case.labels, vector)
    gradient_norm = float(np.linalg.norm(gradient))
    return ParameterShiftQNNLossLandscapePoint(
        params=tuple(float(value) for value in vector),
        loss=loss,
        gradient=tuple(float(value) for value in gradient),
        gradient_norm=gradient_norm,
    )


def _case_result(
    case: _QNNLossLandscapeCase,
    *,
    grid_radius: float,
    points_per_axis: int,
    min_loss_span: float,
    max_features: int,
) -> ParameterShiftQNNLossLandscapeCaseResult:
    if case.center_params.size > max_features:
        raise ValueError(
            f"QNN loss landscape case {case.name!r} has {case.center_params.size} features; "
            f"max_features={max_features} would create an unreviewed grid"
        )
    axes = _axis_values(
        case.center_params, grid_radius=grid_radius, points_per_axis=points_per_axis
    )
    points = tuple(_landscape_point(case, params) for params in product(*axes))
    return ParameterShiftQNNLossLandscapeCaseResult(
        name=case.name,
        n_samples=case.features.shape[0],
        n_features=case.features.shape[1],
        center_params=tuple(float(value) for value in case.center_params),
        grid_radius=grid_radius,
        points_per_axis=points_per_axis,
        axis_values=axes,
        points=points,
        min_loss_span=min_loss_span,
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        production_benchmark=False,
    )


def run_parameter_shift_qnn_loss_landscape_suite(
    *,
    case_names: Sequence[str] | None = None,
    grid_radius: float = 0.25,
    points_per_axis: int = 5,
    min_loss_span: float = 1e-6,
    max_features: int = 2,
) -> ParameterShiftQNNLossLandscapeSuiteResult:
    """Run deterministic bounded phase-QNN loss-landscape evidence."""
    checked_grid_radius = _as_positive_float("grid_radius", grid_radius)
    checked_points_per_axis = _as_points_per_axis(points_per_axis)
    checked_min_loss_span = _as_non_negative_float("min_loss_span", min_loss_span)
    checked_max_features = _as_max_features(max_features)
    cases = tuple(
        _case_result(
            case,
            grid_radius=checked_grid_radius,
            points_per_axis=checked_points_per_axis,
            min_loss_span=checked_min_loss_span,
            max_features=checked_max_features,
        )
        for case in _selected_cases(case_names)
    )
    return ParameterShiftQNNLossLandscapeSuiteResult(
        cases=cases,
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        production_benchmark=False,
    )
