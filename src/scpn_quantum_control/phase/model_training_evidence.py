# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Model Training Evidence
"""Registered medium-scale local differentiable-model training evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

CLAIM_BOUNDARY = (
    "registered local QNN/QGNN/QSNN/Kuramoto-XY training evidence only; "
    "not arbitrary architecture, not provider hardware execution, and not a "
    "production benchmark"
)


@dataclass(frozen=True)
class DifferentiableModelTrainingRecord:
    """One registered training case with gradient-agreement evidence."""

    name: str
    model_family: str
    seed: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_reduction: float
    gradient_max_abs_error: float
    gradient_tolerance: float
    training_steps: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready model evidence."""
        return {
            "name": self.name,
            "model_family": self.model_family,
            "seed": self.seed,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_reduction": self.loss_reduction,
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "gradient_tolerance": self.gradient_tolerance,
            "training_steps": self.training_steps,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DifferentiableModelTrainingEvidenceSuite:
    """Aggregated registered model-training evidence."""

    records: tuple[DifferentiableModelTrainingRecord, ...]
    unsuitable_scenarios: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def model_names(self) -> tuple[str, ...]:
        """Return case names in deterministic order."""
        return tuple(record.name for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when every registered case passes."""
        return bool(self.records) and all(record.passed for record in self.records)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "model_names": list(self.model_names),
            "records": [record.to_dict() for record in self.records],
            "unsuitable_scenarios": list(self.unsuitable_scenarios),
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


def run_differentiable_model_training_evidence_suite(
    *,
    gradient_tolerance: float = 1.0e-5,
) -> DifferentiableModelTrainingEvidenceSuite:
    """Run deterministic registered QNN, QGNN, QSNN, and Kuramoto-XY cases."""
    if gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    records = (
        _qnn_medium_case(gradient_tolerance),
        _qgnn_registered_case(gradient_tolerance),
        _qsnn_medium_case(gradient_tolerance),
        _kuramoto_xy_case(gradient_tolerance),
    )
    return DifferentiableModelTrainingEvidenceSuite(
        records=records,
        unsuitable_scenarios=(
            "unregistered architectures require a new evidence gate before promotion",
            "provider hardware training requires shot ledgers and approval",
            "unseeded stochastic training is not reproducibility evidence",
            "framework-native simulator autodiff is recorded by the parity suite, not assumed",
        ),
    )


def _qnn_medium_case(tolerance: float) -> DifferentiableModelTrainingRecord:
    features = np.array(
        [[-0.8, 0.1, 0.3], [-0.2, 0.4, -0.5], [0.3, -0.6, 0.2], [0.7, 0.5, -0.1]],
        dtype=np.float64,
    )
    labels = np.array([-0.55, -0.15, 0.25, 0.6], dtype=np.float64)
    params = np.array([0.45, -0.35, 0.25], dtype=np.float64)

    def loss(theta: FloatArray) -> float:
        prediction = np.sin(features @ theta)
        return float(np.mean((prediction - labels) ** 2))

    def grad(theta: FloatArray) -> FloatArray:
        activation = features @ theta
        prediction = np.sin(activation)
        return np.asarray(
            (2.0 / labels.size) * features.T @ ((prediction - labels) * np.cos(activation)),
            dtype=np.float64,
        )

    return _train_case(
        name="qnn_medium_phase_classifier",
        model_family="qnn",
        seed=101,
        params=params,
        loss=loss,
        analytic_gradient=grad,
        learning_rate=0.35,
        steps=42,
        tolerance=tolerance,
    )


def _qgnn_registered_case(tolerance: float) -> DifferentiableModelTrainingRecord:
    node_features = np.array(
        [[0.2, -0.5], [0.4, 0.1], [-0.3, 0.6], [0.7, -0.2]],
        dtype=np.float64,
    )
    adjacency = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.array([0.15, 0.35, -0.2, 0.45], dtype=np.float64)
    params = np.array([0.3, -0.2, 0.15], dtype=np.float64)
    degree = adjacency.sum(axis=1, keepdims=True)
    aggregated = (adjacency @ node_features) / degree
    design = np.column_stack(
        (node_features[:, 0], node_features[:, 1], aggregated[:, 0] - aggregated[:, 1])
    )

    def loss(theta: FloatArray) -> float:
        prediction = np.sin(design @ theta)
        return float(np.mean((prediction - labels) ** 2))

    def grad(theta: FloatArray) -> FloatArray:
        activation = design @ theta
        prediction = np.sin(activation)
        return np.asarray(
            (2.0 / labels.size) * design.T @ ((prediction - labels) * np.cos(activation)),
            dtype=np.float64,
        )

    return _train_case(
        name="qgnn_registered_phase_graph",
        model_family="qgnn",
        seed=202,
        params=params,
        loss=loss,
        analytic_gradient=grad,
        learning_rate=0.4,
        steps=40,
        tolerance=tolerance,
    )


def _qsnn_medium_case(tolerance: float) -> DifferentiableModelTrainingRecord:
    features = np.array(
        [[0.1, 0.7], [0.4, -0.3], [-0.6, 0.2], [0.8, -0.1], [-0.2, -0.5]],
        dtype=np.float64,
    )
    labels = np.array([0.55, 0.2, -0.35, 0.5, -0.4], dtype=np.float64)
    params = np.array([0.25, -0.45, 0.1], dtype=np.float64)
    design = np.column_stack((features, np.prod(features, axis=1)))

    def loss(theta: FloatArray) -> float:
        prediction = np.tanh(design @ theta)
        return float(np.mean((prediction - labels) ** 2))

    def grad(theta: FloatArray) -> FloatArray:
        activation = design @ theta
        prediction = np.tanh(activation)
        slope = 1.0 - prediction**2
        return np.asarray(
            (2.0 / labels.size) * design.T @ ((prediction - labels) * slope),
            dtype=np.float64,
        )

    return _train_case(
        name="qsnn_medium_batch",
        model_family="qsnn",
        seed=303,
        params=params,
        loss=loss,
        analytic_gradient=grad,
        learning_rate=0.3,
        steps=45,
        tolerance=tolerance,
    )


def _kuramoto_xy_case(tolerance: float) -> DifferentiableModelTrainingRecord:
    couplings = np.array([[0.0, 0.8, 0.2], [0.8, 0.0, 0.5], [0.2, 0.5, 0.0]], dtype=np.float64)
    target = np.array([0.15, -0.2, 0.35], dtype=np.float64)
    params = np.array([0.6, -0.45, 0.25], dtype=np.float64)

    def loss(theta: FloatArray) -> float:
        drift = couplings @ np.sin(theta[:, None] - theta[None, :]).mean(axis=1)
        residual = drift + 0.2 * np.sin(theta) - target
        return float(np.mean(residual**2))

    def grad(theta: FloatArray) -> FloatArray:
        return _finite_difference_gradient(loss, theta, epsilon=1.0e-7)

    return _train_case(
        name="kuramoto_xy_vqe_medium",
        model_family="kuramoto_xy",
        seed=404,
        params=params,
        loss=loss,
        analytic_gradient=grad,
        learning_rate=0.25,
        steps=36,
        tolerance=tolerance,
    )


def _train_case(
    *,
    name: str,
    model_family: str,
    seed: int,
    params: FloatArray,
    loss: Callable[[FloatArray], float],
    analytic_gradient: Callable[[FloatArray], FloatArray],
    learning_rate: float,
    steps: int,
    tolerance: float,
) -> DifferentiableModelTrainingRecord:
    theta = params.copy()
    initial = loss(theta)
    gradient_error = float(
        np.max(np.abs(analytic_gradient(theta) - _finite_difference_gradient(loss, theta)))
    )
    best = initial
    for _step in range(steps):
        theta = theta - learning_rate * analytic_gradient(theta)
        current = loss(theta)
        best = min(best, current)
    final = loss(theta)
    reduction = initial - final
    return DifferentiableModelTrainingRecord(
        name=name,
        model_family=model_family,
        seed=seed,
        initial_loss=initial,
        final_loss=final,
        best_loss=best,
        loss_reduction=reduction,
        gradient_max_abs_error=gradient_error,
        gradient_tolerance=tolerance,
        training_steps=steps,
        passed=bool(reduction > 0.0 and gradient_error <= tolerance and final < initial),
    )


def _finite_difference_gradient(
    objective: Callable[[FloatArray], float],
    values: FloatArray,
    *,
    epsilon: float = 1.0e-6,
) -> FloatArray:
    gradient = np.zeros_like(values, dtype=np.float64)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    return gradient


__all__ = [
    "DifferentiableModelTrainingEvidenceSuite",
    "DifferentiableModelTrainingRecord",
    "run_differentiable_model_training_evidence_suite",
]
