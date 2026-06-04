# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-Shift QNN Training
"""Deterministic parameter-shift training for bounded phase QNN classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)
from .param_shift import multi_frequency_parameter_shift_rule

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]


@dataclass(frozen=True)
class ParameterShiftQNNPredictionResult:
    """Prediction evidence for the bounded phase QNN classifier."""

    probabilities: FloatArray
    predicted_labels: IntArray
    labels: FloatArray | None
    accuracy: float | None
    decision_threshold: float

    @property
    def n_samples(self) -> int:
        """Return the number of predicted samples."""
        return int(self.probabilities.size)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready prediction evidence."""
        return {
            "probabilities": self.probabilities.tolist(),
            "predicted_labels": self.predicted_labels.tolist(),
            "labels": None if self.labels is None else self.labels.tolist(),
            "accuracy": self.accuracy,
            "decision_threshold": self.decision_threshold,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class ParameterShiftQNNTrainingResult:
    """Full-batch QNN training result with parameter-shift provenance."""

    training: ParameterShiftTrainingResult
    certificate: ParameterShiftTrainingCertificate
    prediction: ParameterShiftQNNPredictionResult
    n_samples: int
    n_features: int
    backend: str
    method: str

    @property
    def best_loss(self) -> float:
        """Return the best full-batch loss observed during training."""
        return self.training.best_value

    @property
    def loss_history(self) -> tuple[float, ...]:
        """Return the optimizer value history as a loss history."""
        return self.training.value_history

    @property
    def best_params(self) -> FloatArray:
        """Return the best parameter vector found during training."""
        return cast(FloatArray, self.training.best_params.copy())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready QNN training evidence."""
        return {
            "training": self.training.to_dict(),
            "certificate": self.certificate.to_dict(),
            "prediction": self.prediction.to_dict(),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "backend": self.backend,
            "method": self.method,
            "best_loss": self.best_loss,
            "loss_history": list(self.loss_history),
            "best_params": self.best_params.tolist(),
        }


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array or a single-column matrix")
    if vector.shape != (n_samples,):
        raise ValueError("features and labels must have the same sample count")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in the closed interval [0, 1]")
    return vector.astype(np.float64, copy=True)


def _as_parameter_vector(initial_params: ArrayLike, *, n_features: int) -> FloatArray:
    vector = np.asarray(initial_params, dtype=float)
    if vector.ndim != 1:
        raise ValueError("initial_params must be a one-dimensional array")
    if vector.shape != (n_features,):
        raise ValueError(
            f"initial_params must have one trainable phase per feature, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial_params must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_threshold(decision_threshold: float) -> float:
    threshold = float(decision_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("decision_threshold must be a finite scalar in (0, 1)")
    return threshold


def _phase_qnn_probabilities(features: FloatArray, params: FloatArray) -> FloatArray:
    if params.shape != (features.shape[1],):
        raise ValueError("initial_params must match the feature width")
    probabilities = 0.5 * (1.0 - np.cos(features + params[None, :]))
    averaged = np.mean(probabilities, axis=1)
    clipped = np.clip(averaged, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, clipped)


def predict_parameter_shift_qnn_classifier(
    features: ArrayLike,
    params: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    decision_threshold: float = 0.5,
) -> ParameterShiftQNNPredictionResult:
    """Predict binary labels with the bounded phase QNN classifier.

    Each feature column is encoded as a phase offset and paired with one
    trainable phase parameter. The output probability is the average one-qubit
    ``0.5 * (1 - cos(feature + parameter))`` response, making the route
    deterministic, bounded, and compatible with explicit multi-frequency
    parameter-shift training when the MSE loss is used.
    """
    feature_matrix = _as_feature_matrix(features)
    parameters = _as_parameter_vector(params, n_features=feature_matrix.shape[1])
    threshold = _as_threshold(decision_threshold)
    label_vector = None
    if labels is not None:
        label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])

    probabilities = _phase_qnn_probabilities(feature_matrix, parameters)
    predicted = (probabilities >= threshold).astype(np.int_)
    accuracy = None
    if label_vector is not None:
        expected = (label_vector >= threshold).astype(np.int_)
        accuracy = float(np.mean(predicted == expected))

    return ParameterShiftQNNPredictionResult(
        probabilities=probabilities,
        predicted_labels=predicted,
        labels=label_vector,
        accuracy=accuracy,
        decision_threshold=threshold,
    )


def train_parameter_shift_qnn_classifier(
    features: ArrayLike,
    labels: ArrayLike,
    *,
    initial_params: ArrayLike,
    backend: str = "statevector_simulator",
    learning_rate: float = 0.25,
    max_steps: int = 80,
    gradient_tolerance: float = 1e-8,
    target_loss: float | None = None,
    target_loss_tolerance: float = 1e-8,
    min_loss_decrease: float | None = None,
    decision_threshold: float = 0.5,
) -> ParameterShiftQNNTrainingResult:
    """Train a small binary phase QNN with parameter-shift gradients.

    The classifier intentionally covers a bounded, auditable surface: local
    deterministic phase responses, full-batch MSE loss, one trainable phase per
    feature, and fail-closed backend policy through
    :func:`parameter_shift_gradient_descent`. The MSE objective introduces
    second harmonics, so the optimizer uses an explicit ``[1, 2]``
    multi-frequency parameter-shift rule instead of the standard two-point
    first-harmonic rule.
    """
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    params = _as_parameter_vector(initial_params, n_features=feature_matrix.shape[1])
    threshold = _as_threshold(decision_threshold)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(candidate: FloatArray) -> float:
        probabilities = _phase_qnn_probabilities(feature_matrix, candidate)
        residual = probabilities - label_vector
        return float(np.mean(residual * residual))

    training = parameter_shift_gradient_descent(
        objective,
        params,
        rule=rule,
        backend=backend,
        learning_rate=learning_rate,
        max_steps=max_steps,
        gradient_tolerance=gradient_tolerance,
    )
    certificate = validate_parameter_shift_training(
        training,
        gradient_tolerance=gradient_tolerance,
        target_value=target_loss,
        target_value_tolerance=target_loss_tolerance,
        min_decrease=min_loss_decrease,
    )
    prediction = predict_parameter_shift_qnn_classifier(
        feature_matrix,
        training.best_params,
        labels=label_vector,
        decision_threshold=threshold,
    )

    return ParameterShiftQNNTrainingResult(
        training=training,
        certificate=certificate,
        prediction=prediction,
        n_samples=feature_matrix.shape[0],
        n_features=feature_matrix.shape[1],
        backend=backend,
        method="multi_frequency_parameter_shift_qnn_classifier",
    )


__all__ = [
    "ParameterShiftQNNPredictionResult",
    "ParameterShiftQNNTrainingResult",
    "predict_parameter_shift_qnn_classifier",
    "train_parameter_shift_qnn_classifier",
]
