# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-Shift QNN Training
"""Deterministic parameter-shift training for bounded phase QNN classifiers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from .param_shift import multi_frequency_parameter_shift_rule, parameter_shift_gradient

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
GradientCallable: TypeAlias = Callable[[FloatArray], ArrayLike]


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


@dataclass(frozen=True)
class ParameterShiftQNNExternalGradientAgreement:
    """Agreement evidence for a named external QNN gradient source."""

    name: str
    gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    source_class: str = "caller_supplied_gradient"
    native_framework_autodiff: bool = False
    claim_boundary: str = (
        "external QNN gradient agreement compares a named caller-supplied "
        "gradient with the bounded parameter-shift reference; it does not "
        "claim native framework autodiff through simulator kernels"
    )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready external-gradient agreement evidence."""
        return {
            "name": self.name,
            "gradient": self.gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "source_class": self.source_class,
            "native_framework_autodiff": self.native_framework_autodiff,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ParameterShiftQNNGradientVerificationResult:
    """QNN gradient verification against finite differences and adapters."""

    loss: float
    parameter_shift_gradient: FloatArray
    finite_difference_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    finite_difference_step: float
    passed: bool
    external_agreements: tuple[ParameterShiftQNNExternalGradientAgreement, ...]
    method: str
    shift_terms: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready QNN gradient verification evidence."""
        return {
            "loss": self.loss,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "finite_difference_gradient": self.finite_difference_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "finite_difference_step": self.finite_difference_step,
            "passed": self.passed,
            "external_agreements": [agreement.to_dict() for agreement in self.external_agreements],
            "method": self.method,
            "shift_terms": self.shift_terms,
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


def _as_positive_step(finite_difference_step: float) -> float:
    step = float(finite_difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite_difference_step must be a finite positive scalar")
    return step


def _as_non_negative_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return value


def _as_external_gradient_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("external gradient name must be non-empty")
    if any(character.isspace() for character in normalized):
        raise ValueError("external gradient name must not contain whitespace")
    return normalized


def _as_external_gradient(name: str, values: ArrayLike, *, width: int) -> FloatArray:
    gradient = np.asarray(values, dtype=float)
    if gradient.ndim != 1 or gradient.shape != (width,):
        raise ValueError(f"external gradient {name!r} must have shape ({width},)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"external gradient {name!r} must contain only finite values")
    return gradient.astype(np.float64, copy=True)


def _phase_qnn_probabilities(features: FloatArray, params: FloatArray) -> FloatArray:
    if params.shape != (features.shape[1],):
        raise ValueError("initial_params must match the feature width")
    probabilities = 0.5 * (1.0 - np.cos(features + params[None, :]))
    averaged = np.mean(probabilities, axis=1)
    clipped = np.clip(averaged, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, clipped)


def _qnn_classifier_loss(
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameters: FloatArray,
) -> float:
    probabilities = _phase_qnn_probabilities(feature_matrix, parameters)
    residual = probabilities - label_vector
    return float(np.mean(residual * residual))


def _central_finite_difference_gradient(
    objective: Callable[[FloatArray], float],
    params: FloatArray,
    *,
    step: float,
) -> FloatArray:
    gradient = np.zeros_like(params, dtype=np.float64)
    for index in range(params.size):
        forward = params.copy()
        backward = params.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (objective(forward) - objective(backward)) / (2.0 * step)
    return gradient


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


def parameter_shift_qnn_classifier_loss(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
) -> float:
    """Return the full-batch MSE loss for the bounded phase QNN classifier."""
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameters = _as_parameter_vector(params, n_features=feature_matrix.shape[1])
    return _qnn_classifier_loss(feature_matrix, label_vector, parameters)


def parameter_shift_qnn_classifier_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
) -> FloatArray:
    """Return the multi-frequency parameter-shift gradient for QNN MSE loss."""
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameters = _as_parameter_vector(params, n_features=feature_matrix.shape[1])
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(candidate: FloatArray) -> float:
        return _qnn_classifier_loss(feature_matrix, label_vector, candidate)

    return cast(FloatArray, parameter_shift_gradient(objective, parameters, rule=rule))


def verify_parameter_shift_qnn_classifier_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    finite_difference_step: float = 1e-6,
    tolerance: float = 1e-5,
    external_gradients: Mapping[str, GradientCallable] | None = None,
    external_tolerance: float | None = None,
) -> ParameterShiftQNNGradientVerificationResult:
    """Verify bounded phase-QNN gradients against independent references.

    The primary reference is a central finite-difference replay of the same
    deterministic QNN loss. Optional named external-gradient callables can be
    used to record JAX, PennyLane, PyTorch, TensorFlow, or other adapter
    agreement without claiming automatic conversion into those frameworks.
    """
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameters = _as_parameter_vector(params, n_features=feature_matrix.shape[1])
    step = _as_positive_step(finite_difference_step)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    external_tolerance_value = (
        tolerance_value
        if external_tolerance is None
        else _as_non_negative_tolerance(external_tolerance)
    )
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(candidate: FloatArray) -> float:
        return _qnn_classifier_loss(feature_matrix, label_vector, candidate)

    shift_gradient = cast(
        FloatArray,
        parameter_shift_gradient(objective, parameters, rule=rule),
    )
    finite_difference_gradient = _central_finite_difference_gradient(
        objective,
        parameters,
        step=step,
    )
    delta = shift_gradient - finite_difference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))

    agreements: list[ParameterShiftQNNExternalGradientAgreement] = []
    for raw_name, gradient_callable in (external_gradients or {}).items():
        name = _as_external_gradient_name(raw_name)
        external_gradient = _as_external_gradient(
            name,
            gradient_callable(parameters.copy()),
            width=parameters.size,
        )
        external_delta = shift_gradient - external_gradient
        external_max_abs_error = (
            float(np.max(np.abs(external_delta))) if external_delta.size else 0.0
        )
        agreements.append(
            ParameterShiftQNNExternalGradientAgreement(
                name=name,
                gradient=external_gradient,
                max_abs_error=external_max_abs_error,
                l2_error=float(np.linalg.norm(external_delta, ord=2)),
                tolerance=external_tolerance_value,
                passed=external_max_abs_error <= external_tolerance_value,
            )
        )

    return ParameterShiftQNNGradientVerificationResult(
        loss=objective(parameters),
        parameter_shift_gradient=shift_gradient,
        finite_difference_gradient=finite_difference_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        finite_difference_step=step,
        passed=bool(
            max_abs_error <= tolerance_value and all(agreement.passed for agreement in agreements)
        ),
        external_agreements=tuple(agreements),
        method="multi_frequency_parameter_shift_qnn_gradient",
        shift_terms=len(rule.terms),
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
        return _qnn_classifier_loss(feature_matrix, label_vector, candidate)

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
    "ParameterShiftQNNExternalGradientAgreement",
    "ParameterShiftQNNGradientVerificationResult",
    "ParameterShiftQNNPredictionResult",
    "ParameterShiftQNNTrainingResult",
    "parameter_shift_qnn_classifier_gradient",
    "parameter_shift_qnn_classifier_loss",
    "predict_parameter_shift_qnn_classifier",
    "train_parameter_shift_qnn_classifier",
    "verify_parameter_shift_qnn_classifier_gradient",
]
