# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Training
"""Tests for phase/qnn_training.py parameter-shift QNN semantics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNGradientVerificationResult,
    ParameterShiftQNNPredictionResult,
    ParameterShiftQNNTrainingResult,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    predict_parameter_shift_qnn_classifier,
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)


def test_parameter_shift_qnn_classifier_converges_on_phase_separable_data() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)

    result = train_parameter_shift_qnn_classifier(
        features,
        labels,
        initial_params=np.array([0.8], dtype=float),
        learning_rate=0.7,
        max_steps=80,
        gradient_tolerance=1e-7,
        target_loss=0.0,
        target_loss_tolerance=1e-4,
    )

    assert isinstance(result, ParameterShiftQNNTrainingResult)
    assert result.n_samples == 2
    assert result.n_features == 1
    assert result.method == "multi_frequency_parameter_shift_qnn_classifier"
    assert result.training.method == "multi_frequency_parameter_shift"
    assert result.training.shift_terms == 2
    assert result.training.best_value < 1e-4
    assert result.certificate.monotone_accepted_values
    assert result.certificate.within_target_value_tolerance
    assert result.prediction.accuracy == 1.0
    assert tuple(result.prediction.predicted_labels.tolist()) == (0, 1)
    payload = result.to_dict()
    prediction_payload = payload["prediction"]
    assert isinstance(prediction_payload, dict)
    assert prediction_payload["accuracy"] == 1.0


def test_parameter_shift_qnn_prediction_is_bounded_and_thresholded() -> None:
    features = np.array([[0.0, np.pi], [np.pi, 0.0]], dtype=float)
    params = np.array([0.0, 0.0], dtype=float)
    labels = np.array([0.5, 0.5], dtype=float)

    prediction = predict_parameter_shift_qnn_classifier(
        features,
        params,
        labels=labels,
        decision_threshold=0.5,
    )

    assert isinstance(prediction, ParameterShiftQNNPredictionResult)
    np.testing.assert_allclose(prediction.probabilities, np.array([0.5, 0.5]))
    assert prediction.predicted_labels.shape == (2,)
    assert np.all((prediction.probabilities >= 0.0) & (prediction.probabilities <= 1.0))
    assert prediction.accuracy == 1.0
    assert prediction.to_dict()["n_samples"] == 2


def test_parameter_shift_qnn_gradient_matches_finite_difference_reference() -> None:
    features = np.array(
        [
            [0.2, -0.4],
            [1.1, 0.7],
            [-0.8, 0.3],
        ],
        dtype=float,
    )
    labels = np.array([0.0, 1.0, 0.25], dtype=float)
    params = np.array([0.4, -0.2], dtype=float)

    loss = parameter_shift_qnn_classifier_loss(features, labels, params)
    gradient = parameter_shift_qnn_classifier_gradient(features, labels, params)
    verification = verify_parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params,
        finite_difference_step=1e-6,
        tolerance=2e-6,
    )

    assert loss == pytest.approx(0.20112073737315475)
    assert isinstance(verification, ParameterShiftQNNGradientVerificationResult)
    assert verification.loss == pytest.approx(loss)
    np.testing.assert_allclose(
        verification.parameter_shift_gradient,
        gradient,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        verification.parameter_shift_gradient,
        verification.finite_difference_gradient,
        rtol=0.0,
        atol=2e-6,
    )
    assert verification.passed
    assert verification.max_abs_error <= 2e-6
    assert verification.shift_terms == 2
    assert verification.method == "multi_frequency_parameter_shift_qnn_gradient"
    payload = verification.to_dict()
    assert payload["passed"] is True
    assert payload["shift_terms"] == 2


def test_parameter_shift_qnn_gradient_records_external_agreements() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)
    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)

    verification = verify_parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params,
        tolerance=1e-6,
        external_gradients={
            "jax": lambda _values: expected.copy(),
            "pennylane": lambda _values: expected + np.array([5e-8], dtype=float),
        },
    )

    assert verification.passed
    assert tuple(agreement.name for agreement in verification.external_agreements) == (
        "jax",
        "pennylane",
    )
    assert all(agreement.passed for agreement in verification.external_agreements)
    assert verification.to_dict()["external_agreements"]


def test_parameter_shift_qnn_training_fails_closed_for_hardware() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        train_parameter_shift_qnn_classifier(
            features,
            labels,
            initial_params=np.array([0.5], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_parameter_shift_qnn_rejects_invalid_inputs() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="same sample count"):
        train_parameter_shift_qnn_classifier(
            features,
            labels[:1],
            initial_params=np.array([0.5], dtype=float),
        )

    with pytest.raises(ValueError, match="initial_params"):
        predict_parameter_shift_qnn_classifier(
            features,
            np.array([0.5, 0.1], dtype=float),
        )

    with pytest.raises(ValueError, match="decision_threshold"):
        predict_parameter_shift_qnn_classifier(
            features,
            np.array([0.5], dtype=float),
            decision_threshold=1.5,
        )

    with pytest.raises(ValueError, match="labels"):
        train_parameter_shift_qnn_classifier(
            features,
            np.array([0.0, np.nan], dtype=float),
            initial_params=np.array([0.5], dtype=float),
        )

    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_parameter_shift_qnn_classifier_gradient(
            features,
            labels,
            np.array([0.5], dtype=float),
            finite_difference_step=0.0,
        )

    with pytest.raises(ValueError, match="external gradient"):
        verify_parameter_shift_qnn_classifier_gradient(
            features,
            labels,
            np.array([0.5], dtype=float),
            external_gradients={"jax": lambda _values: np.array([0.1, 0.2], dtype=float)},
        )
