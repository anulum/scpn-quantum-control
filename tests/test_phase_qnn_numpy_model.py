# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Bounded phase classifier model tests
"""Exercise real classifier probabilities, derivatives and training composition."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import qnn_training as qnn
from scpn_quantum_control.phase_qnn_model_numerics import PhaseModelNumerics
from scpn_quantum_control.phase_qnn_numpy_model import (
    bounded_phase_qnn_probabilities,
    phase_qnn_mean_squared_error,
)


@pytest.mark.parametrize("samples", [1, 3, 5])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_model_and_original_public_routes_match_independent_scalar_reference(
    samples: int, width: int
) -> None:
    """Keep feature averaging and sample averaging on their independent axes.

    Parameters
    ----------
    samples
        Number of independently labelled examples.
    width
        Number of coupled phase features per prediction.

    """
    features = np.linspace(-1.3, 2.7, samples * width).reshape(samples, width)
    params = np.linspace(-0.4, 0.6, width)
    labels = np.linspace(0.0, 1.0, samples)
    original_features, original_params = features.copy(), params.copy()
    expected = np.array(
        [
            min(
                1.0,
                max(
                    0.0,
                    sum(
                        0.5 * (1.0 - math.cos(float(features[row, col] + params[col])))
                        for col in range(width)
                    )
                    / width,
                ),
            )
            for row in range(samples)
        ]
    )
    expected_loss = (
        sum((float(expected[row]) - float(labels[row])) ** 2 for row in range(samples)) / samples
    )
    numerics = PhaseModelNumerics(numpy=lambda: np, scalar=lambda: float)
    probabilities = bounded_phase_qnn_probabilities(features, params, numerics=numerics)
    loss = phase_qnn_mean_squared_error(probabilities, labels, numerics=numerics)
    prediction = qnn.predict_parameter_shift_qnn_classifier(features, params, labels=labels)
    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=3e-16)
    assert probabilities.dtype == np.dtype(np.float64)
    assert loss == pytest.approx(expected_loss, rel=0.0, abs=3e-16)
    np.testing.assert_array_equal(prediction.probabilities, probabilities)
    assert qnn.parameter_shift_qnn_classifier_loss(features, labels, params) == loss
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(params, original_params)
    assert not np.shares_memory(probabilities, features)
    assert not np.shares_memory(probabilities, params)


def test_public_parameter_shift_derivative_retains_feature_coupling() -> None:
    """Differentiate the averaged prediction rather than separate feature losses."""
    features = np.array([[0.2, -0.4, 1.3], [1.1, 0.7, -0.2], [-0.8, 0.3, 0.6]])
    labels = np.array([0.0, 1.0, 0.25])
    params = np.array([0.4, -0.2, 0.15])
    verification = qnn.verify_parameter_shift_qnn_classifier_gradient(
        features, labels, params, tolerance=2e-6
    )
    assert verification.passed
    np.testing.assert_allclose(
        verification.parameter_shift_gradient,
        verification.finite_difference_gradient,
        rtol=0.0,
        atol=2e-6,
    )
    assert verification.shift_terms == 2
    assert verification.loss == qnn.parameter_shift_qnn_classifier_loss(features, labels, params)


def test_original_public_training_reaches_the_model_and_preserves_input_storage() -> None:
    """Train the actual phase model and compare its returned prediction and loss."""
    features = np.array([[0.0], [np.pi]])
    labels = np.array([0.0, 1.0])
    params = np.array([0.8])
    initial = params.copy()
    result = qnn.train_parameter_shift_qnn_classifier(
        features, labels, initial_params=params, learning_rate=0.7, max_steps=8
    )
    assert result.best_loss < result.loss_history[0]
    assert result.certificate.monotone_accepted_values
    assert result.best_loss == qnn.parameter_shift_qnn_classifier_loss(
        features, labels, result.best_params
    )
    prediction = qnn.predict_parameter_shift_qnn_classifier(features, result.best_params)
    np.testing.assert_array_equal(result.prediction.probabilities, prediction.probabilities)
    np.testing.assert_array_equal(params, initial)
    assert result.to_dict()["method"] == "multi_frequency_parameter_shift_qnn_classifier"


def test_original_public_loss_reads_scalar_constructor_after_parameter_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain a real ArrayLike conversion's change to the original module's float.

    Parameters
    ----------
    monkeypatch
        Restore the original module dictionary after the interop probe.

    """

    class ModelScalar(float):
        """Identify the constructor selected after parameter conversion."""

    class ParameterInput:
        """Expose a real NumPy conversion that updates a scalar-constructor hook."""

        def __array__(
            self, dtype: np.dtype[np.float64] | None = None, copy: bool | None = None
        ) -> NDArray[np.float64]:
            """Convert the parameter vector using the already resolved dtype."""
            vars(qnn)["float"] = ModelScalar
            return np.array([0.4, -0.2], dtype=dtype, copy=copy)

    monkeypatch.setitem(vars(qnn), "float", float)
    loss = qnn.parameter_shift_qnn_classifier_loss(
        np.array([[0.2, -0.4], [1.1, 0.7]]), np.array([0.0, 1.0]), ParameterInput()
    )
    assert type(loss) is ModelScalar
    assert math.isfinite(loss)


def test_probabilities_keep_extreme_phase_values_in_the_physical_interval() -> None:
    """Retain exact zero/one endpoints and large finite phase evaluation."""
    features = np.array([[0.0], [np.pi], [1e30], [-1e30]])
    prediction = qnn.predict_parameter_shift_qnn_classifier(features, np.array([0.0]))
    assert prediction.probabilities[0] == 0.0
    assert prediction.probabilities[1] == 1.0
    assert np.all(prediction.probabilities >= 0.0)
    assert np.all(prediction.probabilities <= 1.0)
