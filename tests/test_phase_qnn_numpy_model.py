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
    numpy_phase_classifier_gradient,
    numpy_phase_classifier_loss_and_gradient,
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
    analytic_loss, analytic_gradient = numpy_phase_classifier_loss_and_gradient(
        features, labels, params, numerics=numerics
    )
    expected_gradient = np.array(
        [
            sum(
                (float(expected[row]) - float(labels[row]))
                * math.sin(float(features[row, column] + params[column]))
                for row in range(samples)
            )
            / (samples * width)
            for column in range(width)
        ]
    )
    assert analytic_loss == pytest.approx(expected_loss, rel=0.0, abs=3e-16)
    np.testing.assert_allclose(analytic_gradient, expected_gradient, rtol=0.0, atol=3e-16)
    np.testing.assert_array_equal(
        numpy_phase_classifier_gradient(features, labels, params, numerics=numerics),
        analytic_gradient,
    )
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(params, original_params)
    assert not np.shares_memory(probabilities, features)
    assert not np.shares_memory(probabilities, params)


@pytest.mark.parametrize("gradient_only", [False, True])
def test_analytic_model_retains_scalar_construction_order_and_gradient_only_work(
    gradient_only: bool,
) -> None:
    """Construct the loss before dimension factors and omit it on the gradient route.

    Parameters
    ----------
    gradient_only
        Request only the derivative, which must not construct a scalar loss.

    """
    constructed: list[float] = []

    class ModelScalar(float):
        """Record actual scalar values constructed by the numerical model."""

        def __new__(cls, value: float) -> ModelScalar:
            """Construct the scalar subclass while recording its numerical argument."""
            constructed.append(float(value))
            return super().__new__(cls, value)

    features = np.array([[0.2, -0.4], [1.1, 0.7], [0.3, -0.8]])
    labels = np.array([0.0, 1.0, 0.25])
    params = np.array([0.4, -0.2])
    numerics = PhaseModelNumerics(numpy=lambda: np, scalar=lambda: ModelScalar)
    expected_loss = qnn.parameter_shift_qnn_classifier_loss(features, labels, params)
    if gradient_only:
        gradient = numpy_phase_classifier_gradient(features, labels, params, numerics=numerics)
        assert constructed == [2.0, 3.0]
    else:
        loss, gradient = numpy_phase_classifier_loss_and_gradient(
            features, labels, params, numerics=numerics
        )
        assert type(loss) is ModelScalar
        np.testing.assert_allclose(constructed, [expected_loss, 2.0, 3.0], atol=1e-16)
    np.testing.assert_allclose(
        gradient,
        qnn.parameter_shift_qnn_classifier_gradient(features, labels, params),
        rtol=0.0,
        atol=1e-14,
    )


def test_analytic_loss_resolves_scalar_before_a_real_numpy_underflow_callback() -> None:
    """Retain constructor evaluation order when native arithmetic rebinds it."""

    class OriginalScalar(float):
        """Identify the constructor selected before residual multiplication."""

    class ReboundScalar(float):
        """Identify the constructor selected after NumPy reports underflow."""

    active: type[float] = OriginalScalar
    events: list[tuple[str, int]] = []

    def record_underflow(error: str, flag: int) -> None:
        nonlocal active
        events.append((error, flag))
        active = ReboundScalar

    previous = np.seterrcall(record_underflow)
    try:
        with np.errstate(under="call"):
            loss, gradient = numpy_phase_classifier_loss_and_gradient(
                np.array([[0.0]]),
                np.array([1e-160]),
                np.array([0.0]),
                numerics=PhaseModelNumerics(numpy=lambda: np, scalar=lambda: active),
            )
    finally:
        np.seterrcall(previous)
    assert events and any("underflow" in error for error, _flag in events)
    assert active is ReboundScalar
    assert type(loss) is OriginalScalar
    assert 0.0 < loss < np.finfo(np.float64).tiny
    np.testing.assert_array_equal(gradient, np.array([0.0]))


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
