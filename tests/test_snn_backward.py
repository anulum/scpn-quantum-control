# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Snn Backward
"""Tests for SNN backward pass through quantum layer."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.snn_backward import (
    BackwardResult,
    parameter_shift_gradient,
)
from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer


class TestParameterShiftGradient:
    """Exercise the public SNN quantum parameter-shift bridge."""

    def test_returns_result(self) -> None:
        """Return the public backward-result record."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        assert isinstance(result, BackwardResult)

    def test_grad_shape(self) -> None:
        """Align angle and spike gradients with the input count."""
        layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
        vals = np.array([0.5, 0.3, 0.7])
        target = np.array([0.5, 0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.grad_params.shape == (3,)
        assert result.grad_spikes.shape == (3,)

    def test_loss_non_negative(self) -> None:
        """Return a non-negative mean-squared error."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.loss >= 0

    def test_n_evaluations(self) -> None:
        """2 per parameter."""
        layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
        vals = np.array([0.5, 0.3, 0.7])
        target = np.array([0.5, 0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.n_evaluations == 6  # 2 × 3

    def test_grad_finite(self) -> None:
        """Return finite angle and spike-rate gradients."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        assert np.all(np.isfinite(result.grad_params))
        assert np.all(np.isfinite(result.grad_spikes))

    def test_grad_spikes_scaled(self) -> None:
        """grad_spikes = grad_params times pi."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        np.testing.assert_allclose(result.grad_spikes, result.grad_params * np.pi, atol=1e-12)

    def test_zero_loss_small_gradient(self) -> None:
        """If output matches target, gradients should be small."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.0, 0.0])
        from scpn_quantum_control.bridge.snn_backward import _quantum_forward

        actual_output = _quantum_forward(layer, vals)
        result = parameter_shift_gradient(layer, vals, actual_output)
        assert result.loss < 1e-10


def test_gradient_shape_matches_params() -> None:
    """Match the gradient vector to an asymmetric layer input count."""
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=3)
    vals = np.array([0.5, 0.3, 0.8])
    target = np.array([0.7, 0.2])
    result = parameter_shift_gradient(layer, vals, target)
    assert len(result.grad_params) > 0
    assert result.grad_params.ndim == 1


def test_gradient_finite() -> None:
    """Return finite gradients for a balanced two-input layer."""
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
    vals = np.array([0.5, 0.5])
    target = np.array([0.8, 0.2])
    result = parameter_shift_gradient(layer, vals, target)
    assert np.all(np.isfinite(result.grad_params))


def test_loss_nonnegative() -> None:
    """Keep the standalone backward loss non-negative."""
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
    vals = np.array([0.5, 0.5])
    target = np.array([0.5, 0.5])
    result = parameter_shift_gradient(layer, vals, target)
    assert result.loss >= 0


def test_gradient_3x3() -> None:
    """Evaluate a three-input three-neuron layer."""
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
    vals = np.array([0.1, 0.5, 0.9])
    target = np.array([0.5, 0.5, 0.5])
    result = parameter_shift_gradient(layer, vals, target)
    assert len(result.grad_params) > 0


def test_boundary_zero_shift() -> None:
    """Return zero gradients through the singular zero-shift fallback."""
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    result = parameter_shift_gradient(layer, np.array([1.0, 1.0]), np.array([0.5, 0.5]), shift=0.0)
    np.testing.assert_allclose(result.grad_params, 0.0, atol=1e-12)


def test_single_input_gradient_matches_exact_ry_parameter_shift() -> None:
    """One-input bridge gradient matches the analytic Ry derivative."""
    weight = 0.6
    input_value = 0.3
    target_value = 0.1
    layer = QuantumDenseLayer(
        n_neurons=1,
        n_inputs=1,
        weights=np.array([[weight]], dtype=np.float64),
    )

    result = parameter_shift_gradient(
        layer,
        np.array([input_value], dtype=np.float64),
        np.array([target_value], dtype=np.float64),
    )

    theta = np.pi * input_value
    synapse_probability = np.sin(np.pi * weight / 2.0) ** 2
    output_probability = np.sin(theta / 2.0) ** 2 * synapse_probability
    loss_gradient = 2.0 * (output_probability - target_value)
    expected_dy_dtheta = 0.5 * np.sin(theta) * synapse_probability
    expected_grad_theta = loss_gradient * expected_dy_dtheta

    np.testing.assert_allclose(result.grad_params, [expected_grad_theta], atol=1e-12)
    np.testing.assert_allclose(result.grad_spikes, [expected_grad_theta * np.pi], atol=1e-12)


def test_parameter_shift_does_not_clip_shifted_boundary_angles() -> None:
    """At theta=pi the true Ry derivative is zero despite shifted probes."""
    layer = QuantumDenseLayer(
        n_neurons=1,
        n_inputs=1,
        weights=np.array([[0.6]], dtype=np.float64),
    )

    result = parameter_shift_gradient(
        layer,
        np.array([1.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(result.grad_params, [0.0], atol=1e-12)
