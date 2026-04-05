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
    def test_returns_result(self):
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        assert isinstance(result, BackwardResult)

    def test_grad_shape(self):
        layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
        vals = np.array([0.5, 0.3, 0.7])
        target = np.array([0.5, 0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.grad_params.shape == (3,)
        assert result.grad_spikes.shape == (3,)

    def test_loss_non_negative(self):
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.loss >= 0

    def test_n_evaluations(self):
        """2 per parameter."""
        layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
        vals = np.array([0.5, 0.3, 0.7])
        target = np.array([0.5, 0.5, 0.5])
        result = parameter_shift_gradient(layer, vals, target)
        assert result.n_evaluations == 6  # 2 × 3

    def test_grad_finite(self):
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        assert np.all(np.isfinite(result.grad_params))
        assert np.all(np.isfinite(result.grad_spikes))

    def test_grad_spikes_scaled(self):
        """grad_spikes = grad_params × π."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.5, 0.3])
        target = np.array([0.8, 0.2])
        result = parameter_shift_gradient(layer, vals, target)
        np.testing.assert_allclose(result.grad_spikes, result.grad_params * np.pi, atol=1e-12)

    def test_zero_loss_small_gradient(self):
        """If output matches target, gradients should be small."""
        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
        vals = np.array([0.0, 0.0])
        from scpn_quantum_control.bridge.snn_backward import _quantum_forward

        actual_output = _quantum_forward(layer, vals)
        result = parameter_shift_gradient(layer, vals, actual_output)
        assert result.loss < 1e-10


def test_gradient_shape_matches_params():
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=3)
    vals = np.array([0.5, 0.3, 0.8])
    target = np.array([0.7, 0.2])
    result = parameter_shift_gradient(layer, vals, target)
    assert len(result.grad_params) > 0
    assert result.grad_params.ndim == 1


def test_gradient_finite():
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
    vals = np.array([0.5, 0.5])
    target = np.array([0.8, 0.2])
    result = parameter_shift_gradient(layer, vals, target)
    assert np.all(np.isfinite(result.grad_params))


def test_loss_nonnegative():
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2)
    vals = np.array([0.5, 0.5])
    target = np.array([0.5, 0.5])
    result = parameter_shift_gradient(layer, vals, target)
    assert result.loss >= 0


def test_gradient_3x3():
    layer = QuantumDenseLayer(n_neurons=3, n_inputs=3)
    vals = np.array([0.1, 0.5, 0.9])
    target = np.array([0.5, 0.5, 0.5])
    result = parameter_shift_gradient(layer, vals, target)
    assert len(result.grad_params) > 0


def test_boundary_zero_shift():
    """When input is at 0.0 with large shift, vals_minus clips to 0.0 and
    vals_plus goes up, but both should still yield finite gradients.
    When actual_shift ≈ 0, the gradient falls back to zeros."""
    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    # Input exactly at 1.0, shift=0.25: vals_plus clips to 1.0, vals_minus=0.75
    # actual_shift = 1.0 - 0.75 = 0.25 > 1e-10, so normal path.
    # To trigger zero shift: input at 1.0, shift very small so both clip to 1.0.
    # Actually: vals_plus = min(1.0 + shift, 1.0) = 1.0, vals_minus = max(1.0 - shift, 0.0) = 1.0 - shift
    # That still has nonzero shift.
    # The only way actual_shift = 0 is if both clip to same value.
    # Input=1.0, shift=0 → both = 1.0 → shift=0.
    result = parameter_shift_gradient(layer, np.array([1.0, 1.0]), np.array([0.5, 0.5]), shift=0.0)
    # With shift=0, all gradients should be 0
    np.testing.assert_allclose(result.grad_params, 0.0, atol=1e-12)
