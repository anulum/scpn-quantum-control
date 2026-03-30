# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Snn Backward
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
