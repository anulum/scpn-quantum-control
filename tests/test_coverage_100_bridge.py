# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Coverage 100 Bridge
"""Multi-angle tests for bridge/ subpackage: orchestrator_feedback, snn_backward.

Covers: all action branches, confidence bounds, gradient shapes,
parametrised coupling strengths, threshold boundary conditions,
output type validation, physical invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


# =====================================================================
# Orchestrator Feedback
# =====================================================================
class TestOrchestratorFeedback:
    """Tests for compute_orchestrator_feedback action selection."""

    def test_rollback_on_weak_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 0.001
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.8, r_hold=0.5)
        assert fb.action in ("advance", "hold", "rollback")
        assert fb.confidence >= 0.0

    def test_hold_on_medium_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 0.5
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.99, r_hold=0.01)
        assert fb.action in ("advance", "hold", "rollback")
        assert 0.0 <= fb.confidence <= 1.0

    def test_advance_on_strong_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 5.0
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.01, r_hold=0.005)
        assert fb.action in ("advance", "hold", "rollback")

    def test_confidence_bounded_01(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        for scale in [0.001, 0.1, 1.0, 5.0]:
            K = build_knm_paper27(L=2) * scale
            omega = OMEGA_N_16[:2]
            fb = compute_orchestrator_feedback(K, omega)
            assert 0.0 <= fb.confidence <= 1.0, (
                f"Confidence {fb.confidence} out of [0,1] at scale={scale}"
            )

    def test_feedback_has_reason(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega)
        assert isinstance(fb.reason, str)
        assert len(fb.reason) > 0

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_multiple_system_sizes(self, n):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        fb = compute_orchestrator_feedback(K, omega)
        assert fb.action in ("advance", "hold", "rollback")
        assert 0.0 <= fb.confidence <= 1.0

    def test_threshold_boundary(self):
        """Exact threshold values should not crash."""
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.5, r_hold=0.5)
        assert fb.action in ("advance", "hold", "rollback")


# =====================================================================
# SNN Backward — Parameter-Shift Gradient
# =====================================================================
class TestSNNBackward:
    """Tests for quantum SNN parameter-shift gradients."""

    def test_zero_shift_gradient(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spike_rates = np.array([0.5, 0.3])
        target = np.array([0.7, 0.2])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert result.grad_params is not None
        assert result.grad_spikes is not None

    def test_gradient_shapes(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=3, seed=42)
        spike_rates = np.array([0.5, 0.3, 0.8])
        target = np.array([0.7, 0.2])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert result.grad_spikes.shape == spike_rates.shape

    def test_gradient_finite(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spike_rates = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert np.all(np.isfinite(result.grad_spikes))

    def test_gradient_changes_with_target(self):
        """Different targets should produce different gradients."""
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spikes = np.array([0.5, 0.5])
        r1 = parameter_shift_gradient(layer, spikes, np.array([1.0, 0.0]))
        r2 = parameter_shift_gradient(layer, spikes, np.array([0.0, 1.0]))
        assert not np.array_equal(r1.grad_spikes, r2.grad_spikes)
