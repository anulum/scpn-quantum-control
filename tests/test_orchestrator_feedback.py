# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for orchestrator bidirectional feedback."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.bridge.orchestrator_feedback import (
    OrchestratorFeedback,
    compute_orchestrator_feedback,
)


class TestOrchestratorFeedback:
    def test_returns_feedback(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert isinstance(fb, OrchestratorFeedback)

    def test_action_valid(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert fb.action in ("advance", "hold", "rollback")

    def test_r_global_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert 0 <= fb.r_global <= 1.0

    def test_confidence_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert 0 <= fb.confidence <= 1.0

    def test_reason_non_empty(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert len(fb.reason) > 0

    def test_l16_action_present(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega)
        assert fb.l16_action in ("continue", "adjust", "halt")

    def test_custom_thresholds(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.99, r_hold=0.01)
        assert isinstance(fb.action, str)

    def test_scpn_feedback(self):
        """Record orchestrator feedback at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        fb = compute_orchestrator_feedback(K, omega)
        print("\n  Orchestrator feedback (4 osc):")
        print(f"  Action: {fb.action} (confidence: {fb.confidence:.3f})")
        print(f"  R_global: {fb.r_global:.4f}")
        print(f"  Stability: {fb.stability_score:.4f}")
        print(f"  Reason: {fb.reason}")
        assert isinstance(fb.action, str)
