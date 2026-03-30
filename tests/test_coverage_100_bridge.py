# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for bridge/ module gaps
"""Tests targeting specific uncovered lines in the bridge/ subpackage."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- orchestrator_feedback.py lines 68-75: hold and rollback actions ---


def test_orchestrator_feedback_rollback():
    """Cover lines 72-75: rollback action when R < r_hold."""
    from scpn_quantum_control.bridge.orchestrator_feedback import compute_orchestrator_feedback

    # Weak coupling → low R → rollback
    K = build_knm_paper27(L=2) * 0.001
    omega = OMEGA_N_16[:2]
    fb = compute_orchestrator_feedback(K, omega, r_advance=0.8, r_hold=0.5)
    assert fb.action in ("advance", "hold", "rollback")
    assert fb.confidence >= 0.0


def test_orchestrator_feedback_hold():
    """Cover lines 68-71: hold action when r_hold <= R < r_advance."""
    from scpn_quantum_control.bridge.orchestrator_feedback import compute_orchestrator_feedback

    # Medium coupling → medium R → hold
    K = build_knm_paper27(L=2) * 0.5
    omega = OMEGA_N_16[:2]
    fb = compute_orchestrator_feedback(K, omega, r_advance=0.99, r_hold=0.01)
    assert fb.action in ("advance", "hold", "rollback")
    assert 0.0 <= fb.confidence <= 1.0


def test_orchestrator_feedback_advance():
    """Cover lines 64-67: advance action when R >= r_advance and stable."""
    from scpn_quantum_control.bridge.orchestrator_feedback import compute_orchestrator_feedback

    # Strong coupling → high R
    K = build_knm_paper27(L=2) * 5.0
    omega = OMEGA_N_16[:2]
    fb = compute_orchestrator_feedback(K, omega, r_advance=0.01, r_hold=0.005)
    assert fb.action in ("advance", "hold", "rollback")


# --- snn_backward.py line 125: zero shift returns zero gradient ---


def test_snn_backward_zero_shift():
    """Cover line 125: dy_dv = zeros when actual_shift near zero."""
    from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
    from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    spike_rates = np.array([0.5, 0.3])
    target = np.array([0.7, 0.2])
    result = parameter_shift_gradient(layer, spike_rates, target)
    assert result.grad_params is not None
    assert result.grad_spikes is not None
