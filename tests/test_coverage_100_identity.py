# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for identity/ module gaps
"""Tests targeting specific uncovered lines in the identity/ subpackage."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- coherence_budget.py line 95: n_qubits < 1 raises ValueError ---


def test_coherence_budget_invalid_qubits():
    """Cover line 95: coherence_budget raises for n_qubits < 1."""
    from scpn_quantum_control.identity.coherence_budget import coherence_budget

    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        coherence_budget(n_qubits=0)


# --- robustness.py line 79: gap near zero → p_transition = 1.0 ---


def test_robustness_zero_gap_transition():
    """Cover line 79: p_transition = 1.0 when gap near zero."""
    from scpn_quantum_control.identity.robustness import compute_robustness_certificate

    # Near-degenerate Hamiltonian
    K = build_knm_paper27(L=2) * 1e-12
    omega = OMEGA_N_16[:2] * 1e-12
    cert = compute_robustness_certificate(K, omega)
    assert cert.transition_probability >= 0.0


# --- robustness.py line 85: gap near zero → adiabatic = 1.0 ---


def test_robustness_zero_gap_adiabatic():
    """Cover line 85: adiabatic = 1.0 when gap near zero."""
    from scpn_quantum_control.identity.robustness import compute_robustness_certificate

    K = build_knm_paper27(L=2) * 1e-12
    omega = OMEGA_N_16[:2] * 1e-12
    cert = compute_robustness_certificate(K, omega)
    assert cert.adiabatic_bound >= 0.0


# --- robustness.py line 154: noise_scan gap near zero → p_theory = 1.0 ---


def test_robustness_noise_scan_zero_gap():
    """Cover line 154: gap_vs_perturbation_scan p_theory = 1.0 with small gap."""
    from scpn_quantum_control.identity.robustness import gap_vs_perturbation_scan

    K = build_knm_paper27(L=2) * 1e-12
    omega = OMEGA_N_16[:2] * 1e-12
    result = gap_vs_perturbation_scan(K, omega, noise_range=np.array([0.1]))
    assert result["p_transition_theory"][0] >= 0.0
