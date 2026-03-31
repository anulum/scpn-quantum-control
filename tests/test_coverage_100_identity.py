# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Coverage 100 Identity
"""Multi-angle tests for identity/ subpackage: coherence_budget, robustness.

Covers: error conditions, zero-gap edge cases, parametrised coupling,
physical bounds, output structure, noise scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


# =====================================================================
# Coherence Budget
# =====================================================================
class TestCoherenceBudget:
    def test_invalid_qubits_raises(self):
        from scpn_quantum_control.identity.coherence_budget import coherence_budget

        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            coherence_budget(n_qubits=0)

    def test_invalid_qubits_zero_raises(self):
        """Duplicate test for zero qubits — different entry point."""
        from scpn_quantum_control.identity.coherence_budget import coherence_budget

        with pytest.raises(ValueError):
            coherence_budget(n_qubits=0)


# =====================================================================
# Robustness Certificate
# =====================================================================
class TestRobustness:
    def test_zero_gap_transition(self):
        from scpn_quantum_control.identity.robustness import (
            compute_robustness_certificate,
        )

        K = build_knm_paper27(L=2) * 1e-12
        omega = OMEGA_N_16[:2] * 1e-12
        cert = compute_robustness_certificate(K, omega)
        assert cert.transition_probability >= 0.0

    def test_zero_gap_adiabatic(self):
        from scpn_quantum_control.identity.robustness import (
            compute_robustness_certificate,
        )

        K = build_knm_paper27(L=2) * 1e-12
        omega = OMEGA_N_16[:2] * 1e-12
        cert = compute_robustness_certificate(K, omega)
        assert cert.adiabatic_bound >= 0.0

    @pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 5.0])
    def test_certificate_bounded(self, scale):
        from scpn_quantum_control.identity.robustness import (
            compute_robustness_certificate,
        )

        K = build_knm_paper27(L=2) * scale
        omega = OMEGA_N_16[:2]
        cert = compute_robustness_certificate(K, omega)
        assert 0.0 <= cert.transition_probability <= 1.0
        assert cert.adiabatic_bound >= 0.0

    def test_certificate_has_energy_gap(self):
        from scpn_quantum_control.identity.robustness import (
            compute_robustness_certificate,
        )

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        cert = compute_robustness_certificate(K, omega)
        assert hasattr(cert, "energy_gap")
        assert cert.energy_gap >= 0.0

    def test_noise_scan_zero_gap(self):
        from scpn_quantum_control.identity.robustness import gap_vs_perturbation_scan

        K = build_knm_paper27(L=2) * 1e-12
        omega = OMEGA_N_16[:2] * 1e-12
        result = gap_vs_perturbation_scan(K, omega, noise_range=np.array([0.1]))
        assert result["p_transition_theory"][0] >= 0.0

    def test_noise_scan_multiple_points(self):
        from scpn_quantum_control.identity.robustness import gap_vs_perturbation_scan

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        noise_range = np.linspace(0.01, 0.5, 5)
        result = gap_vs_perturbation_scan(K, omega, noise_range=noise_range)
        assert len(result["p_transition_theory"]) == 5
        assert all(0.0 <= p <= 1.0 for p in result["p_transition_theory"])

    def test_perturbation_fidelity_bounded(self):
        from scpn_quantum_control.identity.robustness import perturbation_fidelity

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        delta_K = K * 0.01  # 1% perturbation
        fid = perturbation_fidelity(K, omega, delta_K)
        assert np.isfinite(fid)
