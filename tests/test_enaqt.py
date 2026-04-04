# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Enaqt
"""Tests for ENAQT noise optimisation."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.enaqt import (
    ENAQTResult,
    enaqt_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestENAQT:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=np.array([0.01, 0.1, 1.0]), n_steps=10)
        assert isinstance(result, ENAQTResult)

    def test_optimal_gamma_positive(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=np.array([0.01, 0.1, 1.0]), n_steps=10)
        assert result.optimal_gamma > 0

    def test_r_bounded(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=np.array([0.01, 0.1, 1.0]), n_steps=10)
        assert 0 <= result.optimal_r <= 1.0
        for r in result.r_values:
            assert 0 <= r <= 1.0 + 1e-6

    def test_enhancement_type(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=np.array([0.01, 0.5, 5.0]), n_steps=10)
        assert isinstance(result.enhancement, float)
        assert result.enhancement > 0

    def test_gamma_values_match(self):
        gammas = np.array([0.01, 0.1, 1.0])
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, gamma_range=gammas, n_steps=10)
        assert len(result.r_values) == 3

    def test_3_oscillators(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = enaqt_scan(K, omega, gamma_range=np.array([0.1, 1.0]), n_steps=5)
        assert isinstance(result, ENAQTResult)

    def test_scpn_enaqt(self):
        """Record ENAQT optimum for SCPN defaults."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = enaqt_scan(K, omega, gamma_range=np.logspace(-2, 1, 8), n_steps=20)
        print("\n  ENAQT (3 osc):")
        print(f"  Optimal γ = {result.optimal_gamma:.4f}")
        print(f"  R at optimum = {result.optimal_r:.4f}")
        print(f"  Coherent R = {result.coherent_r:.4f}")
        print(f"  Enhancement = {result.enhancement:.2f}x")
        assert isinstance(result.optimal_gamma, float)


def test_enaqt_coherent_r_positive():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=np.array([0.0, 0.5]), n_steps=5)
    assert result.coherent_r >= 0


def test_enaqt_r_values_bounded():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=np.linspace(0.01, 2.0, 5), n_steps=5)
    for r in result.r_values:
        assert 0 <= r <= 1.0 + 1e-10


def test_enaqt_optimal_gamma_positive():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=np.logspace(-2, 1, 5), n_steps=5)
    assert result.optimal_gamma >= 0


def test_enaqt_enhancement_finite():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = enaqt_scan(K, omega, gamma_range=np.logspace(-2, 1, 5), n_steps=5)
    assert np.isfinite(result.enhancement)


# ---------------------------------------------------------------------------
# Coverage: internal Lindblad and density matrix helpers
# ---------------------------------------------------------------------------


class TestLindbladEvolve:
    def test_preserves_trace(self):
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        dim = 4
        psi = np.ones(dim, dtype=complex) / 2.0
        rho = np.outer(psi, psi.conj())
        H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
        rho_new = _lindblad_evolve(rho, H, gamma=0.5, dt=0.01, n_qubits=2)
        np.testing.assert_allclose(np.trace(rho_new).real, 1.0, atol=1e-10)

    def test_hermitian_output(self):
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        H = np.diag([0.0, 1.0, 1.0, 2.0]).astype(complex)
        rho_new = _lindblad_evolve(rho, H, gamma=0.1, dt=0.01, n_qubits=2)
        np.testing.assert_allclose(rho_new, rho_new.conj().T, atol=1e-10)

    def test_zero_gamma_unitary(self):
        from scpn_quantum_control.analysis.enaqt import _lindblad_evolve

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
        rho_new = _lindblad_evolve(rho, H, gamma=0.0, dt=0.01, n_qubits=2)
        # Pure state remains pure (purity close to 1)
        purity = float(np.trace(rho_new @ rho_new).real)
        assert purity > 0.99


class TestRFromDensityMatrix:
    def test_pure_state(self):
        from scpn_quantum_control.analysis.enaqt import _r_from_density_matrix

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        r = _r_from_density_matrix(rho, 2)
        assert 0 <= r <= 1.0

    def test_maximally_mixed_phases_degenerate(self):
        from scpn_quantum_control.analysis.enaqt import _r_from_density_matrix

        rho = np.eye(4, dtype=complex) / 4.0
        r = _r_from_density_matrix(rho, 2)
        # Maximally mixed: ⟨X⟩=⟨Y⟩=0 → arctan2(0,0)=0 → all phases=0 → R=1
        # This is an artefact of the phase extraction, not physical coherence
        assert 0 <= r <= 1.0


class TestEnaqtDefaults:
    def test_default_gamma_range(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = enaqt_scan(K, omega, n_steps=5)
        assert len(result.gamma_values) == 20  # default logspace -3 to 1, 20 pts
