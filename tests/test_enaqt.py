# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Enaqt
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
