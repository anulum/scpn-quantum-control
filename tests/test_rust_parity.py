# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Parity tests between Rust engine and Python implementations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_RUST = importlib.util.find_spec("scpn_quantum_engine") is not None
pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_quantum_engine not installed")


class TestRustPythonParity:
    def test_knm_16_exact_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K_rust = np.array(engine.build_knm(16, 0.45, 0.3))
        K_python = build_knm_paper27(L=16)
        np.testing.assert_allclose(K_rust, K_python, atol=1e-12)

    def test_knm_7_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K_rust = np.array(engine.build_knm(7, 0.45, 0.3))
        K_python = build_knm_paper27(L=7)
        np.testing.assert_allclose(K_rust, K_python, atol=1e-12)

    def test_knm_diagonal(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(4, 0.45, 0.3))
        assert K[0, 0] == pytest.approx(0.45, abs=1e-12)

    def test_knm_calibration_anchors(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(16, 0.45, 0.3))
        assert K[0, 1] == pytest.approx(0.302, abs=1e-12)
        assert K[1, 2] == pytest.approx(0.201, abs=1e-12)
        assert K[2, 3] == pytest.approx(0.252, abs=1e-12)
        assert K[3, 4] == pytest.approx(0.154, abs=1e-12)

    def test_knm_cross_hierarchy_max(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(16, 0.45, 0.3))
        # K[4,6] = max(exponential, 0.15) — exponential > 0.15 for these params
        assert K[4, 6] > 0.15
        assert K[0, 15] >= 0.05

    def test_pec_coefficients_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

        for p in [0.001, 0.01, 0.05, 0.1]:
            rust = engine.pec_coefficients(p)
            python = pauli_twirl_decompose(p)
            np.testing.assert_allclose(rust, python, atol=1e-12)

    def test_order_parameter_parity(self):
        import scpn_quantum_engine as engine

        theta = np.array([0.0, 0.1, 0.2, 0.3])
        rust_R = engine.order_parameter(theta)
        z = np.mean(np.exp(1j * theta))
        python_R = float(np.abs(z))
        assert rust_R == pytest.approx(python_R, abs=1e-10)

    def test_kuramoto_euler_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

        n = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n].copy()
        theta0 = np.array([0.1, 0.2, 0.3, 0.4])
        dt = 0.01

        # One Rust step
        theta_rust = np.array(engine.kuramoto_euler(theta0, omega, K, dt, 1))

        # One Python step (manual Euler)
        dtheta = omega.copy()
        for i in range(n):
            for j in range(n):
                dtheta[i] += K[i, j] * np.sin(theta0[j] - theta0[i])
        theta_python = theta0 + dt * dtheta

        np.testing.assert_allclose(theta_rust, theta_python, atol=1e-10)
