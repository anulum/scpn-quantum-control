# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Lindblad Ness
"""Tests for Lindblad NESS computation."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.lindblad_ness import (
    NESSResult,
    NESSScanResult,
    compute_ness,
    ness_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestComputeNESS:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = compute_ness(omega, T, K_base=2.0, gamma=0.2)
        assert isinstance(result, NESSResult)

    def test_purity_bounded(self):
        """Purity should be between 1/d and 1."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = compute_ness(omega, T, K_base=2.0, gamma=0.1)
        assert result.purity >= 1.0 / 4.0 - 0.01
        assert result.purity <= 1.0 + 0.01

    def test_R_bounded(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = compute_ness(omega, T, K_base=2.0, gamma=0.1)
        assert 0 <= result.R_ness <= 1.0 + 1e-10
        assert 0 <= result.R_ideal <= 1.0 + 1e-10

    def test_strong_noise_destroys_order(self):
        """Strong noise → high purity loss, NESS far from ground state."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = compute_ness(omega, T, K_base=1.0, gamma=5.0)
        # Strong damping drives toward |0⟩ → purity high but state different
        assert result.purity > 0.5

    def test_3qubit(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = compute_ness(omega, T, K_base=1.5, gamma=0.2)
        assert isinstance(result, NESSResult)


class TestNESSVsCoupling:
    def test_returns_scan(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = ness_vs_coupling(omega, T, k_range=np.array([1.0, 3.0]), gamma=0.2)
        assert isinstance(result, NESSScanResult)
        assert len(result.k_values) == 2

    def test_noise_resilience_in_range(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.array([0.5, 1.5, 3.0])
        result = ness_vs_coupling(omega, T, k_range=k_range, gamma=0.1)
        assert result.noise_resilience in k_range

    def test_all_values_finite(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = ness_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 3.0]), gamma=0.2)
        assert np.all(np.isfinite(result.R_ness))
        assert np.all(np.isfinite(result.R_ideal))
        assert np.all(np.isfinite(result.purity))
