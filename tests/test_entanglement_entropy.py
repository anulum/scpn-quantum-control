# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for entanglement entropy and Schmidt gap."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.entanglement_entropy import (
    EntanglementResult,
    EntanglementScanResult,
    entanglement_at_coupling,
    entanglement_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestEntanglementAtCoupling:
    def test_returns_result(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_at_coupling(omega, T, K_base=2.0)
        assert isinstance(result, EntanglementResult)

    def test_product_state_zero_entropy(self):
        """Weak coupling → product ground state → S ≈ 0."""
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_at_coupling(omega, T, K_base=0.01)
        assert result.entropy < 0.1

    def test_strong_coupling_nonzero_entropy(self):
        """Strong coupling → entangled ground state → S > 0."""
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_at_coupling(omega, T, K_base=5.0)
        assert result.entropy > 0

    def test_schmidt_gap_bounded(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_at_coupling(omega, T, K_base=2.0)
        assert result.schmidt_gap >= 0
        assert result.schmidt_gap <= 1.0

    def test_entropy_nonnegative(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_at_coupling(omega, T, K_base=2.0)
        assert result.entropy >= -1e-10

    def test_3qubit(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = entanglement_at_coupling(omega, T, K_base=2.0)
        assert isinstance(result, EntanglementResult)
        assert result.spectral_gap > 0


class TestEntanglementVsCoupling:
    def test_returns_scan(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, EntanglementScanResult)
        assert len(result.entropy) == 3

    def test_entropy_varies(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 8))
        assert result.entropy[0] != result.entropy[-1]

    def test_schmidt_gap_varies(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 8))
        assert result.schmidt_gap[0] != result.schmidt_gap[-1]

    def test_peak_and_min_exist(self):
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = entanglement_vs_coupling(omega, T, k_range=np.linspace(0.3, 5.0, 8))
        assert result.schmidt_gap_min_K is not None
