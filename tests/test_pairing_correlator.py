# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Richardson-Gaudin pairing correlators."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.pairing_correlator import (
    PairingResult,
    pairing_map,
    pairing_vs_anisotropy,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestPairingMap:
    def test_returns_result(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_map(omega, T, K_base=2.0, delta=0.0)
        assert isinstance(result, PairingResult)
        assert result.pairing_matrix.shape == (3, 3)

    def test_pairing_bounded(self):
        """Pairing magnitude should be ≤ 0.5 (max for singlet, spin-1/2)."""
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_map(omega, T, K_base=3.0, delta=0.5)
        assert result.max_pairing <= 0.5 + 1e-6

    def test_pairing_changes_with_delta(self):
        """Pairing should differ between XY (Δ=0) and Heisenberg (Δ=1)."""
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        p_xy = pairing_map(omega, T, K_base=2.0, delta=0.0)
        p_heis = pairing_map(omega, T, K_base=2.0, delta=1.0)
        assert not np.allclose(
            np.abs(p_xy.pairing_matrix),
            np.abs(p_heis.pairing_matrix),
        )

    def test_hermitian_conjugate(self):
        """⟨S_i⁺ S_j⁻⟩ = ⟨S_j⁺ S_i⁻⟩*."""
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_map(omega, T, K_base=2.0, delta=0.5)
        P = result.pairing_matrix
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(P[i, j] - P[j, i].conjugate()) < 1e-10

    def test_4qubit(self):
        n = 4
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_map(omega, T, K_base=1.5, delta=0.3)
        assert result.pairing_matrix.shape == (4, 4)
        assert result.n_qubits == 4


class TestPairingVsAnisotropy:
    def test_returns_dict(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_vs_anisotropy(omega, T, K_base=2.0, delta_range=np.array([0.0, 0.5, 1.0]))
        assert "delta" in result
        assert "max_pairing" in result
        assert len(result["delta"]) == 3

    def test_all_values_finite(self):
        n = 3
        T = _ring(n)
        omega = OMEGA_N_16[:n]
        result = pairing_vs_anisotropy(omega, T, K_base=2.0, delta_range=np.array([0.0, 1.0]))
        for key in result:
            assert all(np.isfinite(v) for v in result[key])
