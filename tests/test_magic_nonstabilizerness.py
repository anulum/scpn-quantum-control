# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Magic Nonstabilizerness
"""Tests for magic (non-stabilizerness) at BKT."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.magic_nonstabilizerness import (
    MagicResult,
    MagicScanResult,
    magic_at_coupling,
    magic_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestMagicAtCoupling:
    def test_returns_result(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=2.0)
        assert isinstance(result, MagicResult)
        assert result.n_qubits == 2

    def test_stabilizer_state_zero_magic(self):
        """Product state |00⟩ is a stabilizer state → M_2 ≈ 0."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=0.001)
        # Very weak coupling → ground state ≈ |00⟩ → stabilizer → M_2 ≈ 0
        assert result.sre_m2 < 0.5

    def test_entangled_state_has_magic(self):
        """Strong coupling → entangled ground state → M_2 > 0."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, T, K_base=3.0)
        assert result.sre_m2 > 0

    def test_sre_nonnegative(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, T, K_base=2.0)
        assert result.sre_m2 >= -1e-10

    def test_xi_sum_positive(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=1.0)
        assert result.xi_sum > 0


class TestMagicVsCoupling:
    def test_returns_scan(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_vs_coupling(omega, T, k_range=np.array([0.5, 1.5, 3.0]))
        assert isinstance(result, MagicScanResult)
        assert len(result.sre_m2) == 3

    def test_magic_varies_with_K(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 6))
        assert result.sre_m2[0] != result.sre_m2[-1]

    def test_peak_exists(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 8))
        assert result.peak_magic > 0


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------


class TestMagicInvariants:
    def test_sre_all_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.5, 4.0, 5))
        assert np.all(np.isfinite(result.sre_m2))

    def test_k_values_match_input(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        k_range = np.array([1.0, 2.0, 3.0])
        result = magic_vs_coupling(omega, T, k_range=k_range)
        np.testing.assert_array_equal(result.k_values, k_range)

    def test_peak_magic_at_valid_k(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        k_range = np.linspace(0.5, 5.0, 6)
        result = magic_vs_coupling(omega, T, k_range=k_range)
        assert result.peak_K in k_range


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestMagicPipeline:
    def test_knm_to_magic(self):
        """Pipeline: build_knm_paper27 → magic_at_coupling → SRE."""
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, K, K_base=2.0)
        assert isinstance(result, MagicResult)
        assert result.sre_m2 >= 0
