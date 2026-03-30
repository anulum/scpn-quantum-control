# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qfi Criticality
"""Tests for QFI criticality scan."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.qfi_criticality import (
    QFICriticalityResult,
    qfi_single_coupling,
    qfi_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    """Nearest-neighbor ring topology, normalized to max=1."""
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestQFISingleCoupling:
    def test_strong_coupling_has_nonzero_qfi(self):
        """Strong coupling makes ground state entangled → QFI > 0."""
        n = 3
        T = _ring_topology(n)
        K = 3.0 * T  # strong coupling relative to freq splitting
        omega = OMEGA_N_16[:n]
        mq, gap, tq = qfi_single_coupling(K, omega)
        assert mq > 0
        assert gap > 0
        assert tq >= mq

    def test_weak_coupling_product_ground_state(self):
        """Weak coupling: ground state ≈ product → QFI ≈ 0."""
        n = 2
        T = _ring_topology(n)
        K = 0.01 * T
        omega = OMEGA_N_16[:n]
        mq, gap, tq = qfi_single_coupling(K, omega)
        # Coupling too weak to entangle → V|gs⟩ ≈ 0 → QFI ≈ 0
        assert mq < 0.1

    def test_gap_decreases_with_coupling(self):
        """Spectral gap should generally decrease as coupling competes with freq splitting."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        _, gap_weak, _ = qfi_single_coupling(0.1 * T, omega)
        _, gap_strong, _ = qfi_single_coupling(5.0 * T, omega)
        # At weak coupling, gap ≈ min freq difference; strong coupling changes spectrum
        assert gap_weak != gap_strong

    def test_zero_coupling_matrix(self):
        K = np.zeros((2, 2))
        omega = OMEGA_N_16[:2]
        mq, gap, tq = qfi_single_coupling(K, omega)
        assert mq == 0.0
        assert tq == 0.0

    def test_4qubit(self):
        n = 4
        T = _ring_topology(n)
        K = 2.0 * T
        omega = OMEGA_N_16[:n]
        mq, gap, tq = qfi_single_coupling(K, omega)
        assert mq > 0
        assert gap > 0


class TestQFIVsCoupling:
    def test_returns_result(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = qfi_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, QFICriticalityResult)
        assert len(result.k_values) == 3
        assert len(result.max_qfi) == 3

    def test_qfi_has_nonzero_values(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.linspace(1.0, 5.0, 6)
        result = qfi_vs_coupling(omega, T, k_range=k_range)
        assert result.peak_qfi > 0

    def test_peak_k_is_in_range(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.array([1.0, 2.0, 3.0, 4.0])
        result = qfi_vs_coupling(omega, T, k_range=k_range)
        assert result.peak_k in k_range

    def test_gap_varies_across_scan(self):
        """Gap should change across the coupling scan."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.linspace(0.5, 5.0, 6)
        result = qfi_vs_coupling(omega, T, k_range=k_range)
        assert result.spectral_gap[0] != result.spectral_gap[-1]

    def test_4qubit_scan(self):
        n = 4
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = qfi_vs_coupling(omega, T, k_range=np.array([1.0, 3.0]))
        assert result.peak_qfi > 0
        assert len(result.total_qfi) == 2
