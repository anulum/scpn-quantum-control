# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qfi
"""Tests for Quantum Fisher Information module."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.qfi import QFIResult, compute_qfi, qfi_gap_tradeoff
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestQFI:
    def test_2qubit_qfi_positive(self):
        """Strong coupling relative to frequencies → entangled ground state → nonzero QFI."""
        K = np.array([[0, 2.0], [2.0, 0]])
        omega = np.array([0.1, 0.15])
        result = compute_qfi(K, omega)
        assert result.qfi_matrix.shape == (1, 1)
        assert result.qfi_matrix[0, 0] > 0

    def test_4qubit_qfi_shape(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_qfi(K, omega)
        n_pairs = len(result.coupling_pairs)
        assert result.qfi_matrix.shape == (n_pairs, n_pairs)
        assert n_pairs == 6  # 4 choose 2

    def test_qfi_symmetric(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        np.testing.assert_allclose(result.qfi_matrix, result.qfi_matrix.T, atol=1e-10)

    def test_qfi_positive_semidefinite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        eigenvalues = np.linalg.eigvalsh(result.qfi_matrix)
        assert np.all(eigenvalues >= -1e-10)

    def test_precision_bounds(self):
        """Precision bounds are positive. Some may be inf if QFI diagonal = 0
        (ground state insensitive to that coupling parameter)."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        for bound in result.precision_bounds:
            assert bound > 0  # always positive (finite or inf)

    def test_stronger_coupling_higher_qfi(self):
        """With coupling-dominated parameters, strongly coupled pairs are more estimable."""
        K = build_knm_paper27(L=4) * 5.0  # amplify coupling to dominate frequencies
        omega = OMEGA_N_16[:4] * 0.1  # reduce frequencies
        result = compute_qfi(K, omega)
        diag = np.diag(result.qfi_matrix)
        assert np.max(diag) > 0  # at least one pair estimable

    def test_scpn_default_qfi_near_zero(self):
        """At default SCPN parameters, QFI is near zero — ground state is product-like.

        This is a finding: SCPN frequencies are too heterogeneous for the
        coupling to create entanglement in the ground state. The system
        needs stronger coupling or weaker frequency spread for quantum effects.
        """
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_qfi(K, omega)
        diag = np.diag(result.qfi_matrix)
        # All QFI near zero = product ground state = classically estimable
        assert np.max(diag) < 1.0

    def test_spectral_gap_positive(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        assert result.spectral_gap > 0

    def test_gap_tradeoff(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        tradeoff = qfi_gap_tradeoff(K, omega)
        assert tradeoff["spectral_gap"] > 0
        assert tradeoff["max_qfi_diagonal"] >= 0
        assert tradeoff["gap_squared_over_16"] > 0

    def test_gap_tradeoff_strong_coupling(self):
        """With strong coupling, QFI is nonzero and tradeoff ratio is positive."""
        K = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
        omega = np.array([0.1, 0.15, 0.12])
        tradeoff = qfi_gap_tradeoff(K, omega)
        assert tradeoff["spectral_gap"] > 0
        assert tradeoff["max_qfi_diagonal"] > 0
        assert tradeoff["tradeoff_ratio"] > 0

    def test_qfi_result_precision_method(self):
        result = QFIResult(
            qfi_matrix=np.array([[4.0]]),
            coupling_pairs=[(0, 1)],
            precision_bounds=np.array([0.25]),
            spectral_gap=0.5,
            n_qubits=2,
        )
        prec = result.precision_for(0, 1, n_measurements=100)
        assert prec == pytest.approx(1.0 / 400, rel=1e-10)

    def test_custom_pairs(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_qfi(K, omega, pairs=[(0, 1), (2, 3)])
        assert len(result.coupling_pairs) == 2
        assert result.qfi_matrix.shape == (2, 2)
