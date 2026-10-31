# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qfi
"""Tests for Quantum Fisher Information module."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.qfi as qfi_mod
from scpn_quantum_control.analysis.qfi import QFIResult, compute_qfi, qfi_gap_tradeoff
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError


class TestQFI:
    """Spectral QFI of the XY ground state and its result record."""

    def test_2qubit_qfi_positive(self) -> None:
        """Strong coupling relative to frequencies → entangled ground state → nonzero QFI."""
        K = np.array([[0, 2.0], [2.0, 0]])
        omega = np.array([0.1, 0.15])
        result = compute_qfi(K, omega)
        assert result.qfi_matrix.shape == (1, 1)
        assert result.qfi_matrix[0, 0] > 0

    def test_4qubit_qfi_shape(self) -> None:
        """One row and column per coupling pair; four qubits have six pairs."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_qfi(K, omega)
        n_pairs = len(result.coupling_pairs)
        assert result.qfi_matrix.shape == (n_pairs, n_pairs)
        assert n_pairs == 6  # 4 choose 2

    def test_qfi_symmetric(self) -> None:
        """The QFI matrix is symmetric."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        np.testing.assert_allclose(result.qfi_matrix, result.qfi_matrix.T, atol=1e-10)

    def test_qfi_positive_semidefinite(self) -> None:
        """The QFI matrix has no negative eigenvalue beyond rounding."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        eigenvalues = np.linalg.eigvalsh(result.qfi_matrix)
        assert np.all(eigenvalues >= -1e-10)

    def test_precision_bounds(self) -> None:
        """Precision bounds are positive.

        Some may be infinite when a QFI diagonal is zero, that is when the
        ground state does not respond to that coupling.
        """
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        for bound in result.precision_bounds:
            assert bound > 0  # always positive (finite or inf)

    def test_stronger_coupling_higher_qfi(self) -> None:
        """With coupling-dominated parameters, strongly coupled pairs are more estimable."""
        K = build_knm_paper27(L=4) * 5.0  # amplify coupling to dominate frequencies
        omega = OMEGA_N_16[:4] * 0.1  # reduce frequencies
        result = compute_qfi(K, omega)
        diag = np.diag(result.qfi_matrix)
        assert np.max(diag) > 0  # at least one pair estimable

    def test_scpn_default_qfi_near_zero(self) -> None:
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

    def test_spectral_gap_positive(self) -> None:
        """The admitted ground state has a positive spectral gap."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qfi(K, omega)
        assert result.spectral_gap > 0

    def test_gap_tradeoff(self) -> None:
        """The gap/QFI trade-off report has a positive gap and a non-negative QFI."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        tradeoff = qfi_gap_tradeoff(K, omega)
        assert tradeoff["spectral_gap"] > 0
        assert tradeoff["max_qfi_diagonal"] >= 0
        assert tradeoff["gap_squared_over_16"] > 0

    def test_gap_tradeoff_strong_coupling(self) -> None:
        """With strong coupling, QFI is nonzero and tradeoff ratio is positive."""
        K = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
        omega = np.array([0.1, 0.15, 0.12])
        tradeoff = qfi_gap_tradeoff(K, omega)
        assert tradeoff["spectral_gap"] > 0
        assert tradeoff["max_qfi_diagonal"] > 0
        assert tradeoff["tradeoff_ratio"] > 0

    def test_gap_tradeoff_propagates_dense_budget(self) -> None:
        """The trade-off report refuses before exceeding the dense budget."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        with pytest.raises(DenseAllocationError, match="QFI dense eigensolver"):
            qfi_gap_tradeoff(K, omega, max_dense_gib=1e-6)

    def test_qfi_result_precision_method(self) -> None:
        """The Cramér-Rao bound is ``1 / (M F_ii)``."""
        result = QFIResult(
            qfi_matrix=np.array([[4.0]]),
            coupling_pairs=[(0, 1)],
            precision_bounds=np.array([0.25]),
            spectral_gap=0.5,
            n_qubits=2,
        )
        prec = result.precision_for(0, 1, n_measurements=100)
        assert prec == pytest.approx(1.0 / 400, rel=1e-10)

    def test_precision_method_zero_qfi_is_infinite(self) -> None:
        """A zero QFI entry gives an infinite variance bound."""
        result = QFIResult(
            qfi_matrix=np.array([[0.0]]),
            coupling_pairs=[(0, 1)],
            precision_bounds=np.array([np.inf]),
            spectral_gap=0.0,
            n_qubits=2,
        )

        assert result.precision_for(1, 0, n_measurements=100) == float("inf")

    def test_custom_pairs(self) -> None:
        """Only the requested pairs are parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_qfi(K, omega, pairs=[(0, 1), (2, 3)])
        assert len(result.coupling_pairs) == 2
        assert result.qfi_matrix.shape == (2, 2)

    def test_rejects_dense_qfi_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dense budget is checked before any Hamiltonian is built.

        Parameters
        ----------
        monkeypatch
            Replaces the dense builder with one that fails if called.

        """
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        def fail_if_dense_hamiltonian_is_requested(*args: object, **kwargs: object) -> NoReturn:  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(qfi_mod, "knm_to_dense_matrix", fail_if_dense_hamiltonian_is_requested)

        with pytest.raises(DenseAllocationError, match="QFI dense eigensolver"):
            compute_qfi(K, omega, max_dense_gib=1e-6)

    def test_degenerate_ground_level_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A degenerate ground level has no single ground state, so no QFI is returned.

        Earlier behaviour picked the first eigenvector of the ground space and
        skipped the zero denominators, returning a basis-dependent number.

        Parameters
        ----------
        monkeypatch
            Replaces the Hamiltonian builders with a fixed degenerate spectrum.

        """

        def fake_hamiltonian(K: NDArray[np.float64], omega: NDArray[np.float64]) -> None:
            del K, omega
            return None

        def fake_dense_matrix(
            K: NDArray[np.float64], omega: NDArray[np.float64], **kwargs: object
        ) -> NDArray[np.complex128]:
            del K, omega
            assert kwargs == {"max_dense_gib": None}
            return np.diag([0.0, 0.0, 2.0, 3.0])

        monkeypatch.setattr(qfi_mod, "knm_to_hamiltonian", fake_hamiltonian)
        monkeypatch.setattr(qfi_mod, "knm_to_dense_matrix", fake_dense_matrix)

        with pytest.raises(ValueError, match="ground state is degenerate"):
            compute_qfi(np.zeros((2, 2)), np.zeros(2), pairs=[(0, 1)])
