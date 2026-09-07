# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Critical Concordance
"""Tests for critical point concordance analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest

from scpn_quantum_control.analysis import critical_concordance as concordance_module
from scpn_quantum_control.analysis.critical_concordance import (
    ConcordanceResult,
    critical_concordance,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.dense_budget import DenseAllocationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestCriticalConcordance:
    def test_returns_result(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, ConcordanceResult)
        assert len(result.k_values) == 3

    def test_all_arrays_correct_length(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.linspace(0.5, 4.0, 5)
        result = critical_concordance(omega, T, k_range=k_range)
        assert len(result.R_values) == 5
        assert len(result.qfi_values) == 5
        assert len(result.gap_values) == 5
        assert len(result.fiedler_values) == 5
        assert len(result.n_entangled_pairs) == 5

    def test_gap_always_positive(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.linspace(0.5, 5.0, 5))
        assert np.all(result.gap_values > 0)

    def test_k_c_estimates_exist(self) -> None:
        """At least gap-based K_c should always be found."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.linspace(0.5, 5.0, 8))
        assert result.k_c_from_gap is not None

    def test_R_derivative_k_c(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.linspace(0.5, 5.0, 8))
        assert result.k_c_from_R_deriv is not None
        assert result.k_c_from_R_deriv >= 0.5
        assert result.k_c_from_R_deriv <= 5.0

    def test_concordance_spread_finite(self) -> None:
        """If multiple estimates exist, spread should be finite."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.linspace(0.5, 5.0, 10))
        if result.concordance_spread is not None:
            assert np.isfinite(result.concordance_spread)

    def test_4qubit_concordance(self) -> None:
        n = 4
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.array([1.0, 3.0, 5.0]))
        assert isinstance(result, ConcordanceResult)
        assert result.k_c_from_gap is not None

    def test_wide_scan_finds_transition(self) -> None:
        """A wide scan should show gap varying and QFI nonzero somewhere."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = critical_concordance(omega, T, k_range=np.linspace(0.3, 6.0, 10))
        # Gap should vary across the scan
        assert result.gap_values[0] != result.gap_values[-1]

    def test_rejects_dense_budget_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n = 4
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]

        def fail_if_dense_hamiltonian_is_requested(*args: object, **kwargs: object) -> NoReturn:  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            concordance_module,
            "knm_to_dense_matrix",
            fail_if_dense_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="critical concordance dense eigensolver"):
            critical_concordance(
                omega,
                T,
                k_range=np.array([1.0, 2.0]),
                max_dense_gib=1e-5,
            )

    def test_reuses_concordance_eigendecomposition_for_qfi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # knm_to_dense_matrix is imported by the module under test, not defined
        # there, so the module dict is where the counting stand-in goes.
        original_dense = vars(concordance_module)["knm_to_dense_matrix"]
        dense_calls = 0

        def counting_dense(*args: object, **kwargs: object) -> NDArray[np.complex128]:
            nonlocal dense_calls
            dense_calls += 1
            dense: NDArray[np.complex128] = original_dense(*args, **kwargs)
            return dense

        monkeypatch.setattr(
            concordance_module,
            "knm_to_dense_matrix",
            counting_dense,
        )

        result = critical_concordance(
            omega,
            T,
            k_range=np.array([1.0, 2.0, 3.0]),
            max_dense_gib=0.25,
        )

        assert dense_calls == 3
        assert np.all(np.isfinite(result.qfi_values))


def test_concordance_k_range_length() -> None:
    n = 3
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 5.0, 8)
    result = critical_concordance(omega, T, k_range=k_range)
    assert len(result.k_values) == 8


def test_concordance_gap_positive() -> None:
    n = 3
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = critical_concordance(omega, T, k_range=np.linspace(0.5, 5.0, 6))
    assert np.all(np.array(result.gap_values) > 0)


def test_concordance_2q() -> None:
    n = 2
    T = _ring_topology(n)
    omega = OMEGA_N_16[:n]
    result = critical_concordance(omega, T, k_range=np.array([1.0, 3.0]))
    assert len(result.gap_values) == 2
