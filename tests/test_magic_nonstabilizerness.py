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
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.magic_nonstabilizerness as magic_module
from scpn_quantum_control.analysis.magic_nonstabilizerness import (
    MagicResult,
    MagicScanResult,
    _compute_sre_m2,
    magic_at_coupling,
    magic_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.dense_budget import DenseAllocationError


def _ring(n: int) -> NDArray[np.float64]:
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestMagicAtCoupling:
    """Verify exact single-coupling SRE evaluation and its resource guard."""

    def test_rejects_dense_budget_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject an over-budget request before allocating its Hamiltonian."""
        n = 10
        T = _ring(n)
        omega = OMEGA_N_16[:n]

        def fail_if_dense_hamiltonian_is_requested(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            magic_module, "knm_to_dense_matrix", fail_if_dense_hamiltonian_is_requested
        )

        with pytest.raises(DenseAllocationError, match="magic dense"):
            magic_at_coupling(omega, T, K_base=1.0, max_dense_gib=1e-12)

    def test_passes_dense_budget_to_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forward the exact dense-allocation budget to the bridge."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_dense_matrix(
            _K_arg: NDArray[np.float64],
            _omega_arg: NDArray[np.float64],
            *,
            max_dense_gib: float | None = None,
        ) -> NDArray[np.complex128]:
            seen_budgets.append(max_dense_gib)
            return np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)

        monkeypatch.setattr(magic_module, "knm_to_dense_matrix", fake_dense_matrix)

        magic_at_coupling(omega, T, K_base=1.0, max_dense_gib=0.25)

        assert seen_budgets == [0.25]

    def test_returns_result(self) -> None:
        """Return the typed single-coupling result envelope."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=2.0)
        assert isinstance(result, MagicResult)
        assert result.n_qubits == 2

    def test_stabilizer_state_zero_magic(self) -> None:
        """Product state |00⟩ is a stabilizer state → M_2 ≈ 0."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=0.001)
        # Very weak coupling → ground state ≈ |00⟩ → stabilizer → M_2 ≈ 0
        assert result.sre_m2 < 0.5

    def test_entangled_state_has_magic(self) -> None:
        """Strong coupling → entangled ground state → M_2 > 0."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, T, K_base=3.0)
        assert result.sre_m2 > 0

    def test_sre_nonnegative(self) -> None:
        """Keep the computed SRE numerically non-negative."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, T, K_base=2.0)
        assert result.sre_m2 >= -1e-10

    def test_xi_sum_positive(self) -> None:
        """Return a positive raw Pauli fourth-moment sum."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_at_coupling(omega, T, K_base=1.0)
        assert result.xi_sum > 0

    def test_zero_fourth_moment_returns_maximum_magic(self) -> None:
        """Use the bounded maximum when the fourth moment underflows."""
        sre_m2, xi_sum = _compute_sre_m2(np.zeros(2, dtype=np.complex128), n=1)

        assert xi_sum == 0.0
        assert sre_m2 == 1.0


class TestMagicVsCoupling:
    """Verify finite-grid SRE scans and budget propagation."""

    def test_propagates_dense_budget_to_each_coupling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the dense budget to every sampled coupling."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        seen: list[float | None] = []

        def fake_magic_at_coupling(
            _omega_arg: NDArray[np.float64],
            _K_topology_arg: NDArray[np.float64],
            K_base: float,
            *,
            max_dense_gib: float | None,
        ) -> MagicResult:
            seen.append(max_dense_gib)
            return MagicResult(K_base, 0.1, 1.0, 2)

        monkeypatch.setattr(magic_module, "magic_at_coupling", fake_magic_at_coupling)

        magic_vs_coupling(omega, T, k_range=np.array([0.5, 1.0, 1.5]), max_dense_gib=0.5)

        assert seen == [0.5, 0.5, 0.5]

    def test_returns_scan(self) -> None:
        """Return the typed finite-grid scan envelope."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_vs_coupling(omega, T, k_range=np.array([0.5, 1.5, 3.0]))
        assert isinstance(result, MagicScanResult)
        assert len(result.sre_m2) == 3

    def test_magic_varies_with_K(self) -> None:
        """Distinguish SRE values across the selected coupling range."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 6, dtype=np.float64))
        assert result.sre_m2[0] != result.sre_m2[-1]

    def test_peak_exists(self) -> None:
        """Report a positive finite-grid maximum for the chosen system."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.1, 5.0, 8, dtype=np.float64))
        assert result.peak_magic > 0

    def test_default_k_range_has_documented_size(self) -> None:
        """Use the documented 15-point default coupling grid."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = magic_vs_coupling(omega, T)

        assert len(result.k_values) == 15
        np.testing.assert_allclose(result.k_values[[0, -1]], [0.5, 5.0])


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------


class TestMagicInvariants:
    """Verify numerical and alignment invariants across finite scans."""

    def test_sre_all_finite(self) -> None:
        """Keep every scanned SRE value finite."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = magic_vs_coupling(omega, T, k_range=np.linspace(0.5, 4.0, 5, dtype=np.float64))
        assert np.all(np.isfinite(result.sre_m2))

    def test_k_values_match_input(self) -> None:
        """Preserve the caller-supplied coupling grid exactly."""
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        k_range = np.array([1.0, 2.0, 3.0])
        result = magic_vs_coupling(omega, T, k_range=k_range)
        np.testing.assert_array_equal(result.k_values, k_range)

    def test_peak_magic_at_valid_k(self) -> None:
        """Choose a reported peak only from the supplied grid."""
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        k_range = np.linspace(0.5, 5.0, 6, dtype=np.float64)
        result = magic_vs_coupling(omega, T, k_range=k_range)
        assert result.peak_K in k_range


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestMagicPipeline:
    """Verify the real KNM-to-SRE integration path."""

    def test_knm_to_magic(self) -> None:
        """Pipeline: build_knm_paper27 → magic_at_coupling → SRE."""
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = magic_at_coupling(omega, K, K_base=2.0)
        assert isinstance(result, MagicResult)
        assert result.sre_m2 >= 0
