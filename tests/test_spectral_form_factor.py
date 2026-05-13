# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Spectral Form Factor
"""Tests for Spectral Form Factor at the synchronization transition."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.analysis.spectral_form_factor as sff_mod
from scpn_quantum_control.analysis.magnetisation_sectors import level_spacing_by_magnetisation
from scpn_quantum_control.analysis.spectral_form_factor import (
    SFFResult,
    SFFScanResult,
    _level_spacing_ratio,
    compute_sff,
    sff_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.dense_budget import DenseAllocationError


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestComputeSFF:
    def test_returns_result(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=50)
        assert isinstance(result, SFFResult)
        assert len(result.sff) == 50

    def test_sff_starts_at_one(self):
        """K(0) = |Tr(I)|²/d² = d²/d² = 1."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=50)
        assert abs(result.sff[0] - 1.0) < 1e-10

    def test_sff_bounded(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=10.0, n_times=100)
        assert np.all(result.sff >= 0)
        assert np.all(result.sff <= 1.0 + 1e-10)

    def test_level_spacing_ratio_bounded(self):
        """r̄ should be between 0 and 1."""
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]
        result = compute_sff(K, omega)
        assert 0 <= result.level_spacing_ratio <= 1.0

    def test_4qubit(self):
        K = 1.5 * _ring(4)
        omega = OMEGA_N_16[:4]
        result = compute_sff(K, omega, t_max=5.0, n_times=30)
        assert result.spectral_gap > 0
        assert len(result.times) == 30

    def test_rejects_dense_budget_before_hamiltonian_allocation(self, monkeypatch):
        def fail_if_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(sff_mod, "knm_to_dense_matrix", fail_if_dense_hamiltonian_is_requested)
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]

        with pytest.raises(DenseAllocationError, match="SFF dense eigensolver"):
            compute_sff(K, omega, t_max=1.0, n_times=2, max_dense_gib=1e-5)

    def test_default_level_spacing_uses_magnetisation_sector(self):
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]

        result = compute_sff(K, omega, t_max=1.0, n_times=4)
        expected = level_spacing_by_magnetisation(K, omega, M=None)["r_bar"]

        assert result.level_spacing_basis == "magnetisation"
        assert result.level_spacing_sector == 0
        assert result.level_spacing_sector_dim > 0
        assert result.level_spacing_ratio == pytest.approx(expected)

    def test_full_level_spacing_mode_preserves_full_spectrum_ratio(self):
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]

        result = compute_sff(K, omega, t_max=1.0, n_times=4, level_spacing_basis="full")

        assert result.level_spacing_basis == "full"
        assert result.level_spacing_sector is None
        assert result.level_spacing_sector_dim == 16
        assert result.level_spacing_ratio == pytest.approx(
            result.full_spectrum_level_spacing_ratio
        )

    def test_full_and_sector_level_spacing_are_reported_separately(self):
        K = 2.0 * _ring(4)
        omega = OMEGA_N_16[:4]

        result = compute_sff(K, omega, t_max=1.0, n_times=4)

        assert result.full_spectrum_level_spacing_ratio == pytest.approx(
            _level_spacing_ratio(np.linalg.eigvalsh(sff_mod.knm_to_dense_matrix(K, omega)))
        )
        assert result.level_spacing_ratio == pytest.approx(
            level_spacing_by_magnetisation(K, omega)["r_bar"]
        )


class TestSFFVsCoupling:
    def test_returns_result(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = sff_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, SFFScanResult)
        assert len(result.level_spacing_ratios) == 3

    def test_r_bar_varies(self):
        """Level spacing ratio should change across the scan."""
        T = _ring(4)
        omega = OMEGA_N_16[:4]
        result = sff_vs_coupling(omega, T, k_range=np.linspace(0.5, 5.0, 6))
        assert not np.all(result.level_spacing_ratios == result.level_spacing_ratios[0])

    def test_dip_depth_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = sff_vs_coupling(omega, T, k_range=np.array([1.0, 3.0]))
        assert np.all(np.isfinite(result.sff_dip_depth))
        assert np.all(result.sff_dip_depth >= 0)

    def test_nan_sector_ratios_do_not_trigger_chaos_onset(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]

        result = sff_vs_coupling(
            omega,
            T,
            k_range=np.array([1.0, 2.0]),
            level_spacing_basis="magnetisation",
        )

        assert np.all(np.isnan(result.level_spacing_ratios))
        assert result.chaos_onset_K is None


# ---------------------------------------------------------------------------
# SFF physics invariants
# ---------------------------------------------------------------------------


class TestSFFPhysics:
    def test_sff_at_t_zero_equals_one(self):
        """K(t=0) = 1 exactly (trace of identity squared / d²)."""
        K = 1.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        result = compute_sff(K, omega, t_max=1.0, n_times=10)
        np.testing.assert_allclose(result.sff[0], 1.0, atol=1e-10)

    def test_times_monotonic(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega, t_max=5.0, n_times=20)
        assert np.all(np.diff(result.times) > 0)

    def test_spectral_gap_positive(self):
        """Spectral gap (E1-E0) must be positive for non-trivial coupling."""
        K = 3.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        result = compute_sff(K, omega)
        assert result.spectral_gap > 0

    def test_eigenvalues_real(self):
        """Eigenvalues of Hermitian H must be real (verified via SFF computation)."""
        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        result = compute_sff(K, omega, t_max=1.0, n_times=5)
        # If eigenvalues were complex, sff would diverge
        assert np.all(np.isfinite(result.sff))


# ---------------------------------------------------------------------------
# Pipeline: Knm → SFF → level statistics → RMT wiring
# ---------------------------------------------------------------------------


class TestSFFPipeline:
    def test_knm_to_sff_wired(self):
        """Pipeline: build_knm_paper27 → compute_sff → level_spacing_ratio."""
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = compute_sff(K, omega, t_max=3.0, n_times=20)
        dt = (time.perf_counter() - t0) * 1000

        assert len(result.sff) == 20
        assert 0 <= result.level_spacing_ratio <= 1.0

        print(f"\n  PIPELINE Knm→SFF (4q, 20 time points): {dt:.1f} ms")
        print(f"  r̄ = {result.level_spacing_ratio:.4f}, gap = {result.spectral_gap:.4f}")


class TestSFFSingleTimeDipDepth:
    """Cover line 137: dip_depths = 1.0 when SFF has single element."""

    def test_single_time_point(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = sff_vs_coupling(omega, K, k_range=np.array([1.0, 2.0]), n_times=1)
        # With n_times=1, sff has exactly 1 element → dip_depth = 1.0
        assert np.all(scan.sff_dip_depth == 1.0)
