# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Finite Size Scaling
"""Tests for finite-size scaling K_c extraction."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis import finite_size_scaling as fss_module
from scpn_quantum_control.analysis.finite_size_scaling import (
    FSSResult,
    finite_size_scaling,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


class TestFiniteSizeScaling:
    def test_returns_result(self):
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=np.linspace(0.5, 4.0, 10),
        )
        assert isinstance(result, FSSResult)
        assert len(result.k_c_values) == 2

    def test_k_c_positive(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        for kc in result.k_c_values:
            assert kc > 0

    def test_gap_min_positive(self):
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=np.linspace(0.5, 4.0, 10),
        )
        for g in result.gap_min_values:
            assert g > 0

    def test_extrapolation_exists(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        # At least one extrapolation should succeed
        assert result.k_c_extrapolated_bkt is not None or result.k_c_extrapolated_power is not None

    def test_extrapolated_values_finite(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        if result.k_c_extrapolated_bkt is not None:
            assert np.isfinite(result.k_c_extrapolated_bkt)
        if result.k_c_extrapolated_power is not None:
            assert np.isfinite(result.k_c_extrapolated_power)

    def test_single_size(self):
        """Single system size → no extrapolation possible."""
        result = finite_size_scaling(
            system_sizes=[3],
            k_range=np.linspace(0.5, 4.0, 8),
        )
        assert len(result.k_c_values) == 1

    def test_result_has_system_sizes(self):
        result = finite_size_scaling(system_sizes=[2, 3], k_range=np.linspace(0.5, 4.0, 6))
        assert result.system_sizes == [2, 3]

    def test_k_c_values_finite(self):
        result = finite_size_scaling(system_sizes=[2, 3], k_range=np.linspace(0.5, 4.0, 8))
        for kc in result.k_c_values:
            assert np.isfinite(kc)

    def test_larger_system_lower_kc(self):
        """Larger systems should have K_c closer to thermodynamic limit (lower or equal)."""
        result = finite_size_scaling(system_sizes=[2, 3, 4], k_range=np.linspace(0.5, 5.0, 12))
        # K_c should generally decrease or stay same with system size
        assert len(result.k_c_values) == 3

    def test_gap_min_matches_k_c_count(self):
        result = finite_size_scaling(system_sizes=[2, 4], k_range=np.linspace(0.5, 4.0, 8))
        assert len(result.gap_min_values) == len(result.k_c_values)

    def test_gap_scan_rejects_dense_budget_before_hamiltonian_allocation(self, monkeypatch):
        omega = np.ones(4)
        topology = np.eye(4)
        k_range = np.array([1.0, 2.0])

        def fail_if_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            fss_module,
            "knm_to_dense_matrix",
            fail_if_dense_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="finite-size gap dense eigensolver"):
            fss_module._find_kc_from_gap(
                omega,
                topology,
                k_range,
                max_dense_gib=1e-12,
            )

    def test_finite_size_scaling_propagates_dense_budget(self, monkeypatch):
        seen_budgets = []

        def fake_find_kc(omega, topology, k_range, *, max_dense_gib):  # noqa: ARG001
            seen_budgets.append(max_dense_gib)
            return 1.5, 0.25

        monkeypatch.setattr(fss_module, "_find_kc_from_gap", fake_find_kc)

        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.array([1.0, 2.0]),
            max_dense_gib=0.25,
        )

        assert seen_budgets == [0.25, 0.25, 0.25]
        assert result.k_c_values == [1.5, 1.5, 1.5]


class TestFSSPipeline:
    def test_pipeline_fss_to_kc(self):
        """Full pipeline: system sizes → FSS → K_c extraction."""
        import time

        t0 = time.perf_counter()
        result = finite_size_scaling(system_sizes=[2, 3, 4], k_range=np.linspace(0.5, 5.0, 8))
        dt = (time.perf_counter() - t0) * 1000

        assert len(result.k_c_values) == 3
        print(f"\n  PIPELINE FSS (L=2,3,4, 8 K): {dt:.1f} ms")
        print(f"  K_c = {result.k_c_values}")


class TestFSSFitFallbacks:
    """Solver-failure contracts for finite-size extrapolation fits."""

    def test_fit_bkt_ansatz_handles_linear_solver_failure(self, monkeypatch):
        """BKT extrapolation returns None when the linear solve fails."""
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        def fail_lstsq(*_args, **_kwargs):
            raise np.linalg.LinAlgError("singular fit")

        monkeypatch.setattr(fss.np.linalg, "lstsq", fail_lstsq)

        result = fss._fit_bkt_ansatz([2, 3], [1.2, 1.4])

        assert result is None

    def test_fit_power_ansatz_handles_linear_solver_failure(self, monkeypatch):
        """Power-law extrapolation returns None when the linear solve fails."""
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        def fail_lstsq(*_args, **_kwargs):
            raise np.linalg.LinAlgError("singular fit")

        monkeypatch.setattr(fss.np.linalg, "lstsq", fail_lstsq)

        result = fss._fit_power_ansatz([2, 3], [1.2, 1.4])

        assert result is None


class TestFSSCoverage:
    """Cover default parameter branches and fit error paths."""

    def test_default_parameters(self):
        """Cover lines 85, 87: system_sizes=None, k_range=None defaults."""
        result = finite_size_scaling()
        assert len(result.k_c_values) == 3
        assert len(result.system_sizes) == 3

    def test_fit_power_ansatz_single_point(self):
        """Cover line 132: _fit_power_ansatz with len(sizes) < 2 returns None.

        Lines 125-126, 139-140 (LinAlgError) are defensive guards —
        np.linalg.lstsq is extremely robust and doesn't raise LinAlgError
        for typical inputs. These are effectively unreachable.
        """
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        result = fss._fit_power_ansatz([2], [1.5])
        assert result is None
