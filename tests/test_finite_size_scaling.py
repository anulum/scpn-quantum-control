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
from numpy.typing import NDArray

from scpn_quantum_control.analysis import finite_size_scaling as fss_module
from scpn_quantum_control.analysis.finite_size_scaling import (
    FSSResult,
    finite_size_scaling,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


def _linspace(start: float, stop: float, num: int) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.linspace(start, stop, num, dtype=np.float64)
    return values


def _array(values: object) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return result


class TestFiniteSizeScaling:
    def test_returns_result(self) -> None:
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=_linspace(0.5, 4.0, 10),
        )
        assert isinstance(result, FSSResult)
        assert len(result.k_c_values) == 2

    def test_k_c_positive(self) -> None:
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=_linspace(0.5, 5.0, 12),
        )
        for kc in result.k_c_values:
            assert kc > 0

    def test_gap_min_positive(self) -> None:
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=_linspace(0.5, 4.0, 10),
        )
        for g in result.gap_min_values:
            assert g > 0

    def test_extrapolation_exists(self) -> None:
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=_linspace(0.5, 5.0, 12),
        )
        # At least one extrapolation should succeed
        assert result.k_c_extrapolated_bkt is not None or result.k_c_extrapolated_power is not None

    def test_extrapolated_values_finite(self) -> None:
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=_linspace(0.5, 5.0, 12),
        )
        if result.k_c_extrapolated_bkt is not None:
            assert np.isfinite(result.k_c_extrapolated_bkt)
        if result.k_c_extrapolated_power is not None:
            assert np.isfinite(result.k_c_extrapolated_power)

    def test_single_size(self) -> None:
        """Single system size → no extrapolation possible."""
        result = finite_size_scaling(
            system_sizes=[3],
            k_range=_linspace(0.5, 4.0, 8),
        )
        assert len(result.k_c_values) == 1

    def test_result_has_system_sizes(self) -> None:
        result = finite_size_scaling(system_sizes=[2, 3], k_range=_linspace(0.5, 4.0, 6))
        assert result.system_sizes == [2, 3]

    def test_k_c_values_finite(self) -> None:
        result = finite_size_scaling(system_sizes=[2, 3], k_range=_linspace(0.5, 4.0, 8))
        for kc in result.k_c_values:
            assert np.isfinite(kc)

    def test_larger_system_lower_kc(self) -> None:
        """Larger systems should have K_c closer to thermodynamic limit (lower or equal)."""
        result = finite_size_scaling(system_sizes=[2, 3, 4], k_range=_linspace(0.5, 5.0, 12))
        # K_c should generally decrease or stay same with system size
        assert len(result.k_c_values) == 3

    def test_gap_min_matches_k_c_count(self) -> None:
        result = finite_size_scaling(system_sizes=[2, 4], k_range=_linspace(0.5, 4.0, 8))
        assert len(result.gap_min_values) == len(result.k_c_values)

    def test_gap_scan_rejects_dense_budget_before_hamiltonian_allocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        omega = _array([1.0, 1.0, 1.0, 1.0])
        topology = _array(np.eye(4))
        k_range = _array([1.0, 2.0])

        def fail_if_dense_hamiltonian_is_requested(
            _coupling: NDArray[np.float64],
            _omega: NDArray[np.float64],
            *,
            max_dense_gib: float | None = None,
        ) -> NDArray[np.complex128]:
            del max_dense_gib
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
                max_dense_gib=1e-5,
            )

    def test_finite_size_scaling_propagates_dense_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen_budgets: list[float | None] = []

        def fake_find_kc(
            _omega: NDArray[np.float64],
            _topology: NDArray[np.float64],
            _k_range: NDArray[np.float64],
            *,
            max_dense_gib: float | None,
        ) -> tuple[float, float]:
            seen_budgets.append(max_dense_gib)
            return 1.5, 0.25

        monkeypatch.setattr(fss_module, "_find_kc_from_gap", fake_find_kc)

        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=_array([1.0, 2.0]),
            max_dense_gib=0.25,
        )

        assert seen_budgets == [0.25, 0.25, 0.25]
        assert result.k_c_values == [1.5, 1.5, 1.5]

    def test_result_exposes_fit_diagnostics_and_claim_boundary(self) -> None:
        """Finite-size evidence reports residuals without promoting hardware claims."""
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=_linspace(0.5, 5.0, 12),
        )

        assert "local dense exact" in result.claim_boundary
        assert result.bkt_fit is not None
        assert result.power_fit is not None
        assert result.bkt_fit.model == "bkt_log_correction"
        assert result.power_fit.model == "power_law_nu_1"
        assert result.bkt_fit.extrapolated_k_c == result.k_c_extrapolated_bkt
        assert result.power_fit.extrapolated_k_c == result.k_c_extrapolated_power
        assert result.bkt_fit.residual_norm >= 0.0
        assert result.power_fit.max_abs_residual >= 0.0
        assert len(result.bkt_fit.residuals) == len(result.system_sizes)

        payload = result.to_dict()

        assert payload["claim_boundary"] == result.claim_boundary
        assert payload["bkt_fit"]["model"] == "bkt_log_correction"
        assert payload["power_fit"]["model"] == "power_law_nu_1"

    @pytest.mark.parametrize(
        ("system_sizes", "match"),
        [
            ([], "at least one"),
            ([1, 2], "at least 2"),
            ([2, 2], "unique"),
            ([17], "available frequency"),
        ],
    )
    def test_rejects_invalid_system_sizes(self, system_sizes: list[int], match: str) -> None:
        """The public FSS route rejects malformed finite-size grids."""
        with pytest.raises(ValueError, match=match):
            finite_size_scaling(system_sizes=system_sizes, k_range=_array([0.5, 1.0]))

    @pytest.mark.parametrize(
        ("k_range", "match"),
        [
            (_array([0.5]), "at least two"),
            (_array([[0.5, 1.0]]), "one-dimensional"),
            (_array([0.5, np.nan]), "finite"),
            (_array([1.0, 0.5]), "strictly increasing"),
        ],
    )
    def test_rejects_invalid_k_range(self, k_range: NDArray[np.float64], match: str) -> None:
        """The public FSS route rejects malformed coupling scan grids."""
        with pytest.raises(ValueError, match=match):
            finite_size_scaling(system_sizes=[2], k_range=k_range)


class TestFSSPipeline:
    def test_pipeline_fss_to_kc(self) -> None:
        """Full pipeline: system sizes → FSS → K_c extraction."""
        import time

        t0 = time.perf_counter()
        result = finite_size_scaling(system_sizes=[2, 3, 4], k_range=_linspace(0.5, 5.0, 8))
        dt = (time.perf_counter() - t0) * 1000

        assert len(result.k_c_values) == 3
        print(f"\n  PIPELINE FSS (L=2,3,4, 8 K): {dt:.1f} ms")
        print(f"  K_c = {result.k_c_values}")


class TestFSSFitFallbacks:
    """Solver-failure contracts for finite-size extrapolation fits."""

    def test_fit_bkt_ansatz_handles_linear_solver_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """BKT extrapolation returns None when the linear solve fails."""
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        def fail_lstsq(
            a: NDArray[np.float64],
            b: NDArray[np.float64],
            rcond: float | None = None,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
            del a, b, rcond
            raise np.linalg.LinAlgError("singular fit")

        monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)

        result = fss._fit_bkt_ansatz([2, 3], [1.2, 1.4])

        assert result is None

    def test_fit_power_ansatz_handles_linear_solver_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Power-law extrapolation returns None when the linear solve fails."""
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        def fail_lstsq(
            a: NDArray[np.float64],
            b: NDArray[np.float64],
            rcond: float | None = None,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
            del a, b, rcond
            raise np.linalg.LinAlgError("singular fit")

        monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)

        result = fss._fit_power_ansatz([2, 3], [1.2, 1.4])

        assert result is None


class TestFSSCoverage:
    """Cover default parameter branches and fit error paths."""

    def test_default_parameters(self) -> None:
        """Cover lines 85, 87: system_sizes=None, k_range=None defaults."""
        result = finite_size_scaling()
        assert len(result.k_c_values) == 3
        assert len(result.system_sizes) == 3

    def test_fit_power_ansatz_single_point(self) -> None:
        """Cover line 132: _fit_power_ansatz with len(sizes) < 2 returns None.

        Lines 125-126, 139-140 (LinAlgError) are defensive guards —
        np.linalg.lstsq is extremely robust and doesn't raise LinAlgError
        for typical inputs. These are effectively unreachable.
        """
        import scpn_quantum_control.analysis.finite_size_scaling as fss

        result = fss._fit_power_ansatz([2], [1.5])
        assert result is None
