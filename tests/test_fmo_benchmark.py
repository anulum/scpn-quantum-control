# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Fmo Benchmark
"""Tests for FMO photosynthetic complex benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.fmo_benchmark import (
    FMO_COUPLING,
    FMO_SITE_ENERGIES,
    fmo_benchmark,
    fmo_coupling_matrix,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestFMOData:
    def test_fmo_coupling_symmetric(self):
        np.testing.assert_allclose(FMO_COUPLING, FMO_COUPLING.T, atol=1e-10)

    def test_fmo_coupling_7x7(self):
        assert FMO_COUPLING.shape == (7, 7)

    def test_fmo_site_energies_7(self):
        assert FMO_SITE_ENERGIES.shape == (7,)

    def test_fmo_diagonal_zero(self):
        np.testing.assert_allclose(np.diag(FMO_COUPLING), 0.0)

    def test_builtin_reference_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            fmo_coupling_matrix()

    def test_fmo_coupling_matrix_units(self):
        K, omega = fmo_coupling_matrix(allow_builtin_reference=True)
        assert K.shape == (7, 7)
        assert omega.shape == (7,)
        assert np.all(K >= 0)  # absolute values
        assert np.all(np.diag(K) == 0)


class TestFMOBenchmark:
    def test_benchmark_returns_result(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_oscillators == 7
        assert -1 <= result.topology_correlation <= 1
        assert -1 <= result.frequency_correlation <= 1
        assert result.coupling_ratio > 0
        assert result.frequency_ratio > 0

    def test_benchmark_self_comparison(self):
        """FMO compared against itself should have perfect correlation."""
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        result = fmo_benchmark(K_fmo, omega_fmo, allow_builtin_reference=True)
        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

    def test_benchmark_4_oscillators(self):
        """Works with fewer than 7 oscillators."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_oscillators == 4

    def test_low_topology_match_reports_no_correlation(self):
        rng = np.random.default_rng(1)
        K = rng.random((7, 7))
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 0.0)
        result = fmo_benchmark(K, OMEGA_N_16[:7], allow_builtin_reference=True)
        assert abs(result.topology_correlation) <= 0.3
        assert "no correlation" in result.summary

    def test_benchmark_refuses_implicit_builtin_reference(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            fmo_benchmark(K, omega)

    def test_result_labels_builtin_reference_source_mode(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.source_mode == "builtin_literature_reference"
        assert result.publication_safe is False

    def test_result_labels_measured_reference_source_mode(self):
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        result = fmo_benchmark(
            K_fmo,
            omega_fmo,
            fmo_coupling=K_fmo,
            fmo_frequencies=omega_fmo,
        )
        assert result.source_mode == "measured"
        assert result.publication_safe is True

    def test_reference_matrix_and_frequency_vector_must_be_supplied_together(self):
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        with pytest.raises(ValueError, match="fmo_coupling and fmo_frequencies"):
            fmo_benchmark(K_fmo, omega_fmo, fmo_coupling=K_fmo)

    def test_scpn_vs_fmo_topology(self):
        """Record SCPN vs FMO topology correlation — this is Gap 1 data."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        # The correlation value IS the finding — record it
        print(f"\n  SCPN vs FMO topology correlation: {result.topology_correlation:.3f}")
        print(f"  SCPN vs FMO frequency correlation: {result.frequency_correlation:.3f}")
        print(f"  {result.summary}")
        # No assertion on the value — this is measurement, not validation
        assert isinstance(result.topology_correlation, float)

    def test_summary_string(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert "SCPN vs FMO" in result.summary
        assert "topology" in result.summary

    def test_rejects_non_square_scpn_coupling(self):
        K = np.ones((2, 3))
        omega = np.ones(2)
        with pytest.raises(ValueError, match="K_scpn must be a square"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_scpn_frequency_shape_mismatch(self):
        K = build_knm_paper27(L=4)
        omega = np.ones(3)
        with pytest.raises(ValueError, match="omega_scpn must match"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_non_finite_scpn_coupling(self):
        K = build_knm_paper27(L=4)
        K[0, 1] = np.nan
        omega = OMEGA_N_16[:4]
        with pytest.raises(ValueError, match="K_scpn must contain only finite"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_too_small_comparison_system(self):
        K = np.array([[0.0]])
        omega = np.array([1.0])
        with pytest.raises(ValueError, match="at least two coupled sites"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_fmo_coupling_asymmetry(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.2, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        fmo_omega = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="fmo_coupling must be symmetric"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=fmo_omega)

    def test_rejects_fmo_coupling_nonzero_diagonal(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        fmo_omega = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="fmo_coupling diagonal must be zero"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=fmo_omega)

    def test_rejects_fmo_frequency_shape_mismatch(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="fmo_frequencies must match"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=np.ones(2))


# ---------------------------------------------------------------------------
# FMO physics: biological coupling structure
# ---------------------------------------------------------------------------


class TestFMOPhysics:
    def test_fmo_coupling_non_negative(self):
        """FMO coupling (absolute) must be non-negative."""
        K, _ = fmo_coupling_matrix(allow_builtin_reference=True)
        assert np.all(K >= 0)

    def test_fmo_site_energies_ordered(self):
        """FMO site energies span a range (not all equal)."""
        assert FMO_SITE_ENERGIES.max() > FMO_SITE_ENERGIES.min()


# ---------------------------------------------------------------------------
# Pipeline: FMO coupling → benchmark → correlation → wired
# ---------------------------------------------------------------------------


class TestFMOPipeline:
    def test_pipeline_fmo_to_benchmark(self):
        """Full pipeline: FMO coupling → benchmark → topology correlation.
        Verifies FMO benchmark is wired and produces cross-domain data.
        """
        import time

        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)

        t0 = time.perf_counter()
        result = fmo_benchmark(K_fmo, omega_fmo, allow_builtin_reference=True)
        dt = (time.perf_counter() - t0) * 1000

        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

        print(f"\n  PIPELINE FMO→Benchmark (7 sites): {dt:.1f} ms")
        print(f"  ρ_topo={result.topology_correlation:.4f}")
