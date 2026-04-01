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

    def test_fmo_coupling_matrix_units(self):
        K, omega = fmo_coupling_matrix()
        assert K.shape == (7, 7)
        assert omega.shape == (7,)
        assert np.all(K >= 0)  # absolute values
        assert np.all(np.diag(K) == 0)


class TestFMOBenchmark:
    def test_benchmark_returns_result(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega)
        assert result.n_oscillators == 7
        assert -1 <= result.topology_correlation <= 1
        assert -1 <= result.frequency_correlation <= 1
        assert result.coupling_ratio > 0
        assert result.frequency_ratio > 0

    def test_benchmark_self_comparison(self):
        """FMO compared against itself should have perfect correlation."""
        K_fmo, omega_fmo = fmo_coupling_matrix()
        result = fmo_benchmark(K_fmo, omega_fmo)
        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

    def test_benchmark_4_oscillators(self):
        """Works with fewer than 7 oscillators."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = fmo_benchmark(K, omega)
        assert result.n_oscillators == 4

    def test_scpn_vs_fmo_topology(self):
        """Record SCPN vs FMO topology correlation — this is Gap 1 data."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega)
        # The correlation value IS the finding — record it
        print(f"\n  SCPN vs FMO topology correlation: {result.topology_correlation:.3f}")
        print(f"  SCPN vs FMO frequency correlation: {result.frequency_correlation:.3f}")
        print(f"  {result.summary}")
        # No assertion on the value — this is measurement, not validation
        assert isinstance(result.topology_correlation, float)

    def test_summary_string(self):
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega)
        assert "SCPN vs FMO" in result.summary
        assert "topology" in result.summary


# ---------------------------------------------------------------------------
# FMO physics: biological coupling structure
# ---------------------------------------------------------------------------


class TestFMOPhysics:
    def test_fmo_coupling_non_negative(self):
        """FMO coupling (absolute) must be non-negative."""
        K, _ = fmo_coupling_matrix()
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

        K_fmo, omega_fmo = fmo_coupling_matrix()

        t0 = time.perf_counter()
        result = fmo_benchmark(K_fmo, omega_fmo)
        dt = (time.perf_counter() - t0) * 1000

        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

        print(f"\n  PIPELINE FMO→Benchmark (7 sites): {dt:.1f} ms")
        print(f"  ρ_topo={result.topology_correlation:.4f}")
