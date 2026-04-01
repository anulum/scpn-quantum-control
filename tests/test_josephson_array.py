# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Josephson Array
"""Tests for Josephson junction array mapping."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.applications.josephson_array import (
    JosephsonBenchmarkResult,
    jja_coupling_matrix,
    josephson_benchmark,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestJJACouplingMatrix:
    def test_linear_shape(self):
        K, omega = jja_coupling_matrix(5, topology="linear")
        assert K.shape == (5, 5)
        assert omega.shape == (5,)

    def test_linear_tridiagonal(self):
        K, _ = jja_coupling_matrix(5, topology="linear")
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:
                    assert K[i, j] == 0.0

    def test_symmetric(self):
        for topo in ["linear", "heavy_hex", "all_to_all"]:
            K, _ = jja_coupling_matrix(6, topology=topo)
            np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_non_negative(self):
        for topo in ["linear", "heavy_hex", "all_to_all"]:
            K, _ = jja_coupling_matrix(6, topology=topo)
            assert np.all(K >= 0)

    def test_all_to_all_fully_connected(self):
        K, _ = jja_coupling_matrix(4, topology="all_to_all")
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert K[i, j] > 0


class TestJosephsonBenchmark:
    def test_returns_result(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = josephson_benchmark(K, omega)
        assert isinstance(result, JosephsonBenchmarkResult)

    def test_n_junctions(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = josephson_benchmark(K, omega)
        assert result.n_junctions == 4

    def test_transmon_regime(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = josephson_benchmark(K, omega)
        assert result.is_transmon_regime  # E_J/E_C = 60 > 20

    def test_topology_correlation_bounded(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = josephson_benchmark(K, omega)
        assert -1 <= result.topology_correlation <= 1

    def test_summary_string(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = josephson_benchmark(K, omega)
        assert "SCPN vs JJA" in result.summary

    def test_different_topologies(self):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        for topo in ["linear", "heavy_hex", "all_to_all"]:
            result = josephson_benchmark(K, omega, topology=topo)
            assert isinstance(result.topology_correlation, float)

    def test_scpn_vs_jja(self):
        """Record SCPN vs JJA — self-simulation data."""
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        result = josephson_benchmark(K, omega, topology="all_to_all")
        print(f"\n  {result.summary}")
        assert isinstance(result.topology_correlation, float)
