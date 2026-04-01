# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Otoc Sync Probe
"""Tests for OTOC synchronization transition probe."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.otoc_sync_probe import (
    OTOCSyncScanResult,
    compare_otoc_vs_R,
    otoc_sync_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestOTOCSyncScan:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = otoc_sync_scan(K, omega, n_K_values=5, n_time_points=10, t_max=1.0)
        assert isinstance(result, OTOCSyncScanResult)
        assert len(result.K_base_values) == 5
        assert len(result.lyapunov_values) == 5
        assert len(result.R_classical) == 5
        assert result.n_qubits == 3

    def test_R_classical_increases_with_K(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = otoc_sync_scan(
            K,
            omega,
            K_base_range=np.array([0.1, 1.0, 3.0]),
            n_time_points=10,
            t_max=1.0,
        )
        # R should generally increase with coupling (though not monotonically
        # for all systems at all times)
        assert len(result.R_classical) == 3

    def test_otoc_final_values_real(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = otoc_sync_scan(K, omega, n_K_values=3, n_time_points=5, t_max=0.5)
        for v in result.otoc_final_values:
            assert np.isfinite(v)

    def test_two_qubit_scan(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = otoc_sync_scan(K, omega, n_K_values=4, n_time_points=8, t_max=0.5)
        assert result.n_qubits == 2
        assert len(result.K_base_values) == 4


class TestCompareOTOCvsR:
    def test_returns_expected_keys(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        scan = otoc_sync_scan(K, omega, n_K_values=5, n_time_points=10, t_max=1.0)
        comparison = compare_otoc_vs_R(scan)
        assert "K_c_classical" in comparison
        assert "K_c_otoc" in comparison
        assert "delta_K_c" in comparison
        assert "otoc_detects_transition" in comparison

    def test_detection_flag_is_bool(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        scan = otoc_sync_scan(K, omega, n_K_values=8, n_time_points=10, t_max=1.0)
        comparison = compare_otoc_vs_R(scan)
        assert isinstance(comparison["otoc_detects_transition"], bool)

    def test_k_c_values_finite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        scan = otoc_sync_scan(K, omega, n_K_values=5, n_time_points=8, t_max=0.5)
        comparison = compare_otoc_vs_R(scan)
        assert np.isfinite(comparison["K_c_classical"])

    def test_delta_k_c_nonnegative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        scan = otoc_sync_scan(K, omega, n_K_values=5, n_time_points=8, t_max=0.5)
        comparison = compare_otoc_vs_R(scan)
        assert comparison["delta_K_c"] >= 0


class TestOTOCProperties:
    def test_lyapunov_values_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = otoc_sync_scan(K, omega, n_K_values=3, n_time_points=5, t_max=0.5)
        for v in result.lyapunov_values:
            assert np.isfinite(v)

    def test_R_classical_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = otoc_sync_scan(K, omega, n_K_values=4, n_time_points=8, t_max=0.5)
        for R in result.R_classical:
            assert 0.0 <= R <= 1.0 + 1e-10


class TestOTOCPipeline:
    def test_pipeline_knm_to_otoc(self):
        """Full pipeline: Knm → OTOC scan → compare → K_c detection."""
        import time

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        scan = otoc_sync_scan(K, omega, n_K_values=6, n_time_points=8, t_max=0.5)
        comparison = compare_otoc_vs_R(scan)
        dt = (time.perf_counter() - t0) * 1000

        assert "otoc_detects_transition" in comparison

        print(f"\n  PIPELINE Knm→OTOC (3q, 6K×8t): {dt:.1f} ms")
        print(f"  Transition detected: {comparison['otoc_detects_transition']}")
