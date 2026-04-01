# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Cft Analysis
"""Tests for CFT central charge extraction."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.cft_analysis import (
    CFTResult,
    cft_analysis,
    extract_central_charge,
    find_critical_coupling,
)


class TestFindCriticalCoupling:
    def test_returns_positive_k(self):
        omega = OMEGA_N_16[:4]
        k_c, k_vals, entropies = find_critical_coupling(omega, n_points=10)
        assert k_c > 0

    def test_entropies_length(self):
        omega = OMEGA_N_16[:4]
        _k_c, k_vals, entropies = find_critical_coupling(omega, n_points=15)
        assert len(entropies) == 15
        assert len(k_vals) == 15

    def test_k_c_in_range(self):
        omega = OMEGA_N_16[:4]
        k_c, _k_vals, _entropies = find_critical_coupling(omega, k_range=(0.1, 3.0), n_points=10)
        assert 0.1 <= k_c <= 3.0


class TestExtractCentralCharge:
    def test_returns_float_or_none(self):
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        c = extract_central_charge(K, omega)
        assert c is None or isinstance(c, float)

    def test_small_system_may_return_none(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        c = extract_central_charge(K, omega)
        # n//2 = 1, only 1 data point, fit needs >= 3
        assert c is None

    def test_6_oscillators_returns_value(self):
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        c = extract_central_charge(K, omega)
        assert c is not None
        assert isinstance(c, float)


class TestCFTAnalysis:
    def test_returns_result(self):
        omega = OMEGA_N_16[:4]
        result = cft_analysis(omega, n_points=8)
        assert isinstance(result, CFTResult)

    def test_k_critical_positive(self):
        omega = OMEGA_N_16[:4]
        result = cft_analysis(omega, n_points=8)
        assert result.k_critical > 0

    def test_peak_entropy_positive(self):
        omega = OMEGA_N_16[:4]
        result = cft_analysis(omega, n_points=8)
        assert result.peak_entropy >= 0

    def test_lengths_match(self):
        omega = OMEGA_N_16[:4]
        result = cft_analysis(omega, n_points=10)
        assert len(result.entropy_vs_k) == 10
        assert len(result.k_values) == 10

    def test_scpn_cft_extraction(self):
        """Record central charge for SCPN system."""
        omega = OMEGA_N_16[:6]
        result = cft_analysis(omega, n_points=15)
        print(f"\n  K_critical = {result.k_critical:.4f}")
        print(f"  Peak entropy S(n/2) = {result.peak_entropy:.6f}")
        print(f"  Central charge c = {result.central_charge}")
        if result.deviation_from_c1 is not None:
            print(f"  |c - 1| = {result.deviation_from_c1:.4f}")
        assert isinstance(result.k_critical, float)
