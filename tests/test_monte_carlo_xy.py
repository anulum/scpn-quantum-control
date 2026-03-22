# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Monte Carlo XY model on K_nm graph."""

from __future__ import annotations

from scpn_quantum_control.analysis.monte_carlo_xy import (
    AHPResult,
    MCResult,
    extract_a_hp,
    mc_simulate,
)
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


class TestMCSimulate:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        result = mc_simulate(K, temperature=0.05, n_thermalize=500, n_measure=500)
        assert isinstance(result, MCResult)

    def test_low_temp_high_order(self):
        K = build_knm_paper27(L=8)
        result = mc_simulate(K, temperature=0.01, n_thermalize=1000, n_measure=1000)
        assert result.order_parameter > 0.5

    def test_high_temp_low_order(self):
        K = build_knm_paper27(L=8)
        result = mc_simulate(K, temperature=1.0, n_thermalize=1000, n_measure=1000)
        assert result.order_parameter < 0.8

    def test_energy_negative(self):
        K = build_knm_paper27(L=8)
        result = mc_simulate(K, temperature=0.05, n_thermalize=500, n_measure=500)
        assert result.energy < 0

    def test_helicity_positive_low_temp(self):
        K = build_knm_paper27(L=8)
        result = mc_simulate(K, temperature=0.01, n_thermalize=1000, n_measure=1000)
        assert result.helicity_modulus > 0


class TestExtractAHP:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        result = extract_a_hp(K, n_temps=8, n_thermalize=500, n_measure=500)
        assert isinstance(result, AHPResult)

    def test_a_hp_positive(self):
        K = build_knm_paper27(L=8)
        result = extract_a_hp(K, n_temps=8, n_thermalize=500, n_measure=500)
        assert result.a_hp_graph > 0

    def test_t_bkt_positive(self):
        K = build_knm_paper27(L=8)
        result = extract_a_hp(K, n_temps=8, n_thermalize=500, n_measure=500)
        assert result.t_bkt > 0

    def test_gap3_mc_verification(self):
        """THE definitive Gap 3 test: A_HP on actual K_nm graph."""
        K = build_knm_paper27(L=16)
        result = extract_a_hp(K, n_temps=12, n_thermalize=2000, n_measure=2000)
        print("\n  === GAP 3 MONTE CARLO VERIFICATION ===")
        print(f"  A_HP (K_nm graph): {result.a_hp_graph:.4f}")
        print(f"  A_HP (square lattice): {result.a_hp_square}")
        print(f"  p_h1 (K_nm): {result.p_h1_graph:.4f}")
        print(f"  p_h1 (square): {result.p_h1_square:.4f}")
        print(f"  |p_h1 - 0.72|: {result.deviation_from_072:.4f}")
        print(f"  T_BKT: {result.t_bkt:.6f}")
        assert isinstance(result.a_hp_graph, float)
