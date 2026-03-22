# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for circuit cutting."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.circuit_cutting import (
    CircuitCuttingPlan,
    circuit_cutting_plan,
    count_inter_partition_couplings,
    optimal_partition,
    scaling_analysis,
)


class TestOptimalPartition:
    def test_16_into_16(self):
        K = build_knm_paper27(L=16)
        parts = optimal_partition(K, max_partition_size=16)
        assert len(parts) == 1
        assert len(parts[0]) == 16

    def test_32_into_2x16(self):
        K = build_knm_paper27(L=32)
        parts = optimal_partition(K, max_partition_size=16)
        assert len(parts) == 2

    def test_covers_all(self):
        K = build_knm_paper27(L=20)
        parts = optimal_partition(K, max_partition_size=8)
        all_osc = sorted(i for p in parts for i in p)
        assert all_osc == list(range(20))


class TestCountCuts:
    def test_single_partition_zero_cuts(self):
        K = build_knm_paper27(L=4)
        parts = [[0, 1, 2, 3]]
        assert count_inter_partition_couplings(K, parts) == 0

    def test_two_partitions_nonzero(self):
        K = build_knm_paper27(L=4)
        parts = [[0, 1], [2, 3]]
        cuts = count_inter_partition_couplings(K, parts)
        assert cuts > 0


class TestCircuitCuttingPlan:
    def test_returns_plan(self):
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K)
        assert isinstance(plan, CircuitCuttingPlan)

    def test_16_no_cuts(self):
        K = build_knm_paper27(L=16)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.n_cuts == 0
        assert plan.n_partitions == 1

    def test_32_needs_cuts(self):
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.n_cuts > 0
        assert plan.n_partitions == 2

    def test_fits_heron(self):
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.fits_on_heron


class TestScalingAnalysis:
    def test_returns_keys(self):
        results = scaling_analysis(n_values=[16, 32])
        assert "n_oscillators" in results
        assert "n_cuts" in results
        assert len(results["n_oscillators"]) == 2

    def test_cuts_increase_with_n(self):
        results = scaling_analysis(n_values=[16, 32, 64])
        assert results["n_cuts"][0] <= results["n_cuts"][1] <= results["n_cuts"][2]
