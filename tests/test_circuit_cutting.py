# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Circuit Cutting
"""Tests for circuit cutting."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.circuit_cutting import (
    CircuitCuttingPlan,
    circuit_cutting_plan,
    count_inter_partition_couplings,
    optimal_partition,
    scaling_analysis,
)


class TestOptimalPartition:
    def test_16_into_16(self) -> None:
        K = build_knm_paper27(L=16)
        parts = optimal_partition(K, max_partition_size=16)
        assert len(parts) == 1
        assert len(parts[0]) == 16

    def test_32_into_2x16(self) -> None:
        K = build_knm_paper27(L=32)
        parts = optimal_partition(K, max_partition_size=16)
        assert len(parts) == 2

    def test_128_into_8x16(self) -> None:
        K = build_knm_paper27(L=128)
        parts = optimal_partition(K, max_partition_size=16)
        assert len(parts) == 8
        assert all(len(part) == 16 for part in parts)

    def test_covers_all(self) -> None:
        K = build_knm_paper27(L=20)
        parts = optimal_partition(K, max_partition_size=8)
        all_osc = sorted(i for p in parts for i in p)
        assert all_osc == list(range(20))

    def test_rejects_bad_partition_size(self) -> None:
        K = build_knm_paper27(L=4)
        with pytest.raises(ValueError, match="max_partition_size"):
            optimal_partition(K, max_partition_size=0)
        with pytest.raises(TypeError, match="max_partition_size"):
            optimal_partition(K, max_partition_size=cast(int, "bad"))

    def test_rejects_bad_coupling_matrix(self) -> None:
        with pytest.raises(ValueError, match="square"):
            optimal_partition(np.zeros((2, 3), dtype=np.float64), max_partition_size=2)
        with pytest.raises(ValueError, match="at least two"):
            optimal_partition(np.zeros((1, 1), dtype=np.float64), max_partition_size=1)
        with pytest.raises(ValueError, match="finite"):
            optimal_partition(np.full((2, 2), np.nan, dtype=np.float64), max_partition_size=2)


class TestCountCuts:
    def test_single_partition_zero_cuts(self) -> None:
        K = build_knm_paper27(L=4)
        parts = [[0, 1, 2, 3]]
        assert count_inter_partition_couplings(K, parts) == 0

    def test_two_partitions_nonzero(self) -> None:
        K = build_knm_paper27(L=4)
        parts = [[0, 1], [2, 3]]
        cuts = count_inter_partition_couplings(K, parts)
        assert cuts > 0


class TestCircuitCuttingPlan:
    def test_returns_plan(self) -> None:
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K)
        assert isinstance(plan, CircuitCuttingPlan)

    def test_16_no_cuts(self) -> None:
        K = build_knm_paper27(L=16)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.n_cuts == 0
        assert plan.n_partitions == 1

    def test_32_needs_cuts(self) -> None:
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.n_cuts > 0
        assert plan.n_partitions == 2

    def test_128_plan_is_partitioned_not_single_dense_run(self) -> None:
        K = build_knm_paper27(L=128)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.n_oscillators == 128
        assert plan.n_partitions == 8
        assert plan.partition_sizes == [16] * 8
        assert plan.classical_overhead == float("inf")

    def test_fits_heron(self) -> None:
        K = build_knm_paper27(L=32)
        plan = circuit_cutting_plan(K, max_partition_size=16)
        assert plan.fits_on_heron

    def test_rejects_bad_heron_qubits(self) -> None:
        K = build_knm_paper27(L=4)
        with pytest.raises(ValueError, match="heron_qubits"):
            circuit_cutting_plan(K, heron_qubits=0)
        with pytest.raises(TypeError, match="heron_qubits"):
            circuit_cutting_plan(K, heron_qubits=cast(int, "bad"))


class TestScalingAnalysis:
    def test_returns_keys(self) -> None:
        results = scaling_analysis(n_values=[16, 32])
        assert "n_oscillators" in results
        assert "n_cuts" in results
        assert len(results["n_oscillators"]) == 2

    def test_cuts_increase_with_n(self) -> None:
        results = scaling_analysis(n_values=[16, 32, 64])
        assert results["n_cuts"][0] <= results["n_cuts"][1] <= results["n_cuts"][2]

    def test_default_n_values(self) -> None:
        """Cover n_values=None default branch (line 121)."""
        results = scaling_analysis()
        assert len(results["n_oscillators"]) == 7
        assert results["n_oscillators"] == [16, 24, 32, 48, 64, 96, 128]

    def test_rejects_bad_n_values(self) -> None:
        with pytest.raises(ValueError, match="n_values"):
            scaling_analysis(n_values=[1])
        with pytest.raises(TypeError, match="n_values"):
            scaling_analysis(n_values=cast(list[int], [1.5]))
