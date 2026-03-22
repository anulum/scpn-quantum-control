# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for circuit cutting runner."""

from __future__ import annotations

from scpn_quantum_control.hardware.cutting_runner import (
    CuttingRunResult,
    run_cutting_simulation,
)


class TestCuttingRunner:
    def test_16_single_partition(self):
        result = run_cutting_simulation(n_oscillators=16, reps=2)
        assert isinstance(result, CuttingRunResult)
        assert result.n_partitions == 1
        assert result.n_oscillators == 16

    def test_32_two_partitions(self):
        result = run_cutting_simulation(n_oscillators=32, reps=2)
        assert result.n_partitions == 2
        assert result.partition_sizes == [16, 16]

    def test_r_global_bounded(self):
        result = run_cutting_simulation(n_oscillators=32, reps=2)
        assert 0 <= result.combined_r_global <= 1.0
        for r in result.partition_r_globals:
            assert 0 <= r <= 1.0

    def test_energy_finite(self):
        result = run_cutting_simulation(n_oscillators=32, reps=2)
        assert isinstance(result.total_energy_estimate, float)

    def test_48_three_partitions(self):
        result = run_cutting_simulation(n_oscillators=48, reps=1)
        assert result.n_partitions == 3
        assert result.partition_sizes == [16, 16, 16]

    def test_scpn_32_osc(self):
        """Record 32-oscillator partitioned simulation."""
        result = run_cutting_simulation(n_oscillators=32, reps=3)
        print("\n  32-osc cutting simulation:")
        print(f"  Partitions: {result.n_partitions} × {result.partition_sizes}")
        print(f"  Partition R: {[f'{r:.4f}' for r in result.partition_r_globals]}")
        print(f"  Combined R: {result.combined_r_global:.4f}")
        print(f"  Total energy: {result.total_energy_estimate:.4f}")
        assert result.n_oscillators == 32
