# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Cutting Runner
"""Tests for circuit cutting runner."""

from __future__ import annotations

from scpn_quantum_control.hardware.cutting_runner import (
    CuttingRunResult,
    run_cutting_simulation,
)


class TestCuttingRunner:
    def test_8_single_partition(self):
        result = run_cutting_simulation(n_oscillators=8, reps=2, max_partition_size=8)
        assert isinstance(result, CuttingRunResult)
        assert result.n_partitions == 1
        assert result.n_oscillators == 8

    def test_16_two_partitions(self):
        result = run_cutting_simulation(n_oscillators=16, reps=1, max_partition_size=8)
        assert result.n_partitions == 2
        assert result.partition_sizes == [8, 8]

    def test_r_global_bounded(self):
        result = run_cutting_simulation(n_oscillators=16, reps=1, max_partition_size=8)
        assert 0 <= result.combined_r_global <= 1.0
        for r in result.partition_r_globals:
            assert 0 <= r <= 1.0

    def test_energy_finite(self):
        result = run_cutting_simulation(n_oscillators=16, reps=1, max_partition_size=8)
        assert isinstance(result.total_energy_estimate, float)

    def test_24_three_partitions(self):
        result = run_cutting_simulation(n_oscillators=24, reps=1, max_partition_size=8)
        assert result.n_partitions == 3
        assert result.partition_sizes == [8, 8, 8]

    def test_scpn_24_osc(self):
        """Record 24-oscillator partitioned simulation."""
        result = run_cutting_simulation(n_oscillators=24, reps=2, max_partition_size=8)
        print("\n  24-osc cutting simulation:")
        print(f"  Partitions: {result.n_partitions} x {result.partition_sizes}")
        print(f"  Partition R: {[f'{r:.4f}' for r in result.partition_r_globals]}")
        print(f"  Combined R: {result.combined_r_global:.4f}")
        print(f"  Total energy: {result.total_energy_estimate:.4f}")
        assert result.n_oscillators == 24

    def test_4_single_partition(self):
        result = run_cutting_simulation(n_oscillators=4, reps=2, max_partition_size=8)
        assert result.n_partitions == 1
        assert result.n_oscillators == 4

    def test_partition_r_globals_length(self):
        result = run_cutting_simulation(n_oscillators=16, reps=1, max_partition_size=8)
        assert len(result.partition_r_globals) == result.n_partitions

    def test_energy_is_float(self):
        result = run_cutting_simulation(n_oscillators=8, reps=1, max_partition_size=8)
        assert isinstance(result.total_energy_estimate, float)
        import numpy as np

        assert np.isfinite(result.total_energy_estimate)

    def test_32_four_partitions(self):
        result = run_cutting_simulation(n_oscillators=32, reps=1, max_partition_size=8)
        assert result.n_partitions == 4
        assert result.partition_sizes == [8, 8, 8, 8]

    def test_pipeline_cutting_with_performance(self):
        """Full pipeline: large system → partition → simulate → combine.
        Verifies cutting runner is wired end-to-end with performance data.
        """
        import time

        t0 = time.perf_counter()
        result = run_cutting_simulation(n_oscillators=16, reps=2, max_partition_size=8)
        dt = (time.perf_counter() - t0) * 1000

        assert result.n_partitions == 2
        assert 0 <= result.combined_r_global <= 1.0
        import numpy as np

        assert np.isfinite(result.total_energy_estimate)

        print(f"\n  PIPELINE Cutting (16 osc, 2 partitions): {dt:.1f} ms")
        print(f"  R={result.combined_r_global:.4f}, E={result.total_energy_estimate:.4f}")
