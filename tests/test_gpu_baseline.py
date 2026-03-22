# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for GPU baseline comparison."""

from __future__ import annotations

import pytest

from scpn_quantum_control.benchmarks.gpu_baseline import (
    GPUBaselineResult,
    gate_count_xy_trotter,
    gpu_baseline_comparison,
    scaling_comparison,
    statevector_memory_gb,
)


class TestStatevectorMemory:
    def test_16_qubits(self):
        mem = statevector_memory_gb(16)
        assert mem == pytest.approx(2**16 * 16 / 1e9, rel=0.01)

    def test_exponential(self):
        m20 = statevector_memory_gb(20)
        m30 = statevector_memory_gb(30)
        assert m30 / m20 == pytest.approx(1024.0)


class TestGateCount:
    def test_positive(self):
        assert gate_count_xy_trotter(4, reps=5) > 0

    def test_scales_quadratically(self):
        g4 = gate_count_xy_trotter(4, reps=1)
        g8 = gate_count_xy_trotter(8, reps=1)
        assert g8 > 3 * g4  # roughly quadratic


class TestGPUBaselineComparison:
    def test_returns_result(self):
        result = gpu_baseline_comparison(16)
        assert isinstance(result, GPUBaselineResult)

    def test_small_system_gpu_faster(self):
        result = gpu_baseline_comparison(8)
        assert result.gpu_faster

    def test_memory_finite(self):
        result = gpu_baseline_comparison(16)
        assert result.statevector_memory_gb > 0
        assert result.statevector_memory_gb < 1.0

    def test_crossover_exists(self):
        result = gpu_baseline_comparison(16)
        assert result.crossover_n > 16

    def test_scpn_gpu_comparison(self):
        """Record GPU vs QPU for SCPN sizes."""
        for n in [16, 24, 32]:
            r = gpu_baseline_comparison(n)
            print(
                f"\n  n={n}: GPU {r.estimated_gpu_time_s:.2e}s, "
                f"QPU {r.qpu_time_s:.2e}s, "
                f"mem {r.statevector_memory_gb:.3f} GB, "
                f"faster={'GPU' if r.gpu_faster else 'QPU'}"
            )
        assert True


class TestScalingComparison:
    def test_returns_keys(self):
        results = scaling_comparison(n_values=[4, 8, 16])
        assert "gpu_time_s" in results
        assert len(results["n"]) == 3

    def test_gpu_time_increases(self):
        results = scaling_comparison(n_values=[8, 16, 24])
        assert results["gpu_time_s"][2] > results["gpu_time_s"][0]
