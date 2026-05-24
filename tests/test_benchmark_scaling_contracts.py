# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Benchmark scaling contract tests
"""Contract tests for GPU, MPS, and quantum-advantage benchmark scaling surfaces."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestGPUBaseline:
    def test_crossover_n_at_least_4(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

        result = gpu_baseline_comparison(n=4)
        assert result.crossover_n >= 4

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_comparison_returns_valid_result(self, n):
        from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

        result = gpu_baseline_comparison(n=n)
        assert hasattr(result, "crossover_n")
        assert hasattr(result, "estimated_gpu_time_s")
        assert np.isfinite(result.estimated_gpu_time_s)

    def test_scaling_comparison_shape(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import scaling_comparison

        result = scaling_comparison(n_values=[4, 8])
        assert len(result["n"]) == 2
        assert all(np.isfinite(result["gpu_time_s"]))

    def test_memory_increases_with_n(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import statevector_memory_gb

        m4 = statevector_memory_gb(4)
        m8 = statevector_memory_gb(8)
        m12 = statevector_memory_gb(12)
        assert m4 < m8 < m12

    def test_flops_positive(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import statevector_flops

        f4 = statevector_flops(n=4, n_gates=100)
        assert f4 > 0

    def test_gate_count_positive(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import gate_count_xy_trotter

        for n in [4, 8, 12]:
            g = gate_count_xy_trotter(n)
            assert g > 0

    def test_memory_gb_positive(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import statevector_memory_gb

        mem = statevector_memory_gb(8)
        assert mem > 0


class TestMPSBaseline:
    def test_low_entropy_returns_1000(self):
        from scpn_quantum_control.benchmarks.mps_baseline import quantum_advantage_n

        n = quantum_advantage_n(entropy_per_qubit=0.0)
        assert n == 1000

    @pytest.mark.parametrize("entropy", [0.0, 0.1, 0.5, 1.0])
    def test_advantage_n_positive(self, entropy):
        from scpn_quantum_control.benchmarks.mps_baseline import quantum_advantage_n

        n = quantum_advantage_n(entropy_per_qubit=entropy)
        assert n > 0

    def test_mps_memory_positive(self):
        from scpn_quantum_control.benchmarks.mps_baseline import mps_memory

        for n in [8, 12, 16]:
            m = mps_memory(n, chi=32)
            assert m > 0

    def test_exact_memory_positive(self):
        from scpn_quantum_control.benchmarks.mps_baseline import exact_memory

        for n in [4, 8, 12]:
            m = exact_memory(n)
            assert m > 0


class TestQuantumAdvantage:
    def test_run_scaling_single_size(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            run_scaling_benchmark,
        )

        results = run_scaling_benchmark(sizes=[4])
        assert len(results) == 1
        assert results[0].n_qubits == 4

    def test_scaling_results_structure(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            run_scaling_benchmark,
        )

        results = run_scaling_benchmark(sizes=[4])
        r = results[0]
        assert hasattr(r, "n_qubits")
        assert r.n_qubits == 4


class TestBenchmarkEdgeCases:
    def test_quantum_advantage_scaling(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        result = run_scaling_benchmark(sizes=[2, 3])
        assert len(result) == 2

    def test_gpu_baseline_comparison(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

        result = gpu_baseline_comparison(n=3)
        assert hasattr(result, "estimated_gpu_time_s")

    def test_gpu_baseline_scaling(self):
        from scpn_quantum_control.benchmarks.gpu_baseline import scaling_comparison

        results = scaling_comparison(n_values=[2, 3])
        assert "n" in results
        assert len(results["n"]) == 2


class TestQuantumAdvantageEdgeCases:
    """Verify behaviours in benchmarks/quantum_advantage.py."""

    def test_classical_benchmark_infeasible(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import classical_benchmark

        result = classical_benchmark(16, t_max=0.1, dt=0.1)
        assert result["t_total_ms"] == float("inf")
        assert result["ground_energy"] is None

    def test_estimate_crossover_too_few_points(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [AdvantageResult(4, 1.0, 2.0), AdvantageResult(6, 2.0, 3.0)]
        assert estimate_crossover(results) is None

    def test_estimate_crossover_with_inf(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, float("inf"), 4.0),
        ]
        cross = estimate_crossover(results)
        # Only 2 feasible points — not enough for fit
        assert cross is None

    def test_run_scaling_benchmark_few_sizes(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        results = run_scaling_benchmark(sizes=[4, 6], t_max=0.1, dt=0.1)
        assert len(results) == 2
        assert results[0].crossover_predicted is None  # < 3 results


class TestQuantumAdvantageMoreEdges:
    def test_estimate_crossover_curve_fit_fails(self):

        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, 20.0, 4.0),
        ]
        with patch(
            "scpn_quantum_control.benchmarks.quantum_advantage.curve_fit",
            side_effect=RuntimeError("fit failed"),
        ):
            assert estimate_crossover(results) is None

    def test_estimate_crossover_quantum_grows_faster(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        # Quantum grows faster than classical — no crossover
        results = [
            AdvantageResult(4, 10.0, 1.0),
            AdvantageResult(6, 15.0, 5.0),
            AdvantageResult(8, 20.0, 50.0),
        ]
        cross = estimate_crossover(results)
        # b_c <= b_q → None
        assert cross is None

    def test_run_scaling_default_sizes(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        results = run_scaling_benchmark(sizes=[4], t_max=0.1, dt=0.1)
        assert len(results) == 1
        assert results[0].crossover_predicted is None  # single size

    def test_classical_benchmark_n15(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import classical_benchmark

        result = classical_benchmark(15, t_max=0.05, dt=0.05)
        assert result["t_total_ms"] == float("inf")

    def test_estimate_crossover_negative_ratio(self):

        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, 20.0, 4.0),
        ]
        with patch(
            "scpn_quantum_control.benchmarks.quantum_advantage.curve_fit",
            side_effect=[
                (np.array([1.0, 0.5]), None),
                (np.array([-1.0, 0.1]), None),
            ],
        ):
            assert estimate_crossover(results) is None

    def test_run_scaling_uses_default_sizes(self, monkeypatch):
        from scpn_quantum_control.benchmarks import quantum_advantage as qa

        observed: list[int] = []

        def fake_classical(n, t_max=1.0, dt=0.1):
            observed.append(n)
            return {"t_total_ms": float(n), "ground_energy": 0.0, "R_final": 0.0}

        def fake_quantum(n, t_max=1.0, dt=0.1, trotter_reps=5):
            return {"t_total_ms": float(n) / 2.0, "n_trotter_steps": trotter_reps}

        monkeypatch.setattr(qa, "classical_benchmark", fake_classical)
        monkeypatch.setattr(qa, "quantum_benchmark", fake_quantum)

        results = qa.run_scaling_benchmark(sizes=None, t_max=0.05, dt=0.05)

        assert observed == [4, 8, 12, 16, 20]
        assert [r.n_qubits for r in results] == observed

    def test_run_scaling_with_warning(self, monkeypatch):
        import warnings

        from scpn_quantum_control.benchmarks import quantum_advantage as qa

        monkeypatch.setattr(
            qa,
            "classical_benchmark",
            lambda n, t_max=1.0, dt=0.1: {
                "t_total_ms": float("inf"),
                "ground_energy": None,
                "R_final": None,
            },
        )
        monkeypatch.setattr(
            qa,
            "quantum_benchmark",
            lambda n, t_max=1.0, dt=0.1: {
                "t_total_ms": 1.0,
                "n_trotter_steps": 1,
            },
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = qa.run_scaling_benchmark(sizes=[24], t_max=0.05, dt=0.05)

        assert len(results) == 1
        assert results[0].n_qubits == 24
