# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Coverage 100 Bench
"""Multi-angle tests for benchmarks/ subpackage: gpu_baseline, mps_baseline,
quantum_advantage.

Covers: scaling, memory estimates, crossover detection, edge cases,
parametrised system sizes, output validation.
"""

from __future__ import annotations

import numpy as np
import pytest


# =====================================================================
# GPU Baseline
# =====================================================================
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


# =====================================================================
# MPS Baseline
# =====================================================================
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


# =====================================================================
# Quantum Advantage
# =====================================================================
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
