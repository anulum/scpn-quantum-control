# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Gpu Baseline
"""Tests for GPU baseline comparison."""

from __future__ import annotations

import math

import pytest

from scpn_quantum_control.benchmarks.gpu_baseline import (
    A100_MEMORY_GB,
    A100_PROFILE,
    A100_TFLOPS,
    GPUBaselineResult,
    GPUProfile,
    gate_count_xy_trotter,
    gpu_baseline_comparison,
    scaling_comparison,
    statevector_flops,
    statevector_memory_gb,
)


class TestStatevectorMemory:
    """Statevector memory follows ``2^n * 16`` bytes."""

    def test_16_qubits(self) -> None:
        """Sixteen qubits need ``2^16 * 16`` bytes."""
        mem = statevector_memory_gb(16)
        assert mem == pytest.approx(2**16 * 16 / 1e9, rel=0.01)

    def test_exponential(self) -> None:
        """Ten more qubits multiply the memory by 1024."""
        m20 = statevector_memory_gb(20)
        m30 = statevector_memory_gb(30)
        assert m30 / m20 == pytest.approx(1024.0)


class TestGateCount:
    """XY Trotter gate counts."""

    def test_positive(self) -> None:
        """A non-trivial circuit has gates."""
        assert gate_count_xy_trotter(4, reps=5) > 0

    def test_scales_quadratically(self) -> None:
        """Doubling the qubits more than triples the gate count."""
        g4 = gate_count_xy_trotter(4, reps=1)
        g8 = gate_count_xy_trotter(8, reps=1)
        assert g8 > 3 * g4  # roughly quadratic


class TestGPUProfile:
    """The estimator takes its device from an explicit profile."""

    def test_reference_profile_keeps_the_historical_figures(self) -> None:
        """The default profile is the named A100 reference with the module figures."""
        assert GPUProfile("NVIDIA A100 80 GB", A100_TFLOPS, A100_MEMORY_GB) == A100_PROFILE
        assert gpu_baseline_comparison(16).gpu_profile == "NVIDIA A100 80 GB"

    @pytest.mark.parametrize(
        ("name", "flops", "memory", "message"),
        [
            (" ", 1e12, 1.0, "nonblank"),
            ("gpu", 0.0, 1.0, "peak_flops"),
            ("gpu", math.inf, 1.0, "peak_flops"),
            ("gpu", 1e12, -1.0, "memory_gb"),
            ("gpu", 1e12, math.nan, "memory_gb"),
        ],
    )
    def test_malformed_profiles_are_refused(
        self, name: str, flops: float, memory: float, message: str
    ) -> None:
        """Blank names and non-finite or non-positive figures are refused.

        Parameters
        ----------
        name
            Device name under test.
        flops
            Peak throughput under test.
        memory
            Device memory under test.
        message
            Expected refusal text.

        """
        with pytest.raises(ValueError, match=message):
            GPUProfile(name, flops, memory)

    def test_profile_throughput_sets_the_time_estimate(self) -> None:
        """Time is FLOPs over the profile's peak throughput, whatever the device."""
        profile = GPUProfile("half-speed reference", A100_TFLOPS / 2, A100_MEMORY_GB)
        result = gpu_baseline_comparison(20, profile=profile)
        assert result.estimated_gpu_time_s == pytest.approx(
            statevector_flops(20, result.n_gates) / (A100_TFLOPS / 2)
        )
        assert result.gpu_profile == "half-speed reference"

    def test_profile_memory_ends_the_crossover_search(self) -> None:
        """A device too small for 18 qubits ends the search at 18."""
        small = GPUProfile("small memory", 1e30, statevector_memory_gb(17))
        assert gpu_baseline_comparison(16, profile=small).crossover_n == 18


class TestGPUBaselineComparison:
    """GPU against QPU estimates for one size."""

    def test_returns_result(self) -> None:
        """The comparison returns its result record."""
        result = gpu_baseline_comparison(16)
        assert isinstance(result, GPUBaselineResult)

    def test_small_system_gpu_faster(self) -> None:
        """A small system is estimated faster on the GPU."""
        result = gpu_baseline_comparison(8)
        assert result.gpu_faster

    def test_memory_finite(self) -> None:
        """Sixteen qubits fit well under one GB."""
        result = gpu_baseline_comparison(16)
        assert result.statevector_memory_gb > 0
        assert result.statevector_memory_gb < 1.0

    def test_crossover_exists(self) -> None:
        """The crossover lies above the starting size."""
        result = gpu_baseline_comparison(16)
        assert result.crossover_n > 16

    def test_scpn_gpu_comparison(self) -> None:
        """Memory grows with every SCPN size compared."""
        memory = [gpu_baseline_comparison(n).statevector_memory_gb for n in (16, 24, 32)]
        assert memory == sorted(memory)
        assert len(set(memory)) == 3


class TestScalingComparison:
    """Batch comparison across sizes."""

    def test_default_values(self) -> None:
        """The default sizes are 4 to 40."""
        results = scaling_comparison()
        assert results["n"] == [4, 8, 16, 24, 32, 40]

    def test_returns_keys(self) -> None:
        """Every requested size gets one row."""
        results = scaling_comparison(n_values=[4, 8, 16])
        assert "gpu_time_s" in results
        assert len(results["n"]) == 3

    def test_gpu_time_increases(self) -> None:
        """GPU time grows with the size."""
        results = scaling_comparison(n_values=[8, 16, 24])
        assert results["gpu_time_s"][2] > results["gpu_time_s"][0]

    def test_profile_is_passed_through(self) -> None:
        """A slower profile gives proportionally longer times for every size."""
        slow = GPUProfile("tenth speed", A100_TFLOPS / 10, A100_MEMORY_GB)
        reference = scaling_comparison(n_values=[8, 16])
        scaled = scaling_comparison(n_values=[8, 16], profile=slow)
        assert scaled["gpu_time_s"] == pytest.approx([10 * t for t in reference["gpu_time_s"]])
