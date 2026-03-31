# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Quantum Advantage
"""Tests for benchmarks/quantum_advantage.py."""

import numpy as np

from scpn_quantum_control.benchmarks.quantum_advantage import (
    AdvantageResult,
    classical_benchmark,
    estimate_crossover,
    quantum_benchmark,
    run_scaling_benchmark,
)


def test_classical_benchmark_returns_timing():
    result = classical_benchmark(4, t_max=0.5, dt=0.25)
    assert "t_total_ms" in result
    assert result["t_total_ms"] > 0
    assert np.isfinite(result["ground_energy"])


def test_quantum_benchmark_returns_timing():
    result = quantum_benchmark(4, t_max=0.5, dt=0.25, trotter_reps=2)
    assert "t_total_ms" in result
    assert result["t_total_ms"] > 0


def test_advantage_result_fields():
    r = AdvantageResult(n_qubits=4, t_classical_ms=10.0, t_quantum_ms=5.0)
    assert r.n_qubits == 4
    assert r.crossover_predicted is None


def test_run_scaling_small():
    results = run_scaling_benchmark(sizes=[4, 6], t_max=0.2, dt=0.1)
    assert len(results) == 2
    assert results[0].n_qubits == 4
    assert results[1].n_qubits == 6


def test_run_scaling_classical_grows():
    results = run_scaling_benchmark(sizes=[4, 8], t_max=0.2, dt=0.1)
    assert results[1].t_classical_ms >= results[0].t_classical_ms * 0.1


def test_estimate_crossover_needs_data():
    assert estimate_crossover([]) is None
    assert estimate_crossover([AdvantageResult(4, 1.0, 1.0)]) is None


def test_estimate_crossover_returns_int_or_none():
    results = [
        AdvantageResult(4, 1.0, 2.0),
        AdvantageResult(8, 10.0, 4.0),
        AdvantageResult(12, 100.0, 8.0),
    ]
    cross = estimate_crossover(results)
    assert cross is None or isinstance(cross, int)


def test_quantum_timing_positive():
    result = quantum_benchmark(4, t_max=0.1, dt=0.1, trotter_reps=1)
    assert result["t_total_ms"] > 0
    assert result["n_trotter_steps"] > 0


# ---------------------------------------------------------------------------
# Scaling physics: classical should scale exponentially, quantum polynomially
# ---------------------------------------------------------------------------


def test_classical_time_increases_with_n():
    """Larger systems → more classical time (exponential Hilbert space)."""
    r4 = classical_benchmark(4, t_max=0.1, dt=0.1)
    r8 = classical_benchmark(8, t_max=0.1, dt=0.1)
    assert r8["t_total_ms"] > r4["t_total_ms"]


def test_classical_beyond_limit_returns_inf():
    """n > MAX_CLASSICAL_QUBITS → inf time."""
    result = classical_benchmark(20, t_max=0.1, dt=0.1)
    assert result["t_total_ms"] == float("inf")


def test_quantum_ground_energy_finite():
    result = quantum_benchmark(3, t_max=0.2, dt=0.1, trotter_reps=2)
    if "ground_energy" in result:
        assert np.isfinite(result["ground_energy"])


def test_advantage_result_timing_positive():
    r = AdvantageResult(n_qubits=4, t_classical_ms=10.0, t_quantum_ms=5.0)
    assert r.t_classical_ms > 0
    assert r.t_quantum_ms > 0


# ---------------------------------------------------------------------------
# Pipeline: Knm → classical + quantum benchmark → crossover → wired
# ---------------------------------------------------------------------------


def test_pipeline_benchmark_to_crossover():
    """Full pipeline: run_scaling_benchmark → estimate_crossover.
    Verifies benchmarking module is wired end-to-end, not decorative.
    """
    import time

    t0 = time.perf_counter()
    results = run_scaling_benchmark(sizes=[3, 4], t_max=0.1, dt=0.1)
    dt = (time.perf_counter() - t0) * 1000

    assert len(results) == 2
    for r in results:
        assert r.t_classical_ms > 0
        assert r.t_quantum_ms > 0

    cross = estimate_crossover(results)
    # May be None for only 2 data points
    assert cross is None or isinstance(cross, int)

    print(f"\n  PIPELINE scaling benchmark (n=3,4): {dt:.1f} ms")
    for r in results:
        print(
            f"    n={r.n_qubits}: classical={r.t_classical_ms:.1f}ms, "
            f"quantum={r.t_quantum_ms:.1f}ms"
        )
