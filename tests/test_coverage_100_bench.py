# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for benchmarks/ module gaps
"""Tests targeting specific uncovered lines in the benchmarks/ subpackage."""

from __future__ import annotations

# --- gpu_baseline.py lines 103-104: crossover at memory limit ---


def test_gpu_baseline_crossover():
    """Cover lines 103-104: crossover_n at A100 memory boundary."""
    from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

    result = gpu_baseline_comparison(n=4)
    assert result.crossover_n >= 4


# --- gpu_baseline.py line 123-124: scaling_comparison default n_values ---


def test_gpu_scaling_comparison():
    """Cover line 124: n_values defaults when None."""
    from scpn_quantum_control.benchmarks.gpu_baseline import scaling_comparison

    result = scaling_comparison(n_values=[4, 8])
    assert len(result["n"]) == 2


# --- mps_baseline.py line 88-89: entropy_per_qubit near zero ---


def test_mps_quantum_advantage_n_low_entropy():
    """Cover line 89: quantum_advantage_n returns 1000 for near-zero entropy."""
    from scpn_quantum_control.benchmarks.mps_baseline import quantum_advantage_n

    n = quantum_advantage_n(entropy_per_qubit=0.0)
    assert n == 1000


# --- quantum_advantage.py line 150-151: sizes default ---


def test_quantum_advantage_run_scaling():
    """Cover line 151: sizes defaults to [4,8,12,16,20]."""
    from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

    results = run_scaling_benchmark(sizes=[4])
    assert len(results) == 1
    assert results[0].n_qubits == 4
