# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Ansatz Bench
"""Tests for ansatz benchmark — elite multi-angle coverage."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.ansatz_bench import benchmark_ansatz, run_ansatz_benchmark


@pytest.fixture
def small_system():
    n = 4
    return build_knm_paper27(L=n), OMEGA_N_16[:n]


def test_knm_informed_produces_finite_energy(small_system):
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=50)
    assert result["energy"] < 0


def test_two_local_produces_finite_energy(small_system):
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "two_local", maxiter=50)
    assert result["energy"] < 0


def test_efficient_su2_produces_finite_energy(small_system):
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "efficient_su2", maxiter=50)
    assert result["energy"] < 0


def test_knm_fewer_params_than_two_local(small_system):
    K, omega = small_system
    knm = benchmark_ansatz(K, omega, "knm_informed", maxiter=10)
    tl = benchmark_ansatz(K, omega, "two_local", maxiter=10)
    assert knm["n_params"] <= tl["n_params"]


def test_unknown_ansatz_raises(small_system):
    K, omega = small_system
    with pytest.raises(ValueError, match="Unknown ansatz"):
        benchmark_ansatz(K, omega, "nonexistent", maxiter=10)


def test_run_benchmark_returns_three():
    results = run_ansatz_benchmark(n_qubits=3, maxiter=30)
    assert len(results) == 3
    names = {r["ansatz"] for r in results}
    assert names == {"knm_informed", "two_local", "efficient_su2"}


def test_benchmark_result_keys(small_system):
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=10)
    assert "energy" in result
    assert "n_params" in result
    assert "ansatz" in result


def test_all_energies_finite(small_system):
    K, omega = small_system
    for name in ("knm_informed", "two_local", "efficient_su2"):
        result = benchmark_ansatz(K, omega, name, maxiter=10)
        assert np.isfinite(result["energy"])


@pytest.mark.parametrize("n", [2, 3, 4])
def test_run_benchmark_various_sizes(n):
    results = run_ansatz_benchmark(n_qubits=n, maxiter=10)
    assert len(results) == 3
    for r in results:
        assert np.isfinite(r["energy"])


def test_knm_informed_negative_energy_2q():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=30)
    assert result["energy"] < 0


# ---------------------------------------------------------------------------
# Ansatz physics: parameter efficiency and expressibility
# ---------------------------------------------------------------------------


def test_knm_informed_depth_lower_than_two_local(small_system):
    """Knm-informed ansatz should use fewer or equal parameters than two_local."""
    K, omega = small_system
    r_knm = benchmark_ansatz(K, omega, "knm_informed", maxiter=5)
    r_tl = benchmark_ansatz(K, omega, "two_local", maxiter=5)
    assert r_knm["n_params"] <= r_tl["n_params"]


def test_all_ansatz_names_in_result(small_system):
    """Each result must contain the ansatz name."""
    K, omega = small_system
    for name in ("knm_informed", "two_local", "efficient_su2"):
        result = benchmark_ansatz(K, omega, name, maxiter=5)
        assert result["ansatz"] == name


# ---------------------------------------------------------------------------
# Pipeline: Knm → ansatz benchmark → comparison → wired
# ---------------------------------------------------------------------------


def test_pipeline_ansatz_comparison():
    """Full pipeline: build_knm → benchmark 3 ansätze → compare.
    Verifies ansatz benchmark is wired and produces comparative data.
    """
    import time

    t0 = time.perf_counter()
    results = run_ansatz_benchmark(n_qubits=3, maxiter=20)
    dt = (time.perf_counter() - t0) * 1000

    assert len(results) == 3
    energies = {r["ansatz"]: r["energy"] for r in results}
    params = {r["ansatz"]: r["n_params"] for r in results}

    print(f"\n  PIPELINE AnsatzBenchmark (3q, 3 ansätze): {dt:.1f} ms")
    for name in ("knm_informed", "two_local", "efficient_su2"):
        print(f"    {name}: E={energies[name]:.4f}, params={params[name]}")
