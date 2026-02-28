"""Tests for ansatz benchmark."""

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
