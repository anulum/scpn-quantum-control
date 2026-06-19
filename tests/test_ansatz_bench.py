# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Ansatz Bench
"""Tests for ansatz benchmark contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import ansatz_bench as ansatz_bench_module
from scpn_quantum_control.phase import benchmark_ansatz as exported_benchmark_ansatz
from scpn_quantum_control.phase import run_ansatz_benchmark as exported_run_ansatz_benchmark
from scpn_quantum_control.phase.ansatz_bench import benchmark_ansatz, run_ansatz_benchmark

_REAL_VQE_ENERGY = ansatz_bench_module._vqe_energy


@pytest.fixture
def small_system() -> tuple[np.ndarray, np.ndarray]:
    n = 4
    return build_knm_paper27(L=n), OMEGA_N_16[:n]


@pytest.fixture(autouse=True)
def deterministic_vqe_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_hamiltonian(K: np.ndarray, omega: np.ndarray) -> object:
        return {"shape": K.shape, "omega": omega.tolist()}

    def fake_vqe(ansatz: Any, hamiltonian: object, maxiter: int) -> tuple[float, int, list[float]]:
        del hamiltonian
        energy = -float(ansatz.num_parameters)
        return energy, maxiter, [energy + 1.0, energy]

    monkeypatch.setattr(ansatz_bench_module, "knm_to_hamiltonian", fake_hamiltonian)
    monkeypatch.setattr(ansatz_bench_module, "_vqe_energy", fake_vqe)


def test_knm_informed_produces_finite_energy(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=50)
    assert result["energy"] < 0


def test_two_local_produces_finite_energy(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "two_local", maxiter=50)
    assert result["energy"] < 0


def test_efficient_su2_produces_finite_energy(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "efficient_su2", maxiter=50)
    assert result["energy"] < 0


def test_knm_fewer_params_than_two_local(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    knm = benchmark_ansatz(K, omega, "knm_informed", maxiter=10)
    tl = benchmark_ansatz(K, omega, "two_local", maxiter=10)
    assert knm["n_params"] <= tl["n_params"]


def test_unknown_ansatz_raises_before_hamiltonian_build(
    small_system: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    K, omega = small_system

    def fail_hamiltonian(K: np.ndarray, omega: np.ndarray) -> object:
        raise AssertionError("Hamiltonian build should not run for unsupported ansatz names")

    monkeypatch.setattr(ansatz_bench_module, "knm_to_hamiltonian", fail_hamiltonian)
    with pytest.raises(ValueError, match="Unknown ansatz"):
        benchmark_ansatz(K, omega, "nonexistent", maxiter=10)


def test_run_benchmark_returns_three() -> None:
    results = run_ansatz_benchmark(n_qubits=3, maxiter=30)
    assert len(results) == 3
    names = {r["ansatz"] for r in results}
    assert names == {"knm_informed", "two_local", "efficient_su2"}


def test_benchmark_result_keys(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=10)
    assert set(result) == {
        "ansatz",
        "n_qubits",
        "n_params",
        "energy",
        "n_evals",
        "history",
        "reps",
    }
    assert result["n_qubits"] == len(omega)
    assert result["n_evals"] == 10
    assert result["history"][-1] == result["energy"]


def test_all_energies_finite(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    K, omega = small_system
    for name in ("knm_informed", "two_local", "efficient_su2"):
        result = benchmark_ansatz(K, omega, name, maxiter=10)
        assert np.isfinite(result["energy"])


@pytest.mark.parametrize("n", [2, 3, 4])
def test_run_benchmark_various_sizes(n: int) -> None:
    results = run_ansatz_benchmark(n_qubits=n, maxiter=10)
    assert len(results) == 3
    for r in results:
        assert np.isfinite(r["energy"])


def test_knm_informed_negative_energy_2q() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = benchmark_ansatz(K, omega, "knm_informed", maxiter=30)
    assert result["energy"] < 0


# ---------------------------------------------------------------------------
# Ansatz physics: parameter efficiency and expressibility
# ---------------------------------------------------------------------------


def test_knm_informed_depth_lower_than_two_local(
    small_system: tuple[np.ndarray, np.ndarray],
) -> None:
    """Knm-informed ansatz should use fewer or equal parameters than two_local."""
    K, omega = small_system
    r_knm = benchmark_ansatz(K, omega, "knm_informed", maxiter=5)
    r_tl = benchmark_ansatz(K, omega, "two_local", maxiter=5)
    assert r_knm["n_params"] <= r_tl["n_params"]


def test_all_ansatz_names_in_result(small_system: tuple[np.ndarray, np.ndarray]) -> None:
    """Each result must contain the ansatz name."""
    K, omega = small_system
    for name in ("knm_informed", "two_local", "efficient_su2"):
        result = benchmark_ansatz(K, omega, name, maxiter=5)
        assert result["ansatz"] == name


# ---------------------------------------------------------------------------
# Pipeline: Knm → ansatz benchmark → comparison → wired
# ---------------------------------------------------------------------------


def test_phase_namespace_exports_ansatz_benchmark_contracts() -> None:
    assert exported_benchmark_ansatz is benchmark_ansatz
    assert exported_run_ansatz_benchmark is run_ansatz_benchmark


def test_vqe_energy_records_history(monkeypatch: pytest.MonkeyPatch) -> None:
    class Ansatz:
        num_parameters = 2

        def assign_parameters(self, params: np.ndarray) -> np.ndarray:
            return params

    class Statevector:
        @classmethod
        def from_instruction(cls, bound: np.ndarray) -> Statevector:
            instance = cls()
            instance.bound = bound
            return instance

        def expectation_value(self, hamiltonian: object) -> complex:
            del hamiltonian
            return complex(np.sum(self.bound), 0.0)

    def fake_minimize(
        cost: Any,
        x0: np.ndarray,
        method: str,
        options: dict[str, int],
    ) -> SimpleNamespace:
        assert method == "COBYLA"
        assert options == {"maxiter": 7}
        first = cost(x0)
        second = cost(np.array([1.0, 2.0], dtype=np.float64))
        return SimpleNamespace(fun=second, nfev=2, first=first)

    monkeypatch.setattr(ansatz_bench_module, "Statevector", Statevector)
    monkeypatch.setattr(ansatz_bench_module, "minimize", fake_minimize)
    monkeypatch.setattr(ansatz_bench_module, "_vqe_energy", _REAL_VQE_ENERGY)

    energy, n_evals, history = ansatz_bench_module._vqe_energy(Ansatz(), object(), 7)

    assert energy == 3.0
    assert n_evals == 2
    assert len(history) == 2
    assert history[-1] == 3.0


def test_pipeline_ansatz_comparison() -> None:
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
