# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Cross Domain Transfer
"""Tests for cross-domain VQE transfer learning."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scpn_quantum_control.hardware.classical as classical_backend
import scpn_quantum_control.phase.cross_domain_transfer as transfer_module
from scpn_quantum_control.phase.cross_domain_transfer import (
    PhysicalSystem,
    TransferResult,
    build_systems,
    run_transfer_matrix,
    summarize_transfer,
    transfer_experiment,
)


class _FakeAnsatz:
    def __init__(self, n_params: int) -> None:
        self.num_parameters = n_params

    def assign_parameters(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)


class _FakeStatevector:
    @classmethod
    def from_instruction(cls, instruction: np.ndarray) -> _FakeStatevector:
        instance = cls()
        instance._instruction = np.asarray(instruction, dtype=np.float64)
        return instance

    def expectation_value(self, _hamiltonian: object) -> complex:
        return complex(float(np.sum(self._instruction**2)), 0.0)


def _install_fake_transfer_boundaries(monkeypatch, *, n_params: int = 4) -> list[np.ndarray]:
    calls: list[np.ndarray] = []

    def fake_ansatz(_K: np.ndarray, *, reps: int = 2) -> _FakeAnsatz:
        return _FakeAnsatz(n_params + reps - 1)

    def fake_hamiltonian(_K: np.ndarray, _omega: np.ndarray) -> object:
        return object()

    def fake_exact_diag(_n: int, *, K: np.ndarray, omega: np.ndarray) -> dict[str, float]:
        return {"ground_energy": -float(K.shape[0]) - float(np.mean(omega))}

    def fake_vqe(
        _ansatz: _FakeAnsatz,
        _hamiltonian: object,
        init_params: np.ndarray,
        maxiter: int,
    ) -> tuple[float, int, np.ndarray]:
        params = np.asarray(init_params, dtype=np.float64)
        calls.append(params.copy())
        energy = float(np.mean(params * params))
        return energy, max(1, min(maxiter, 3)), params * 0.5

    monkeypatch.setattr(transfer_module, "knm_to_ansatz", fake_ansatz)
    monkeypatch.setattr(transfer_module, "knm_to_hamiltonian", fake_hamiltonian)
    monkeypatch.setattr(transfer_module, "_vqe_optimize", fake_vqe)
    monkeypatch.setattr(classical_backend, "classical_exact_diag", fake_exact_diag)
    return calls


class TestBuildSystems:
    def test_returns_four_systems(self):
        systems = build_systems(n_qubits=3)
        assert len(systems) == 4

    def test_system_shapes(self):
        systems = build_systems(n_qubits=4)
        for s in systems:
            assert isinstance(s, PhysicalSystem)
            assert s.K.shape == (4, 4)
            assert len(s.omega) == 4

    def test_names_unique(self):
        systems = build_systems(n_qubits=3)
        names = [s.name for s in systems]
        assert len(names) == len(set(names))

    def test_K_symmetric(self):
        systems = build_systems(n_qubits=4)
        for s in systems:
            np.testing.assert_array_almost_equal(s.K, s.K.T)


class TestTransferExperiment:
    def test_returns_result(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=10)
        assert isinstance(result, TransferResult)
        assert result.source_system == systems[0].name
        assert result.target_system == systems[1].name

    def test_energies_finite(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=10)
        assert np.isfinite(result.random_init_energy)
        assert np.isfinite(result.transfer_init_energy)
        assert np.isfinite(result.exact_energy)

    def test_exact_is_lower_bound(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=20)
        # VQE energy should be >= exact ground state
        assert result.random_init_energy >= result.exact_energy - 0.1
        assert result.transfer_init_energy >= result.exact_energy - 0.1

    def test_speedup_positive(self, monkeypatch):
        calls = _install_fake_transfer_boundaries(monkeypatch, n_params=3)
        systems = build_systems(n_qubits=3)
        result = transfer_experiment(systems[0], systems[1], reps=1, maxiter=15)
        assert result.speedup > 0
        assert len(calls) == 3
        assert calls[2].shape == calls[1].shape

    def test_vqe_optimize_uses_cost_function(self, monkeypatch):
        def fake_minimize(cost, init_params, *, method, options):
            assert method == "COBYLA"
            assert options["maxiter"] == 7
            value = cost(np.asarray(init_params, dtype=np.float64))
            return SimpleNamespace(fun=value, nfev=2, x=np.asarray(init_params) * 0.25)

        monkeypatch.setattr(transfer_module, "Statevector", _FakeStatevector)
        monkeypatch.setattr(transfer_module, "minimize", fake_minimize)

        energy, nfev, params = transfer_module._vqe_optimize(
            _FakeAnsatz(2),
            object(),
            np.array([2.0, -1.0], dtype=np.float64),
            maxiter=7,
        )

        assert energy == 5.0
        assert nfev == 2
        np.testing.assert_allclose(params, np.array([0.5, -0.25]))


class TestRunTransferMatrix:
    def test_all_pairs(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        # 4 systems × 3 targets each = 12 pairs
        assert len(results) == 12

    def test_no_self_transfer(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        for r in results:
            assert r.source_system != r.target_system


class TestSummarizeTransfer:
    def test_summary_keys(self, monkeypatch):
        _install_fake_transfer_boundaries(monkeypatch)
        results = run_transfer_matrix(n_qubits=2, reps=1, maxiter=5)
        summary = summarize_transfer(results)
        assert "n_pairs" in summary
        assert "n_positive_transfer" in summary
        assert "best_speedup" in summary
        assert "mean_speedup" in summary
        assert summary["n_pairs"] == 12

    def test_empty_results(self):
        summary = summarize_transfer([])
        assert summary["n_pairs"] == 0
