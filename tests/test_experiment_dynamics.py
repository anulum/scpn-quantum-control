# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Experiment Dynamics
"""Tests for Kuramoto hardware dynamics experiment wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scpn_quantum_control.hardware.experiment_dynamics as dynamics


class CountingRunner:
    def __init__(self):
        self.calls = []
        self.saved = []

    def run_sampler(self, circuits, shots=100, name="test"):
        if not isinstance(circuits, list):
            circuits = [circuits]
        self.calls.append({"name": name, "count": len(circuits), "shots": shots})
        return [
            SimpleNamespace(counts={"0" * circuit.num_qubits: shots}, job_id=f"{name}_{idx}")
            for idx, circuit in enumerate(circuits)
        ]

    def save_result(self, result, filename):
        self.saved.append((result.job_id, filename))


def _fake_classical(n, t_total, dt, K, omega):
    steps = max(1, int(round(t_total / dt)))
    return {
        "times": np.linspace(dt, t_total, steps),
        "R": np.linspace(0.1, 0.2, steps),
    }


def test_kuramoto_4osc_batches_xyz_circuits_and_saves(monkeypatch):
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_4osc_experiment(runner, shots=50, n_time_steps=2, dt=0.05)

    assert result["experiment"] == "kuramoto_4osc"
    assert result["hw_times"] == [0.05, 0.1]
    assert len(result["hw_R"]) == 2
    assert len(result["hw_expectations"]) == 2
    assert runner.calls == [{"name": "kuramoto_4osc", "count": 6, "shots": 50}]
    assert runner.saved == [("kuramoto_4osc_0", "kuramoto_4osc.json")]


def test_kuramoto_8osc_batches_xyz_circuits_and_saves(monkeypatch):
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_8osc_experiment(runner, shots=60, n_time_steps=2, dt=0.05)

    assert result["experiment"] == "kuramoto_8osc"
    assert result["n_oscillators"] == 8
    assert len(result["hw_R_std"]) == 2
    assert runner.calls == [{"name": "kuramoto_8osc", "count": 6, "shots": 60}]
    assert runner.saved == [("kuramoto_8osc_0", "kuramoto_8osc.json")]


def test_kuramoto_4osc_trotter2_reports_order(monkeypatch):
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_4osc_trotter2_experiment(runner, shots=70, n_time_steps=2, dt=0.05)

    assert result["experiment"] == "kuramoto_4osc_trotter2"
    assert result["trotter_order"] == 2
    assert len(result["hw_expectations"]) == 2
    assert runner.calls == [{"name": "kuramoto_4osc_trotter2", "count": 6, "shots": 70}]
    assert runner.saved == []


def test_sync_threshold_uses_default_k_sweep(monkeypatch):
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.sync_threshold_experiment(runner, shots=80)

    assert result["experiment"] == "sync_threshold"
    assert result["k_values"] == [0.05, 0.15, 0.30, 0.45, 0.60, 0.80]
    assert len(result["results"]) == 6
    assert [call["count"] for call in runner.calls] == [3, 3, 3, 3, 3, 3]
    assert all(call["shots"] == 80 for call in runner.calls)
