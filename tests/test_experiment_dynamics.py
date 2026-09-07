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
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.hardware.experiment_dynamics as dynamics
from scpn_quantum_control.hardware.runner import HardwareRunner


class CountingRunner:
    """Record bounded sampler calls and saved-result requests."""

    def __init__(self) -> None:
        """Create empty call and save ledgers."""
        self.calls: list[dict[str, object]] = []
        self.saved: list[tuple[str, str | None]] = []

    def run_sampler(
        self, circuits: Any, shots: int = 100, name: str = "test"
    ) -> list[SimpleNamespace]:
        """Return deterministic aligned counts for every submitted circuit."""
        if not isinstance(circuits, list):
            circuits = [circuits]
        self.calls.append({"name": name, "count": len(circuits), "shots": shots})
        return [
            SimpleNamespace(counts={"0" * circuit.num_qubits: shots}, job_id=f"{name}_{idx}")
            for idx, circuit in enumerate(circuits)
        ]

    def save_result(self, result: Any, filename: str | None = None) -> None:
        """Record the job identifier and requested result filename."""
        self.saved.append((result.job_id, filename))


def _fake_classical(
    n: int,
    t_total: float,
    dt: float,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return a deterministic classical reference with the requested grid."""
    steps = max(1, int(round(t_total / dt)))
    return {
        "times": np.linspace(dt, t_total, steps),
        "R": np.linspace(0.1, 0.2, steps),
    }


def test_kuramoto_4osc_batches_xyz_circuits_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch three bases per four-oscillator step and save one receipt."""
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_4osc_experiment(
        cast(HardwareRunner, runner), shots=50, n_time_steps=2, dt=0.05
    )

    assert result["experiment"] == "kuramoto_4osc"
    assert result["hw_times"] == [0.05, 0.1]
    assert len(result["hw_R"]) == 2
    assert len(result["hw_expectations"]) == 2
    assert runner.calls == [{"name": "kuramoto_4osc", "count": 6, "shots": 50}]
    assert runner.saved == [("kuramoto_4osc_0", "kuramoto_4osc.json")]


def test_kuramoto_8osc_batches_xyz_circuits_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch three bases per eight-oscillator step and save one receipt."""
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_8osc_experiment(
        cast(HardwareRunner, runner), shots=60, n_time_steps=2, dt=0.05
    )

    assert result["experiment"] == "kuramoto_8osc"
    assert result["n_oscillators"] == 8
    assert len(result["hw_R_std"]) == 2
    assert runner.calls == [{"name": "kuramoto_8osc", "count": 6, "shots": 60}]
    assert runner.saved == [("kuramoto_8osc_0", "kuramoto_8osc.json")]


def test_kuramoto_4osc_trotter2_reports_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report second-order evolution without persisting an implicit receipt."""
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.kuramoto_4osc_trotter2_experiment(
        cast(HardwareRunner, runner), shots=70, n_time_steps=2, dt=0.05
    )

    assert result["experiment"] == "kuramoto_4osc_trotter2"
    assert result["trotter_order"] == 2
    assert len(result["hw_expectations"]) == 2
    assert runner.calls == [{"name": "kuramoto_4osc_trotter2", "count": 6, "shots": 70}]
    assert runner.saved == []


def test_sync_threshold_uses_default_k_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the bounded default coupling sweep through three-basis batches."""
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()
    result = dynamics.sync_threshold_experiment(cast(HardwareRunner, runner), shots=80)

    assert result["experiment"] == "sync_threshold"
    assert result["k_values"] == [0.05, 0.15, 0.30, 0.45, 0.60, 0.80]
    assert len(result["results"]) == 6
    assert [call["count"] for call in runner.calls] == [3, 3, 3, 3, 3, 3]
    assert all(call["shots"] == 80 for call in runner.calls)


def test_sync_threshold_preserves_custom_k_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluate only caller-supplied couplings and preserve their order."""
    monkeypatch.setattr(dynamics, "classical_exact_evolution", _fake_classical)
    runner = CountingRunner()

    result = dynamics.sync_threshold_experiment(
        cast(HardwareRunner, runner), shots=40, k_values=[0.6, 0.1]
    )

    assert result["k_values"] == [0.6, 0.1]
    assert [row["K_base"] for row in result["results"]] == [0.6, 0.1]
    assert runner.calls == [
        {"name": "sync_K0.60", "count": 3, "shots": 40},
        {"name": "sync_K0.10", "count": 3, "shots": 40},
    ]
