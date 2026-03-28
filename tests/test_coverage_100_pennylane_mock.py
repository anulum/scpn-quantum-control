# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — PennyLane Adapter Mock Tests
"""Mock-based tests for PennyLane adapter covering all code paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_quantum_control.hardware import pennylane_adapter as pl_mod


class _MockQml:
    """Minimal PennyLane mock that satisfies all adapter calls."""

    def __init__(self):
        self._call_count = 0

    def PauliX(self, wire):
        m = MagicMock(name=f"PauliX({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name=f"XX_{wire}")
        return m

    def PauliY(self, wire):
        m = MagicMock(name=f"PauliY({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name=f"YY_{wire}")
        return m

    def PauliZ(self, wire):
        return MagicMock(name=f"PauliZ({wire})")

    def Hamiltonian(self, coeffs, ops):
        return MagicMock(name="Hamiltonian")

    def device(self, name, wires=None, shots=None, **kwargs):
        return MagicMock(name="device")

    def qnode(self, dev):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                self._call_count += 1
                return 0.5 + 0.01 * self._call_count

            wrapper.__name__ = fn.__name__
            return wrapper

        return decorator

    def ApproxTimeEvolution(self, H, dt, n):
        pass

    def expval(self, op):
        return MagicMock(name="expval")

    def Rot(self, a, b, c, wires=None):
        pass

    def CNOT(self, wires=None):
        pass

    def GradientDescentOptimizer(self, stepsize=0.1):
        opt = MagicMock()
        opt.step = lambda fn, p: p + np.random.default_rng(0).normal(0, 0.001, size=len(p))
        return opt


@pytest.fixture()
def mock_pl(monkeypatch):
    """Patch pennylane_adapter to think PennyLane is available."""
    qml = _MockQml()
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", True)
    monkeypatch.setattr(pl_mod, "qml", qml)
    return qml


def test_is_pennylane_available_true(mock_pl, monkeypatch):
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", True)
    assert pl_mod.is_pennylane_available() is True


def test_is_pennylane_available_false(monkeypatch):
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    assert pl_mod.is_pennylane_available() is False


def test_xy_hamiltonian_pl_raises_without_pl(monkeypatch):
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    with pytest.raises(ImportError, match="PennyLane"):
        pl_mod._xy_hamiltonian_pl(np.eye(2), np.ones(2))


def test_xy_hamiltonian_pl(mock_pl):
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    result = pl_mod._xy_hamiltonian_pl(K, omega)
    assert result is not None


def test_runner_init_raises_without_pl(monkeypatch):
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    with pytest.raises(ImportError, match="PennyLane"):
        pl_mod.PennyLaneRunner(np.eye(2), np.ones(2))


def test_runner_init(mock_pl):
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega, device="default.qubit")
    assert runner.n == 2
    assert runner.device_name == "default.qubit"


def test_runner_run_trotter(mock_pl):
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=1.0, reps=2)
    assert isinstance(result, pl_mod.PennyLaneResult)
    assert result.n_qubits == 2
    assert result.device_name == "default.qubit"
    assert result.statevector is None
    assert isinstance(result.energy, float)
    assert isinstance(result.order_parameter, float)


def test_runner_run_vqe(mock_pl):
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_vqe(ansatz_depth=1, maxiter=3, seed=42)
    assert isinstance(result, pl_mod.PennyLaneResult)
    assert result.n_qubits == 2
    assert result.order_parameter == 0.0


def test_runner_shots_param(mock_pl):
    K = np.array([[0, 0.3], [0.3, 0]])
    omega = np.array([1.0, 1.5])
    runner = pl_mod.PennyLaneRunner(K, omega, shots=1024)
    assert runner.shots == 1024


def test_runner_3qubit(mock_pl):
    K = np.array([[0, 0.5, 0.2], [0.5, 0, 0.3], [0.2, 0.3, 0]])
    omega = np.array([1.0, 2.0, 3.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=0.5, reps=1)
    assert result.n_qubits == 3
