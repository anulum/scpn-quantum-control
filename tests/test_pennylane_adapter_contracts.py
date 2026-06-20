# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PennyLane adapter contract tests
"""Contract tests for PennyLane adapter availability, Hamiltonian construction, runner execution, and pipeline output boundaries."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import GradientResult
from scpn_quantum_control.hardware import pennylane_adapter as pl_mod

FloatArray = NDArray[np.float64]


class _MockQml:
    """Minimal PennyLane mock that satisfies all adapter calls."""

    def __init__(self) -> None:
        self._call_count = 0
        self.operations: list[tuple[str, object]] = []
        self.device_calls: list[dict[str, object]] = []

    def PauliX(self, wire: int) -> MagicMock:
        self.operations.append(("paulix", wire))
        m = MagicMock(name=f"PauliX({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name=f"XX_{wire}")
        return m

    def PauliY(self, wire: int) -> MagicMock:
        self.operations.append(("pauliy", wire))
        m = MagicMock(name=f"PauliY({wire})")
        m.__matmul__ = lambda s, o: MagicMock(name=f"YY_{wire}")
        return m

    def PauliZ(self, wire: int) -> MagicMock:
        self.operations.append(("pauliz", wire))
        return MagicMock(name=f"PauliZ({wire})")

    def Hamiltonian(self, coeffs: list[float], ops: list[Any]) -> MagicMock:
        del coeffs, ops
        return MagicMock(name="Hamiltonian")

    def device(
        self,
        name: str,
        wires: int | None = None,
        shots: int | None = None,
        **kwargs: object,
    ) -> MagicMock:
        self.device_calls.append(
            {"name": name, "wires": wires, "shots": shots, "kwargs": dict(kwargs)}
        )
        return MagicMock(name="device")

    def qnode(self, dev: object) -> Any:
        del dev

        def decorator(fn: Any) -> Any:
            def wrapper(*args: object, **kwargs: object) -> float:
                self._call_count += 1
                fn(*args, **kwargs)
                return 0.5 + 0.01 * self._call_count

            wrapper.__name__ = str(fn.__name__)
            return wrapper

        return decorator

    def ApproxTimeEvolution(self, H: object, dt: float, n: int) -> None:
        del H
        self.operations.append(("evolution", (dt, n)))

    def expval(self, op: object) -> MagicMock:
        self.operations.append(("expval", op))
        return MagicMock(name="expval")

    def Rot(self, a: object, b: object, c: object, wires: int | None = None) -> None:
        del a, b, c
        self.operations.append(("rot", wires))

    def CNOT(self, wires: list[int] | None = None) -> None:
        self.operations.append(("cnot", tuple(wires or ())))

    def GradientDescentOptimizer(self, stepsize: float = 0.1) -> MagicMock:
        del stepsize
        opt = MagicMock()
        opt.step = lambda fn, p: p + np.random.default_rng(0).normal(0, 0.001, size=len(p))
        return opt

    def grad(self, fn: Any) -> Any:
        """Return a deterministic finite gradient function for mock QNodes."""
        del fn

        def gradient(current_params: object) -> NDArray[np.float64]:
            params = np.asarray(current_params, dtype=np.float64)
            return np.full(params.shape, 0.25, dtype=np.float64)

        return gradient


@pytest.fixture()
def mock_pl(monkeypatch: pytest.MonkeyPatch) -> _MockQml:
    """Patch pennylane_adapter to think PennyLane is available."""
    qml = _MockQml()
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", True)
    monkeypatch.setattr(pl_mod, "qml", qml)
    return qml


def test_is_pennylane_available_true(mock_pl: _MockQml, monkeypatch: pytest.MonkeyPatch) -> None:
    del mock_pl
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", True)
    assert pl_mod.is_pennylane_available() is True


def test_is_pennylane_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    assert pl_mod.is_pennylane_available() is False


def test_xy_hamiltonian_pl_raises_without_pl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    with pytest.raises(ImportError, match="PennyLane"):
        pl_mod._xy_hamiltonian_pl(np.eye(2), np.ones(2))


def test_module_import_guard_sets_qml_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import guard preserves a usable module when PennyLane import fails."""
    import builtins

    module_name = "scpn_quantum_control.hardware.pennylane_adapter"
    original_module = sys.modules[module_name]
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pennylane":
            raise RuntimeError("blocked PennyLane import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(module_name, None)
    try:
        reloaded = importlib.import_module(module_name)
        assert reloaded.is_pennylane_available() is False
        assert reloaded.qml is None
    finally:
        sys.modules[module_name] = original_module


def test_module_import_guard_sets_pl_available_when_import_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import guard records PennyLane availability during module load."""
    module_name = "scpn_quantum_control.hardware.pennylane_adapter"
    original_module = sys.modules[module_name]
    fake_qml = MagicMock(name="pennylane")

    monkeypatch.setitem(sys.modules, "pennylane", fake_qml)
    sys.modules.pop(module_name, None)
    try:
        reloaded = importlib.import_module(module_name)
        assert reloaded.is_pennylane_available() is True
        assert reloaded.qml is fake_qml
    finally:
        sys.modules[module_name] = original_module


def test_xy_hamiltonian_pl(mock_pl: _MockQml) -> None:
    del mock_pl
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    result = pl_mod._xy_hamiltonian_pl(K, omega)
    assert result is not None


def test_runner_init_raises_without_pl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl_mod, "_PL_AVAILABLE", False)
    with pytest.raises(ImportError, match="PennyLane"):
        pl_mod.PennyLaneRunner(np.eye(2), np.ones(2))


def test_runner_init(mock_pl: _MockQml) -> None:
    del mock_pl
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega, device="default.qubit")
    assert runner.n == 2
    assert runner.device_name == "default.qubit"


@pytest.mark.parametrize(
    ("K", "omega", "match"),
    [
        (np.ones((2, 3), dtype=np.float64), np.ones(2, dtype=np.float64), "square"),
        (np.eye(2, dtype=np.float64), np.ones(3, dtype=np.float64), "omega"),
        (
            np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64),
            np.ones(2, dtype=np.float64),
            "finite",
        ),
    ],
)
def test_runner_rejects_invalid_physics_inputs(
    mock_pl: _MockQml,
    K: FloatArray,
    omega: FloatArray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pl_mod.PennyLaneRunner(K, omega)
    assert mock_pl.device_calls == []


@pytest.mark.parametrize("shots", [0, -1, True])
def test_runner_rejects_invalid_shots_before_device_dispatch(
    mock_pl: _MockQml,
    shots: int | bool,
) -> None:
    with pytest.raises(ValueError, match="shots"):
        pl_mod.PennyLaneRunner(np.eye(2, dtype=np.float64), np.ones(2), shots=shots)
    assert mock_pl.device_calls == []


def test_runner_run_trotter(mock_pl: _MockQml) -> None:
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
    assert [op for op, _payload in mock_pl.operations].count("evolution") == 10


def test_runner_run_vqe(mock_pl: _MockQml) -> None:
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_vqe(ansatz_depth=1, maxiter=3, seed=42)
    assert isinstance(result, pl_mod.PennyLaneResult)
    assert result.n_qubits == 2
    assert 0.0 <= result.order_parameter <= 1.0
    assert result.order_parameter != 0.0
    assert any(op == "rot" for op, _payload in mock_pl.operations)
    assert any(op == "cnot" for op, _payload in mock_pl.operations)
    assert any(op == "paulix" for op, _payload in mock_pl.operations)
    assert any(op == "pauliy" for op, _payload in mock_pl.operations)


def test_runner_exposes_vqe_value_and_grad(mock_pl: _MockQml) -> None:
    """PennyLane adapter should expose differentiable VQE parameters."""

    K = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    omega = np.array([0.1, -0.2], dtype=np.float64)
    runner = pl_mod.PennyLaneRunner(K, omega, device="default.qubit")
    params = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06], dtype=np.float64)

    result = runner.vqe_value_and_grad(params, ansatz_depth=1)

    assert isinstance(result, GradientResult)
    assert result.method == "pennylane_autodiff"
    assert result.gradient.shape == params.shape
    assert result.parameter_names[0] == "vqe_0"
    assert np.isfinite(result.value)
    assert np.all(np.isfinite(result.gradient))
    assert [op for op, _payload in mock_pl.operations].count("rot") >= runner.n


def test_runner_shots_param(mock_pl: _MockQml) -> None:
    K = np.array([[0, 0.3], [0.3, 0]])
    omega = np.array([1.0, 1.5])
    runner = pl_mod.PennyLaneRunner(K, omega, shots=1024)
    assert runner.shots == 1024


def test_runner_3qubit(mock_pl: _MockQml) -> None:
    del mock_pl
    K = np.array([[0, 0.5, 0.2], [0.5, 0, 0.3], [0.2, 0.3, 0]])
    omega = np.array([1.0, 2.0, 3.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=0.5, reps=1)
    assert result.n_qubits == 3


def test_result_energy_type(mock_pl: _MockQml) -> None:
    """Energy from Trotter must be a float, not complex or None."""
    del mock_pl
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=0.5, reps=1)
    assert isinstance(result.energy, float)
    assert np.isfinite(result.energy)


def test_result_order_parameter_type(mock_pl: _MockQml) -> None:
    """R from Trotter must be a float."""
    del mock_pl
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=0.5, reps=1)
    assert isinstance(result.order_parameter, float)


def test_vqe_energy_is_float(mock_pl: _MockQml) -> None:
    """VQE must return float energy."""
    del mock_pl
    K = np.array([[0, 0.3], [0.3, 0]])
    omega = np.array([1.0, 1.5])
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_vqe(ansatz_depth=1, maxiter=2, seed=0)
    assert isinstance(result.energy, float)


def test_vqe_order_parameter_is_measured_from_ansatz(mock_pl: _MockQml) -> None:
    """VQE must report a measured Kuramoto order parameter, not a sentinel."""
    K = np.array([[0, 0.3], [0.3, 0]])
    omega = np.array([1.0, 1.5])
    runner = pl_mod.PennyLaneRunner(K, omega)

    result = runner.run_vqe(ansatz_depth=1, maxiter=1, seed=0)

    assert np.isfinite(result.order_parameter)
    assert 0.0 <= result.order_parameter <= 1.0
    assert result.order_parameter != 0.0
    assert [op for op, _payload in mock_pl.operations].count("paulix") >= runner.n
    assert [op for op, _payload in mock_pl.operations].count("pauliy") >= runner.n


def test_pipeline_knm_to_pennylane(mock_pl: _MockQml) -> None:
    del mock_pl
    """Full pipeline: build_knm → PennyLane runner → Trotter → result.
    Verifies PennyLane adapter is wired end-to-end (via mock).
    """
    import time

    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]

    t0 = time.perf_counter()
    runner = pl_mod.PennyLaneRunner(K, omega)
    result = runner.run_trotter(t=0.5, reps=2)
    dt = (time.perf_counter() - t0) * 1000

    assert result.n_qubits == 3
    assert isinstance(result.energy, float)

    print(f"\n  PIPELINE Knm→PennyLane (3q, mock): {dt:.1f} ms")
    print(f"  E={result.energy:.4f}, R={result.order_parameter:.4f}")
