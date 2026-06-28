# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external framework comparison edges.
"""Optional-framework edge tests for external differentiable comparisons."""

from __future__ import annotations

import math
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from _differentiable_external_comparison_edges import (
    FakeTensorFlowTape,
    FakeTorchTensor,
    set_module_attr,
    tensor_float,
)
from numpy.typing import NDArray

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison


def test_framework_row_classifies_import_error_as_dependency_gap() -> None:
    """Framework runners that raise ImportError should become dependency gaps."""

    def missing_runner(
        values: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        raise ImportError("optional framework missing")

    row = comparison._framework_row(
        "jax",
        missing_runner,
        "vmap",
        "value_and_grad",
        "Install JAX.",
    )

    assert row.status == "hard_gap"
    assert row.failure_class == "dependency_missing"
    assert row.setup_instructions == "Install JAX."


def test_jsonify_operations_normalises_numpy_scalars() -> None:
    """Operation JSON payloads should normalize NumPy scalar values."""
    payload = comparison._jsonify_operations(
        ((np.int64(1), (np.float64(0.25), np.int64(2))),),
    )

    assert payload == [[1, [0.25, 2]]]


def test_fake_jax_reference_executes_value_and_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    """The JAX reference path should consume value_and_grad output."""
    jax_module = ModuleType("jax")
    jnp_module = ModuleType("jax.numpy")
    update_calls: list[tuple[str, bool]] = []

    def update(key: str, value: bool) -> None:
        update_calls.append((key, value))

    def value_and_grad(
        objective: Callable[[NDArray[np.float64]], float],
    ) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]:
        def wrapped(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            return float(objective(values)), comparison._bounded_phase_gradient(values)

        return wrapped

    set_module_attr(jax_module, "config", SimpleNamespace(update=update))
    set_module_attr(jax_module, "value_and_grad", value_and_grad)
    set_module_attr(
        jnp_module, "asarray", lambda values, dtype: np.asarray(values, dtype=np.float64)
    )
    set_module_attr(jnp_module, "cos", np.cos)
    set_module_attr(jnp_module, "sin", np.sin)
    set_module_attr(jnp_module, "float64", np.float64)
    monkeypatch.setitem(__import__("sys").modules, "jax", jax_module)
    monkeypatch.setitem(__import__("sys").modules, "jax.numpy", jnp_module)

    values = np.array([0.2, -0.4], dtype=np.float64)
    value, gradient = comparison._run_jax_reference(values)

    assert update_calls == [("jax_enable_x64", True)]
    assert value == pytest.approx(comparison._bounded_phase_objective(values))
    np.testing.assert_allclose(gradient, comparison._bounded_phase_gradient(values))


def test_fake_pytorch_reference_executes_func_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyTorch reference path should consume torch.func.grad output."""
    torch_module = ModuleType("torch")

    def tensor(values: NDArray[np.float64], *, dtype: object) -> FakeTorchTensor:
        assert dtype is np.float64
        return FakeTorchTensor(values)

    def grad(
        objective: Callable[[FakeTorchTensor], FakeTorchTensor],
    ) -> Callable[[FakeTorchTensor], FakeTorchTensor]:
        def wrapped(values: FakeTorchTensor) -> FakeTorchTensor:
            _ = objective
            return FakeTorchTensor(comparison._bounded_phase_gradient(values.numpy()))

        return wrapped

    set_module_attr(torch_module, "float64", np.float64)
    set_module_attr(torch_module, "tensor", tensor)
    set_module_attr(torch_module, "func", SimpleNamespace(grad=grad))
    set_module_attr(
        torch_module,
        "cos",
        lambda value: FakeTorchTensor(math.cos(tensor_float(value))),
    )
    set_module_attr(
        torch_module,
        "sin",
        lambda value: FakeTorchTensor(math.sin(tensor_float(value))),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_module)

    values = np.array([0.2, -0.4], dtype=np.float64)
    value, gradient = comparison._run_pytorch_reference(values)

    assert value == pytest.approx(comparison._bounded_phase_objective(values))
    np.testing.assert_allclose(gradient, comparison._bounded_phase_gradient(values))


def test_fake_tensorflow_reference_executes_gradient_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TensorFlow reference path should consume GradientTape output."""
    tf_module = ModuleType("tensorflow")
    set_module_attr(tf_module, "float64", np.float64)
    set_module_attr(
        tf_module,
        "Variable",
        lambda values, dtype: FakeTorchTensor(np.asarray(values, dtype=dtype)),
    )
    set_module_attr(tf_module, "GradientTape", FakeTensorFlowTape)
    set_module_attr(tf_module, "constant", lambda value, dtype: FakeTorchTensor(float(value)))
    set_module_attr(
        tf_module,
        "cos",
        lambda value: FakeTorchTensor(math.cos(tensor_float(value))),
    )
    set_module_attr(
        tf_module,
        "sin",
        lambda value: FakeTorchTensor(math.sin(tensor_float(value))),
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", tf_module)

    values = np.array([0.2, -0.4], dtype=np.float64)
    value, gradient = comparison._run_tensorflow_reference(values)

    assert value == pytest.approx(comparison._bounded_phase_objective(values))
    np.testing.assert_allclose(gradient, comparison._bounded_phase_gradient(values))


def test_tensorflow_reference_rejects_missing_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """TensorFlow GradientTape should fail closed when no gradient is returned."""

    class MissingGradientTape(FakeTensorFlowTape):
        def gradient(
            self,
            value: FakeTorchTensor,
            tensor: FakeTorchTensor,
        ) -> None:
            del value, tensor
            return None

    tf_module = ModuleType("tensorflow")
    set_module_attr(tf_module, "float64", np.float64)
    set_module_attr(
        tf_module,
        "Variable",
        lambda values, dtype: FakeTorchTensor(np.asarray(values, dtype=dtype)),
    )
    set_module_attr(tf_module, "GradientTape", MissingGradientTape)
    set_module_attr(tf_module, "constant", lambda value, dtype: FakeTorchTensor(float(value)))
    set_module_attr(
        tf_module,
        "cos",
        lambda value: FakeTorchTensor(math.cos(tensor_float(value))),
    )
    set_module_attr(
        tf_module,
        "sin",
        lambda value: FakeTorchTensor(math.sin(tensor_float(value))),
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", tf_module)

    with pytest.raises(RuntimeError, match="returned no gradient"):
        comparison._run_tensorflow_reference(np.array([0.2, -0.4], dtype=np.float64))


def test_fake_pennylane_reference_executes_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PennyLane reference path should consume QNode and autograd output."""
    qml_module = ModuleType("pennylane")
    pnp_module = ModuleType("pennylane.numpy")

    def qnode(
        _device: object, *, interface: str
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], tuple[float, float]]]:
        assert interface == "autograd"

        def decorator(_body: Callable[[Any], Any]) -> Callable[[Any], tuple[float, float]]:
            def wrapped(values: Any) -> tuple[float, float]:
                raw = np.asarray(values, dtype=np.float64)
                _body(raw)
                return float(np.cos(raw[0])), float(np.sin(raw[1]))

            return wrapped

        return decorator

    def grad(
        _objective: Callable[[Any], Any],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return lambda values: comparison._bounded_phase_gradient(
            np.asarray(values, dtype=np.float64)
        )

    set_module_attr(qml_module, "device", lambda name, wires: {"name": name, "wires": wires})
    set_module_attr(qml_module, "RY", lambda theta, wires: None)
    set_module_attr(qml_module, "PauliZ", lambda wire: ("Z", wire))
    set_module_attr(qml_module, "PauliX", lambda wire: ("X", wire))
    set_module_attr(qml_module, "expval", lambda observable: observable)
    set_module_attr(qml_module, "qnode", qnode)
    set_module_attr(qml_module, "grad", grad)
    set_module_attr(qml_module, "numpy", pnp_module)
    set_module_attr(
        pnp_module, "array", lambda values, requires_grad: np.asarray(values, dtype=np.float64)
    )
    monkeypatch.setitem(__import__("sys").modules, "pennylane", qml_module)
    monkeypatch.setitem(__import__("sys").modules, "pennylane.numpy", pnp_module)

    values = np.array([0.2, -0.4], dtype=np.float64)
    value, gradient = comparison._run_pennylane_reference(values)

    assert value == pytest.approx(comparison._bounded_phase_objective(values))
    np.testing.assert_allclose(gradient, comparison._bounded_phase_gradient(values))


def test_pennylane_reference_reports_missing_numpy_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PennyLane import should fail closed when the NumPy interface is absent."""
    qml_module = ModuleType("pennylane")
    monkeypatch.setitem(__import__("sys").modules, "pennylane", qml_module)
    monkeypatch.delitem(__import__("sys").modules, "pennylane.numpy", raising=False)

    with pytest.raises(ImportError, match="NumPy interface"):
        comparison._run_pennylane_reference(np.array([0.2, -0.4], dtype=np.float64))
