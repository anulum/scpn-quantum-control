# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native axis compiler integration tests
"""Check actual compiled QNode gradients, numerical order and dependency guards."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.differentiable_axis_compilation import (
    numpy_axis_order,
    register_torch_axis_compilation,
    trace_numpy_axis_order,
)
from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

pytestmark = pytest.mark.framework_imports


@pytest.fixture(autouse=True)
def _fresh_compiler_caches() -> Iterable[None]:
    """Give every case its own Dynamo caches.

    Compiled audits specialise on each circuit's structure, and PyTorch limits
    recompilations per function within one process, so cases must not depend
    on which compiled circuits ran before them.
    """
    torch = importlib.import_module("torch")
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("axes", [[], [0], [1, 0], [2, 0, 1]])
def test_native_permutation_lowering_agrees_with_actual_numpy(axes: list[int]) -> None:
    """Retain NumPy order and eager scalar types across admitted dimension orders.

    Parameters
    ----------
    axes
        Empty, singleton, reversed or reordered admitted dimension permutation.

    """
    expected = np.argsort(axes)
    actual = numpy_axis_order(np.argsort, axes)
    assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(tuple(trace_numpy_axis_order(np.argsort, axes)), expected)


def test_replacement_sorter_retains_actual_return_object_and_calls() -> None:
    """Call the supplied numerical dependency in both eager and lowering paths."""
    expected = np.argsort([2, 0, 1])[::-1]
    seen: list[list[int]] = []

    def replacement(axes: list[int]) -> Iterable[int]:
        seen.append(axes)
        return expected

    axes = [2, 0, 1]
    assert numpy_axis_order(replacement, axes) is expected
    assert trace_numpy_axis_order(replacement, axes) is expected
    assert seen == [axes, axes]


def test_real_sdk_registration_is_idempotent_and_handles_feature_withdrawal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register only the actual SDK's supported compiler API without replacing it.

    Parameters
    ----------
    monkeypatch
        Restore the actual SDK polyfill API after the withdrawal check.

    """
    torch = importlib.import_module("torch")
    assert not register_torch_axis_compilation(object())
    assert not register_torch_axis_compilation(np)
    compiler = torch.compile
    sorter = np.argsort
    assert register_torch_axis_compilation(torch)
    assert register_torch_axis_compilation(torch)
    assert torch.compile is compiler and np.argsort is sorter
    with monkeypatch.context() as change:
        change.delattr(torch.compiler, "substitute_in_graph")
        assert not register_torch_axis_compilation(torch)


@pytest.mark.parametrize(
    "gate,qubits,parameter_index",
    [("ry", (2,), 3), ("rxx", (2, 0), 3), ("ccnot", (2, 0, 1), None)],
)
def test_original_public_fullgraph_gradient_matches_registered_numpy_execution(
    gate: str,
    qubits: tuple[int, ...],
    parameter_index: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture original public values and gradients on reordered one-to-three wires.

    Parameters
    ----------
    gate
        Actual registered local gate, including controlled three-wire execution.
    qubits
        Ordered wire dimensions used by the admitted circuit.
    parameter_index
        Original parameter binding, absent for the fixed controlled gate.
    monkeypatch
        Restore actual Torch compilation after selecting the eager capture backend.

    """
    torch = importlib.import_module("torch")
    module = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    observable = PauliTerm(0.7, ((0, "z"), (1, "z"), (2, "x")))
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("ry", (2,), parameter_index=2),
            PhaseQNodeOperation(gate, qubits, parameter_index=parameter_index),
        ),
        observable=observable,
    )
    params = np.array([0.31, -0.23, 0.47, 0.41])
    expected = execute_phase_qnode_circuit(circuit, params)
    derivative = parameter_shift_phase_qnode_gradient(circuit, params)
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    result = module.torch_phase_qnode_compile_audit(
        circuit, params, fullgraph=True, tolerance=1e-10
    )
    assert result.passed and result.fullgraph
    np.testing.assert_allclose(result.value, expected.value, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.gradient, derivative.gradient, rtol=0, atol=1e-13)


def test_cached_original_graph_revalidates_actual_sorter_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse the same compiled functions after rebinding the original NumPy sorter.

    Parameters
    ----------
    monkeypatch
        Restore actual compiler and sorter dependencies after the guard check.

    """
    torch = importlib.import_module("torch")
    module = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    compile_actual = torch.compile
    captured: list[Any] = []

    def compiler(function: Any, **kwargs: Any) -> Any:
        result = compile_actual(function, backend="eager", **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(torch, "compile", compiler)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(PhaseQNodeOperation("ry", (1,), parameter_index=0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    result = module.torch_phase_qnode_compile_audit(
        circuit, [0.23], fullgraph=True, tolerance=1e-10
    )
    assert result.passed
    np.testing.assert_allclose(result.value, 1.0, atol=1e-14)
    np.testing.assert_allclose(result.gradient, [0.0], atol=1e-14)
    assert len(captured) == 2

    def identity_order(axes: list[int]) -> tuple[int, ...]:
        return tuple(range(len(axes)))

    monkeypatch.setattr(np, "argsort", identity_order)
    parameters = torch.tensor([0.23], dtype=torch.float64)
    value = captured[0](parameters)
    gradient = captured[1](parameters)
    np.testing.assert_allclose(float(value), math.cos(0.23), atol=1e-14)
    np.testing.assert_allclose(gradient.detach().numpy(), [-math.sin(0.23)], atol=1e-14)


@pytest.mark.parametrize(
    "gate,qubits,parameter_index",
    [("ry", (2,), 3), ("rxx", (2, 0), 3), ("ccnot", (2, 0, 1), None)],
)
def test_original_public_fullgraph_dense_gradient_matches_registered_numpy_execution(
    gate: str,
    qubits: tuple[int, ...],
    parameter_index: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture original public values and gradients on reordered one-to-three wires.

    Parameters
    ----------
    gate
        Actual registered local gate, including controlled three-wire execution.
    qubits
        Ordered wire dimensions used by the admitted circuit.
    parameter_index
        Original parameter binding, absent for the fixed controlled gate.
    monkeypatch
        Restore actual Torch compilation after selecting the eager capture backend.

    """
    torch = importlib.import_module("torch")
    module = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    observable = DenseHermitianObservable(
        np.asarray((values + values.T + 1j * (values - values.T)) / 80, dtype=np.complex128)
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("ry", (2,), parameter_index=2),
            PhaseQNodeOperation(gate, qubits, parameter_index=parameter_index),
        ),
        observable=observable,
    )
    params = np.array([0.31, -0.23, 0.47, 0.41])
    expected = execute_phase_qnode_circuit(circuit, params)
    derivative = parameter_shift_phase_qnode_gradient(circuit, params)
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    result = module.torch_phase_qnode_compile_audit(
        circuit, params, fullgraph=True, tolerance=1e-10
    )
    assert result.passed and result.fullgraph
    np.testing.assert_allclose(result.value, expected.value, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.gradient, derivative.gradient, rtol=0, atol=1e-13)
