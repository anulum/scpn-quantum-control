# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native local-operator evolution tests
"""Exercise lazy native evolution, dependency rebinding and real circuit routes."""

from __future__ import annotations

import importlib
import math
import weakref
from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
)
from scpn_quantum_control.phase_qnode_gate_numerics import pauli_rotation
from scpn_quantum_control.phase_qnode_state_evolution import (
    NativeLocalOperator,
    evolve_local_operators,
    jax_zero_state,
    torch_zero_state,
)

pytestmark = pytest.mark.framework_imports


@pytest.fixture(autouse=True)
def _precision(backend: str) -> Iterator[None]:
    """Save and restore actual JAX precision for each admitted SDK case."""
    if backend != "jax":
        yield
        return
    jax = importlib.import_module("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_pure_native_evolution_retains_derivatives_and_empty_identity(backend: str) -> None:
    """Differentiate the supplied-matrix engine without importing a legacy adapter.

    Parameters
    ----------
    backend
        Actual CPU SDK providing numerical tensors and transformations.

    """
    runtime = importlib.import_module(backend)
    ns = importlib.import_module("jax.numpy") if backend == "jax" else runtime

    def tensor(values: object) -> Any:
        if backend == "jax":
            return ns.asarray(values, dtype=ns.complex128)
        return ns.as_tensor(values, dtype=ns.complex128)

    def zero() -> Any:
        return (
            jax_zero_state(ns, lambda: 1) if backend == "jax" else torch_zero_state(ns, lambda: 1)
        )

    def multiply(
        namespace: Any, state: Any, width: int, qubits: tuple[int, ...], matrix: Any
    ) -> Any:
        assert namespace is ns and width == 1 and qubits == (0,)
        return matrix @ state

    def objective(theta: Any) -> Any:
        matrix = pauli_rotation(
            ns,
            theta,
            tensor(1j),
            identity=lambda: ns.eye(2, dtype=ns.complex128),
            operator=lambda: tensor([[0, -1j], [1j, 0]]),
        )
        state, last = evolve_local_operators(
            zero(), (NativeLocalOperator(multiply, ns, 1, (0,), matrix),)
        )
        assert last is matrix
        return ns.real(ns.vdot(state, tensor([[1, 0], [0, -1]]) @ state))

    initial = zero()
    unchanged, matrix = evolve_local_operators(initial, ())
    assert unchanged is initial and matrix is None
    if backend == "jax":
        theta = ns.asarray(0.37)
        value = runtime.jit(objective)(theta)
        gradient = runtime.jit(runtime.grad(objective))(theta)
        hessian = runtime.jit(runtime.hessian(objective))(theta)
    else:
        theta = ns.tensor(0.37, dtype=ns.float64)
        value = runtime.compile(objective, backend="eager", fullgraph=True)(theta)
        gradient = runtime.func.jacrev(objective)(theta)
        hessian = runtime.func.jacrev(runtime.func.jacrev(objective))(theta)
    np.testing.assert_allclose(
        [float(value), float(gradient), float(hessian)],
        [math.cos(0.37), -math.sin(0.37), -math.cos(0.37)],
        atol=1e-14,
    )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_stream_resolves_each_application_after_the_previous_step(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebind a real matrix application between instructions on the original route.

    Parameters
    ----------
    backend
        Actual native adapter whose natural dependency bindings are observed.
    monkeypatch
        Restore application functions and native SDK operations after execution.

    """
    runtime = importlib.import_module(backend)
    ns = importlib.import_module("jax.numpy") if backend == "jax" else runtime
    original = importlib.import_module(
        "scpn_quantum_control.phase." + backend + "_qnode_transforms"
    )
    name = "_" + backend + "_apply_gate_matrix"
    application = getattr(original, name)
    actual_cast = original.cast
    actual_zeros = ns.zeros
    events: list[str] = []

    def zeros(*args: Any, **kwargs: Any) -> Any:
        events.append("zeros")
        return actual_zeros(*args, **kwargs)

    def cast(type_alias: object, value: Any) -> Any:
        events.append("operations")
        return actual_cast(type_alias, value)

    def second(*args: Any) -> Any:
        events.append("second")
        return application(*args)

    def first(*args: Any) -> Any:
        events.append("first")
        result = application(*args)
        monkeypatch.setattr(original, name, second)
        return result

    monkeypatch.setattr(ns, "zeros", zeros)
    monkeypatch.setattr(original, "cast", cast)
    monkeypatch.setattr(original, name, first)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
        ),
        observable=PauliTerm(0.7, ((0, "x"), (1, "z"))),
    )
    expected = execute_phase_qnode_circuit(circuit, [0.23, -0.41])
    actual_value, actual_state = getattr(original, "_" + backend + "_phase_qnode_value_and_state")(
        ns, circuit, np.array([0.23, -0.41])
    )
    if backend == "torch":
        actual_value, actual_state = actual_value.detach().numpy(), actual_state.detach().numpy()
    np.testing.assert_allclose(actual_state, expected.state, atol=1e-14)
    np.testing.assert_allclose(actual_value, expected.value, atol=1e-14)
    assert events[:4] == ["zeros", "operations", "first", "second"]
    assert events.count("first") == 1 and events.count("second") == 3


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_zero_state_resolves_sdk_callable_before_shape_lookup(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain native callable capture when the supplied dimension lookup rebinds it.

    Parameters
    ----------
    backend
        Actual native array namespace to allocate the state.
    monkeypatch
        Restore the SDK allocation operation after the dimension side effect.

    """
    runtime = importlib.import_module(backend)
    ns = importlib.import_module("jax.numpy") if backend == "jax" else runtime
    actual_zeros = ns.zeros
    calls: list[str] = []

    def before(*args: Any, **kwargs: Any) -> Any:
        calls.append("before")
        return actual_zeros(*args, **kwargs)

    def after(*args: Any, **kwargs: Any) -> Any:
        calls.append("after")
        return actual_zeros(*args, **kwargs)

    def width() -> int:
        monkeypatch.setattr(ns, "zeros", after)
        return 2

    monkeypatch.setattr(ns, "zeros", before)
    zero = jax_zero_state(ns, width) if backend == "jax" else torch_zero_state(ns, width)
    np.testing.assert_array_equal(zero, [1, 0, 0, 0])
    assert calls == ["before"]


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_last_gate_matrix_remains_alive_through_observable_evaluation(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe the actual final native matrix lifetime at the original boundary.

    Parameters
    ----------
    backend
        Actual native circuit execution adapter.
    monkeypatch
        Restore the original mathematical gate and observable functions.

    """
    runtime = importlib.import_module(backend)
    ns = importlib.import_module("jax.numpy") if backend == "jax" else runtime
    original = importlib.import_module(
        "scpn_quantum_control.phase." + backend + "_qnode_transforms"
    )
    gate_name, expectation_name = (
        "_" + backend + "_gate_matrix",
        "_" + backend + "_expectation_value",
    )
    actual_gate, actual_expectation = (
        getattr(original, gate_name),
        getattr(original, expectation_name),
    )
    references: list[weakref.ReferenceType[Any]] = []
    observed: list[bool] = []

    def gate(*args: Any) -> Any:
        matrix = actual_gate(*args)
        references.append(weakref.ref(matrix))
        return matrix

    def expectation(*args: Any) -> Any:
        observed.append(references[-1]() is not None)
        return actual_expectation(*args)

    monkeypatch.setattr(original, gate_name, gate)
    monkeypatch.setattr(original, expectation_name, expectation)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("ry", (0,), parameter_index=0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    value, _state = getattr(original, "_" + backend + "_phase_qnode_value_and_state")(
        ns, circuit, np.array([0.2])
    )
    assert observed == [True]
    assert float(value) == pytest.approx(math.cos(0.2))


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_public_compiled_value_and_gradient_execute_the_lazy_stream(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute JAX JIT and the admitted Torch non-fullgraph eager compiler route.

    Parameters
    ----------
    backend
        Actual CPU framework whose public registered-circuit route executes.
    monkeypatch
        Restore the Torch compiler after selecting its real eager capture backend.

    """
    runtime = importlib.import_module(backend)
    original = importlib.import_module(
        "scpn_quantum_control.phase." + backend + "_qnode_transforms"
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (1,), parameter_index=0),
            PhaseQNodeOperation("cnot", (1, 0)),
        ),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    if backend == "jax":
        result = original.jax_phase_qnode_value_and_grad(
            circuit, [0.23], jit=True, tolerance=1e-10
        )
        assert result.jitted
    else:
        monkeypatch.setattr(runtime, "compile", partial(runtime.compile, backend="eager"))
        result = original.torch_phase_qnode_compile_audit(
            circuit, [0.23], fullgraph=False, tolerance=1e-10
        )
        assert not result.fullgraph
    assert result.passed
    np.testing.assert_allclose(result.value, math.cos(0.23), atol=1e-14)
    np.testing.assert_allclose(result.gradient, [-math.sin(0.23)], atol=1e-14)
