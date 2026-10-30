# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native controlled and permutation operator tests
"""Exercise real native matrix arithmetic and its original registered circuits."""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)
from scpn_quantum_control.phase_qnode_native_operators import (
    jax_basis_transposition,
    jax_controlled_operator,
    pauli_operator_entries,
    torch_basis_transposition,
    torch_controlled_operator_rows,
)

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("first,second", [(6, 7), (5, 6), (0, 4)])
def test_native_basis_transposition_exchanges_only_the_requested_states(
    backend: str, first: int, second: int
) -> None:
    """Compare every basis column against an independently permuted basis vector.

    Parameters
    ----------
    backend
        Real native CPU runtime.
    first
        First exchanged basis state.
    second
        Second exchanged basis state.

    """
    if backend == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        namespace = importlib.import_module("jax.numpy")
        actual = jax_basis_transposition(namespace, first, second)
    else:
        namespace = importlib.import_module("torch")
        actual = torch_basis_transposition(namespace, first, second)
    matrix = np.asarray(actual)
    assert matrix.dtype == np.dtype(np.complex128)
    for source in range(8):
        target = second if source == first else first if source == second else source
        expected = np.zeros(8, dtype=np.complex128)
        expected[target] = 1.0
        np.testing.assert_array_equal(matrix[:, source], expected)
    np.testing.assert_array_equal(matrix @ matrix, np.eye(8))


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_controlled_operator_preserves_complex_targets_and_native_hessians(
    backend: str,
) -> None:
    """Retain target amplitudes and two derivatives through real native execution.

    Parameters
    ----------
    backend
        Runtime used for the complex target and parameterized rotation.

    """
    target = np.array([[0.2 + 0.3j, -0.4j], [0.7, -0.5 + 0.1j]])
    expected = np.eye(4, dtype=np.complex128)
    expected[2:, 2:] = target
    theta = 0.37
    if backend == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        namespace = importlib.import_module("jax.numpy")
        original = namespace.asarray(target)
        matrix = jax_controlled_operator(namespace, original)

        def amplitude(angle: Any) -> Any:
            """Read the active-control diagonal of a traced real rotation."""
            c, s = namespace.cos(angle), namespace.sin(angle)
            rotation = namespace.asarray([[c, -s], [s, c]], dtype=namespace.complex128)
            return namespace.real(jax_controlled_operator(namespace, rotation)[2, 2])

        value = jax.jit(amplitude)(theta)
        gradient = jax.jit(jax.grad(amplitude))(theta)
        hessian = jax.jit(jax.grad(jax.grad(amplitude)))(theta)
    else:
        namespace = importlib.import_module("torch")
        original = namespace.as_tensor(target)
        matrix = namespace.stack(
            tuple(
                torch_controlled_operator_rows(
                    namespace,
                    original,
                    complex_tensor=lambda values: namespace.as_tensor(
                        values, dtype=namespace.complex128
                    ),
                )
            )
        )
        angle = namespace.tensor(theta, dtype=namespace.float64, requires_grad=True)
        c, s = namespace.cos(angle), namespace.sin(angle)
        rotation = namespace.stack((namespace.stack((c, -s)), namespace.stack((s, c))))
        rows = torch_controlled_operator_rows(
            namespace,
            rotation.to(namespace.complex128),
            complex_tensor=lambda values: namespace.as_tensor(values, dtype=namespace.complex128),
        )
        value = namespace.real(namespace.stack(tuple(rows))[2, 2])
        gradient = namespace.autograd.grad(value, angle, create_graph=True)[0]
        hessian = namespace.autograd.grad(gradient, angle)[0]
        value, gradient, hessian = value.detach(), gradient.detach(), hessian.detach()
    np.testing.assert_array_equal(np.asarray(matrix), expected)
    np.testing.assert_array_equal(np.asarray(original), target)
    np.testing.assert_allclose(
        [value, gradient, hessian],
        [math.cos(theta), -math.sin(theta), -math.cos(theta)],
        rtol=0.0,
        atol=2e-15,
    )


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("gate", ["ch", "cs", "ct", "crx", "cry", "crz", "ccnot", "cswap"])
def test_original_registered_circuits_use_native_operator_arithmetic(
    backend: str, gate: str
) -> None:
    """Compare original public state and gradient routes with the NumPy executor.

    Parameters
    ----------
    backend
        Native public bridge used to execute the registered circuit.
    gate
        Controlled or three-qubit operation whose numerical owner was separated.

    """
    operation = PhaseQNodeOperation(
        gate,
        (2, 0, 1) if gate in ("ccnot", "cswap") else (2, 0),
        parameter_index=3 if gate in ("crx", "cry", "crz") else None,
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("ry", (2,), parameter_index=2),
            operation,
        ),
        observable=PauliTerm(0.7, ((0, "x"), (1, "y"), (2, "z"))),
    )
    parameters = np.array([0.37, -0.21, 0.61, -0.43])
    reference = parameter_shift_phase_qnode_gradient(circuit, parameters)
    expected_state = execute_phase_qnode_circuit(circuit, parameters).state
    bridge = importlib.import_module("scpn_quantum_control.phase." + backend + "_bridge")
    actual = getattr(bridge, backend + "_phase_qnode_value_and_grad")(
        circuit, parameters, tolerance=1e-10
    )
    assert actual.passed
    np.testing.assert_allclose(actual.state, expected_state, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.value, reference.value, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, reference.gradient, rtol=0.0, atol=1e-14)


def test_original_torch_controlled_helper_keeps_late_primitives_and_tuple_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe late rebinding while actual Torch rows and a public circuit execute.

    Parameters
    ----------
    monkeypatch
        Restore original globals after observing real tensor construction.

    """
    namespace = importlib.import_module("torch")
    owner = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    bridge = importlib.import_module("scpn_quantum_control.phase.torch_bridge")
    original = owner._torch_complex_tensor
    observed: list[str] = []

    class RowsTuple(tuple[object, ...]):
        """Observe real controlled rows converted through the original module global."""

        def __new__(cls, values: Iterable[object] = ()) -> RowsTuple:
            """Keep actual tuple values and observe four real Torch row tensors."""
            if isinstance(values, list) and len(values) == 4:
                assert all(namespace.is_tensor(row) for row in values)
                observed.append("tuple")
            return super().__new__(cls, values)

    def second(runtime: Any, values: object) -> Any:
        """Use the real active constructor after the first one rebinds its global."""
        observed.append("second")
        return original(runtime, values)

    def first(runtime: Any, values: object) -> Any:
        """Rebind both genuine primitives during the first controlled identity row."""
        if values == [1.0, 0.0, 0.0, 0.0]:
            observed.append("first")
            monkeypatch.setitem(vars(owner), "_torch_complex_tensor", second)
            monkeypatch.setitem(vars(owner), "tuple", RowsTuple)
        return original(runtime, values)

    monkeypatch.setitem(vars(owner), "_torch_complex_tensor", first)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("h", (0,)),
            PhaseQNodeOperation("ch", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((1, "x"),)),
    )
    actual = bridge.torch_phase_qnode_value_and_grad(circuit, np.array([0.2]))
    expected = execute_phase_qnode_circuit(circuit, np.array([0.2]))
    assert actual.passed
    np.testing.assert_allclose(actual.state, expected.state, rtol=0.0, atol=1e-14)
    start = observed.index("first")
    assert observed[start : start + 5] == ["first", "second", "second", "second", "tuple"]


def test_pauli_entries_obey_operator_algebra_and_are_fresh() -> None:
    """Check Pauli products and ensure mutation cannot alter later gate construction."""
    x, y, z = (np.array(pauli_operator_entries(label)) for label in ("x", "y", "z"))
    np.testing.assert_array_equal(x @ y, 1j * z)
    np.testing.assert_array_equal(y @ z, 1j * x)
    np.testing.assert_array_equal(z @ x, 1j * y)
    for matrix in (x, y, z):
        np.testing.assert_array_equal(matrix @ matrix, np.eye(2))
    rows = pauli_operator_entries("x")
    rows[0][1] = 5.0
    rows.append([7.0, 8.0])
    np.testing.assert_array_equal(pauli_operator_entries("x"), x)
    with pytest.raises(ValueError, match="unsupported Pauli label: identity"):
        pauli_operator_entries("identity")


def test_original_torch_pauli_factories_keep_their_lexical_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe original nested factories while the real public circuit executes.

    Parameters
    ----------
    monkeypatch
        Restore the original active tensor-construction helper after observation.

    """
    owner = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    bridge = importlib.import_module("scpn_quantum_control.phase.torch_bridge")
    original = owner._torch_complex_tensor
    scopes: list[str] = []

    def observed(runtime: Any, values: object) -> Any:
        """Observe the natural caller before running the exact real constructor."""
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        scopes.append(frame.f_back.f_code.co_qualname)
        del frame
        return original(runtime, values)

    monkeypatch.setitem(vars(owner), "_torch_complex_tensor", observed)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("x", (0,)),
            PhaseQNodeOperation("y", (0,)),
            PhaseQNodeOperation("z", (0,)),
        ),
        observable=PauliTerm(1.0, ((0, "x"),)),
    )
    actual = bridge.torch_phase_qnode_value_and_grad(circuit, np.array([0.37]))
    assert actual.passed
    np.testing.assert_allclose(
        actual.state,
        execute_phase_qnode_circuit(circuit, np.array([0.37])).state,
        rtol=0.0,
        atol=1e-14,
    )
    for name in ("x_matrix", "y_matrix", "z_matrix"):
        assert "_torch_gate_matrix.<locals>." + name in scopes
