# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native binary tensor axis operator tests
"""Check native axis arithmetic against bit-index sums and real circuit consumers."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from types import GeneratorType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_axis_operators import (
    jax_binary_axis_matmul,
    restore_binary_axis_order,
    torch_binary_axis_matmul,
)
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

pytestmark = pytest.mark.framework_imports


def _bit_reference(
    state: NDArray[np.complex128],
    matrix: NDArray[np.complex128],
    dimensions: int,
    qubits: tuple[int, ...],
) -> NDArray[np.complex128]:
    result = np.zeros_like(state)
    for source, amplitude in enumerate(state):
        bits = [(source >> (dimensions - 1 - axis)) & 1 for axis in range(dimensions)]
        column = sum(bits[axis] << (len(qubits) - 1 - index) for index, axis in enumerate(qubits))
        for row in range(2 ** len(qubits)):
            target = bits.copy()
            for index, axis in enumerate(qubits):
                target[axis] = (row >> (len(qubits) - 1 - index)) & 1
            destination = sum(bit << (dimensions - 1 - axis) for axis, bit in enumerate(target))
            result[destination] += matrix[row, column] * amplitude
    return result


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize(
    "dimensions,qubits", [(1, (0,)), (2, (1,)), (2, (1, 0)), (3, (2, 0)), (3, (0, 1, 2))]
)
def test_native_axis_products_preserve_complex_values_and_order(
    backend: str, dimensions: int, qubits: tuple[int, ...]
) -> None:
    """Apply nontrivial operators on adjacent, reversed and separated dimensions.

    Parameters
    ----------
    backend
        Actual CPU array runtime to execute.
    dimensions
        Number of binary tensor axes.
    qubits
        Ordered axes acted on by the supplied matrix.

    """
    values = np.arange(1, 2**dimensions + 1, dtype=np.float64)
    state = np.asarray(values + 1j * values[::-1], dtype=np.complex128)
    state /= np.linalg.norm(state)
    entries = np.arange(2 ** (2 * len(qubits)), dtype=np.float64).reshape(
        2 ** len(qubits), 2 ** len(qubits)
    )
    matrix = np.asarray(np.sin(entries + 0.3) + 1j * np.cos(entries.T + 0.7))
    expected = _bit_reference(state, matrix, dimensions, qubits)
    original_state, original_matrix = state.copy(), matrix.copy()
    axes = list(qubits) + [axis for axis in range(dimensions) if axis not in qubits]
    inverse = tuple(int(axis) for axis in np.argsort(axes))
    if backend == "jax":
        namespace = importlib.import_module("jax.numpy")
        updated = jax_binary_axis_matmul(
            namespace,
            namespace.asarray(state),
            namespace.asarray(matrix),
            dimensions=dimensions,
            width=len(qubits),
            axes=axes,
        )
        actual = restore_binary_axis_order(
            updated, inverse, reshape=namespace.reshape, permute=namespace.transpose
        )
    else:
        namespace = importlib.import_module("torch")
        updated = torch_binary_axis_matmul(
            namespace,
            namespace.as_tensor(state),
            namespace.as_tensor(matrix),
            dimensions=dimensions,
            width=len(qubits),
            axes=lambda: tuple(axes),
        )
        actual = restore_binary_axis_order(
            updated, inverse, reshape=namespace.reshape, permute=namespace.permute
        )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=2e-15)
    assert np.asarray(actual).dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(state, original_state)
    np.testing.assert_array_equal(matrix, original_matrix)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_axis_products_retain_the_parameter_gradient(backend: str) -> None:
    """Keep the derivative of a real native rotation through the complex tensor path.

    Parameters
    ----------
    backend
        Actual runtime whose native differentiation graph is checked.

    """
    theta = 0.37
    if backend == "jax":
        jax = importlib.import_module("jax")
        namespace = importlib.import_module("jax.numpy")

        def objective(parameter: object) -> object:
            matrix = namespace.cos(parameter) * namespace.eye(2, dtype=namespace.complex128)
            matrix += namespace.sin(parameter) * namespace.asarray(
                [[0.0, -1.0], [1.0, 0.0]], dtype=namespace.complex128
            )
            updated = jax_binary_axis_matmul(
                namespace,
                namespace.asarray([1.0, 0.0], dtype=namespace.complex128),
                matrix,
                dimensions=1,
                width=1,
                axes=(0,),
            )
            result = restore_binary_axis_order(
                updated, (0,), reshape=namespace.reshape, permute=namespace.transpose
            )
            return cast(object, namespace.real(result[1]))

        value, gradient = jax.jit(jax.value_and_grad(objective))(namespace.asarray(theta))
    else:
        namespace = importlib.import_module("torch")
        parameter = namespace.tensor(theta, dtype=namespace.float64, requires_grad=True)
        matrix = namespace.cos(parameter) * namespace.eye(2, dtype=namespace.complex128)
        matrix = matrix + namespace.sin(parameter) * namespace.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=namespace.complex128
        )
        updated = torch_binary_axis_matmul(
            namespace,
            namespace.tensor([1.0, 0.0], dtype=namespace.complex128),
            matrix,
            dimensions=1,
            width=1,
            axes=lambda: (0,),
        )
        result = restore_binary_axis_order(
            updated, (0,), reshape=namespace.reshape, permute=namespace.permute
        )
        native_value = namespace.real(result[1])
        value = native_value.detach().numpy()
        gradient = namespace.autograd.grad(native_value, parameter)[0].detach().numpy()
    np.testing.assert_allclose(value, math.sin(theta), rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(gradient, math.cos(theta), rtol=0.0, atol=2e-15)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_axis_products_propagate_incompatible_matrix_shapes(backend: str) -> None:
    """Expose the actual SDK matrix-shape error without manufacturing a result.

    Parameters
    ----------
    backend
        Runtime used to apply an incompatible three-row matrix to a binary axis.

    """
    with pytest.raises((ValueError, TypeError, RuntimeError)) as caught:
        if backend == "jax":
            namespace = importlib.import_module("jax.numpy")
            jax_binary_axis_matmul(
                namespace,
                namespace.zeros(4),
                namespace.eye(3),
                dimensions=2,
                width=1,
                axes=(0, 1),
            )
        else:
            namespace = importlib.import_module("torch")
            torch_binary_axis_matmul(
                namespace,
                namespace.zeros(4),
                namespace.eye(3),
                dimensions=2,
                width=1,
                axes=lambda: (0, 1),
            )
    assert "shape" in str(caught.value).lower()


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_registered_qnode_preserves_inverse_generator_scope(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the real original public circuit and observe its tuple construction.

    Parameters
    ----------
    backend
        Native registered-circuit route to execute.
    monkeypatch
        Restore the original module's tuple constructor after the namespace probe.

    """
    scopes: list[str] = []

    class ObservedTuple(tuple[object, ...]):
        """Construct real tuples while observing a supplied generator's origin."""

        def __new__(cls, values: Iterable[object] = ()) -> ObservedTuple:
            """Preserve native tuple contents and record the original lexical scope."""
            if isinstance(values, GeneratorType):
                scopes.append(values.gi_code.co_qualname)
            return super().__new__(cls, values)

    owner = importlib.import_module("scpn_quantum_control.phase." + backend + "_qnode_transforms")
    bridge = importlib.import_module("scpn_quantum_control.phase." + backend + "_bridge")
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("h", (0,)),
            PhaseQNodeOperation("ry", (2,), parameter_index=0),
            PhaseQNodeOperation("cnot", (2, 0)),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("crz", (1, 2), parameter_index=2),
            PhaseQNodeOperation("phase", (0,), parameter_index=3),
        ),
        observable=PauliTerm(1.0, ((0, "x"), (2, "z"))),
    )
    parameters = np.array([0.37, -0.21, 0.17, -0.4])
    reference = parameter_shift_phase_qnode_gradient(circuit, parameters)
    expected_state = execute_phase_qnode_circuit(circuit, parameters).state
    monkeypatch.setitem(vars(owner), "tuple", ObservedTuple)
    result = getattr(bridge, backend + "_phase_qnode_value_and_grad")(
        circuit, parameters, tolerance=1e-9
    )
    assert result.passed
    np.testing.assert_allclose(result.state, expected_state, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.gradient, reference.gradient, rtol=0.0, atol=1e-14)
    assert f"_{backend}_apply_gate_matrix.<locals>.<genexpr>" in scopes
