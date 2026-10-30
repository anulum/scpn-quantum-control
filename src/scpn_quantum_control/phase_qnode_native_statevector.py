# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Phase-QNode statevector kernels
"""Supply PyTorch and JAX kernels to the Phase-QNode statevector engine.

Each runtime module is supplied by the caller; nothing here imports PyTorch or
JAX, loads a framework or admits inputs. The value functions return native
tensors that keep the parameter gradient graph of the supplied runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .differentiable_axis_compilation import numpy_axis_order
from .differentiable_axis_operators import (
    jax_binary_axis_matmul,
    restore_binary_axis_order,
    torch_binary_axis_matmul,
)
from .phase.qnode_circuit import PhaseQNodeCircuit, PhaseQNodeOperation
from .phase_qnode_native_gates import jax_gate_operator, torch_gate_operator
from .phase_qnode_native_operators import pauli_operator_entries
from .phase_qnode_state_evolution import jax_zero_state, torch_zero_state
from .phase_qnode_statevector_engine import (
    StatevectorKernels,
    native_statevector_kernels,
    refuse_unregistered_observable,
    statevector_value_and_state,
)

_PAULI_LABELS = frozenset({"x", "y", "z"})


def torch_apply_local_operator(
    torch_module: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
    *,
    inverse_order: Callable[[list[int]], Iterable[int]] | None = None,
) -> Any:
    """Apply a native PyTorch matrix to ordered qubit dimensions of a state.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    state
        Native statevector of ``2 ** n_qubits`` amplitudes.
    n_qubits
        Admitted number of qubits of the state.
    qubits
        Ordered qubit dimensions acted on by ``matrix``.
    matrix
        Native operator of dimension ``2 ** len(qubits)``.
    inverse_order
        Inverse axis order of a permutation; NumPy ``argsort`` by default.

    Returns
    -------
    Any
        Transformed native statevector in the original qubit order.

    """
    width = len(qubits)
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = tuple(
        int(axis)
        for axis in (
            numpy_axis_order(np.argsort, axes) if inverse_order is None else inverse_order(axes)
        )
    )
    updated = torch_binary_axis_matmul(
        torch_module,
        state,
        matrix,
        dimensions=n_qubits,
        width=width,
        axes=lambda: tuple(axes),
    )
    reshape = torch_module.reshape
    permute = torch_module.permute
    inverse_axes = inverse
    return restore_binary_axis_order(updated, inverse_axes, reshape=reshape, permute=permute)


def jax_apply_local_operator(
    jnp: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
    *,
    inverse_order: Callable[[list[int]], Iterable[int]] | None = None,
) -> Any:
    """Apply a native JAX matrix to ordered qubit dimensions of a state.

    Parameters
    ----------
    jnp
        Supplied ``jax.numpy`` namespace.
    state
        Native statevector of ``2 ** n_qubits`` amplitudes.
    n_qubits
        Admitted number of qubits of the state.
    qubits
        Ordered qubit dimensions acted on by ``matrix``.
    matrix
        Native operator of dimension ``2 ** len(qubits)``.
    inverse_order
        Inverse axis order of a permutation; NumPy ``argsort`` by default.

    Returns
    -------
    Any
        Transformed native statevector in the original qubit order.

    """
    width = len(qubits)
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = np.argsort(axes) if inverse_order is None else inverse_order(axes)
    updated = jax_binary_axis_matmul(
        jnp,
        state,
        matrix,
        dimensions=n_qubits,
        width=width,
        axes=axes,
    )
    reshape = jnp.reshape
    permute = jnp.transpose
    inverse_axes = tuple(int(axis) for axis in inverse)
    return restore_binary_axis_order(updated, inverse_axes, reshape=reshape, permute=permute)


def torch_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    """Return the PyTorch angle of one operation.

    Parameters
    ----------
    operation
        Registered circuit operation.
    parameter_tensor
        Native PyTorch parameter vector.

    Returns
    -------
    Any
        The indexed parameter, or a zero tensor of the same dtype and device
        for a fixed gate.

    """
    if operation.parameter_index is None:
        return parameter_tensor.new_tensor(0.0)
    return parameter_tensor[operation.parameter_index]


def jax_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    """Return the JAX angle of one operation.

    Parameters
    ----------
    operation
        Registered circuit operation.
    parameter_tensor
        Native JAX parameter vector.

    Returns
    -------
    Any
        The indexed parameter, or ``0.0`` for a fixed gate.

    """
    if operation.parameter_index is None:
        return 0.0
    return parameter_tensor[operation.parameter_index]


def _pauli_entries(label: str, runtime: str) -> list[list[complex]]:
    if label not in _PAULI_LABELS:
        raise ValueError(f"unsupported {runtime} Pauli label: {label}")
    return pauli_operator_entries(label)


def torch_statevector_kernels(torch_module: Any) -> StatevectorKernels:
    """Return the PyTorch kernels of the statevector engine.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.

    Returns
    -------
    StatevectorKernels
        Complex128/float64 PyTorch kernels whose routes are the engine's own.

    """

    def complex_tensor(values: object) -> Any:
        return torch_module.as_tensor(values, dtype=torch_module.complex128)

    return native_statevector_kernels(
        namespace=torch_module,
        parameter_tensor=lambda params: torch_module.as_tensor(params, dtype=torch_module.float64),
        zero_state=torch_zero_state,
        operation_theta=torch_operation_theta,
        gate_operator=lambda gate, theta: torch_gate_operator(torch_module, gate, theta),
        apply_operator=torch_apply_local_operator,
        pauli_operator=lambda label: complex_tensor(_pauli_entries(label, "PyTorch")),
        dense_operator=complex_tensor,
        real_zero=lambda: torch_module.as_tensor(0.0, dtype=torch_module.float64),
        other_observable=refuse_unregistered_observable(
            "unsupported PyTorch Phase-QNode observable"
        ),
    )


def jax_statevector_kernels(jnp: Any) -> StatevectorKernels:
    """Return the JAX kernels of the statevector engine.

    Parameters
    ----------
    jnp
        Supplied ``jax.numpy`` namespace; complex128 needs JAX 64-bit mode.

    Returns
    -------
    StatevectorKernels
        Complex128 JAX kernels whose routes are the engine's own.

    """
    return native_statevector_kernels(
        namespace=jnp,
        parameter_tensor=lambda params: jnp.asarray(params),
        zero_state=jax_zero_state,
        operation_theta=jax_operation_theta,
        gate_operator=lambda gate, theta: jax_gate_operator(jnp, gate, theta),
        apply_operator=jax_apply_local_operator,
        pauli_operator=lambda label: jnp.asarray(
            _pauli_entries(label, "JAX"), dtype=jnp.complex128
        ),
        dense_operator=lambda matrix: jnp.asarray(matrix, dtype=jnp.complex128),
        real_zero=lambda: jnp.asarray(0.0),
        other_observable=refuse_unregistered_observable("unsupported JAX Phase-QNode observable"),
    )


def torch_statevector_value_and_state(
    torch_module: Any,
    circuit: PhaseQNodeCircuit,
    parameter_tensor: Any,
    *,
    dense_matrix: Any = None,
) -> tuple[Any, Any]:
    """Return the PyTorch expectation value and final state of a circuit.

    Parameters
    ----------
    torch_module
        Supplied PyTorch module.
    circuit
        Admitted registered Phase-QNode circuit.
    parameter_tensor
        Admitted parameters; a float64 tensor keeps its gradient graph.
    dense_matrix
        Optional pre-converted native matrix of a dense observable.

    Returns
    -------
    tuple[Any, Any]
        Native real expectation value and final complex128 statevector.

    """
    return statevector_value_and_state(
        torch_statevector_kernels(torch_module), circuit, parameter_tensor, dense_matrix
    )


def jax_statevector_value_and_state(
    jnp: Any, circuit: PhaseQNodeCircuit, parameter_tensor: Any
) -> tuple[Any, Any]:
    """Return the JAX expectation value and final state of a circuit.

    Parameters
    ----------
    jnp
        Supplied ``jax.numpy`` namespace.
    circuit
        Admitted registered Phase-QNode circuit.
    parameter_tensor
        Admitted parameters; JAX transformations trace through them.

    Returns
    -------
    tuple[Any, Any]
        Native real expectation value and final complex128 statevector.

    """
    return statevector_value_and_state(jax_statevector_kernels(jnp), circuit, parameter_tensor)


__all__ = [
    "jax_apply_local_operator",
    "jax_operation_theta",
    "jax_statevector_kernels",
    "jax_statevector_value_and_state",
    "torch_apply_local_operator",
    "torch_operation_theta",
    "torch_statevector_kernels",
    "torch_statevector_value_and_state",
]
