# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Execution
"""Numerical statevector and density execution for Phase-QNode circuits.

This one-way leaf executes registered gates and Kraus channels, applies local
operators, and evaluates registered observables. It contains no parameter
shift, derivative propagation, Fisher/metric orchestration, framework,
compiler, provider, hardware, or benchmark logic.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .qnode_circuit_contracts import (
    _H,
    _I,
    _PARAMETRIC_GATES,
    _PAULI_MATRICES,
    _S,
    _SX,
    _T,
    _X,
    _Y,
    _Z,
    ComplexArray,
    DenseHermitianObservable,
    FloatArray,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeDensityExecutionResult,
    PhaseQNodeExecutionResult,
    PhaseQNodeNoiseChannel,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
)
from .qnode_circuit_support import (
    _as_parameter_vector,
    _parsed_density_operations,
    _parsed_operations,
    phase_qnode_density_support_report,
    phase_qnode_support_report,
)


def execute_phase_qnode_circuit(
    circuit: PhaseQNodeCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeExecutionResult:
    """Execute a registered local Phase-QNode circuit with a statevector simulator."""
    values = _as_parameter_vector(parameters)
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    state = np.zeros(2**circuit.n_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    for operation in _parsed_operations(circuit):
        state = _apply_operation(state, circuit.n_qubits, operation, values)
    value = _expectation_value(state, circuit.n_qubits, circuit.observable)
    return PhaseQNodeExecutionResult(value=value, state=state, support_report=report)


def execute_phase_qnode_density_matrix(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeDensityExecutionResult:
    """Execute a registered local Phase-QNode through a density-matrix simulator."""
    density_circuit = _as_density_circuit(circuit)
    values = _as_parameter_vector(parameters)
    report = phase_qnode_density_support_report(density_circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    dimension = 2**density_circuit.n_qubits
    density = np.zeros((dimension, dimension), dtype=np.complex128)
    density[0, 0] = 1.0 + 0.0j
    for operation in _parsed_density_operations(density_circuit):
        if isinstance(operation, PhaseQNodeOperation):
            matrix = _operation_matrix(operation, values)
            density = _apply_unitary_density_matrix(
                density,
                density_circuit.n_qubits,
                operation.qubits,
                matrix,
            )
        else:
            density = _apply_noise_channel_density_matrix(
                density,
                density_circuit.n_qubits,
                operation,
            )
    trace = float(np.real_if_close(np.trace(density)).real)
    purity = float(np.real_if_close(np.trace(density @ density)).real)
    value = _density_expectation_value(
        density, density_circuit.n_qubits, density_circuit.observable
    )
    return PhaseQNodeDensityExecutionResult(
        value=value,
        density_matrix=density,
        trace=trace,
        purity=purity,
        support_report=report,
        claim_boundary=(
            "local density-matrix Phase-QNode execution for registered unitary "
            "gates and registered single-qubit Kraus noise channels; no "
            "parameter-shift gradient, pure-state metric, finite-shot, provider, "
            "hardware, or benchmark-promotion claim"
        ),
    )


def _as_density_circuit(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
) -> PhaseQNodeDensityCircuit:
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        return circuit
    return PhaseQNodeDensityCircuit(
        n_qubits=circuit.n_qubits,
        operations=circuit.operations,
        observable=circuit.observable,
    )


def _apply_operation(
    state: ComplexArray,
    n_qubits: int,
    operation: PhaseQNodeOperation,
    parameters: FloatArray,
) -> ComplexArray:
    gate = operation.gate
    if gate in {"rx", "ry", "rz", "phase", "crx", "cry", "crz", "rxx", "ryy", "rzz"}:
        theta = float(parameters[cast(int, operation.parameter_index)])
    else:
        theta = 0.0
    matrix = _gate_matrix(gate, theta)
    return _apply_gate_matrix(state, n_qubits, operation.qubits, matrix)


def _operation_matrix(
    operation: PhaseQNodeOperation,
    parameters: FloatArray,
) -> ComplexArray:
    theta = 0.0
    if operation.gate in _PARAMETRIC_GATES:
        theta = float(parameters[cast(int, operation.parameter_index)])
    return _gate_matrix(operation.gate, theta)


def _gate_matrix(gate: str, theta: float) -> ComplexArray:
    if gate == "h":
        return np.asarray(_H, dtype=np.complex128)
    if gate == "x":
        return np.asarray(_X, dtype=np.complex128)
    if gate == "y":
        return np.asarray(_Y, dtype=np.complex128)
    if gate == "z":
        return np.asarray(_Z, dtype=np.complex128)
    if gate == "s":
        return np.asarray(_S, dtype=np.complex128)
    if gate == "t":
        return np.asarray(_T, dtype=np.complex128)
    if gate == "sx":
        return np.asarray(_SX, dtype=np.complex128)
    if gate == "rx":
        return np.asarray(np.cos(theta / 2.0) * _I - 1.0j * np.sin(theta / 2.0) * _X)
    if gate == "ry":
        return np.asarray(np.cos(theta / 2.0) * _I - 1.0j * np.sin(theta / 2.0) * _Y)
    if gate == "rz":
        return np.asarray(np.cos(theta / 2.0) * _I - 1.0j * np.sin(theta / 2.0) * _Z)
    if gate == "phase":
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * theta)]], dtype=np.complex128)
    if gate == "cnot":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
    if gate == "cz":
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    if gate == "cy":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1.0j], [0, 0, 1.0j, 0]],
            dtype=np.complex128,
        )
    if gate == "swap":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
    if gate == "ch":
        return _controlled(_H)
    if gate == "cs":
        return _controlled(_S)
    if gate == "ct":
        return _controlled(_T)
    if gate == "ccnot":
        return _ccnot_matrix()
    if gate == "ccz":
        return np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    if gate == "cswap":
        return _cswap_matrix()
    if gate == "crx":
        return _controlled(_gate_matrix("rx", theta))
    if gate == "cry":
        return _controlled(_gate_matrix("ry", theta))
    if gate == "crz":
        return _controlled(_gate_matrix("rz", theta))
    if gate == "rxx":
        return np.asarray(
            np.cos(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 1.0j * np.sin(theta / 2.0) * np.kron(_X, _X),
            dtype=np.complex128,
        )
    if gate == "ryy":
        return np.asarray(
            np.cos(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 1.0j * np.sin(theta / 2.0) * np.kron(_Y, _Y),
            dtype=np.complex128,
        )
    if gate == "rzz":
        return np.asarray(
            np.cos(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 1.0j * np.sin(theta / 2.0) * np.kron(_Z, _Z),
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported gate matrix: {gate}")


def _controlled(target: ComplexArray) -> ComplexArray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix[2:4, 2:4] = target
    return matrix


def _ccnot_matrix() -> ComplexArray:
    matrix = np.eye(8, dtype=np.complex128)
    matrix[6, 6] = 0.0
    matrix[7, 7] = 0.0
    matrix[6, 7] = 1.0
    matrix[7, 6] = 1.0
    return matrix


def _cswap_matrix() -> ComplexArray:
    matrix = np.eye(8, dtype=np.complex128)
    matrix[5, 5] = 0.0
    matrix[6, 6] = 0.0
    matrix[5, 6] = 1.0
    matrix[6, 5] = 1.0
    return matrix


def _apply_gate_matrix(
    state: ComplexArray,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: ComplexArray,
) -> ComplexArray:
    width = len(qubits)
    if matrix.shape != (2**width, 2**width):
        raise ValueError("gate matrix shape does not match target qubits")
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = np.argsort(axes)
    tensor = state.reshape((2,) * n_qubits).transpose(axes)
    front = tensor.reshape(2**width, -1)
    updated = (matrix @ front).reshape((2,) * n_qubits).transpose(inverse)
    return cast(ComplexArray, updated.reshape(-1).astype(np.complex128, copy=False))


def _expanded_operator(
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: ComplexArray,
) -> ComplexArray:
    dimension = 2**n_qubits
    expanded = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        basis = np.zeros(dimension, dtype=np.complex128)
        basis[column] = 1.0 + 0.0j
        expanded[:, column] = _apply_gate_matrix(basis, n_qubits, qubits, matrix)
    return expanded


def _apply_unitary_density_matrix(
    density: ComplexArray,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: ComplexArray,
) -> ComplexArray:
    expanded = _expanded_operator(n_qubits, qubits, matrix)
    return cast(ComplexArray, (expanded @ density @ expanded.conj().T).astype(np.complex128))


def _apply_noise_channel_density_matrix(
    density: ComplexArray,
    n_qubits: int,
    channel: PhaseQNodeNoiseChannel,
) -> ComplexArray:
    updated = np.zeros_like(density)
    for kraus in _noise_channel_kraus(channel):
        expanded = _expanded_operator(n_qubits, channel.qubits, kraus)
        updated += expanded @ density @ expanded.conj().T
    return cast(ComplexArray, updated.astype(np.complex128, copy=False))


def _noise_channel_kraus(channel: PhaseQNodeNoiseChannel) -> tuple[ComplexArray, ...]:
    probability = channel.probability
    if channel.channel == "bit_flip":
        return (
            np.sqrt(1.0 - probability) * _I,
            np.sqrt(probability) * _X,
        )
    if channel.channel == "phase_flip":
        return (
            np.sqrt(1.0 - probability) * _I,
            np.sqrt(probability) * _Z,
        )
    if channel.channel == "depolarizing":
        return (
            np.sqrt(1.0 - probability) * _I,
            np.sqrt(probability / 3.0) * _X,
            np.sqrt(probability / 3.0) * _Y,
            np.sqrt(probability / 3.0) * _Z,
        )
    if channel.channel == "amplitude_damping":
        return (
            np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - probability)]], dtype=np.complex128),
            np.array([[0.0, np.sqrt(probability)], [0.0, 0.0]], dtype=np.complex128),
        )
    raise ValueError(f"unsupported noise channel: {channel.channel}")


def _expectation_value(
    state: ComplexArray,
    n_qubits: int,
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    ),
) -> float:
    if isinstance(observable, DenseHermitianObservable):
        value = np.vdot(state, observable.matrix @ state)
        return float(np.real_if_close(value).real)
    if isinstance(observable, PauliCovarianceObservable):
        return _covariance_expectation(state, n_qubits, observable)
    if isinstance(observable, SparsePauliHamiltonian):
        return float(sum(_term_expectation(state, n_qubits, term) for term in observable.terms))
    if isinstance(observable, PauliTerm):
        return _term_expectation(state, n_qubits, observable)
    raise ValueError(f"unsupported observable: {observable}")


def _density_expectation_value(
    density: ComplexArray,
    n_qubits: int,
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    ),
) -> float:
    if isinstance(observable, DenseHermitianObservable):
        value = np.trace(observable.matrix @ density)
        return float(np.real_if_close(value).real)
    if isinstance(observable, PauliCovarianceObservable):
        return _density_covariance_expectation(density, n_qubits, observable)
    if isinstance(observable, SparsePauliHamiltonian):
        return float(
            sum(_density_term_expectation(density, n_qubits, term) for term in observable.terms)
        )
    if isinstance(observable, PauliTerm):
        return _density_term_expectation(density, n_qubits, observable)
    raise ValueError(f"unsupported observable: {observable}")


def _density_term_expectation(
    density: ComplexArray,
    n_qubits: int,
    term: PauliTerm,
) -> float:
    operator = _term_operator(n_qubits, term)
    value = term.coefficient * np.trace(operator @ density)
    return float(np.real_if_close(value).real)


def _term_expectation(state: ComplexArray, n_qubits: int, term: PauliTerm) -> float:
    transformed = state.copy()
    for qubit, label in term.factors:
        transformed = _apply_gate_matrix(transformed, n_qubits, (qubit,), _PAULI_MATRICES[label])
    value = term.coefficient * np.vdot(state, transformed)
    return float(np.real_if_close(value).real)


def _execute_state(circuit: PhaseQNodeCircuit, values: FloatArray) -> ComplexArray:
    state = np.zeros(2**circuit.n_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    for operation in _parsed_operations(circuit):
        state = _apply_operation(state, circuit.n_qubits, operation, values)
    return state


def _covariance_expectation(
    state: ComplexArray,
    n_qubits: int,
    observable: PauliCovarianceObservable,
) -> float:
    symmetrized = _symmetrized_product_expectation(
        state,
        n_qubits,
        observable.left,
        observable.right,
    )
    left_mean = _term_expectation(state, n_qubits, observable.left)
    right_mean = _term_expectation(state, n_qubits, observable.right)
    return float(symmetrized - left_mean * right_mean)


def _symmetrized_product_expectation(
    state: ComplexArray,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> float:
    left_right = _term_product_expectation(state, n_qubits, left, right)
    right_left = _term_product_expectation(state, n_qubits, right, left)
    return float(np.real_if_close(0.5 * (left_right + right_left)).real)


def _term_product_expectation(
    state: ComplexArray,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> complex:
    transformed = _apply_term_operator(state, n_qubits, right)
    transformed = _apply_term_operator(transformed, n_qubits, left)
    return complex(left.coefficient * right.coefficient * np.vdot(state, transformed))


def _density_covariance_expectation(
    density: ComplexArray,
    n_qubits: int,
    observable: PauliCovarianceObservable,
) -> float:
    left_operator = _term_operator(n_qubits, observable.left)
    right_operator = _term_operator(n_qubits, observable.right)
    symmetrized = 0.5 * np.trace(
        (left_operator @ right_operator + right_operator @ left_operator) @ density
    )
    left_mean = _density_term_expectation(density, n_qubits, observable.left)
    right_mean = _density_term_expectation(density, n_qubits, observable.right)
    return float(np.real_if_close(symmetrized).real - left_mean * right_mean)


def _term_operator(n_qubits: int, term: PauliTerm) -> ComplexArray:
    operator = np.eye(2**n_qubits, dtype=np.complex128)
    for qubit, label in term.factors:
        expanded = _expanded_operator(n_qubits, (qubit,), _PAULI_MATRICES[label])
        operator = expanded @ operator
    return cast(ComplexArray, operator.astype(np.complex128, copy=False))


def _apply_term_operator(state: ComplexArray, n_qubits: int, term: PauliTerm) -> ComplexArray:
    state_copy: Any = state.copy()
    transformed = cast(ComplexArray, state_copy)
    for qubit, label in term.factors:
        transformed = _apply_gate_matrix(transformed, n_qubits, (qubit,), _PAULI_MATRICES[label])
    return transformed


__all__ = [
    "execute_phase_qnode_circuit",
    "execute_phase_qnode_density_matrix",
]
