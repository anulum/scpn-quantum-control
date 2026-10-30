# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Phase-QNode statevector evaluation
"""Evaluate a registered Phase-QNode circuit on a supplied native runtime.

One algorithm evolves the computational zero state through the circuit's gate
operators and measures its observable. Every native operation is a supplied
kernel, so each runtime keeps its own tensor types, constant creation order and
differentiation graph. Route kernels let a caller keep its own named entry
points for every recursive step. Circuit admission, support reports, loaders
and model logic remain with their owners.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from .phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    SparsePauliHamiltonian,
)
from .phase_qnode_native_expectations import (
    covariance_from_moments,
    real_quadratic_form,
    real_symmetrized_overlap,
    weighted_complex_overlap,
    weighted_real_overlap,
)
from .phase_qnode_state_evolution import (
    MatrixApplication,
    NativeLocalOperator,
    evolve_local_operators,
)


@dataclass(frozen=True)
class StatevectorKernels:
    """Native operations and route entry points used by the statevector engine.

    Parameters
    ----------
    namespace
        Native array namespace passed to every numerical leaf.
    parameter_tensor
        Convert admitted parameters to the native parameter tensor.
    zero_state
        Allocate the zero state; the qubit count is read through the callable.
    circuit_operations
        Read the circuit's admitted operation sequence; it is called once,
        after the zero state is allocated.
    operation_theta
        Return the native angle of one operation from the parameter tensor.
    gate_operator
        Build the native operator for a gate name and angle.
    apply_operator
        Apply a native matrix to ordered qubit dimensions of a state.
    pauli_operator
        Build the native single-qubit Pauli operator for a label.
    dense_operator
        Convert dense observable entries to a native complex matrix.
    real_zero
        Return the native real zero that starts a Hamiltonian sum.
    other_observable
        Evaluate, or refuse, an observable outside the registered family.
    dense_class, covariance_class, hamiltonian_class, term_class
        Resolvers of the observable classes, read in dispatch order.
    observable_expectation, term_expectation, symmetrized_product_expectation
        Route entry points for the observable and Pauli-term moments.
    term_product_expectation, apply_term_operator
        Route entry points for ordered Pauli products and term application.

    """

    namespace: Any
    parameter_tensor: Callable[[object], Any]
    zero_state: Callable[[Any, Callable[[], int]], Any]
    circuit_operations: Callable[[PhaseQNodeCircuit], tuple[PhaseQNodeOperation, ...]]
    operation_theta: Callable[[PhaseQNodeOperation, Any], Any]
    gate_operator: Callable[[str, Any], Any]
    apply_operator: MatrixApplication
    pauli_operator: Callable[[str], Any]
    dense_operator: Callable[[object], Any]
    real_zero: Callable[[], Any]
    other_observable: Callable[[Any, int, object], Any]
    dense_class: Callable[[], type]
    covariance_class: Callable[[], type]
    hamiltonian_class: Callable[[], type]
    term_class: Callable[[], type]
    observable_expectation: Callable[[Any, int, object, Any], Any]
    term_expectation: Callable[[Any, int, PauliTerm], Any]
    symmetrized_product_expectation: Callable[[Any, int, PauliTerm, PauliTerm], Any]
    term_product_expectation: Callable[[Any, int, PauliTerm, PauliTerm], Any]
    apply_term_operator: Callable[[Any, int, PauliTerm], Any]


def registered_operations(circuit: PhaseQNodeCircuit) -> tuple[PhaseQNodeOperation, ...]:
    """Return the admitted operation sequence of a registered circuit.

    Parameters
    ----------
    circuit
        Admitted registered Phase-QNode circuit.

    Returns
    -------
    tuple[PhaseQNodeOperation, ...]
        The circuit's parsed operations, in order.

    """
    return cast(tuple[PhaseQNodeOperation, ...], circuit.operations)


def native_statevector_kernels(
    *,
    namespace: Any,
    parameter_tensor: Callable[[object], Any],
    zero_state: Callable[[Any, Callable[[], int]], Any],
    operation_theta: Callable[[PhaseQNodeOperation, Any], Any],
    gate_operator: Callable[[str, Any], Any],
    apply_operator: MatrixApplication,
    pauli_operator: Callable[[str], Any],
    dense_operator: Callable[[object], Any],
    real_zero: Callable[[], Any],
    other_observable: Callable[[Any, int, object], Any],
) -> StatevectorKernels:
    """Return kernels whose route entry points are this module's own algorithm.

    Parameters
    ----------
    namespace, parameter_tensor, zero_state, operation_theta, gate_operator
        Native operations, as documented on :class:`StatevectorKernels`.
    apply_operator, pauli_operator, dense_operator, real_zero, other_observable
        Native operations, as documented on :class:`StatevectorKernels`.

    Returns
    -------
    StatevectorKernels
        Kernels for a runtime that has no separate named route entry points;
        operations are read with :func:`registered_operations`.

    """
    kernels: StatevectorKernels
    kernels = StatevectorKernels(
        namespace=namespace,
        parameter_tensor=parameter_tensor,
        zero_state=zero_state,
        circuit_operations=registered_operations,
        operation_theta=operation_theta,
        gate_operator=gate_operator,
        apply_operator=apply_operator,
        pauli_operator=pauli_operator,
        dense_operator=dense_operator,
        real_zero=real_zero,
        other_observable=other_observable,
        dense_class=lambda: DenseHermitianObservable,
        covariance_class=lambda: PauliCovarianceObservable,
        hamiltonian_class=lambda: SparsePauliHamiltonian,
        term_class=lambda: PauliTerm,
        observable_expectation=lambda state, n_qubits, observable, dense: observable_expectation(
            kernels, state, n_qubits, observable, dense
        ),
        term_expectation=lambda state, n_qubits, term: term_expectation(
            kernels, state, n_qubits, term
        ),
        symmetrized_product_expectation=lambda state, n_qubits, left, right: (
            symmetrized_product_expectation(kernels, state, n_qubits, left, right)
        ),
        term_product_expectation=lambda state, n_qubits, left, right: term_product_expectation(
            kernels, state, n_qubits, left, right
        ),
        apply_term_operator=lambda state, n_qubits, term: apply_term_operator(
            kernels, state, n_qubits, term
        ),
    )
    return kernels


def statevector_value_and_state(
    kernels: StatevectorKernels,
    circuit: PhaseQNodeCircuit,
    params: object,
    dense_matrix: Any = None,
) -> tuple[Any, Any]:
    """Evolve the zero state through a circuit and measure its observable.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    circuit
        Admitted registered Phase-QNode circuit.
    params
        Admitted parameter values, converted by ``kernels.parameter_tensor``.
    dense_matrix
        Optional pre-converted native matrix of a dense observable.

    Returns
    -------
    tuple[Any, Any]
        Native real expectation value and final native statevector.

    """
    parameter_tensor = kernels.parameter_tensor(params)
    # Defer circuit operand reads until after the native basis-state allocation.
    operators = (
        NativeLocalOperator(
            kernels.apply_operator,
            kernels.namespace,
            circuit.n_qubits,
            operation.qubits,
            matrix,
        )
        for _ in (None,)
        for operation in kernels.circuit_operations(circuit)
        for matrix in (
            kernels.gate_operator(
                operation.gate, kernels.operation_theta(operation, parameter_tensor)
            ),
        )
    )
    state, _matrix = evolve_local_operators(
        kernels.zero_state(kernels.namespace, lambda: circuit.n_qubits), operators
    )
    return kernels.observable_expectation(
        state, circuit.n_qubits, circuit.observable, dense_matrix
    ), state


def observable_expectation(
    kernels: StatevectorKernels,
    state: Any,
    n_qubits: int,
    observable: object,
    dense_matrix: Any = None,
) -> Any:
    """Return the real expectation of a registered observable in a native state.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    state
        Native statevector.
    n_qubits
        Admitted number of qubits of the state.
    observable
        Dense, covariance, sparse-Hamiltonian or Pauli-term observable.
    dense_matrix
        Optional pre-converted native matrix of a dense observable.

    Returns
    -------
    Any
        Native real expectation value.

    """
    if isinstance(observable, kernels.dense_class()):
        matrix = (
            kernels.dense_operator(cast(DenseHermitianObservable, observable).matrix)
            if dense_matrix is None
            else dense_matrix
        )
        return real_quadratic_form(kernels.namespace, state, matrix)
    if isinstance(observable, kernels.covariance_class()):
        symmetrized = kernels.symmetrized_product_expectation(
            state,
            n_qubits,
            cast(PauliCovarianceObservable, observable).left,
            cast(PauliCovarianceObservable, observable).right,
        )
        left_mean = kernels.term_expectation(
            state, n_qubits, cast(PauliCovarianceObservable, observable).left
        )
        right_mean = kernels.term_expectation(
            state, n_qubits, cast(PauliCovarianceObservable, observable).right
        )
        return covariance_from_moments(symmetrized, left_mean, right_mean)
    if isinstance(observable, kernels.hamiltonian_class()):
        total = kernels.real_zero()
        for term in cast(SparsePauliHamiltonian, observable).terms:
            total = total + kernels.term_expectation(state, n_qubits, term)
        return total
    if isinstance(observable, kernels.term_class()):
        return kernels.term_expectation(state, n_qubits, cast(PauliTerm, observable))
    return kernels.other_observable(state, n_qubits, observable)


def term_expectation(
    kernels: StatevectorKernels, state: Any, n_qubits: int, term: PauliTerm
) -> Any:
    """Return the weighted real expectation of one Pauli term.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    state
        Native statevector.
    n_qubits
        Admitted number of qubits of the state.
    term
        Pauli term whose coefficient is read after its operator is applied.

    Returns
    -------
    Any
        Native real expectation value.

    """
    transformed = kernels.apply_term_operator(state, n_qubits, term)
    return weighted_real_overlap(kernels.namespace, state, transformed, term.coefficient)


def symmetrized_product_expectation(
    kernels: StatevectorKernels,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    """Return the real symmetrised expectation of two Pauli-term products.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    state
        Native statevector.
    n_qubits
        Admitted number of qubits of the state.
    left, right
        Pauli terms; both orders of their product are evaluated.

    Returns
    -------
    Any
        Native real symmetrised moment.

    """
    left_right = kernels.term_product_expectation(state, n_qubits, left, right)
    right_left = kernels.term_product_expectation(state, n_qubits, right, left)
    return real_symmetrized_overlap(kernels.namespace, left_right, right_left)


def term_product_expectation(
    kernels: StatevectorKernels,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    """Return the weighted complex expectation of ``left`` times ``right``.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    state
        Native statevector.
    n_qubits
        Admitted number of qubits of the state.
    left, right
        Pauli terms; ``right`` is applied first.

    Returns
    -------
    Any
        Native complex overlap weighted by both coefficients.

    """
    transformed = kernels.apply_term_operator(state, n_qubits, right)
    transformed = kernels.apply_term_operator(transformed, n_qubits, left)
    return weighted_complex_overlap(
        kernels.namespace, state, transformed, left.coefficient, right.coefficient
    )


def apply_term_operator(
    kernels: StatevectorKernels, state: Any, n_qubits: int, term: PauliTerm
) -> Any:
    """Apply the unweighted Pauli operators of one term to a native state.

    Parameters
    ----------
    kernels
        Native operations and route entry points of one runtime.
    state
        Native statevector.
    n_qubits
        Admitted number of qubits of the state.
    term
        Pauli term whose factors are applied in order, one qubit at a time.

    Returns
    -------
    Any
        Transformed native statevector.

    """
    transformed, _matrix = evolve_local_operators(
        state,
        (
            NativeLocalOperator(
                kernels.apply_operator,
                kernels.namespace,
                n_qubits,
                (qubit,),
                kernels.pauli_operator(label),
            )
            for qubit, label in term.factors
        ),
    )
    return transformed


def refuse_unregistered_observable(message: str) -> Callable[[Any, int, object], NoReturn]:
    """Return an ``other_observable`` kernel raising ``ValueError``.

    Parameters
    ----------
    message
        Refusal prefix; the observable follows it after ``": "``.

    Returns
    -------
    Callable[[Any, int, object], NoReturn]
        Kernel that raises ``ValueError(f"{message}: {observable}")``.

    """

    def refuse(state: Any, n_qubits: int, observable: object) -> NoReturn:
        raise ValueError(f"{message}: {observable}")

    return refuse


__all__ = [
    "StatevectorKernels",
    "apply_term_operator",
    "native_statevector_kernels",
    "observable_expectation",
    "refuse_unregistered_observable",
    "registered_operations",
    "statevector_value_and_state",
    "symmetrized_product_expectation",
    "term_expectation",
    "term_product_expectation",
]
