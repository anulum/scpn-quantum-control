# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Circuit Registry
"""Bounded statevector Phase-QNode circuit execution and support reports."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
OperationSpec: TypeAlias = tuple[object, ...]

_NON_PARAMETRIC_GATES = frozenset(("h", "x", "y", "z", "s", "t", "sx", "cnot", "cz", "cy", "swap"))
_PARAMETRIC_GATES = frozenset(
    ("rx", "ry", "rz", "phase", "crx", "cry", "crz", "rxx", "ryy", "rzz")
)
_REGISTERED_GATES = tuple(sorted(_NON_PARAMETRIC_GATES | _PARAMETRIC_GATES))
_REGISTERED_OBSERVABLES = (
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "weighted_pauli_sum",
    "pauli_product",
    "pauli_covariance",
    "dense_hermitian",
    "sparse_pauli_hamiltonian",
)

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
_S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
_T = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)
_SX = 0.5 * np.array(
    [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
    dtype=np.complex128,
)
_PAULI = Mapping[str, ComplexArray]
_PAULI_MATRICES: _PAULI = {"x": _X, "y": _Y, "z": _Z}


@dataclass(frozen=True)
class PhaseQNodeOperation:
    """One registered Phase-QNode gate application."""

    gate: str
    qubits: tuple[int, ...]
    parameter_index: int | None = None

    def __post_init__(self) -> None:
        gate = str(self.gate).strip().lower()
        if not gate:
            raise ValueError("gate must be non-empty")
        qubits = tuple(self.qubits)
        if not qubits:
            raise ValueError("operation qubits must be non-empty")
        if any(isinstance(qubit, bool) or qubit < 0 for qubit in qubits):
            raise ValueError("operation qubits must be non-negative integers")
        if len(set(qubits)) != len(qubits):
            raise ValueError("operation qubits must be unique")
        if self.parameter_index is not None and (
            isinstance(self.parameter_index, bool) or self.parameter_index < 0
        ):
            raise ValueError("parameter_index must be a non-negative integer or None")
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "qubits", qubits)


@dataclass(frozen=True)
class PauliTerm:
    """Weighted Pauli-product observable term."""

    coefficient: float
    factors: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        coefficient = _as_finite_scalar("coefficient", self.coefficient)
        factors: list[tuple[int, str]] = []
        seen: set[int] = set()
        if not self.factors:
            raise ValueError("PauliTerm factors must be non-empty")
        for qubit, label in self.factors:
            if isinstance(qubit, bool) or qubit < 0:
                raise ValueError("PauliTerm qubits must be non-negative integers")
            normalized = str(label).strip().lower().replace("pauli_", "")
            if normalized not in _PAULI_MATRICES:
                raise ValueError("PauliTerm labels must be x, y, or z")
            if qubit in seen:
                raise ValueError("PauliTerm cannot repeat a qubit")
            seen.add(qubit)
            factors.append((int(qubit), normalized))
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "factors", tuple(factors))

    @property
    def observable_kind(self) -> str:
        """Return the public observable family represented by this term."""
        return "pauli_product" if len(self.factors) > 1 else f"pauli_{self.factors[0][1]}"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready Pauli term metadata."""
        return {"coefficient": self.coefficient, "factors": [list(item) for item in self.factors]}


@dataclass(frozen=True)
class SparsePauliHamiltonian:
    """Sparse weighted Pauli Hamiltonian expectation observable."""

    terms: tuple[PauliTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("SparsePauliHamiltonian terms must be non-empty")
        object.__setattr__(self, "terms", tuple(self.terms))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready sparse Hamiltonian metadata."""
        return {"terms": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True)
class DenseHermitianObservable:
    """Dense finite-dimensional Hermitian observable."""

    matrix: ComplexArray
    label: str = "dense_hermitian"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("DenseHermitianObservable matrix must be square")
        if matrix.shape[0] == 0 or matrix.shape[0] & (matrix.shape[0] - 1):
            raise ValueError("DenseHermitianObservable dimension must be a positive power of two")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("DenseHermitianObservable matrix must contain finite values")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
            raise ValueError("DenseHermitianObservable matrix must be Hermitian")
        label = str(self.label).strip() or "dense_hermitian"
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "label", label)

    @property
    def observable_kind(self) -> str:
        """Return the public observable family represented by this matrix."""
        return "dense_hermitian"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dense observable metadata."""
        return {
            "label": self.label,
            "dimension": int(self.matrix.shape[0]),
            "matrix_real": self.matrix.real.tolist(),
            "matrix_imag": self.matrix.imag.tolist(),
        }


@dataclass(frozen=True)
class PauliCovarianceObservable:
    """Symmetrised covariance between two Pauli-product observables."""

    left: PauliTerm
    right: PauliTerm

    @property
    def observable_kind(self) -> str:
        """Return the public observable family represented by this covariance."""
        return "pauli_covariance"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready covariance metadata."""
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True)
class PhaseQNodeSupportReport:
    """Structured support report for local Phase-QNode execution."""

    supported: bool
    gates: tuple[str, ...]
    observable_kind: str
    differentiable_parameters: tuple[int, ...]
    unsupported_gates: tuple[str, ...]
    unsupported_observables: tuple[str, ...]
    unsupported_parameters: tuple[int, ...]
    failure_reason: str
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready support metadata."""
        return {
            "supported": self.supported,
            "gates": list(self.gates),
            "observable_kind": self.observable_kind,
            "differentiable_parameters": list(self.differentiable_parameters),
            "unsupported_gates": list(self.unsupported_gates),
            "unsupported_observables": list(self.unsupported_observables),
            "unsupported_parameters": list(self.unsupported_parameters),
            "failure_reason": self.failure_reason,
            "alternatives": list(self.alternatives),
        }


class PhaseQNodeSupportError(ValueError):
    """Raised when a Phase-QNode route is outside the registered local subset."""

    def __init__(self, report: PhaseQNodeSupportReport) -> None:
        super().__init__(report.failure_reason)
        self.report = report


@dataclass(frozen=True)
class PhaseQNodeCircuit:
    """Bounded statevector Phase-QNode circuit declaration."""

    n_qubits: int
    operations: tuple[PhaseQNodeOperation | OperationSpec, ...]
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    )

    def __post_init__(self) -> None:
        if isinstance(self.n_qubits, bool) or self.n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        parsed = tuple(_parse_operation(operation) for operation in self.operations)
        if not parsed:
            raise ValueError("operations must be non-empty")
        for operation in parsed:
            if any(qubit >= self.n_qubits for qubit in operation.qubits):
                raise ValueError("operation qubit exceeds n_qubits")
        observable = _normalise_observable(self.observable, self.n_qubits)
        object.__setattr__(self, "operations", parsed)
        object.__setattr__(self, "observable", observable)


@dataclass(frozen=True)
class PhaseQNodeExecutionResult:
    """Statevector execution result for a supported Phase-QNode circuit."""

    value: float
    state: ComplexArray
    support_report: PhaseQNodeSupportReport

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready execution evidence."""
        return {
            "value": self.value,
            "state_real": self.state.real.tolist(),
            "state_imag": self.state.imag.tolist(),
            "support_report": self.support_report.to_dict(),
        }


@dataclass(frozen=True)
class PhaseQNodeGradientResult:
    """Parameter-shift gradient result for a supported Phase-QNode circuit."""

    value: float
    gradient: FloatArray
    support_report: PhaseQNodeSupportReport
    parameter_shift_evaluations: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gradient evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "support_report": self.support_report.to_dict(),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
        }


@dataclass(frozen=True)
class PhaseQNodeMetricTensorResult:
    """Pure-state metric tensor evidence for a supported Phase-QNode circuit."""

    fubini_study_metric: FloatArray
    quantum_fisher_information: FloatArray
    derivative_norms: FloatArray
    support_report: PhaseQNodeSupportReport
    parameter_derivative_evaluations: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready metric tensor evidence."""
        return {
            "fubini_study_metric": self.fubini_study_metric.tolist(),
            "quantum_fisher_information": self.quantum_fisher_information.tolist(),
            "derivative_norms": self.derivative_norms.tolist(),
            "support_report": self.support_report.to_dict(),
            "parameter_derivative_evaluations": self.parameter_derivative_evaluations,
            "claim_boundary": self.claim_boundary,
        }


def registered_phase_qnode_gates() -> tuple[str, ...]:
    """Return the local Phase-QNode gate family."""
    return _REGISTERED_GATES


def registered_phase_qnode_observables() -> tuple[str, ...]:
    """Return the local Phase-QNode observable family."""
    return _REGISTERED_OBSERVABLES


def phase_qnode_support_report(
    circuit: PhaseQNodeCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeSupportReport:
    """Return a fail-closed support report for a circuit and parameter vector."""
    values = _as_parameter_vector(parameters)
    operations = _parsed_operations(circuit)
    gates = tuple(operation.gate for operation in operations)
    unsupported_gates = tuple(gate for gate in gates if gate not in _REGISTERED_GATES)
    unsupported_parameters = tuple(
        operation.parameter_index
        for operation in operations
        if operation.gate in _PARAMETRIC_GATES
        and operation.parameter_index is not None
        and operation.parameter_index >= values.size
    )
    missing_parameters = tuple(
        index
        for index, operation in enumerate(operations)
        if operation.gate in _PARAMETRIC_GATES and operation.parameter_index is None
    )
    unsupported_observables: tuple[str, ...] = ()
    observable_kind = _observable_kind(circuit.observable)
    if observable_kind not in _REGISTERED_OBSERVABLES:
        unsupported_observables = (observable_kind,)
    differentiable = tuple(
        sorted(
            {
                cast(int, operation.parameter_index)
                for operation in operations
                if operation.gate in _PARAMETRIC_GATES and operation.parameter_index is not None
            }
        )
    )
    reasons: list[str] = []
    if unsupported_gates:
        reasons.append(f"unsupported gates: {', '.join(unsupported_gates)}")
    if unsupported_observables:
        reasons.append(f"unsupported observables: {', '.join(unsupported_observables)}")
    if unsupported_parameters:
        reasons.append(f"parameter indices outside supplied vector: {unsupported_parameters}")
    if missing_parameters:
        reasons.append(
            f"parametric gates missing parameter indices at operations: {missing_parameters}"
        )
    return PhaseQNodeSupportReport(
        supported=not reasons,
        gates=gates,
        observable_kind=observable_kind,
        differentiable_parameters=differentiable,
        unsupported_gates=unsupported_gates,
        unsupported_observables=unsupported_observables,
        unsupported_parameters=tuple(int(item) for item in unsupported_parameters),
        failure_reason="; ".join(reasons),
        alternatives=(
            "use registered_phase_qnode_gates for the supported local gate family",
            "route provider or dynamic circuits through explicit provider-boundary records",
        ),
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


def parameter_shift_phase_qnode_gradient(
    circuit: PhaseQNodeCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeGradientResult:
    """Evaluate the analytic parameter-shift gradient for registered generators."""
    values = _as_parameter_vector(parameters)
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    gradient = np.zeros_like(values)
    base_result = execute_phase_qnode_circuit(circuit, values)
    for index in report.differentiable_parameters:
        plus = values.copy()
        minus = values.copy()
        plus[index] += np.pi / 2.0
        minus[index] -= np.pi / 2.0
        if isinstance(circuit.observable, PauliCovarianceObservable):
            plus_state = _execute_state(circuit, plus)
            minus_state = _execute_state(circuit, minus)
            gradient[index] = _covariance_product_rule_gradient(
                base_result.state,
                plus_state,
                minus_state,
                circuit.n_qubits,
                circuit.observable,
            )
        else:
            gradient[index] = 0.5 * (
                execute_phase_qnode_circuit(circuit, plus).value
                - execute_phase_qnode_circuit(circuit, minus).value
            )
    return PhaseQNodeGradientResult(
        value=base_result.value,
        gradient=gradient,
        support_report=report,
        parameter_shift_evaluations=2 * len(report.differentiable_parameters),
    )


def phase_qnode_quantum_fisher_information(
    circuit: PhaseQNodeCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeMetricTensorResult:
    """Compute the pure-state QFI and Fubini-Study metric for a local QNode."""
    values = _as_parameter_vector(parameters)
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    state, derivatives = _execute_state_and_parameter_derivatives(circuit, values)
    width = values.size
    metric = np.zeros((width, width), dtype=np.float64)
    overlaps = np.array([np.vdot(state, derivative) for derivative in derivatives])
    for row in range(width):
        for column in range(row, width):
            raw = (
                np.vdot(derivatives[row], derivatives[column])
                - np.conj(overlaps[row]) * overlaps[column]
            )
            value = float(np.real_if_close(raw).real)
            metric[row, column] = value
            metric[column, row] = value
    symmetrized_metric: FloatArray = np.asarray(0.5 * (metric + metric.T), dtype=np.float64)
    qfi: FloatArray = np.asarray(4.0 * symmetrized_metric, dtype=np.float64)
    derivative_norms = np.asarray(
        [np.linalg.norm(derivative) for derivative in derivatives],
        dtype=np.float64,
    )
    return PhaseQNodeMetricTensorResult(
        fubini_study_metric=symmetrized_metric,
        quantum_fisher_information=qfi,
        derivative_norms=derivative_norms,
        support_report=report,
        parameter_derivative_evaluations=len(_parsed_operations(circuit)),
        claim_boundary=(
            "pure-state local Phase-QNode Fubini-Study metric and QFI for the "
            "registered statevector gate family; no finite-shot classical Fisher, "
            "density-matrix, noisy-channel, provider, or hardware metric claim"
        ),
    )


def phase_qnode_natural_gradient_metric(
    circuit: PhaseQNodeCircuit,
) -> Callable[[FloatArray], FloatArray]:
    """Return a metric provider for quantum natural-gradient optimisation."""

    def metric(parameters: FloatArray) -> FloatArray:
        result = phase_qnode_quantum_fisher_information(circuit, parameters)
        return cast(FloatArray, result.fubini_study_metric.copy())

    return metric


def _parse_operation(operation: PhaseQNodeOperation | OperationSpec) -> PhaseQNodeOperation:
    if isinstance(operation, PhaseQNodeOperation):
        return operation
    if len(operation) not in {2, 3}:
        raise ValueError(
            "operation specs must be (gate, qubits) or (gate, qubits, parameter_index)"
        )
    gate = str(operation[0])
    qubits_raw = operation[1]
    if not isinstance(qubits_raw, Iterable):
        raise ValueError("operation qubits must be an iterable of integer qubits")
    qubits = tuple(int(qubit) for qubit in cast(Iterable[int], qubits_raw))
    parameter_index = None if len(operation) == 2 else int(cast(int, operation[2]))
    return PhaseQNodeOperation(gate=gate, qubits=qubits, parameter_index=parameter_index)


def _parsed_operations(circuit: PhaseQNodeCircuit) -> tuple[PhaseQNodeOperation, ...]:
    return cast(tuple[PhaseQNodeOperation, ...], circuit.operations)


def _normalise_observable(
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    ),
    n_qubits: int,
) -> (
    str | PauliTerm | SparsePauliHamiltonian | PauliCovarianceObservable | DenseHermitianObservable
):
    if isinstance(observable, str):
        normalized = observable.strip().lower()
        if normalized in {"x", "pauli_x"}:
            return PauliTerm(1.0, ((0, "x"),))
        if normalized in {"y", "pauli_y"}:
            return PauliTerm(1.0, ((0, "y"),))
        if normalized in {"z", "pauli_z"}:
            return PauliTerm(1.0, ((0, "z"),))
        return normalized
    if isinstance(observable, DenseHermitianObservable):
        expected_dimension = 2**n_qubits
        if observable.matrix.shape != (expected_dimension, expected_dimension):
            raise ValueError("DenseHermitianObservable dimension must match n_qubits")
        return observable
    max_qubit = -1
    terms: tuple[PauliTerm, ...]
    if isinstance(observable, PauliCovarianceObservable):
        terms = (observable.left, observable.right)
    else:
        terms = (
            observable.terms if isinstance(observable, SparsePauliHamiltonian) else (observable,)
        )
    for term in terms:
        max_qubit = max(max_qubit, *(qubit for qubit, _label in term.factors))
    if max_qubit >= n_qubits:
        raise ValueError("observable qubit exceeds n_qubits")
    return observable


def _observable_kind(
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    ),
) -> str:
    if isinstance(observable, DenseHermitianObservable):
        return observable.observable_kind
    if isinstance(observable, PauliCovarianceObservable):
        return observable.observable_kind
    if isinstance(observable, SparsePauliHamiltonian):
        return "sparse_pauli_hamiltonian"
    if isinstance(observable, PauliTerm):
        return observable.observable_kind
    return str(observable).strip().lower()


def _as_parameter_vector(parameters: ArrayLike) -> FloatArray:
    raw = np.asarray(parameters)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError("parameters must contain finite real numeric values")
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("parameters must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain finite real numeric values")
    return values.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


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


def _execute_state_and_parameter_derivatives(
    circuit: PhaseQNodeCircuit,
    values: FloatArray,
) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
    state = np.zeros(2**circuit.n_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    derivatives = tuple(np.zeros_like(state) for _ in range(values.size))
    for operation in _parsed_operations(circuit):
        matrix = _operation_matrix(operation, values)
        derivative_matrix = _operation_derivative_matrix(operation, values)
        previous_state = state
        state = _apply_gate_matrix(previous_state, circuit.n_qubits, operation.qubits, matrix)
        updated: list[ComplexArray] = []
        for index, derivative in enumerate(derivatives):
            propagated = _apply_gate_matrix(derivative, circuit.n_qubits, operation.qubits, matrix)
            if operation.parameter_index == index:
                propagated = propagated + _apply_gate_matrix(
                    previous_state,
                    circuit.n_qubits,
                    operation.qubits,
                    derivative_matrix,
                )
            updated.append(cast(ComplexArray, propagated.astype(np.complex128, copy=False)))
        derivatives = tuple(updated)
    return state, derivatives


def _operation_matrix(
    operation: PhaseQNodeOperation,
    parameters: FloatArray,
) -> ComplexArray:
    theta = 0.0
    if operation.gate in _PARAMETRIC_GATES:
        theta = float(parameters[cast(int, operation.parameter_index)])
    return _gate_matrix(operation.gate, theta)


def _operation_derivative_matrix(
    operation: PhaseQNodeOperation,
    parameters: FloatArray,
) -> ComplexArray:
    if operation.gate not in _PARAMETRIC_GATES:
        return np.zeros(
            (2 ** len(operation.qubits), 2 ** len(operation.qubits)), dtype=np.complex128
        )
    theta = float(parameters[cast(int, operation.parameter_index)])
    return _gate_derivative_matrix(operation.gate, theta)


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


def _gate_derivative_matrix(gate: str, theta: float) -> ComplexArray:
    if gate == "rx":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * _I - 0.5j * np.cos(theta / 2.0) * _X,
            dtype=np.complex128,
        )
    if gate == "ry":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * _I - 0.5j * np.cos(theta / 2.0) * _Y,
            dtype=np.complex128,
        )
    if gate == "rz":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * _I - 0.5j * np.cos(theta / 2.0) * _Z,
            dtype=np.complex128,
        )
    if gate == "phase":
        return np.array(
            [[0.0, 0.0], [0.0, 1.0j * np.exp(1.0j * theta)]],
            dtype=np.complex128,
        )
    if gate == "crx":
        return _controlled(_gate_derivative_matrix("rx", theta))
    if gate == "cry":
        return _controlled(_gate_derivative_matrix("ry", theta))
    if gate == "crz":
        return _controlled(_gate_derivative_matrix("rz", theta))
    if gate == "rxx":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 0.5j * np.cos(theta / 2.0) * np.kron(_X, _X),
            dtype=np.complex128,
        )
    if gate == "ryy":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 0.5j * np.cos(theta / 2.0) * np.kron(_Y, _Y),
            dtype=np.complex128,
        )
    if gate == "rzz":
        return np.asarray(
            -0.5 * np.sin(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 0.5j * np.cos(theta / 2.0) * np.kron(_Z, _Z),
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported gate derivative matrix: {gate}")


def _controlled(target: ComplexArray) -> ComplexArray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix[2:4, 2:4] = target
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


def _covariance_product_rule_gradient(
    base_state: ComplexArray,
    plus_state: ComplexArray,
    minus_state: ComplexArray,
    n_qubits: int,
    observable: PauliCovarianceObservable,
) -> float:
    base_left = _term_expectation(base_state, n_qubits, observable.left)
    base_right = _term_expectation(base_state, n_qubits, observable.right)
    left_grad = 0.5 * (
        _term_expectation(plus_state, n_qubits, observable.left)
        - _term_expectation(minus_state, n_qubits, observable.left)
    )
    right_grad = 0.5 * (
        _term_expectation(plus_state, n_qubits, observable.right)
        - _term_expectation(minus_state, n_qubits, observable.right)
    )
    sym_grad = 0.5 * (
        _symmetrized_product_expectation(
            plus_state,
            n_qubits,
            observable.left,
            observable.right,
        )
        - _symmetrized_product_expectation(
            minus_state,
            n_qubits,
            observable.left,
            observable.right,
        )
    )
    return float(sym_grad - left_grad * base_right - base_left * right_grad)


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


def _apply_term_operator(state: ComplexArray, n_qubits: int, term: PauliTerm) -> ComplexArray:
    transformed = cast(ComplexArray, state.copy())
    for qubit, label in term.factors:
        transformed = _apply_gate_matrix(transformed, n_qubits, (qubit,), _PAULI_MATRICES[label])
    return transformed


__all__ = [
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PauliTerm",
    "PhaseQNodeCircuit",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeGradientResult",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeOperation",
    "PhaseQNodeSupportError",
    "PhaseQNodeSupportReport",
    "SparsePauliHamiltonian",
    "execute_phase_qnode_circuit",
    "parameter_shift_phase_qnode_gradient",
    "phase_qnode_natural_gradient_metric",
    "phase_qnode_quantum_fisher_information",
    "phase_qnode_support_report",
    "registered_phase_qnode_gates",
    "registered_phase_qnode_observables",
]
