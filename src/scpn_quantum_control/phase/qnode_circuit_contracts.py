# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Contracts
"""Phase-QNode declarations, result records, registries, and validation.

This NumPy/stdlib-only leaf owns the immutable circuit vocabulary shared by
builders, support analysis, execution, gradients, measurements, and framework
bridges. It has no dependency on the executable QNode facade.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
OperationSpec: TypeAlias = tuple[object, ...]
DensityOperationSpec: TypeAlias = tuple[object, ...]
DensityOperation: TypeAlias = "PhaseQNodeOperation | PhaseQNodeNoiseChannel"

_NON_PARAMETRIC_GATES = frozenset(
    (
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "sx",
        "cnot",
        "cz",
        "cy",
        "swap",
        "ch",
        "cs",
        "ct",
        "ccnot",
        "ccz",
        "cswap",
    )
)
_PARAMETRIC_GATES = frozenset(
    ("rx", "ry", "rz", "phase", "crx", "cry", "crz", "rxx", "ryy", "rzz")
)
_REGISTERED_GATES = tuple(sorted(_NON_PARAMETRIC_GATES | _PARAMETRIC_GATES))
_GATE_ARITY = {
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "s": 1,
    "t": 1,
    "sx": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "phase": 1,
    "cnot": 2,
    "cz": 2,
    "cy": 2,
    "swap": 2,
    "ch": 2,
    "cs": 2,
    "ct": 2,
    "crx": 2,
    "cry": 2,
    "crz": 2,
    "rxx": 2,
    "ryy": 2,
    "rzz": 2,
    "ccnot": 3,
    "ccz": 3,
    "cswap": 3,
}
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
_REGISTERED_TEMPLATES = (
    "ghz_chain",
    "hardware_efficient_ry",
    "hardware_efficient_ryrz",
)
_REGISTERED_DECOMPOSITIONS = ("ccnot", "cswap")
_REGISTERED_NOISE_CHANNELS = (
    "amplitude_damping",
    "bit_flip",
    "depolarizing",
    "phase_flip",
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
class PhaseQNodeNoiseChannel:
    """One registered local density-matrix noise channel."""

    channel: str
    qubits: tuple[int, ...]
    probability: float

    def __post_init__(self) -> None:
        channel = str(self.channel).strip().lower()
        if not channel:
            raise ValueError("noise channel must be non-empty")
        qubits = tuple(self.qubits)
        if not qubits:
            raise ValueError("noise channel qubits must be non-empty")
        if any(isinstance(qubit, bool) or qubit < 0 for qubit in qubits):
            raise ValueError("noise channel qubits must be non-negative integers")
        if len(set(qubits)) != len(qubits):
            raise ValueError("noise channel qubits must be unique")
        probability = _as_probability(self.probability)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "probability", probability)


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
class PhaseQNodeDensityCircuit:
    """Bounded density-matrix Phase-QNode declaration with local noise."""

    n_qubits: int
    operations: tuple[DensityOperation | DensityOperationSpec, ...]
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
        parsed = tuple(_parse_density_operation(operation) for operation in self.operations)
        if not parsed:
            raise ValueError("operations must be non-empty")
        for operation in parsed:
            if any(qubit >= self.n_qubits for qubit in operation.qubits):
                raise ValueError("operation qubit exceeds n_qubits")
        observable = _normalise_observable(self.observable, self.n_qubits)
        object.__setattr__(self, "operations", parsed)
        object.__setattr__(self, "observable", observable)


@dataclass(frozen=True)
class PhaseQNodeTemplateSpec:
    """Registered multi-qubit Phase-QNode template declaration."""

    name: str
    n_qubits: int
    n_layers: int
    entangler: str
    parameter_count: int
    operations: tuple[PhaseQNodeOperation, ...]
    observable: (
        PauliTerm | SparsePauliHamiltonian | PauliCovarianceObservable | DenseHermitianObservable
    )
    claim_boundary: str

    def circuit(self) -> PhaseQNodeCircuit:
        """Return the executable circuit represented by this template."""
        return PhaseQNodeCircuit(
            n_qubits=self.n_qubits,
            operations=self.operations,
            observable=self.observable,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready template metadata."""
        observable: object
        if hasattr(self.observable, "to_dict"):
            observable = self.observable.to_dict()
        else:
            observable = str(self.observable)
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "entangler": self.entangler,
            "parameter_count": self.parameter_count,
            "operations": [
                {
                    "gate": operation.gate,
                    "qubits": list(operation.qubits),
                    "parameter_index": operation.parameter_index,
                }
                for operation in self.operations
            ],
            "observable": observable,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseQNodeDepthProfile:
    """Depth and resource profile for a registered local Phase-QNode circuit."""

    n_qubits: int
    operation_count: int
    depth: int
    operation_layers: tuple[int, ...]
    gate_counts: Mapping[str, int]
    parameter_count: int
    differentiable_parameters: tuple[int, ...]
    two_qubit_gate_count: int
    entangling_pairs: tuple[tuple[int, int], ...]
    max_operation_arity: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready circuit-depth metadata."""
        return {
            "n_qubits": self.n_qubits,
            "operation_count": self.operation_count,
            "depth": self.depth,
            "operation_layers": list(self.operation_layers),
            "gate_counts": dict(self.gate_counts),
            "parameter_count": self.parameter_count,
            "differentiable_parameters": list(self.differentiable_parameters),
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "entangling_pairs": [list(pair) for pair in self.entangling_pairs],
            "max_operation_arity": self.max_operation_arity,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseQNodeRegisteredCircuitSpec:
    """Validated arbitrary-depth registered Phase-QNode circuit declaration."""

    circuit: PhaseQNodeCircuit
    depth_profile: PhaseQNodeDepthProfile
    support_report: PhaseQNodeSupportReport
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered-depth circuit metadata."""
        return {
            "n_qubits": self.circuit.n_qubits,
            "depth_profile": self.depth_profile.to_dict(),
            "support_report": self.support_report.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


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
class PhaseQNodeDensityExecutionResult:
    """Density-matrix execution result for a supported local Phase-QNode."""

    value: float
    density_matrix: ComplexArray
    trace: float
    purity: float
    support_report: PhaseQNodeSupportReport
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready density-matrix execution evidence."""
        return {
            "value": self.value,
            "density_matrix_real": self.density_matrix.real.tolist(),
            "density_matrix_imag": self.density_matrix.imag.tolist(),
            "trace": self.trace,
            "purity": self.purity,
            "support_report": self.support_report.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseQNodeGradientResult:
    """Parameter-shift gradient result for a supported Phase-QNode circuit."""

    value: float
    gradient: FloatArray
    support_report: PhaseQNodeSupportReport
    parameter_shift_evaluations: int
    evaluation_plan: PhaseQNodeGradientEvaluationPlan | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gradient evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "support_report": self.support_report.to_dict(),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "evaluation_plan": None
            if self.evaluation_plan is None
            else self.evaluation_plan.to_dict(),
        }


@dataclass(frozen=True)
class PhaseQNodeGradientEvaluationGroup:
    """One logical-parameter shift group for a registered Phase-QNode circuit."""

    parameter_index: int
    operation_indices: tuple[int, ...]
    gates: tuple[str, ...]
    qubits: tuple[tuple[int, ...], ...]
    generator_keys: tuple[str, ...]
    frequency_gaps: tuple[float, ...]
    commuting: bool
    shifted_evaluations: int
    naive_operation_shifted_evaluations: int
    saved_shifted_evaluations: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready logical-parameter shift metadata."""
        return {
            "parameter_index": self.parameter_index,
            "operation_indices": list(self.operation_indices),
            "gates": list(self.gates),
            "qubits": [list(item) for item in self.qubits],
            "generator_keys": list(self.generator_keys),
            "frequency_gaps": list(self.frequency_gaps),
            "commuting": self.commuting,
            "shifted_evaluations": self.shifted_evaluations,
            "naive_operation_shifted_evaluations": self.naive_operation_shifted_evaluations,
            "saved_shifted_evaluations": self.saved_shifted_evaluations,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PhaseQNodeGradientEvaluationPlan:
    """Evaluation-count plan for registered Phase-QNode parameter-shift gradients."""

    supported: bool
    method: str
    parameter_count: int
    differentiable_parameters: tuple[int, ...]
    operation_level_naive_evaluations: int
    planned_shifted_evaluations: int
    saved_shifted_evaluations: int
    groups: tuple[PhaseQNodeGradientEvaluationGroup, ...]
    fallback_reason: str
    claim_boundary: str

    @property
    def parameter_shift_evaluations(self) -> int:
        """Return shifted evaluations required by the plan."""
        return self.planned_shifted_evaluations

    @property
    def generic_scalar_objective_evaluations(self) -> int:
        """Return the matching generic-callable fallback count."""
        return 2 * self.parameter_count

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gate-aware evaluation planning metadata."""
        return {
            "supported": self.supported,
            "method": self.method,
            "parameter_count": self.parameter_count,
            "differentiable_parameters": list(self.differentiable_parameters),
            "operation_level_naive_evaluations": self.operation_level_naive_evaluations,
            "planned_shifted_evaluations": self.planned_shifted_evaluations,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "saved_shifted_evaluations": self.saved_shifted_evaluations,
            "generic_scalar_objective_evaluations": self.generic_scalar_objective_evaluations,
            "groups": [group.to_dict() for group in self.groups],
            "fallback_reason": self.fallback_reason,
            "claim_boundary": self.claim_boundary,
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


@dataclass(frozen=True)
class PhaseQNodeClassicalFisherResult:
    """Computational-basis Fisher evidence for a supported Phase-QNode.

    ``classical_fisher_information`` is always the exact statevector reference.
    When ``shot_count`` or ``observed_counts`` is supplied, the optional
    finite-shot fields carry a multinomial plug-in estimate, delta-method
    standard errors, and confidence radii for the same computational-basis
    measurement route.
    """

    classical_fisher_information: FloatArray
    probabilities: FloatArray
    probability_derivatives: FloatArray
    measurement: str
    min_probability: float
    support_report: PhaseQNodeSupportReport
    claim_boundary: str
    shot_count: int | None = None
    count_record: tuple[int, ...] | None = None
    empirical_probabilities: FloatArray | None = None
    finite_shot_classical_fisher_information: FloatArray | None = None
    fisher_standard_error: FloatArray | None = None
    fisher_confidence_radius: FloatArray | None = None
    confidence_level: float | None = None
    confidence_z: float | None = None
    sampling_model: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready classical Fisher evidence."""
        return {
            "classical_fisher_information": self.classical_fisher_information.tolist(),
            "probabilities": self.probabilities.tolist(),
            "probability_derivatives": self.probability_derivatives.tolist(),
            "measurement": self.measurement,
            "min_probability": self.min_probability,
            "support_report": self.support_report.to_dict(),
            "claim_boundary": self.claim_boundary,
            "shot_count": self.shot_count,
            "count_record": None if self.count_record is None else list(self.count_record),
            "empirical_probabilities": _optional_float_array_to_list(self.empirical_probabilities),
            "finite_shot_classical_fisher_information": _optional_float_array_to_list(
                self.finite_shot_classical_fisher_information
            ),
            "fisher_standard_error": _optional_float_array_to_list(self.fisher_standard_error),
            "fisher_confidence_radius": _optional_float_array_to_list(
                self.fisher_confidence_radius
            ),
            "confidence_level": self.confidence_level,
            "confidence_z": self.confidence_z,
            "sampling_model": self.sampling_model,
        }


@dataclass(frozen=True)
class _FiniteShotFisherEvidence:
    """Optional finite-shot Fisher evidence attached to an exact Fisher result."""

    shot_count: int | None
    count_record: tuple[int, ...] | None
    empirical_probabilities: FloatArray | None
    finite_shot_classical_fisher_information: FloatArray | None
    fisher_standard_error: FloatArray | None
    fisher_confidence_radius: FloatArray | None
    confidence_level: float | None
    confidence_z: float | None
    sampling_model: str | None


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


def _parse_density_operation(
    operation: DensityOperation | DensityOperationSpec,
) -> DensityOperation:
    if isinstance(operation, PhaseQNodeOperation | PhaseQNodeNoiseChannel):
        return operation
    if len(operation) not in {2, 3}:
        raise ValueError(
            "density operation specs must be (gate, qubits), "
            "(gate, qubits, parameter_index), or (noise_channel, qubits, probability)"
        )
    name = str(operation[0]).strip().lower()
    qubits_raw = operation[1]
    if not isinstance(qubits_raw, Iterable):
        raise ValueError("density operation qubits must be an iterable of integer qubits")
    qubits = tuple(int(qubit) for qubit in cast(Iterable[int], qubits_raw))
    if name in _REGISTERED_NOISE_CHANNELS:
        if len(operation) != 3:
            raise ValueError("noise channel specs must include a probability")
        return PhaseQNodeNoiseChannel(name, qubits, float(cast(float, operation[2])))
    return _parse_operation(operation)


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


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _as_probability(value: object) -> float:
    probability = _as_finite_scalar("probability", value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be between 0 and 1")
    return probability


def _optional_float_array_to_list(value: FloatArray | None) -> object:
    """Return a JSON-ready list for an optional float array."""
    if value is None:
        return None
    return value.tolist()


__all__ = [
    "PhaseQNodeOperation",
    "PhaseQNodeNoiseChannel",
    "PauliTerm",
    "SparsePauliHamiltonian",
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PhaseQNodeSupportReport",
    "PhaseQNodeSupportError",
    "PhaseQNodeCircuit",
    "PhaseQNodeDensityCircuit",
    "PhaseQNodeTemplateSpec",
    "PhaseQNodeDepthProfile",
    "PhaseQNodeRegisteredCircuitSpec",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeDensityExecutionResult",
    "PhaseQNodeGradientResult",
    "PhaseQNodeGradientEvaluationGroup",
    "PhaseQNodeGradientEvaluationPlan",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeClassicalFisherResult",
]
