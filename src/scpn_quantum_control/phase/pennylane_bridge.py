# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PennyLane Bridge
"""Optional PennyLane agreement checks for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import Parameter, ParameterShiftRule, value_and_parameter_shift_grad
from .qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
    plan_phase_qnode_parameter_shift_evaluations,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class PennyLaneGradientAgreementResult:
    """Agreement report between SCPN and PennyLane-style gradient callables."""

    value: float
    scpn_gradient: FloatArray
    pennylane_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable agreement metadata."""
        return {
            "value": self.value,
            "scpn_gradient": self.scpn_gradient.copy(),
            "pennylane_gradient": self.pennylane_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PennyLaneRoundTripResult:
    """Round-trip value and gradient agreement report for PennyLane QNode adapters."""

    scpn_value: float
    pennylane_value: float
    value_abs_error: float
    scpn_gradient: FloatArray
    pennylane_gradient: FloatArray
    gradient_max_abs_error: float
    gradient_l2_error: float
    value_tolerance: float
    gradient_tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable round-trip metadata."""
        return {
            "scpn_value": self.scpn_value,
            "pennylane_value": self.pennylane_value,
            "value_abs_error": self.value_abs_error,
            "scpn_gradient": self.scpn_gradient.tolist(),
            "pennylane_gradient": self.pennylane_gradient.tolist(),
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "gradient_l2_error": self.gradient_l2_error,
            "value_tolerance": self.value_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PennyLaneQNodeConversionResult:
    """Bounded conversion metadata for a registered Phase-QNode PennyLane QNode."""

    qnode: Callable[[ArrayLike], object]
    gradient: Callable[[ArrayLike], object]
    device_name: str
    n_qubits: int
    shots: int | None
    interface: str
    diff_method: str
    gates: tuple[str, ...]
    observable_kind: str
    differentiable_parameters: tuple[int, ...]
    hardware_execution: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready conversion metadata without raw callables."""
        return {
            "device_name": self.device_name,
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "interface": self.interface,
            "diff_method": self.diff_method,
            "gates": list(self.gates),
            "observable_kind": self.observable_kind,
            "differentiable_parameters": list(self.differentiable_parameters),
            "hardware_execution": self.hardware_execution,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PennyLaneMaturityAuditResult:
    """Aggregate PennyLane parity evidence and explicit provider-plugin blockers."""

    identical_circuit_ready: bool
    ready_for_provider_exceedance: bool
    evidence: dict[str, object]
    required_capabilities: dict[str, str]
    promotion_metadata: dict[str, object]
    open_gaps: tuple[str, ...]
    claim_boundary: str = "bounded_pennylane_provider_maturity_audit"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PennyLane maturity evidence."""
        return {
            "identical_circuit_ready": self.identical_circuit_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "evidence": {name: _result_to_dict(result) for name, result in self.evidence.items()},
            "required_capabilities": dict(self.required_capabilities),
            "promotion_metadata": dict(self.promotion_metadata),
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


def _load_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is unavailable; install scpn-quantum-control[pennylane]"
        ) from exc
    return qml


def is_phase_pennylane_available() -> bool:
    """Return whether the optional phase PennyLane bridge can import PennyLane."""
    try:
        _load_pennylane()
    except ImportError:
        return False
    return True


def _as_parameter_vector(name: str, values: object, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def check_pennylane_parameter_shift_agreement(
    objective: ScalarObjective,
    pennylane_gradient: GradientCallable,
    values: ArrayLike,
    *,
    tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PennyLaneGradientAgreementResult:
    """Compare SCPN parameter-shift gradients with a PennyLane gradient callable.

    ``pennylane_gradient`` is intentionally caller-supplied. It can be
    ``qml.grad(qnode)`` or any strict PennyLane-derived gradient function with
    the same one-dimensional parameter vector. This keeps the bridge honest:
    it verifies cross-framework agreement without claiming automatic QNode
    generation for every SCPN ansatz.
    """
    _load_pennylane()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_gradient = _as_parameter_vector(
        "PennyLane gradient",
        pennylane_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PennyLaneGradientAgreementResult(
        value=float(scpn.value),
        scpn_gradient=scpn.gradient.copy(),
        pennylane_gradient=external_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
    )


def check_pennylane_qnode_round_trip(
    scpn_objective: ScalarObjective,
    pennylane_objective: ScalarObjective,
    pennylane_gradient: GradientCallable,
    values: ArrayLike,
    *,
    value_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PennyLaneRoundTripResult:
    """Verify SCPN value/gradient parity against a caller-supplied PennyLane QNode.

    The PennyLane callables are supplied by the caller so the bridge stays
    explicit about what is actually compared. In a real PennyLane workflow,
    ``pennylane_objective`` should be a QNode and ``pennylane_gradient`` should
    come from PennyLane autodiff, for example ``qml.grad(qnode)``.
    """
    _load_pennylane()
    value_tol = _as_non_negative_tolerance(value_tolerance)
    gradient_tol = _as_non_negative_tolerance(gradient_tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        scpn_objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_value = _as_finite_scalar(
        "PennyLane objective",
        pennylane_objective(parameter_values.copy()),
    )
    external_gradient = _as_parameter_vector(
        "PennyLane gradient",
        pennylane_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    value_abs_error = abs(float(scpn.value) - external_value)
    gradient_max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    gradient_l2_error = float(np.linalg.norm(delta, ord=2))

    return PennyLaneRoundTripResult(
        scpn_value=float(scpn.value),
        pennylane_value=external_value,
        value_abs_error=value_abs_error,
        scpn_gradient=scpn.gradient.copy(),
        pennylane_gradient=external_gradient,
        gradient_max_abs_error=gradient_max_abs_error,
        gradient_l2_error=gradient_l2_error,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        passed=bool(value_abs_error <= value_tol and gradient_max_abs_error <= gradient_tol),
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
    )


def build_pennylane_qnode_from_phase_qnode(
    circuit: PhaseQNodeCircuit,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    interface: str = "autograd",
    diff_method: str = "parameter-shift",
) -> PennyLaneQNodeConversionResult:
    """Build a PennyLane QNode for the registered local Phase-QNode subset.

    The converter is intentionally bounded. It maps the registered static gate
    family and expectation observables that have direct PennyLane equivalents,
    then records the exact device and shot policy used by the generated QNode.
    Provider submission, hardware execution, dynamic circuits, and covariance
    observables remain explicit non-claims.
    """
    qml = _load_pennylane()
    shot_policy = _as_optional_shots(shots)
    parameter_count = _phase_qnode_parameter_count(circuit)
    support_report = phase_qnode_support_report(circuit, np.zeros(parameter_count, dtype=float))
    if not support_report.supported:
        raise PhaseQNodeSupportError(support_report)
    if isinstance(circuit.observable, PauliCovarianceObservable):
        raise ValueError(
            "PennyLane QNode conversion does not support Pauli covariance observables; "
            "use local Phase-QNode execution for covariance product-rule gradients"
        )
    device = qml.device(device_name, wires=circuit.n_qubits, shots=shot_policy)

    def qnode_body(parameters: ArrayLike) -> object:
        values = _as_qnode_parameter_argument("values", parameters, width=parameter_count)
        for operation in circuit.operations:
            _apply_pennylane_operation(qml, cast(PhaseQNodeOperation, operation), values)
        return qml.expval(_pennylane_observable(qml, circuit.observable, circuit.n_qubits))

    qnode = qml.qnode(device, interface=interface, diff_method=diff_method)(qnode_body)
    _attach_phase_qnode_metadata(qnode, circuit)
    try:
        gradient = qml.grad(qnode, argnum=0)
    except TypeError:
        gradient = qml.grad(qnode)
    return PennyLaneQNodeConversionResult(
        qnode=qnode,
        gradient=gradient,
        device_name=device_name,
        n_qubits=circuit.n_qubits,
        shots=shot_policy,
        interface=interface,
        diff_method=diff_method,
        gates=support_report.gates,
        observable_kind=support_report.observable_kind,
        differentiable_parameters=support_report.differentiable_parameters,
        hardware_execution=False,
        claim_boundary=(
            "bounded PennyLane QNode conversion for registered local Phase-QNode "
            "gates and expectation observables on an explicitly named PennyLane "
            "device; no provider submission, hardware execution, dynamic-circuit, "
            "noise-model, or covariance-observable conversion claim"
        ),
    )


def check_pennylane_phase_qnode_round_trip(
    circuit: PhaseQNodeCircuit,
    values: ArrayLike,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    interface: str = "autograd",
    diff_method: str = "parameter-shift",
    value_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-6,
) -> PennyLaneRoundTripResult:
    """Verify a generated PennyLane QNode against local Phase-QNode execution."""
    qml = _load_pennylane()
    conversion = build_pennylane_qnode_from_phase_qnode(
        circuit,
        device_name=device_name,
        shots=shots,
        interface=interface,
        diff_method=diff_method,
    )
    value_tol = _as_non_negative_tolerance(value_tolerance)
    gradient_tol = _as_non_negative_tolerance(gradient_tolerance)
    parameter_count = _phase_qnode_parameter_count(circuit)
    parameter_values = _as_parameter_vector("values", values, width=parameter_count)
    scpn_value = execute_phase_qnode_circuit(circuit, parameter_values).value
    scpn_gradient = parameter_shift_phase_qnode_gradient(circuit, parameter_values)
    pennylane_value = _as_finite_scalar(
        "PennyLane generated QNode",
        conversion.qnode(parameter_values.copy()),
    )
    pennylane_gradient = _as_parameter_vector(
        "PennyLane generated QNode gradient",
        conversion.gradient(_pennylane_trainable_vector(qml, parameter_values)),
        width=parameter_count,
    )
    delta = scpn_gradient.gradient - pennylane_gradient
    value_abs_error = abs(float(scpn_value) - pennylane_value)
    gradient_max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    gradient_l2_error = float(np.linalg.norm(delta, ord=2))
    return PennyLaneRoundTripResult(
        scpn_value=float(scpn_value),
        pennylane_value=pennylane_value,
        value_abs_error=value_abs_error,
        scpn_gradient=scpn_gradient.gradient.copy(),
        pennylane_gradient=pennylane_gradient,
        gradient_max_abs_error=gradient_max_abs_error,
        gradient_l2_error=gradient_l2_error,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        passed=bool(value_abs_error <= value_tol and gradient_max_abs_error <= gradient_tol),
        evaluations=1 + scpn_gradient.parameter_shift_evaluations,
        method="phase_qnode_pennylane_conversion",
        shift_terms=1,
    )


def _pennylane_trainable_vector(qml: Any, values: FloatArray) -> Any:
    qml_numpy = getattr(qml, "numpy", None)
    array = getattr(qml_numpy, "array", None)
    if callable(array):
        try:
            return array(values.copy(), requires_grad=True)
        except TypeError:
            return array(values.copy())
    return values.copy()


def _as_qnode_parameter_argument(name: str, values: object, *, width: int) -> Any:
    shape = getattr(values, "shape", None)
    if shape is None:
        shape = np.asarray(values).shape
    if tuple(shape) != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {tuple(shape)}")
    return values


def run_pennylane_maturity_audit(
    *,
    objective: ScalarObjective,
    pennylane_objective: ScalarObjective,
    pennylane_gradient: GradientCallable,
    values: ArrayLike,
    circuit: PhaseQNodeCircuit,
    phase_qnode_values: ArrayLike,
    import_tape: object | None = None,
    device_name: str = "default.qubit",
    shots: int | None = None,
    interface: str = "autograd",
    diff_method: str = "parameter-shift",
    value_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PennyLaneMaturityAuditResult:
    """Aggregate PennyLane agreement, export, and optional import evidence.

    The audit records the bounded surfaces available today: caller-supplied
    PennyLane gradient agreement, caller-supplied QNode round-trip parity,
    generated registered-Phase-QNode export parity, and optional PennyLane tape
    import parity. It does not promote plugin ecosystems, provider execution,
    hardware execution, or benchmark claims until those routes have their own
    artefacts.
    """

    value_tol = _as_non_negative_tolerance(value_tolerance)
    gradient_tol = _as_non_negative_tolerance(gradient_tolerance)
    scalar_values = _as_parameter_vector("values", values)
    phase_values = _as_parameter_vector("phase_qnode_values", phase_qnode_values)

    gradient_agreement = check_pennylane_parameter_shift_agreement(
        objective,
        pennylane_gradient,
        scalar_values,
        tolerance=gradient_tol,
        parameters=parameters,
        rule=rule,
    )
    caller_qnode_round_trip = check_pennylane_qnode_round_trip(
        objective,
        pennylane_objective,
        pennylane_gradient,
        scalar_values,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        parameters=parameters,
        rule=rule,
    )
    export_conversion = build_pennylane_qnode_from_phase_qnode(
        circuit,
        device_name=device_name,
        shots=shots,
        interface=interface,
        diff_method=diff_method,
    )
    phase_qnode_export_round_trip = check_pennylane_phase_qnode_round_trip(
        circuit,
        phase_values,
        device_name=device_name,
        shots=shots,
        interface=interface,
        diff_method=diff_method,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
    )
    phase_qnode_import_round_trip: object | None = None
    if import_tape is not None:
        from .pennylane_import import check_pennylane_phase_qnode_import_round_trip

        phase_qnode_import_round_trip = check_pennylane_phase_qnode_import_round_trip(
            import_tape,
            value_tolerance=value_tol,
            gradient_tolerance=gradient_tol,
        )

    evaluation_plan = plan_phase_qnode_parameter_shift_evaluations(circuit, phase_values)
    promotion_metadata: dict[str, object] = {
        "device_name": export_conversion.device_name,
        "shots": export_conversion.shots,
        "interface": export_conversion.interface,
        "diff_method": export_conversion.diff_method,
        "hardware_execution": export_conversion.hardware_execution,
        "observable_kind": export_conversion.observable_kind,
        "differentiable_parameters": list(export_conversion.differentiable_parameters),
        "phase_qnode_parameter_shift_evaluations": evaluation_plan.parameter_shift_evaluations,
        "phase_qnode_generic_scalar_evaluations": (
            evaluation_plan.generic_scalar_objective_evaluations
        ),
        "phase_qnode_saved_shifted_evaluations": evaluation_plan.saved_shifted_evaluations,
        "phase_qnode_evaluation_groups": [group.to_dict() for group in evaluation_plan.groups],
        "export_claim_boundary": export_conversion.claim_boundary,
    }
    if phase_qnode_import_round_trip is not None:
        promotion_metadata["import_round_trip_parameters"] = int(
            getattr(phase_qnode_import_round_trip, "n_parameters", 0)
        )

    import_passed = bool(
        phase_qnode_import_round_trip is not None
        and getattr(phase_qnode_import_round_trip, "value_match", False)
        and getattr(phase_qnode_import_round_trip, "gradient_match", False)
    )
    evidence: dict[str, object] = {
        "gradient_agreement": gradient_agreement,
        "caller_qnode_round_trip": caller_qnode_round_trip,
        "phase_qnode_export_conversion": export_conversion,
        "phase_qnode_export_round_trip": phase_qnode_export_round_trip,
        "phase_qnode_import_round_trip": phase_qnode_import_round_trip,
    }
    required_capabilities = {
        "gradient_agreement": "passed" if gradient_agreement.passed else "failed",
        "caller_qnode_round_trip": "passed" if caller_qnode_round_trip.passed else "failed",
        "phase_qnode_export_conversion": "passed",
        "phase_qnode_export_round_trip": (
            "passed" if phase_qnode_export_round_trip.passed else "failed"
        ),
        "phase_qnode_import_round_trip": "passed" if import_passed else "blocked",
        "device_metadata": "passed",
        "shot_policy_metadata": "passed",
        "diff_method_metadata": "passed",
        "grouped_parameter_shift_evaluation_counts": (
            "passed" if evaluation_plan.supported else "failed"
        ),
        "pennylane_plugin_matrix": "blocked",
        "provider_plugin_execution": "blocked",
        "hardware_execution": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    identical_circuit_ready = all(
        required_capabilities[name] == "passed"
        for name in (
            "gradient_agreement",
            "caller_qnode_round_trip",
            "phase_qnode_export_conversion",
            "phase_qnode_export_round_trip",
            "phase_qnode_import_round_trip",
            "device_metadata",
            "shot_policy_metadata",
            "diff_method_metadata",
            "grouped_parameter_shift_evaluation_counts",
        )
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PennyLaneMaturityAuditResult(
        identical_circuit_ready=identical_circuit_ready,
        ready_for_provider_exceedance=identical_circuit_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        promotion_metadata=promotion_metadata,
        open_gaps=open_gaps,
    )


def _result_to_dict(result: object) -> object:
    if result is None:
        return None
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    return result


_PENNYLANE_OPERATION_NAMES: dict[str, str] = {
    "h": "Hadamard",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "s": "S",
    "t": "T",
    "sx": "SX",
    "cnot": "CNOT",
    "cz": "CZ",
    "cy": "CY",
    "swap": "SWAP",
    "ch": "CH",
    "ccnot": "Toffoli",
    "cswap": "CSWAP",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "phase": "PhaseShift",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "rxx": "IsingXX",
    "ryy": "IsingYY",
    "rzz": "IsingZZ",
}


def _phase_qnode_parameter_count(circuit: PhaseQNodeCircuit) -> int:
    parameter_indices = [
        parsed.parameter_index
        for operation in circuit.operations
        if (parsed := cast(PhaseQNodeOperation, operation)).parameter_index is not None
    ]
    return 0 if not parameter_indices else max(parameter_indices) + 1


def _as_optional_shots(value: int | None) -> int | None:
    if value is None:
        return None
    shots = int(value)
    if isinstance(value, bool) or shots < 1:
        raise ValueError("shots must be a positive integer or None")
    return shots


def _attach_phase_qnode_metadata(
    qnode: Callable[[ArrayLike], object], circuit: PhaseQNodeCircuit
) -> None:
    qnode_with_metadata: Any = qnode
    with suppress(AttributeError):
        qnode_with_metadata._scpn_phase_qnode_circuit = circuit


def _apply_pennylane_operation(qml: Any, operation: PhaseQNodeOperation, values: Any) -> None:
    wire_sequence = list(operation.qubits)
    wires = operation.qubits[0] if len(operation.qubits) == 1 else wire_sequence
    if operation.gate == "cs":
        qml.ControlledPhaseShift(np.pi / 2.0, wires=wires)
        return
    if operation.gate == "ct":
        qml.ControlledPhaseShift(np.pi / 4.0, wires=wires)
        return
    if operation.gate == "ccz":
        qml.ControlledQubitUnitary(
            np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128),
            control_wires=wire_sequence[:2],
            wires=wire_sequence[2],
        )
        return
    operation_name = _PENNYLANE_OPERATION_NAMES.get(operation.gate)
    if operation_name is None:
        raise ValueError(f"PennyLane QNode conversion does not support gate {operation.gate!r}")
    qml_operation = getattr(qml, operation_name)
    if operation.parameter_index is None:
        qml_operation(wires=wires)
        return
    qml_operation(values[operation.parameter_index], wires=wires)


def _pennylane_observable(qml: Any, observable: object, n_qubits: int) -> object:
    if isinstance(observable, PauliTerm):
        return _pennylane_pauli_term(qml, observable)
    if isinstance(observable, SparsePauliHamiltonian):
        terms = [_pennylane_pauli_product(qml, term) for term in observable.terms]
        coefficients = [float(term.coefficient) for term in observable.terms]
        return qml.Hamiltonian(coefficients, terms)
    if isinstance(observable, DenseHermitianObservable):
        return qml.Hermitian(observable.matrix, wires=range(n_qubits))
    if isinstance(observable, PauliCovarianceObservable):
        raise ValueError(
            "PennyLane QNode conversion does not support Pauli covariance observables"
        )
    raise ValueError(
        f"PennyLane QNode conversion does not support observable {type(observable).__name__}"
    )


def _pennylane_pauli_term(qml: Any, term: PauliTerm) -> object:
    observable: Any = _pennylane_pauli_product(qml, term)
    return float(term.coefficient) * observable


def _pennylane_pauli_product(qml: Any, term: PauliTerm) -> object:
    factors = [_pennylane_single_pauli(qml, qubit, label) for qubit, label in term.factors]
    product: Any = factors[0]
    for factor in factors[1:]:
        product = product @ factor
    return product


def _pennylane_single_pauli(qml: Any, qubit: int, label: str) -> object:
    if label == "x":
        return qml.PauliX(wires=qubit)
    if label == "y":
        return qml.PauliY(wires=qubit)
    if label == "z":
        return qml.PauliZ(wires=qubit)
    raise ValueError(f"PennyLane QNode conversion does not support Pauli label {label!r}")


__all__ = [
    "PennyLaneGradientAgreementResult",
    "PennyLaneMaturityAuditResult",
    "PennyLaneQNodeConversionResult",
    "PennyLaneRoundTripResult",
    "build_pennylane_qnode_from_phase_qnode",
    "check_pennylane_parameter_shift_agreement",
    "check_pennylane_phase_qnode_round_trip",
    "check_pennylane_qnode_round_trip",
    "is_phase_pennylane_available",
    "run_pennylane_maturity_audit",
]
