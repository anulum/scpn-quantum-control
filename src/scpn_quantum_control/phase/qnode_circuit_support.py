# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Support
"""Declarative support analysis and shift planning for Phase-QNode circuits.

This one-way leaf validates registered circuits, profiles circuit depth and
resources, reports statevector/density/gradient/metric support, and constructs
gate-aware parameter-shift evaluation plans. It contains no numerical circuit
execution, measurement, framework, compiler, provider, or hardware orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..differentiable import multi_frequency_parameter_shift_rule
from .qnode_circuit_contracts import (
    _GATE_ARITY,
    _PARAMETRIC_GATES,
    _REGISTERED_GATES,
    _REGISTERED_NOISE_CHANNELS,
    _REGISTERED_OBSERVABLES,
    DenseHermitianObservable,
    DensityOperation,
    FloatArray,
    OperationSpec,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeDepthProfile,
    PhaseQNodeGradientEvaluationGroup,
    PhaseQNodeGradientEvaluationPlan,
    PhaseQNodeNoiseChannel,
    PhaseQNodeOperation,
    PhaseQNodeRegisteredCircuitSpec,
    PhaseQNodeSupportError,
    PhaseQNodeSupportReport,
    SparsePauliHamiltonian,
    _parse_operation,
)


def build_registered_phase_qnode_circuit(
    n_qubits: int,
    operations: tuple[PhaseQNodeOperation | OperationSpec, ...],
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
    ),
    *,
    max_depth: int | None = None,
    max_operations: int | None = None,
) -> PhaseQNodeRegisteredCircuitSpec:
    """Build and validate an arbitrary-depth registered local Phase-QNode circuit."""
    circuit = PhaseQNodeCircuit(n_qubits=n_qubits, operations=operations, observable=observable)
    profile = phase_qnode_depth_profile(circuit)
    operation_budget = _as_optional_positive_int("max_operations", max_operations)
    depth_budget = _as_optional_positive_int("max_depth", max_depth)
    if operation_budget is not None and profile.operation_count > operation_budget:
        raise ValueError(
            f"registered Phase-QNode operation count {profile.operation_count} "
            f"exceeds max_operations={operation_budget}"
        )
    if depth_budget is not None and profile.depth > depth_budget:
        raise ValueError(
            f"registered Phase-QNode depth {profile.depth} exceeds max_depth={depth_budget}"
        )
    report = phase_qnode_support_report(
        circuit,
        np.zeros(profile.parameter_count, dtype=np.float64),
    )
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    return PhaseQNodeRegisteredCircuitSpec(
        circuit=circuit,
        depth_profile=profile,
        support_report=report,
        claim_boundary=(
            "arbitrary-depth registered local Phase-QNode circuit over the "
            "bounded statevector gate and observable family with deterministic "
            "depth/resource accounting; no dynamic-circuit, provider, finite-shot, "
            "hardware, or native framework autodiff-through-simulator claim"
        ),
    )


def phase_qnode_depth_profile(circuit: PhaseQNodeCircuit) -> PhaseQNodeDepthProfile:
    """Return deterministic depth and resource metadata for a registered circuit."""
    operations = _parsed_operations(circuit)
    last_layer_by_qubit = [0 for _ in range(circuit.n_qubits)]
    operation_layers: list[int] = []
    gate_counts: dict[str, int] = {}
    parameter_indices: set[int] = set()
    entangling_pairs: set[tuple[int, int]] = set()
    max_arity = 0
    two_qubit_gate_count = 0
    for operation in operations:
        layer = 1 + max(last_layer_by_qubit[qubit] for qubit in operation.qubits)
        operation_layers.append(layer)
        for qubit in operation.qubits:
            last_layer_by_qubit[qubit] = layer
        gate_counts[operation.gate] = gate_counts.get(operation.gate, 0) + 1
        if operation.parameter_index is not None:
            parameter_indices.add(operation.parameter_index)
        arity = len(operation.qubits)
        max_arity = max(max_arity, arity)
        if arity == 2:
            two_qubit_gate_count += 1
            left, right = operation.qubits
            entangling_pairs.add((min(left, right), max(left, right)))
    ordered_parameters = tuple(sorted(parameter_indices))
    parameter_count = 0 if not ordered_parameters else ordered_parameters[-1] + 1
    return PhaseQNodeDepthProfile(
        n_qubits=circuit.n_qubits,
        operation_count=len(operations),
        depth=max(operation_layers, default=0),
        operation_layers=tuple(operation_layers),
        gate_counts=dict(sorted(gate_counts.items())),
        parameter_count=parameter_count,
        differentiable_parameters=ordered_parameters,
        two_qubit_gate_count=two_qubit_gate_count,
        entangling_pairs=tuple(sorted(entangling_pairs)),
        max_operation_arity=max_arity,
        claim_boundary=(
            "ordered local Phase-QNode depth/resource profile for registered "
            "statevector circuits; no hardware duration, pulse schedule, noise, "
            "provider transpilation, or isolated-performance claim"
        ),
    )


def phase_qnode_support_report(
    circuit: PhaseQNodeCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeSupportReport:
    """Return a fail-closed support report for a circuit and parameter vector."""
    values = _as_parameter_vector(parameters)
    operations = _parsed_operations(circuit)
    gates = tuple(operation.gate for operation in operations)
    unsupported_gates = tuple(gate for gate in gates if gate not in _REGISTERED_GATES)
    invalid_arities = tuple(
        f"{index}:{operation.gate}/{len(operation.qubits)}"
        for index, operation in enumerate(operations)
        if operation.gate in _GATE_ARITY and len(operation.qubits) != _GATE_ARITY[operation.gate]
    )
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
                operation.parameter_index
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
    if invalid_arities:
        reasons.append(f"gate arity mismatches at operations: {invalid_arities}")
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


def phase_qnode_density_support_report(
    circuit: PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeSupportReport:
    """Return a fail-closed support report for density-matrix execution."""
    values = _as_parameter_vector(parameters)
    operations = _parsed_density_operations(circuit)
    unitary_operations = tuple(
        operation for operation in operations if isinstance(operation, PhaseQNodeOperation)
    )
    noise_operations = tuple(
        operation for operation in operations if isinstance(operation, PhaseQNodeNoiseChannel)
    )
    gates = tuple(
        operation.gate if isinstance(operation, PhaseQNodeOperation) else operation.channel
        for operation in operations
    )
    unsupported_gates = tuple(
        operation.gate
        for operation in unitary_operations
        if operation.gate not in _REGISTERED_GATES
    )
    unsupported_noise = tuple(
        operation.channel
        for operation in noise_operations
        if operation.channel not in _REGISTERED_NOISE_CHANNELS
    )
    invalid_arities = tuple(
        f"{index}:{operation.gate}/{len(operation.qubits)}"
        for index, operation in enumerate(operations)
        if isinstance(operation, PhaseQNodeOperation)
        and operation.gate in _GATE_ARITY
        and len(operation.qubits) != _GATE_ARITY[operation.gate]
    )
    invalid_noise_arities = tuple(
        f"{index}:{operation.channel}/{len(operation.qubits)}"
        for index, operation in enumerate(operations)
        if isinstance(operation, PhaseQNodeNoiseChannel) and len(operation.qubits) != 1
    )
    unsupported_parameters = tuple(
        operation.parameter_index
        for operation in unitary_operations
        if operation.gate in _PARAMETRIC_GATES
        and operation.parameter_index is not None
        and operation.parameter_index >= values.size
    )
    missing_parameters = tuple(
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, PhaseQNodeOperation)
        and operation.gate in _PARAMETRIC_GATES
        and operation.parameter_index is None
    )
    observable_kind = _observable_kind(circuit.observable)
    unsupported_observables: tuple[str, ...] = ()
    if observable_kind not in _REGISTERED_OBSERVABLES:
        unsupported_observables = (observable_kind,)
    differentiable = tuple(
        sorted(
            {
                operation.parameter_index
                for operation in unitary_operations
                if operation.gate in _PARAMETRIC_GATES and operation.parameter_index is not None
            }
        )
    )
    reasons: list[str] = []
    if unsupported_gates:
        reasons.append(f"unsupported unitary gates: {', '.join(unsupported_gates)}")
    if unsupported_noise:
        reasons.append(f"unsupported noise channels: {', '.join(unsupported_noise)}")
    if unsupported_observables:
        reasons.append(f"unsupported observables: {', '.join(unsupported_observables)}")
    if unsupported_parameters:
        reasons.append(f"parameter indices outside supplied vector: {unsupported_parameters}")
    if missing_parameters:
        reasons.append(
            f"parametric gates missing parameter indices at operations: {missing_parameters}"
        )
    if invalid_arities:
        reasons.append(f"gate arity mismatches at operations: {invalid_arities}")
    if invalid_noise_arities:
        reasons.append(f"noise channel arity mismatches at operations: {invalid_noise_arities}")
    return PhaseQNodeSupportReport(
        supported=not reasons,
        gates=gates,
        observable_kind=observable_kind,
        differentiable_parameters=differentiable,
        unsupported_gates=unsupported_gates + unsupported_noise,
        unsupported_observables=unsupported_observables,
        unsupported_parameters=tuple(int(item) for item in unsupported_parameters),
        failure_reason="; ".join(reasons),
        alternatives=(
            "use registered_phase_qnode_gates for unitary density operations",
            "use registered_phase_qnode_noise_channels for local noisy channels",
            "route provider, hardware, and finite-shot noise through explicit policy records",
        ),
    )


def phase_qnode_gradient_support_report(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeSupportReport:
    """Return support metadata for the analytic parameter-shift route."""
    values = _as_parameter_vector(parameters)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        return _blocked_density_route_support_report(
            circuit,
            values,
            route_name="parameter-shift gradients",
            alternatives=(
                "use PhaseQNodeCircuit for pure-state parameter-shift gradients",
                "use execute_phase_qnode_density_matrix for noisy value, trace, and purity evidence",
                "route finite-shot, provider, or hardware gradients through explicit policy records",
            ),
        )
    return phase_qnode_support_report(circuit, values)


def plan_phase_qnode_parameter_shift_evaluations(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeGradientEvaluationPlan:
    """Plan shifted evaluations for a registered Phase-QNode gradient.

    The planner is gate-aware at the registered-circuit level and parameter-aware
    at the logical parameter level. It does not claim PennyLane-style generator
    simplification for an opaque callable, and it does not collapse distinct
    logical parameters into simultaneous shifts because that would produce a
    directional derivative rather than independent partial derivatives.
    """
    values = _as_parameter_vector(parameters)
    report = phase_qnode_gradient_support_report(circuit, values)
    if isinstance(circuit, PhaseQNodeDensityCircuit) or not report.supported:
        return PhaseQNodeGradientEvaluationPlan(
            supported=False,
            method="unsupported_phase_qnode_parameter_shift",
            parameter_count=values.size,
            differentiable_parameters=report.differentiable_parameters,
            operation_level_naive_evaluations=0,
            planned_shifted_evaluations=0,
            saved_shifted_evaluations=0,
            groups=(),
            fallback_reason=report.failure_reason,
            claim_boundary=(
                "fail-closed Phase-QNode parameter-shift evaluation plan; no "
                "shifted evaluations are authorised for unsupported or density/noise routes"
            ),
        )
    operations = _parsed_operations(circuit)
    parametric_by_index: dict[int, list[tuple[int, PhaseQNodeOperation]]] = {}
    for operation_index, operation in enumerate(operations):
        if operation.gate in _PARAMETRIC_GATES and operation.parameter_index is not None:
            parametric_by_index.setdefault(operation.parameter_index, []).append(
                (operation_index, operation)
            )

    groups: list[PhaseQNodeGradientEvaluationGroup] = []
    for parameter_index in report.differentiable_parameters:
        records = tuple(parametric_by_index.get(parameter_index, ()))
        operation_indices = tuple(index for index, _operation in records)
        parameter_operations = tuple(operation for _index, operation in records)
        generator_keys = tuple(
            _parameter_generator_key(operation) for operation in parameter_operations
        )
        commuting = _operations_commute_for_shared_parameter(parameter_operations)
        shift_terms = _parameter_shift_terms_for_group(parameter_operations)
        shifted_evaluations = 2 * len(shift_terms)
        naive_operation_evaluations = 2 * len(parameter_operations)
        saved = max(0, naive_operation_evaluations - shifted_evaluations)
        reason = (
            "shared logical parameter across a collapsible registered generator group; "
            "one frequency-adjusted plus/minus pair shifts the parameter once and updates all tied gates"
            if len(parameter_operations) > 1 and len(shift_terms) == 1 and commuting
            else "single logical parameter group; no distinct-parameter simultaneous shift claimed"
        )
        if len(parameter_operations) > 1 and len(shift_terms) > 1:
            reason = (
                "shared logical parameter requires a multi-frequency shift plan; "
                "the planner avoids the invalid single-gate pi/2 rule"
            )
        if len(parameter_operations) > 1 and not commuting:
            reason = (
                "shared logical parameter appears on non-commuting generators; "
                "frequency-aware shifted evaluations are used and no commuting-generator "
                "reuse claim is made"
            )
        groups.append(
            PhaseQNodeGradientEvaluationGroup(
                parameter_index=parameter_index,
                operation_indices=operation_indices,
                gates=tuple(operation.gate for operation in parameter_operations),
                qubits=tuple(operation.qubits for operation in parameter_operations),
                generator_keys=generator_keys,
                frequency_gaps=tuple(frequency for frequency, _shift, _coefficient in shift_terms),
                commuting=commuting,
                shifted_evaluations=shifted_evaluations,
                naive_operation_shifted_evaluations=naive_operation_evaluations,
                saved_shifted_evaluations=saved,
                reason=reason,
            )
        )
    operation_level_naive = sum(group.naive_operation_shifted_evaluations for group in groups)
    planned = sum(group.shifted_evaluations for group in groups)
    return PhaseQNodeGradientEvaluationPlan(
        supported=True,
        method="registered_phase_qnode_gate_aware_parameter_shift",
        parameter_count=values.size,
        differentiable_parameters=report.differentiable_parameters,
        operation_level_naive_evaluations=operation_level_naive,
        planned_shifted_evaluations=planned,
        saved_shifted_evaluations=max(0, operation_level_naive - planned),
        groups=tuple(groups),
        fallback_reason=(
            "generic ScalarObjective callables expose only values, so the generic "
            "route must plan independent 2N shifts; registered PhaseQNodeCircuit "
            "metadata is required for operation-level evaluation-count evidence"
        ),
        claim_boundary=(
            "registered local Phase-QNode evaluation-count planning for logical "
            "parameter-shift gradients; no opaque-callable generator inference, "
            "distinct-parameter simultaneous-shift gradient extraction, provider "
            "submission, hardware execution, or native framework autodiff claim"
        ),
    )


def phase_qnode_metric_support_report(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeSupportReport:
    """Return support metadata for pure-state Fubini-Study and QFI routes."""
    values = _as_parameter_vector(parameters)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        return _blocked_density_route_support_report(
            circuit,
            values,
            route_name="pure-state Fubini-Study and QFI metrics",
            alternatives=(
                "use PhaseQNodeCircuit for pure-state metric and QFI diagnostics",
                "use execute_phase_qnode_density_matrix for noisy value, trace, and purity evidence",
                "route density-matrix, finite-shot, provider, or hardware metrics through explicit policy records",
            ),
        )
    return phase_qnode_support_report(circuit, values)


def _parsed_operations(circuit: PhaseQNodeCircuit) -> tuple[PhaseQNodeOperation, ...]:
    return tuple(_parse_operation(operation) for operation in circuit.operations)


def _parsed_density_operations(circuit: PhaseQNodeDensityCircuit) -> tuple[DensityOperation, ...]:
    return cast(tuple[DensityOperation, ...], circuit.operations)


def _blocked_density_route_support_report(
    circuit: PhaseQNodeDensityCircuit,
    values: FloatArray,
    *,
    route_name: str,
    alternatives: tuple[str, ...],
) -> PhaseQNodeSupportReport:
    density_report = phase_qnode_density_support_report(circuit, values)
    noise_channels = tuple(
        operation.channel
        for operation in _parsed_density_operations(circuit)
        if isinstance(operation, PhaseQNodeNoiseChannel)
    )
    reasons = [
        f"{route_name} require the pure-state PhaseQNodeCircuit route",
    ]
    if noise_channels:
        reasons.append(
            f"{route_name} are not registered for density-matrix noise channels: "
            f"{', '.join(noise_channels)}"
        )
    if density_report.failure_reason:
        reasons.append(density_report.failure_reason)
    return PhaseQNodeSupportReport(
        supported=False,
        gates=density_report.gates,
        observable_kind=density_report.observable_kind,
        differentiable_parameters=density_report.differentiable_parameters,
        unsupported_gates=density_report.unsupported_gates,
        unsupported_observables=density_report.unsupported_observables,
        unsupported_parameters=density_report.unsupported_parameters,
        failure_reason="; ".join(reasons),
        alternatives=alternatives,
    )


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


def _as_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _parameter_generator_key(operation: PhaseQNodeOperation) -> str:
    if operation.gate in {"rx", "crx"}:
        return "x"
    if operation.gate in {"ry", "cry"}:
        return "y"
    if operation.gate in {"rz", "crz", "phase"}:
        return "z"
    if operation.gate == "rxx":
        return "xx"
    if operation.gate == "ryy":
        return "yy"
    if operation.gate == "rzz":
        return "zz"
    return operation.gate


def _operations_commute_for_shared_parameter(
    operations: tuple[PhaseQNodeOperation, ...],
) -> bool:
    for left_index, left in enumerate(operations):
        for right in operations[left_index + 1 :]:
            if not _parameter_generators_commute(left, right):
                return False
    return True


def _base_generator_frequencies(operation: PhaseQNodeOperation) -> tuple[float, ...]:
    """Spectral gaps of a single registered generator.

    A single-Pauli rotation (``rx``/``ry``/``rz``/``phase``/``rxx``/``ryy``/
    ``rzz``) has generator eigenvalues ``±1/2``, so the only positive gap is
    ``1`` and the canonical two-term rule is exact. A controlled rotation
    (``crx``/``cry``/``crz``) has generator eigenvalues ``{0, 0, +1/2, -1/2}``
    (the rotation only acts in the control-on subspace), so it carries two
    distinct gaps ``{1/2, 1}`` and needs the four-term rule — the two-term rule
    is wrong whenever the observable couples the control-on and control-off
    sectors.
    """
    if _is_controlled_parametric_gate(operation.gate):
        return (0.5, 1.0)
    return (1.0,)


def _group_generator_frequencies(
    operations: tuple[PhaseQNodeOperation, ...],
) -> tuple[float, ...]:
    """Positive spectral gaps of the shared-parameter generator group."""
    if len(operations) == 1 or _collapsible_shared_parameter_group(operations):
        multiplicity = float(len(operations))
        return tuple(
            frequency * multiplicity for frequency in _base_generator_frequencies(operations[0])
        )
    # Distinct commuting single-Pauli generators: eigenvalue gaps 1..n.
    return tuple(float(index) for index in range(1, len(operations) + 1))


def _parameter_shift_terms_for_group(
    operations: tuple[PhaseQNodeOperation, ...],
) -> tuple[tuple[float, float, float], ...]:
    if not operations:
        return ()
    frequencies = _group_generator_frequencies(operations)
    if len(frequencies) == 1:
        frequency = frequencies[0]
        return ((frequency, float(np.pi / (2.0 * frequency)), 0.5 * frequency),)
    rule = multi_frequency_parameter_shift_rule(frequencies)
    return tuple(
        (frequency, float(term[0]), float(term[1]))
        for frequency, term in zip(frequencies, rule.terms, strict=True)
    )


def _collapsible_shared_parameter_group(
    operations: tuple[PhaseQNodeOperation, ...],
) -> bool:
    if len(operations) <= 1:
        return False
    first = operations[0]
    return all(
        operation.gate == first.gate and operation.qubits == first.qubits
        for operation in operations[1:]
    )


def _parameter_generators_commute(
    left: PhaseQNodeOperation,
    right: PhaseQNodeOperation,
) -> bool:
    if set(left.qubits).isdisjoint(right.qubits):
        return True
    if _is_controlled_parametric_gate(left.gate) or _is_controlled_parametric_gate(right.gate):
        return left.gate == right.gate and left.qubits == right.qubits
    left_paulis = _generator_pauli_map(left)
    right_paulis = _generator_pauli_map(right)
    shared = set(left_paulis).intersection(right_paulis)
    if not shared:
        return True
    anti_commuting_overlap = sum(
        1 for qubit in shared if left_paulis[qubit] != right_paulis[qubit]
    )
    return anti_commuting_overlap % 2 == 0


def _is_controlled_parametric_gate(gate: str) -> bool:
    return gate in {"crx", "cry", "crz"}


def _generator_pauli_map(operation: PhaseQNodeOperation) -> Mapping[int, str]:
    key = _parameter_generator_key(operation)
    if key in {"x", "y", "z"}:
        return {operation.qubits[-1]: key}
    if key in {"xx", "yy", "zz"}:
        return {qubit: key[0] for qubit in operation.qubits}
    return {}


__all__ = [
    "build_registered_phase_qnode_circuit",
    "phase_qnode_depth_profile",
    "phase_qnode_support_report",
    "phase_qnode_density_support_report",
    "phase_qnode_gradient_support_report",
    "plan_phase_qnode_parameter_shift_evaluations",
    "phase_qnode_metric_support_report",
]
