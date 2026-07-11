# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Circuit Registry
"""Executable facade for registered Phase-QNode circuit routes.

Shared declarations, result records, registries, and constructor validation live in
:mod:`.qnode_circuit_contracts`. This module retains builders, support analysis,
execution, gradients, metrics, measurements, and numerical kernels while those
responsibilities undergo bounded decomposition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from ..differentiable import multi_frequency_parameter_shift_rule
from .qnode_circuit_contracts import (
    _GATE_ARITY as _GATE_ARITY,
)
from .qnode_circuit_contracts import (
    _H as _H,
)
from .qnode_circuit_contracts import (
    _I as _I,
)
from .qnode_circuit_contracts import (
    _NON_PARAMETRIC_GATES as _NON_PARAMETRIC_GATES,
)
from .qnode_circuit_contracts import (
    _PARAMETRIC_GATES as _PARAMETRIC_GATES,
)
from .qnode_circuit_contracts import (
    _PAULI as _PAULI,
)
from .qnode_circuit_contracts import (
    _PAULI_MATRICES as _PAULI_MATRICES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_DECOMPOSITIONS as _REGISTERED_DECOMPOSITIONS,
)
from .qnode_circuit_contracts import (
    _REGISTERED_GATES as _REGISTERED_GATES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_NOISE_CHANNELS as _REGISTERED_NOISE_CHANNELS,
)
from .qnode_circuit_contracts import (
    _REGISTERED_OBSERVABLES as _REGISTERED_OBSERVABLES,
)
from .qnode_circuit_contracts import (
    _REGISTERED_TEMPLATES as _REGISTERED_TEMPLATES,
)
from .qnode_circuit_contracts import (
    _S as _S,
)
from .qnode_circuit_contracts import (
    _SX as _SX,
)
from .qnode_circuit_contracts import (
    _T as _T,
)
from .qnode_circuit_contracts import (
    _X as _X,
)
from .qnode_circuit_contracts import (
    _Y as _Y,
)
from .qnode_circuit_contracts import (
    _Z as _Z,
)
from .qnode_circuit_contracts import (
    ComplexArray as ComplexArray,
)
from .qnode_circuit_contracts import (
    DenseHermitianObservable as DenseHermitianObservable,
)
from .qnode_circuit_contracts import (
    DensityOperation as DensityOperation,
)
from .qnode_circuit_contracts import (
    DensityOperationSpec as DensityOperationSpec,
)
from .qnode_circuit_contracts import (
    FloatArray as FloatArray,
)
from .qnode_circuit_contracts import (
    OperationSpec as OperationSpec,
)
from .qnode_circuit_contracts import (
    PauliCovarianceObservable as PauliCovarianceObservable,
)
from .qnode_circuit_contracts import (
    PauliTerm as PauliTerm,
)
from .qnode_circuit_contracts import (
    PhaseQNodeCircuit as PhaseQNodeCircuit,
)
from .qnode_circuit_contracts import (
    PhaseQNodeClassicalFisherResult as PhaseQNodeClassicalFisherResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDensityCircuit as PhaseQNodeDensityCircuit,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDensityExecutionResult as PhaseQNodeDensityExecutionResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeDepthProfile as PhaseQNodeDepthProfile,
)
from .qnode_circuit_contracts import (
    PhaseQNodeExecutionResult as PhaseQNodeExecutionResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientEvaluationGroup as PhaseQNodeGradientEvaluationGroup,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientEvaluationPlan as PhaseQNodeGradientEvaluationPlan,
)
from .qnode_circuit_contracts import (
    PhaseQNodeGradientResult as PhaseQNodeGradientResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeMetricTensorResult as PhaseQNodeMetricTensorResult,
)
from .qnode_circuit_contracts import (
    PhaseQNodeNoiseChannel as PhaseQNodeNoiseChannel,
)
from .qnode_circuit_contracts import (
    PhaseQNodeOperation as PhaseQNodeOperation,
)
from .qnode_circuit_contracts import (
    PhaseQNodeRegisteredCircuitSpec as PhaseQNodeRegisteredCircuitSpec,
)
from .qnode_circuit_contracts import (
    PhaseQNodeSupportError as PhaseQNodeSupportError,
)
from .qnode_circuit_contracts import (
    PhaseQNodeSupportReport as PhaseQNodeSupportReport,
)
from .qnode_circuit_contracts import (
    PhaseQNodeTemplateSpec as PhaseQNodeTemplateSpec,
)
from .qnode_circuit_contracts import (
    SparsePauliHamiltonian as SparsePauliHamiltonian,
)
from .qnode_circuit_contracts import (
    _as_finite_scalar as _as_finite_scalar,
)
from .qnode_circuit_contracts import (
    _as_probability as _as_probability,
)
from .qnode_circuit_contracts import (
    _FiniteShotFisherEvidence as _FiniteShotFisherEvidence,
)
from .qnode_circuit_contracts import (
    _normalise_observable as _normalise_observable,
)
from .qnode_circuit_contracts import (
    _optional_float_array_to_list as _optional_float_array_to_list,
)
from .qnode_circuit_contracts import (
    _parse_density_operation as _parse_density_operation,
)
from .qnode_circuit_contracts import (
    _parse_operation as _parse_operation,
)


def registered_phase_qnode_gates() -> tuple[str, ...]:
    """Return the local Phase-QNode gate family."""
    return _REGISTERED_GATES


def registered_phase_qnode_observables() -> tuple[str, ...]:
    """Return the local Phase-QNode observable family."""
    return _REGISTERED_OBSERVABLES


def registered_phase_qnode_templates() -> tuple[str, ...]:
    """Return the registered local multi-qubit Phase-QNode templates."""
    return _REGISTERED_TEMPLATES


def registered_phase_qnode_decompositions() -> tuple[str, ...]:
    """Return gates with exact registered operation-list decompositions."""
    return _REGISTERED_DECOMPOSITIONS


def registered_phase_qnode_noise_channels() -> tuple[str, ...]:
    """Return the registered local density-matrix noise channels."""
    return _REGISTERED_NOISE_CHANNELS


def build_sparse_ising_chain_hamiltonian(
    n_qubits: int,
    *,
    x_field: float | ArrayLike = 0.0,
    z_field: float | ArrayLike = 0.0,
    zz_coupling: float | ArrayLike = 1.0,
    periodic: bool = False,
) -> SparsePauliHamiltonian:
    """Build a sparse nearest-neighbour Ising-chain Pauli Hamiltonian.

    Coefficients are direct observable weights. Scalar values are broadcast
    across all sites or edges; arrays must match the site or edge count exactly.
    """
    width = _as_ising_chain_width(n_qubits)
    if not isinstance(periodic, bool):
        raise ValueError("periodic must be a boolean")
    edge_count = width if periodic else width - 1
    x_coefficients = _as_broadcast_coefficients("x_field", x_field, width)
    z_coefficients = _as_broadcast_coefficients("z_field", z_field, width)
    zz_coefficients = _as_broadcast_coefficients("zz_coupling", zz_coupling, edge_count)
    terms: list[PauliTerm] = []
    for qubit, coefficient in enumerate(x_coefficients):
        if coefficient != 0.0:
            terms.append(PauliTerm(coefficient, ((qubit, "x"),)))
    for qubit, coefficient in enumerate(z_coefficients):
        if coefficient != 0.0:
            terms.append(PauliTerm(coefficient, ((qubit, "z"),)))
    for edge, coefficient in enumerate(zz_coefficients):
        if coefficient != 0.0:
            left = edge
            right = (edge + 1) % width
            terms.append(PauliTerm(coefficient, ((left, "z"), (right, "z"))))
    if not terms:
        raise ValueError("sparse Ising Hamiltonian must contain at least one non-zero term")
    return SparsePauliHamiltonian(tuple(terms))


def decompose_phase_qnode_controlled_gate(
    operation: PhaseQNodeOperation | OperationSpec,
) -> tuple[PhaseQNodeOperation, ...]:
    """Return an exact registered decomposition for supported controlled gates."""
    parsed = _parse_operation(operation)
    if parsed.parameter_index is not None:
        raise ValueError("controlled-gate decompositions do not accept trainable parameters")
    if parsed.gate == "ccnot":
        _require_qubit_width(parsed, 3)
        control_a, control_b, target = parsed.qubits
        return (
            PhaseQNodeOperation("h", (target,)),
            PhaseQNodeOperation("ccz", (control_a, control_b, target)),
            PhaseQNodeOperation("h", (target,)),
        )
    if parsed.gate == "cswap":
        _require_qubit_width(parsed, 3)
        control, target_a, target_b = parsed.qubits
        return (
            PhaseQNodeOperation("cnot", (target_a, target_b)),
            PhaseQNodeOperation("ccnot", (control, target_b, target_a)),
            PhaseQNodeOperation("cnot", (target_a, target_b)),
        )
    raise ValueError(
        "no registered operation-list decomposition for gate "
        f"{parsed.gate!r}; use registered_phase_qnode_decompositions()"
    )


def build_phase_qnode_template(
    name: str,
    n_qubits: int,
    *,
    n_layers: int = 1,
    entangler: str = "chain",
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
        | None
    ) = None,
) -> PhaseQNodeTemplateSpec:
    """Build a registered multi-qubit template as an executable circuit spec.

    The templates are deterministic local statevector declarations. They do not
    imply hardware execution, dynamic circuits, finite-shot sampling, or native
    framework autodiff-through-simulator support.
    """
    normalized = str(name).strip().lower()
    if normalized not in _REGISTERED_TEMPLATES:
        raise ValueError(
            "unsupported Phase-QNode template; use registered_phase_qnode_templates()"
        )
    width = _as_template_width(n_qubits)
    layers = _as_template_layers(n_layers)
    topology = _as_template_entangler(entangler, width)
    if normalized == "ghz_chain" and topology != "chain":
        raise ValueError("ghz_chain template only supports chain entanglement")
    parsed_observable = _normalise_template_observable(observable, width)
    if normalized == "ghz_chain":
        operations = _ghz_chain_operations(width)
        parameter_count = 0
        effective_layers = 1
    else:
        rotation_gates = ("ry",) if normalized == "hardware_efficient_ry" else ("ry", "rz")
        operations, parameter_count = _hardware_efficient_operations(
            width,
            layers,
            topology,
            rotation_gates,
        )
        effective_layers = layers
    return PhaseQNodeTemplateSpec(
        name=normalized,
        n_qubits=width,
        n_layers=effective_layers,
        entangler=topology,
        parameter_count=parameter_count,
        operations=operations,
        observable=parsed_observable,
        claim_boundary=(
            "registered local multi-qubit Phase-QNode template over the bounded "
            "statevector gate family; no dynamic-circuit, provider, finite-shot, "
            "hardware, or native framework autodiff-through-simulator claim"
        ),
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


def phase_qnode_computational_basis_fisher_support_report(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
    *,
    min_probability: float = 1e-15,
) -> PhaseQNodeSupportReport:
    """Return support metadata for exact computational-basis Fisher diagnostics."""
    values = _as_parameter_vector(parameters)
    threshold = _as_min_probability(min_probability)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        return _blocked_density_route_support_report(
            circuit,
            values,
            route_name="exact computational-basis Fisher diagnostics",
            alternatives=(
                "use PhaseQNodeCircuit away from zero-probability boundaries",
                "use phase_qnode_quantum_fisher_information for pure-state QFI diagnostics",
                "route noisy, finite-shot, provider, or hardware Fisher estimators through explicit policy records",
            ),
        )
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        return report
    state, _derivatives = _execute_state_and_parameter_derivatives(circuit, values)
    probabilities = np.asarray(np.abs(state) ** 2, dtype=np.float64)
    if np.any(probabilities <= threshold):
        return PhaseQNodeSupportReport(
            supported=False,
            gates=report.gates,
            observable_kind=report.observable_kind,
            differentiable_parameters=report.differentiable_parameters,
            unsupported_gates=report.unsupported_gates,
            unsupported_observables=report.unsupported_observables,
            unsupported_parameters=report.unsupported_parameters,
            failure_reason=(
                "computational-basis Fisher information is singular at a "
                "zero-probability outcome; choose parameters away from the boundary "
                "or use QFI/Fubini-Study diagnostics"
            ),
            alternatives=(
                "increase min_probability only when the boundary is intentionally excluded",
                "use phase_qnode_quantum_fisher_information for pure-state QFI diagnostics",
                "use parameter_shift_phase_qnode_gradient for expectation-value gradients",
            ),
        )
    return report


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


def parameter_shift_phase_qnode_gradient(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeGradientResult:
    """Evaluate the analytic parameter-shift gradient for registered generators."""
    values = _as_parameter_vector(parameters)
    report = phase_qnode_gradient_support_report(circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        raise PhaseQNodeSupportError(report)
    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, values)
    gradient = np.zeros_like(values)
    base_result = execute_phase_qnode_circuit(circuit, values)
    operations_by_parameter = _parametric_operations_by_parameter(circuit)
    for index in report.differentiable_parameters:
        shift_terms = _parameter_shift_terms_for_group(operations_by_parameter[index])
        if isinstance(circuit.observable, PauliCovarianceObservable):
            if len(shift_terms) != 1:
                raise ValueError(
                    "Pauli covariance gradients with repeated logical parameters require "
                    "an explicit product-rule implementation for each frequency term"
                )
            _frequency, shift, coefficient = shift_terms[0]
            plus = values.copy()
            minus = values.copy()
            plus[index] += shift
            minus[index] -= shift
            plus_state = _execute_state(circuit, plus)
            minus_state = _execute_state(circuit, minus)
            gradient[index] = _covariance_product_rule_gradient(
                base_result.state,
                plus_state,
                minus_state,
                circuit.n_qubits,
                circuit.observable,
            ) * (2.0 * coefficient)
        else:
            total = 0.0
            for _frequency, shift, coefficient in shift_terms:
                plus = values.copy()
                minus = values.copy()
                plus[index] += shift
                minus[index] -= shift
                total += coefficient * (
                    execute_phase_qnode_circuit(circuit, plus).value
                    - execute_phase_qnode_circuit(circuit, minus).value
                )
            gradient[index] = total
    return PhaseQNodeGradientResult(
        value=base_result.value,
        gradient=gradient,
        support_report=report,
        parameter_shift_evaluations=plan.parameter_shift_evaluations,
        evaluation_plan=plan,
    )


def phase_qnode_quantum_fisher_information(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
) -> PhaseQNodeMetricTensorResult:
    """Compute the pure-state QFI and Fubini-Study metric for a local QNode."""
    values = _as_parameter_vector(parameters)
    report = phase_qnode_metric_support_report(circuit, values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
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


def phase_qnode_computational_basis_fisher_information(
    circuit: PhaseQNodeCircuit | PhaseQNodeDensityCircuit,
    parameters: ArrayLike,
    *,
    min_probability: float = 1e-15,
    shot_count: int | None = None,
    observed_counts: ArrayLike | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
) -> PhaseQNodeClassicalFisherResult:
    """Compute computational-basis classical Fisher information.

    The returned ``classical_fisher_information`` is always the exact local
    statevector reference. ``shot_count`` adds a multinomial delta-method
    uncertainty model around that reference distribution. ``observed_counts``
    replays a strictly positive raw-count record as a plug-in finite-shot
    Fisher estimate while preserving the exact analytic reference matrix.
    """
    values = _as_parameter_vector(parameters)
    threshold = _as_min_probability(min_probability)
    shots = _as_shot_count(shot_count)
    confidence = _as_confidence_level(confidence_level)
    z_value = _as_confidence_z(confidence_z)
    report = phase_qnode_computational_basis_fisher_support_report(
        circuit,
        values,
        min_probability=threshold,
    )
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    if isinstance(circuit, PhaseQNodeDensityCircuit):
        raise PhaseQNodeSupportError(report)
    state, derivatives = _execute_state_and_parameter_derivatives(circuit, values)
    probabilities = np.asarray(np.abs(state) ** 2, dtype=np.float64)
    probability_derivatives = np.asarray(
        [2.0 * np.real(np.conj(state) * derivative) for derivative in derivatives],
        dtype=np.float64,
    )
    fisher = _classical_fisher_from_probabilities(
        probability_derivatives,
        probabilities,
    )
    finite_shot = _finite_shot_fisher_evidence(
        probabilities,
        probability_derivatives,
        shots,
        observed_counts,
        confidence,
        z_value,
    )
    claim_boundary = (
        "exact classical Fisher information for computational-basis "
        "probabilities from the registered local statevector Phase-QNode "
        "family; no finite-shot estimator, hardware sampling, adaptive "
        "measurement, or optimal-measurement claim"
    )
    if finite_shot.sampling_model is not None:
        claim_boundary = (
            "exact computational-basis classical Fisher reference plus "
            "finite-shot multinomial uncertainty/replay evidence for the "
            "registered local statevector Phase-QNode family; no hardware "
            "submission, backend calibration, adaptive measurement, "
            "optimal-measurement, or provider-runtime claim"
        )
    return PhaseQNodeClassicalFisherResult(
        classical_fisher_information=fisher,
        probabilities=probabilities,
        probability_derivatives=probability_derivatives,
        measurement="computational_basis",
        min_probability=threshold,
        support_report=report,
        claim_boundary=claim_boundary,
        shot_count=finite_shot.shot_count,
        count_record=finite_shot.count_record,
        empirical_probabilities=finite_shot.empirical_probabilities,
        finite_shot_classical_fisher_information=(
            finite_shot.finite_shot_classical_fisher_information
        ),
        fisher_standard_error=finite_shot.fisher_standard_error,
        fisher_confidence_radius=finite_shot.fisher_confidence_radius,
        confidence_level=finite_shot.confidence_level,
        confidence_z=finite_shot.confidence_z,
        sampling_model=finite_shot.sampling_model,
    )


def phase_qnode_natural_gradient_metric(
    circuit: PhaseQNodeCircuit,
) -> Callable[[FloatArray], FloatArray]:
    """Return a metric provider for quantum natural-gradient optimisation."""

    def metric(parameters: FloatArray) -> FloatArray:
        result = phase_qnode_quantum_fisher_information(circuit, parameters)
        metric_copy: Any = result.fubini_study_metric.copy()
        return cast(FloatArray, metric_copy)

    return metric


def _parsed_operations(circuit: PhaseQNodeCircuit) -> tuple[PhaseQNodeOperation, ...]:
    return tuple(_parse_operation(operation) for operation in circuit.operations)


def _parametric_operations_by_parameter(
    circuit: PhaseQNodeCircuit,
) -> dict[int, tuple[PhaseQNodeOperation, ...]]:
    grouped: dict[int, list[PhaseQNodeOperation]] = {}
    for operation in _parsed_operations(circuit):
        if operation.gate in _PARAMETRIC_GATES and operation.parameter_index is not None:
            grouped.setdefault(operation.parameter_index, []).append(operation)
    return {index: tuple(operations) for index, operations in grouped.items()}


def _parsed_density_operations(circuit: PhaseQNodeDensityCircuit) -> tuple[DensityOperation, ...]:
    return cast(tuple[DensityOperation, ...], circuit.operations)


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


def _as_template_width(value: int) -> int:
    if isinstance(value, bool) or value < 2:
        raise ValueError("Phase-QNode templates require at least two qubits")
    return int(value)


def _as_ising_chain_width(value: int) -> int:
    if isinstance(value, bool) or value < 2:
        raise ValueError("sparse Ising chain Hamiltonians require at least two qubits")
    return int(value)


def _as_template_layers(value: int) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError("n_layers must be a positive integer")
    return int(value)


def _as_template_entangler(value: str, n_qubits: int) -> str:
    topology = str(value).strip().lower()
    if topology not in {"chain", "ring"}:
        raise ValueError("entangler must be 'chain' or 'ring'")
    if topology == "ring" and n_qubits < 3:
        raise ValueError("ring entanglement requires at least three qubits")
    return topology


def _normalise_template_observable(
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
        | None
    ),
    n_qubits: int,
) -> PauliTerm | SparsePauliHamiltonian | PauliCovarianceObservable | DenseHermitianObservable:
    if observable is None:
        return _z_magnetization_observable(n_qubits)
    if isinstance(observable, str):
        normalized = observable.strip().lower()
        if normalized in {"z_magnetization", "pauli_z_magnetization"}:
            return _z_magnetization_observable(n_qubits)
        if normalized in {"z_parity", "pauli_z_parity"}:
            return PauliTerm(1.0, tuple((qubit, "z") for qubit in range(n_qubits)))
    parsed = _normalise_observable(observable, n_qubits)
    if isinstance(parsed, str):
        raise ValueError("template observable strings must be z_magnetization or z_parity")
    return parsed


def _z_magnetization_observable(n_qubits: int) -> SparsePauliHamiltonian:
    weight = 1.0 / float(n_qubits)
    return SparsePauliHamiltonian(
        tuple(PauliTerm(weight, ((qubit, "z"),)) for qubit in range(n_qubits))
    )


def _ghz_chain_operations(n_qubits: int) -> tuple[PhaseQNodeOperation, ...]:
    operations = [PhaseQNodeOperation("h", (0,))]
    operations.extend(
        PhaseQNodeOperation("cnot", (control, control + 1)) for control in range(n_qubits - 1)
    )
    return tuple(operations)


def _hardware_efficient_operations(
    n_qubits: int,
    n_layers: int,
    entangler: str,
    rotation_gates: tuple[str, ...],
) -> tuple[tuple[PhaseQNodeOperation, ...], int]:
    operations: list[PhaseQNodeOperation] = []
    parameter_index = 0
    for _layer in range(n_layers):
        for gate in rotation_gates:
            for qubit in range(n_qubits):
                operations.append(PhaseQNodeOperation(gate, (qubit,), parameter_index))
                parameter_index += 1
        operations.extend(_entangler_operations(n_qubits, entangler))
    return tuple(operations), parameter_index


def _entangler_operations(n_qubits: int, entangler: str) -> tuple[PhaseQNodeOperation, ...]:
    pairs = [(control, control + 1) for control in range(n_qubits - 1)]
    if entangler == "ring":
        pairs.append((n_qubits - 1, 0))
    return tuple(PhaseQNodeOperation("cnot", pair) for pair in pairs)


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


def _as_broadcast_coefficients(name: str, value: object, width: int) -> tuple[float, ...]:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must contain finite real numeric values")
    values = np.asarray(value, dtype=np.float64)
    if values.shape == ():
        scalar = float(values.item())
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must contain finite real numeric values")
        return tuple(scalar for _ in range(width))
    if values.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    if values.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return tuple(float(item) for item in values)


def _as_min_probability(value: float) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError("min_probability must be a non-negative finite scalar")
    scalar = float(raw.item())
    if scalar < 0.0 or not np.isfinite(scalar):
        raise ValueError("min_probability must be a non-negative finite scalar")
    return scalar


def _as_shot_count(value: int | None) -> int | None:
    """Return a validated optional positive shot count."""
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("shot_count must be a positive integer")
    count = int(raw.item())
    if count < 1:
        raise ValueError("shot_count must be a positive integer")
    return count


def _as_confidence_level(value: float) -> float:
    """Return a validated open-interval confidence level."""
    confidence = _as_finite_scalar("confidence_level", value)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    return confidence


def _as_confidence_z(value: float) -> float:
    """Return a validated positive normal-approximation multiplier."""
    z_value = _as_finite_scalar("confidence_z", value)
    if z_value <= 0.0:
        raise ValueError("confidence_z must be finite and positive")
    return z_value


def _as_observed_count_record(
    observed_counts: ArrayLike,
    width: int,
    shot_count: int | None,
) -> tuple[tuple[int, ...], int]:
    """Return a strict positive raw-count record and its total shots."""
    raw = np.asarray(observed_counts)
    if raw.shape != (width,):
        raise ValueError(f"observed_counts must have shape ({width},), got {raw.shape}")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("observed_counts must be integer counts")
    counts = np.asarray(raw, dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError("observed_counts must be strictly positive for finite-shot Fisher replay")
    total = int(np.sum(counts, dtype=np.int64))
    if shot_count is not None and total != shot_count:
        raise ValueError("observed_counts sum must equal shot_count")
    return tuple(int(item) for item in counts.tolist()), total


def _classical_fisher_from_probabilities(
    probability_derivatives: FloatArray,
    probabilities: FloatArray,
) -> FloatArray:
    """Return the computational-basis Fisher matrix for probability derivatives."""
    weighted = probability_derivatives / probabilities[np.newaxis, :]
    fisher = np.asarray(probability_derivatives @ weighted.T, dtype=np.float64)
    return np.asarray(0.5 * (fisher + fisher.T), dtype=np.float64)


def _classical_fisher_delta_method_standard_error(
    probability_derivatives: FloatArray,
    probabilities: FloatArray,
    shot_count: int,
) -> FloatArray:
    """Return multinomial delta-method standard errors for Fisher entries."""
    width = probability_derivatives.shape[0]
    standard_error = np.zeros((width, width), dtype=np.float64)
    for row in range(width):
        for column in range(row, width):
            weights = probability_derivatives[row] * probability_derivatives[column]
            sensitivity = -weights / np.square(probabilities)
            mean = float(np.dot(probabilities, sensitivity))
            second_moment = float(np.dot(probabilities, sensitivity * sensitivity))
            variance = max(0.0, (second_moment - mean * mean) / float(shot_count))
            value = float(np.sqrt(variance))
            standard_error[row, column] = value
            standard_error[column, row] = value
    return standard_error


def _finite_shot_fisher_evidence(
    probabilities: FloatArray,
    probability_derivatives: FloatArray,
    shot_count: int | None,
    observed_counts: ArrayLike | None,
    confidence_level: float,
    confidence_z: float,
) -> _FiniteShotFisherEvidence:
    """Return optional finite-shot Fisher uncertainty and replay evidence."""
    if observed_counts is None and shot_count is None:
        return _FiniteShotFisherEvidence(
            shot_count=None,
            count_record=None,
            empirical_probabilities=None,
            finite_shot_classical_fisher_information=None,
            fisher_standard_error=None,
            fisher_confidence_radius=None,
            confidence_level=None,
            confidence_z=None,
            sampling_model=None,
        )
    count_record: tuple[int, ...] | None = None
    effective_shots = shot_count
    if observed_counts is not None:
        count_record, effective_shots = _as_observed_count_record(
            observed_counts,
            probabilities.size,
            shot_count,
        )
        counts = np.asarray(count_record, dtype=np.float64)
        empirical_probabilities = np.asarray(counts / float(effective_shots), dtype=np.float64)
        sampling_model = "multinomial_delta_method_raw_count_replay"
    else:
        if effective_shots is None:
            raise ValueError("shot_count is required when observed_counts are not supplied")
        empirical_probabilities = np.asarray(probabilities.copy(), dtype=np.float64)
        sampling_model = "multinomial_delta_method_expected_counts"
    finite_shot_fisher = _classical_fisher_from_probabilities(
        probability_derivatives,
        empirical_probabilities,
    )
    standard_error = _classical_fisher_delta_method_standard_error(
        probability_derivatives,
        empirical_probabilities,
        effective_shots,
    )
    confidence_radius = np.asarray(confidence_z * standard_error, dtype=np.float64)
    return _FiniteShotFisherEvidence(
        shot_count=effective_shots,
        count_record=count_record,
        empirical_probabilities=empirical_probabilities,
        finite_shot_classical_fisher_information=finite_shot_fisher,
        fisher_standard_error=standard_error,
        fisher_confidence_radius=confidence_radius,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
        sampling_model=sampling_model,
    )


def _as_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _require_qubit_width(operation: PhaseQNodeOperation, width: int) -> None:
    if len(operation.qubits) != width:
        raise ValueError(f"{operation.gate} decomposition expects {width} qubits")


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
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PauliTerm",
    "PhaseQNodeClassicalFisherResult",
    "PhaseQNodeCircuit",
    "PhaseQNodeDepthProfile",
    "PhaseQNodeDensityCircuit",
    "PhaseQNodeDensityExecutionResult",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeGradientEvaluationGroup",
    "PhaseQNodeGradientEvaluationPlan",
    "PhaseQNodeGradientResult",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeNoiseChannel",
    "PhaseQNodeOperation",
    "PhaseQNodeRegisteredCircuitSpec",
    "PhaseQNodeSupportError",
    "PhaseQNodeSupportReport",
    "PhaseQNodeTemplateSpec",
    "SparsePauliHamiltonian",
    "build_sparse_ising_chain_hamiltonian",
    "build_registered_phase_qnode_circuit",
    "build_phase_qnode_template",
    "decompose_phase_qnode_controlled_gate",
    "execute_phase_qnode_circuit",
    "execute_phase_qnode_density_matrix",
    "parameter_shift_phase_qnode_gradient",
    "plan_phase_qnode_parameter_shift_evaluations",
    "phase_qnode_computational_basis_fisher_information",
    "phase_qnode_computational_basis_fisher_support_report",
    "phase_qnode_density_support_report",
    "phase_qnode_depth_profile",
    "phase_qnode_gradient_support_report",
    "phase_qnode_metric_support_report",
    "phase_qnode_natural_gradient_metric",
    "phase_qnode_quantum_fisher_information",
    "phase_qnode_support_report",
    "registered_phase_qnode_gates",
    "registered_phase_qnode_observables",
    "registered_phase_qnode_decompositions",
    "registered_phase_qnode_noise_channels",
    "registered_phase_qnode_templates",
]
