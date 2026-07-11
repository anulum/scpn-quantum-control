# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Local Gradients
"""Qiskit shifted-circuit generation and local gradient execution.

This one-way leaf owns deterministic Statevector parameter-shift gradients and
finite-shot surrogate uncertainty. It contains no Runtime capture, provider
evidence assembly, hardware preparation, maturity, benchmark, or publication
orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector

from ..differentiable import ParameterShiftRule
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
)
from .qiskit_bridge_contracts import (
    FloatArray,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    _as_finite_scalar,
    _as_finite_vector,
    _as_positive_scalar,
    _normalise_shots,
)


def generate_qiskit_parameter_shift_circuits(
    circuit: QuantumCircuit,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> tuple[QiskitParameterShiftRecord, ...]:
    """Generate fully bound Qiskit plus/minus circuits for parameter shift."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    terms = _parameter_shift_terms(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)

    records: list[QiskitParameterShiftRecord] = []
    for index, parameter in enumerate(parameter_tuple):
        for shift_index, (shift_value, coefficient) in enumerate(terms):
            plus_values = values_vector.copy()
            minus_values = values_vector.copy()
            plus_values[index] += shift_value
            minus_values[index] -= shift_value
            records.append(
                QiskitParameterShiftRecord(
                    parameter_index=index,
                    shift_index=shift_index,
                    shift=shift_value,
                    coefficient=coefficient,
                    parameter_name=parameter.name,
                    plus_values=plus_values,
                    minus_values=minus_values,
                    plus_circuit=_bind_circuit(circuit, parameter_tuple, plus_values),
                    minus_circuit=_bind_circuit(circuit, parameter_tuple, minus_values),
                )
            )
    return tuple(records)


def execute_qiskit_statevector_parameter_shift(
    circuit: QuantumCircuit,
    observable: object,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> QiskitParameterShiftGradientResult:
    """Evaluate a local Qiskit Statevector value and parameter-shift gradient."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    terms = _parameter_shift_terms(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)
    base_circuit = _bind_circuit(circuit, parameter_tuple, values_vector)
    value = _expectation(base_circuit, observable)
    records = generate_qiskit_parameter_shift_circuits(
        circuit,
        parameter_tuple,
        values_vector,
        rule=rule,
        shift=shift,
    )
    gradient = np.zeros(values_vector.size, dtype=np.float64)
    for record in records:
        gradient[record.parameter_index] += record.coefficient * (
            _expectation(record.plus_circuit, observable)
            - _expectation(record.minus_circuit, observable)
        )
    return QiskitParameterShiftGradientResult(
        value=value,
        gradient=gradient,
        records=records,
        method="qiskit_statevector_parameter_shift"
        if len(terms) == 1
        else "qiskit_statevector_multi_frequency_parameter_shift",
        evaluations=1 + 2 * len(records),
        claim_boundary=(
            "local Qiskit Statevector parameter-shift execution for fully bound circuits; "
            "not hardware execution, provider submission, or finite-shot evidence"
        ),
    )


def execute_qiskit_finite_shot_parameter_shift(
    circuit: QuantumCircuit,
    observable: object,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    shots: int,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
) -> ProviderGradientExecutionResult:
    """Evaluate Qiskit shifted circuits through the provider-safe finite-shot contract."""
    shot_count = _normalise_shots(shots)
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    _validate_circuit_parameters(circuit, parameter_tuple)

    def sampler(shifted_values: FloatArray, sample_shots: int | None) -> ProviderExpectationSample:
        bound = _bind_circuit(circuit, parameter_tuple, shifted_values)
        value, variance = _expectation_and_variance(bound, observable)
        shot_batch_id = "qiskit-statevector-surrogate:" + ",".join(
            f"{float(entry):.17g}" for entry in shifted_values
        )
        return ProviderExpectationSample(
            value=value,
            variance=variance,
            shots=sample_shots,
            metadata={
                "engine": "qiskit_statevector_finite_shot_surrogate",
                "observable_type": type(observable).__name__,
                "sample_seed": "deterministic-statevector-surrogate",
                "shot_batch_id": shot_batch_id,
                "source_class": "local_simulator",
            },
        )

    return execute_provider_parameter_shift_gradient(
        sampler,
        values_vector,
        backend="finite_shot_simulator",
        shots=shot_count,
        rule=rule,
        shift=shift,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
    )


def _expectation(circuit: QuantumCircuit, observable: object) -> float:
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(observable)
    return _as_finite_scalar("Qiskit expectation value", np.real(value))


def _expectation_and_variance(circuit: QuantumCircuit, observable: object) -> tuple[float, float]:
    state = Statevector.from_instruction(circuit)
    value = _as_finite_scalar(
        "Qiskit expectation value",
        np.real(state.expectation_value(observable)),
    )
    matrix = Operator(observable).data
    vector = np.asarray(state.data, dtype=np.complex128)
    second_moment = _as_finite_scalar(
        "Qiskit expectation second moment",
        np.real(np.vdot(matrix @ vector, matrix @ vector)),
    )
    variance = max(0.0, second_moment - value * value)
    return value, variance


def _bind_circuit(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
    values: FloatArray,
) -> QuantumCircuit:
    mapping = {parameter: float(values[index]) for index, parameter in enumerate(parameters)}
    bound = circuit.assign_parameters(mapping, inplace=False)
    if bound.num_parameters != 0:
        raise ValueError("all circuit parameters must be supplied before Qiskit execution")
    return bound


def _validate_circuit_parameters(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
) -> None:
    supplied = set(parameters)
    circuit_parameters = set(circuit.parameters)
    if not circuit_parameters:
        raise ValueError("circuit must contain at least one trainable parameter")
    if not circuit_parameters.issubset(supplied):
        raise ValueError("all circuit parameters must be listed in parameters")


def _normalise_parameters(parameters: Sequence[Parameter]) -> tuple[Parameter, ...]:
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("parameters must contain at least one Qiskit Parameter")
    if len(set(parameter_tuple)) != len(parameter_tuple):
        raise ValueError("parameters must not contain duplicates")
    for parameter in parameter_tuple:
        if not isinstance(parameter, Parameter):
            raise ValueError("parameters must contain Qiskit Parameter objects")
    return parameter_tuple


def _parameter_shift_terms(
    rule: ParameterShiftRule | None,
    shift: float,
) -> tuple[tuple[float, float], ...]:
    if rule is not None:
        return tuple(
            (
                _as_positive_scalar("rule.shift", shift_value),
                _as_finite_scalar("rule.coefficient", coefficient),
            )
            for shift_value, coefficient in rule.terms
        )
    shift_value = _as_positive_scalar("shift", shift)
    denominator = 2.0 * np.sin(shift_value)
    if abs(denominator) <= 1.0e-15:
        raise ValueError("shift must not make the parameter-shift denominator singular")
    return ((shift_value, float(1.0 / denominator)),)


__all__ = [
    "generate_qiskit_parameter_shift_circuits",
    "execute_qiskit_statevector_parameter_shift",
    "execute_qiskit_finite_shot_parameter_shift",
]
