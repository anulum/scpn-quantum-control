# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Qiskit Bridge
"""Qiskit parameter-shift circuit generation and local gradient execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector

from ..differentiable import ParameterShiftRule
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class QiskitParameterShiftRecord:
    """One Qiskit plus/minus shifted-circuit pair."""

    parameter_index: int
    shift_index: int
    shift: float
    coefficient: float
    parameter_name: str
    plus_values: FloatArray
    minus_values: FloatArray
    plus_circuit: QuantumCircuit
    minus_circuit: QuantumCircuit

    def __post_init__(self) -> None:
        if isinstance(self.parameter_index, bool) or self.parameter_index < 0:
            raise ValueError("parameter_index must be a non-negative integer")
        if isinstance(self.shift_index, bool) or self.shift_index < 0:
            raise ValueError("shift_index must be a non-negative integer")
        shift = _as_positive_scalar("shift", self.shift)
        coefficient = _as_finite_scalar("coefficient", self.coefficient)
        if not self.parameter_name:
            raise ValueError("parameter_name must be non-empty")
        plus_values = _as_finite_vector("plus_values", self.plus_values)
        minus_values = _as_finite_vector("minus_values", self.minus_values)
        if plus_values.shape != minus_values.shape:
            raise ValueError("plus_values and minus_values must have matching shapes")
        if self.plus_circuit.num_parameters != 0 or self.minus_circuit.num_parameters != 0:
            raise ValueError("shifted Qiskit circuits must be fully bound")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "plus_values", plus_values)
        object.__setattr__(self, "minus_values", minus_values)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible shifted-circuit metadata."""
        return {
            "parameter_index": self.parameter_index,
            "shift_index": self.shift_index,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "parameter_name": self.parameter_name,
            "plus_values": self.plus_values.tolist(),
            "minus_values": self.minus_values.tolist(),
            "plus_depth": self.plus_circuit.depth(),
            "minus_depth": self.minus_circuit.depth(),
            "plus_size": self.plus_circuit.size(),
            "minus_size": self.minus_circuit.size(),
        }


@dataclass(frozen=True)
class QiskitParameterShiftGradientResult:
    """Local Qiskit Statevector parameter-shift gradient result."""

    value: float
    gradient: FloatArray
    records: tuple[QiskitParameterShiftRecord, ...]
    method: str
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        value = _as_finite_scalar("value", self.value)
        gradient = _as_finite_vector("gradient", self.gradient)
        if self.evaluations <= 0:
            raise ValueError("evaluations must be positive")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit gradient metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "records": [record.to_dict() for record in self.records],
            "method": self.method,
            "evaluations": self.evaluations,
            "claim_boundary": self.claim_boundary,
        }


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
        return ProviderExpectationSample(
            value=value,
            variance=variance,
            shots=sample_shots,
            metadata={
                "engine": "qiskit_statevector_finite_shot_surrogate",
                "observable_type": type(observable).__name__,
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


def _as_finite_vector(
    name: str,
    value: ArrayLike,
    *,
    width: int | None = None,
) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_positive_scalar(name: str, value: object) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _normalise_shots(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("shots must be a positive integer")
    return value


__all__ = [
    "QiskitParameterShiftGradientResult",
    "QiskitParameterShiftRecord",
    "execute_qiskit_finite_shot_parameter_shift",
    "execute_qiskit_statevector_parameter_shift",
    "generate_qiskit_parameter_shift_circuits",
]
